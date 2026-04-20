"""Local Wan 2.2 I2V HTTP backend.

Two backends (switch via WAN_BACKEND_DEFAULT below or `WAN_BACKEND=...` env):

  stub : portable ffmpeg crossfade. No TT hardware required.
  wan  : real WanPipelineI2V via wan_i2v_core, one shared pipeline per process.

Usage:
    uvicorn server.server:app --host 0.0.0.0 --port 8000 \
        --workers 1 --timeout-graceful-shutdown 60

Request shape is the same for both backends (POST /predictions multipart).
See server/README.md for curl examples.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse


# --- Server defaults (edit here; env vars with the same names override) ----
# "stub" = portable ffmpeg crossfade (no TT hardware).
# "wan"  = real WanPipelineI2V on Tenstorrent mesh.
WAN_BACKEND_DEFAULT = "wan"
# Mesh preset from wan_i2v_core.CONFIGS (bh_4x8sp1tp0 = Blackhole Galaxy 4x8).
WAN_SERVER_CONFIG = "bh_4x8sp1tp0"
# Pinned output resolution. Clients can request a different one and the server
# will resize the output via ffmpeg; inference always runs at this size.
WAN_SERVER_RESOLUTION = "720p"
# Pinned frame count. Must satisfy (n - 1) % 4 == 0.
WAN_SERVER_NUM_FRAMES = 81
# fcntl flock /tmp/tt_device.lock at startup. Set False only if you are sure
# no other TT job can run on this machine concurrently.
WAN_SERVER_USE_LOCK = True
# ---------------------------------------------------------------------------

WAN_BACKEND = os.environ.get("WAN_BACKEND", WAN_BACKEND_DEFAULT)

# (height, width) — server wire contract is H, W.
# wan_i2v_core.RESOLUTIONS is (W, H). Do not mix them.
RESOLUTION_MAP = {
    "480p": (480, 832),
    "720p": (720, 1280),
}

INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")

_WAN_STATE: dict = {
    "pipeline": None,
    "mesh_cm": None,
    "infer_lock": None,
    "pinned_height": None,
    "pinned_width": None,
    "pinned_num_frames": None,
    "ready": False,
}


def _log(msg: str) -> None:
    print(f"[server] {msg}", file=sys.stderr, flush=True)


def _banner(lines: list[str]) -> None:
    w = max(len(line) for line in lines) + 4
    print("=" * w, file=sys.stderr, flush=True)
    for line in lines:
        print(f"  {line}", file=sys.stderr, flush=True)
    print("=" * w, file=sys.stderr, flush=True)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if WAN_BACKEND == "wan":
        try:
            from wan_i2v_core import (
                CONFIGS,
                RESOLUTIONS,
                create_pipeline,
                open_mesh,
                round_up_num_frames,
            )
        except ImportError as e:
            raise RuntimeError(
                f"WAN_BACKEND=wan but wan_i2v_core is not importable: {e}. "
                "Run uvicorn from the tt-metal repo root (or set PYTHONPATH)."
            )

        cfg_name = os.environ.get("WAN_SERVER_CONFIG", WAN_SERVER_CONFIG)
        res_name = os.environ.get("WAN_SERVER_RESOLUTION", WAN_SERVER_RESOLUTION)
        num_frames = int(os.environ.get("WAN_SERVER_NUM_FRAMES", WAN_SERVER_NUM_FRAMES))
        use_lock = os.environ.get("WAN_SERVER_USE_LOCK", "1" if WAN_SERVER_USE_LOCK else "0") == "1"

        if cfg_name not in CONFIGS:
            raise RuntimeError(f"unknown WAN_SERVER_CONFIG: {cfg_name!r}")
        if res_name not in RESOLUTIONS:
            raise RuntimeError(f"unknown WAN_SERVER_RESOLUTION: {res_name!r}")
        rounded = round_up_num_frames(num_frames)
        if rounded != num_frames:
            _log(f"WAN_SERVER_NUM_FRAMES={num_frames} does not satisfy " f"(n-1) %% 4 == 0; rounding up to {rounded}")
            num_frames = rounded

        width, height = RESOLUTIONS[res_name]

        _banner(
            [
                "Wan 2.2 I2V server: STARTING",
                f"backend={WAN_BACKEND}  config={cfg_name}  resolution={res_name} ({width}x{height})",
                f"num_frames={num_frames}  device_lock={'on' if use_lock else 'off'}",
                "",
                "This takes several minutes (mesh init + pipeline warmup).",
                "Watch for the 'SERVER IS READY' banner.",
            ]
        )

        def _setup():
            t0 = time.monotonic()
            _log("acquiring mesh device and configuring fabric…")
            cm = open_mesh(cfg_name, use_lock=use_lock)
            mesh_device, cfg = cm.__enter__()
            _log(f"mesh open (t+{time.monotonic()-t0:.1f}s); building pipeline…")
            try:
                pipeline = create_pipeline(
                    mesh_device,
                    cfg,
                    target_height=height,
                    target_width=width,
                    num_frames=num_frames,
                )
            except BaseException:
                cm.__exit__(*sys.exc_info())
                raise
            _log(f"pipeline built (t+{time.monotonic()-t0:.1f}s); warmup complete")
            return cm, pipeline

        cm, pipeline = await asyncio.to_thread(_setup)
        _WAN_STATE["mesh_cm"] = cm
        _WAN_STATE["pipeline"] = pipeline
        _WAN_STATE["infer_lock"] = anyio.Lock()
        _WAN_STATE["pinned_height"] = height
        _WAN_STATE["pinned_width"] = width
        _WAN_STATE["pinned_num_frames"] = num_frames
        _WAN_STATE["ready"] = True

        _banner(
            [
                "SERVER IS READY",
                f"backend={WAN_BACKEND}  pinned={width}x{height}  num_frames={num_frames}",
                "POST /predictions   GET /healthz   GET /files/<name>",
            ]
        )

        try:
            yield
        finally:
            _WAN_STATE["ready"] = False
            _log("shutting down — closing mesh + fabric (can take ~10s)")
            try:
                await asyncio.to_thread(cm.__exit__, None, None, None)
            finally:
                _WAN_STATE["pipeline"] = None
                _WAN_STATE["mesh_cm"] = None
                _WAN_STATE["infer_lock"] = None
            _log("shutdown complete")
    else:
        _banner(
            [
                "SERVER IS READY",
                f"backend={WAN_BACKEND} (stub, no hardware)",
                "POST /predictions   GET /healthz   GET /files/<name>",
            ]
        )
        _WAN_STATE["ready"] = True
        yield
        _WAN_STATE["ready"] = False


app = FastAPI(title="Local Wan 2.2 I2V", version="0.3.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request, exc):
    msg = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors())
    return JSONResponse(
        status_code=400,
        content={"id": None, "status": "failed", "error": msg},
    )


@app.exception_handler(HTTPException)
async def _http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"id": None, "status": "failed", "error": str(exc.detail)},
    )


@app.get("/healthz")
@app.get("/health")
def healthz() -> dict:
    info = {"status": "ok", "backend": WAN_BACKEND, "ready": _WAN_STATE["ready"]}
    if WAN_BACKEND == "wan":
        info["pinned_height"] = _WAN_STATE["pinned_height"]
        info["pinned_width"] = _WAN_STATE["pinned_width"]
        info["pinned_num_frames"] = _WAN_STATE["pinned_num_frames"]
    return info


def _probe_frame_count(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return int(out.decode().strip())


def _stub_generate(
    first_image_path: Path,
    last_image_path: Path | None,
    num_frames: int,
    fps: int,
    height: int,
    width: int,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if last_image_path is None:
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(first_image_path),
            "-vf",
            f"scale={width}:{height},format=yuv420p",
            "-frames:v",
            str(num_frames),
            "-c:v",
            "libx264",
            str(out_path),
        ]
    else:
        duration = num_frames / fps
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            str(duration),
            "-i",
            str(first_image_path),
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            str(duration),
            "-i",
            str(last_image_path),
            "-filter_complex",
            f"[0:v]scale={width}:{height}[a];"
            f"[1:v]scale={width}:{height}[b];"
            f"[a][b]xfade=transition=fade:duration={duration}:offset=0,format=yuv420p",
            "-frames:v",
            str(num_frames),
            "-c:v",
            "libx264",
            str(out_path),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    actual = _probe_frame_count(out_path)
    if actual != num_frames:
        raise RuntimeError(f"stub produced {actual} frames, expected {num_frames}; cmd: {' '.join(cmd)}")
    return out_path


def _run_wan_real(
    pred_id: str,
    prompt: str,
    negative_prompt: str,
    first_image_path: Path,
    last_image_path: Path | None,
    image_prompt_paths: list | None,
    fps: int,
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    guidance_scale_2: Optional[float],
    seed: Optional[int],
    out_path: Path,
    encode_height: int,
    encode_width: int,
) -> Path:
    """Blocking. Runs under infer_lock, inside asyncio.to_thread.

    Always runs inference at the server's pinned dims / num_frames. Any
    client-requested resize is handled by the caller after this returns.
    """
    from wan_i2v_core import generate_video

    pipeline = _WAN_STATE["pipeline"]
    if pipeline is None:
        raise RuntimeError("pipeline not initialized (lifespan did not run)")

    pinned_h = _WAN_STATE["pinned_height"]
    pinned_w = _WAN_STATE["pinned_width"]
    pinned_n = _WAN_STATE["pinned_num_frames"]

    image_prompts = None
    if image_prompt_paths is not None:
        image_prompts = [(p, fp) for p, fp in image_prompt_paths]

    _log(f"[{pred_id}] inference START ({pinned_w}x{pinned_h}, {pinned_n} frames)")
    t0 = time.monotonic()
    result = generate_video(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        first_image=first_image_path if image_prompts is None else None,
        last_image=last_image_path if image_prompts is None else None,
        image_prompts=image_prompts,
        num_frames=pinned_n,
        height=pinned_h,
        width=pinned_w,
        encode_height=encode_height,
        encode_width=encode_width,
        steps=num_inference_steps,
        seed=seed,
        guidance=guidance_scale,
        guidance_2=guidance_scale_2,
        fps=fps,
        out_path=out_path,
    )
    _log(f"[{pred_id}] inference DONE in {time.monotonic()-t0:.1f}s")
    return result


async def run_backend_async(
    *,
    pred_id: str,
    prompt: str,
    negative_prompt: str,
    first_image_path: Path,
    last_image_path: Path | None,
    image_prompt_paths: list | None,
    num_frames: int,
    fps: int,
    height: int,
    width: int,
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    guidance_scale_2: Optional[float],
    seed: Optional[int],
    out_path: Path,
) -> tuple[int, int]:
    """Run inference and return the (height, width) of the file at `out_path`."""
    if WAN_BACKEND == "stub":
        await asyncio.to_thread(
            _stub_generate,
            first_image_path,
            last_image_path,
            num_frames,
            fps,
            height,
            width,
            out_path,
        )
        return height, width

    if WAN_BACKEND == "wan":
        lock = _WAN_STATE["infer_lock"]
        if lock is None:
            raise RuntimeError("pipeline not initialized (lifespan did not run)")

        async with lock:
            await asyncio.to_thread(
                _run_wan_real,
                pred_id,
                prompt,
                negative_prompt,
                first_image_path,
                last_image_path,
                image_prompt_paths,
                fps,
                num_inference_steps,
                guidance_scale,
                guidance_scale_2,
                seed,
                out_path,
                height,
                width,  # encode_height, encode_width (in-process PIL resize)
            )
        return height, width

    raise NotImplementedError(f"backend '{WAN_BACKEND}' not wired up in this checkout")


def _resolve_request_dims(
    resolution: Optional[str],
    height: Optional[int],
    width: Optional[int],
) -> tuple[int, int, Optional[str]]:
    """Resolve what the CLIENT asked for. Returns (h, w, resolved_resolution_label).

    If the client didn't specify anything, fall back to the server's pinned
    dims (for wan) or 480p (for stub).
    """
    if height is not None and width is not None:
        return height, width, resolution
    if resolution is not None:
        if resolution not in RESOLUTION_MAP:
            raise HTTPException(status_code=400, detail=f"invalid resolution: {resolution}")
        h, w = RESOLUTION_MAP[resolution]
        return h, w, resolution
    # Nothing provided: default
    if WAN_BACKEND == "wan" and _WAN_STATE["pinned_height"] is not None:
        return _WAN_STATE["pinned_height"], _WAN_STATE["pinned_width"], None
    # Stub fallback
    return RESOLUTION_MAP["480p"][0], RESOLUTION_MAP["480p"][1], "480p"


def _new_prediction_id() -> str:
    return f"pred_{uuid.uuid4().hex[:12]}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.post("/predictions")
async def create_prediction(request: Request) -> dict:
    form = await request.form()

    def get(name: str, default=None):
        v = form.get(name)
        return v if v is not None else default

    prompt = get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="prompt is required")

    negative_prompt = get("negative_prompt", "") or ""
    fps = int(get("frames_per_second", 16))
    resolution = get("resolution")  # None if not provided
    height = int(form["height"]) if form.get("height") else None
    width = int(form["width"]) if form.get("width") else None
    num_inference_steps = int(form["num_inference_steps"]) if form.get("num_inference_steps") else None
    guidance_scale = float(form["guidance_scale"]) if form.get("guidance_scale") else None
    guidance_scale_2 = float(form["guidance_scale_2"]) if form.get("guidance_scale_2") else None
    seed = int(form["seed"]) if form.get("seed") else None

    # num_frames: default to pinned for wan, 81 for stub.
    if WAN_BACKEND == "wan" and _WAN_STATE["pinned_num_frames"] is not None:
        num_frames_default = _WAN_STATE["pinned_num_frames"]
    else:
        num_frames_default = 81
    num_frames = int(get("num_frames", num_frames_default))

    image = form.get("image")
    last_image = form.get("last_image")
    image_prompts_raw = form.get("image_prompts")

    if image_prompts_raw is not None and (image is not None or last_image is not None):
        raise HTTPException(
            status_code=400,
            detail="cannot combine image/last_image with image_prompts",
        )
    if image_prompts_raw is None and image is None:
        raise HTTPException(status_code=400, detail="image file is required")

    # Resolve dims — no mismatch errors; resize happens server-side if needed.
    h, w, resolved_res = _resolve_request_dims(resolution, height, width)

    if WAN_BACKEND == "wan":
        # Round up to the nearest (n-1)%%4==0 value first (Wan requires
        # num_frames align with VAE temporal latent stride = 4).
        from wan_i2v_core import round_up_num_frames as _round

        original = num_frames
        num_frames = _round(num_frames)
        if num_frames != original:
            _log(f"[handler] rounded requested num_frames {original} -> " f"{num_frames} to satisfy (n-1) %% 4 == 0")
        if num_frames != _WAN_STATE["pinned_num_frames"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"num_frames={num_frames} "
                    f"(rounded from {original}) != server pinned "
                    f"{_WAN_STATE['pinned_num_frames']}. Restart the server with "
                    "WAN_SERVER_NUM_FRAMES set to the value you want."
                ),
            )

    pred_id = _new_prediction_id()
    created_at = _iso_now()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = INPUTS_DIR / pred_id
    in_dir.mkdir(parents=True, exist_ok=True)

    first_path: Path
    last_path: Path | None = None
    image_prompt_paths: list[tuple[Path, int]] | None = None

    if image_prompts_raw is not None:
        import json as _json

        try:
            ip = _json.loads(image_prompts_raw)
        except _json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"invalid image_prompts JSON: {e}")
        if not isinstance(ip, list) or not ip:
            raise HTTPException(status_code=400, detail="image_prompts must be a non-empty list")

        positions = [entry["frame_pos"] for entry in ip]
        if len(set(positions)) != len(positions):
            raise HTTPException(status_code=400, detail="duplicate frame_pos in image_prompts")

        saved_paths: dict[int, Path] = {}
        for entry in ip:
            idx = entry["image_index"]
            upload = form.get(f"image_prompt_{idx}")
            if upload is None:
                raise HTTPException(status_code=400, detail=f"missing file for image_prompt_{idx}")
            p = in_dir / f"ip_{idx}{Path(getattr(upload, 'filename', '') or '.png').suffix or '.png'}"
            with open(p, "wb") as fout:
                fout.write(await upload.read())
            saved_paths[idx] = p

        sorted_entries = sorted(ip, key=lambda e: e["frame_pos"])
        first_path = saved_paths[sorted_entries[0]["image_index"]]
        if len(sorted_entries) > 1:
            last_path = saved_paths[sorted_entries[-1]["image_index"]]
        image_prompt_paths = [(saved_paths[entry["image_index"]], int(entry["frame_pos"])) for entry in ip]
    else:
        first_path = in_dir / f"image{Path(getattr(image, 'filename', '') or 'image.png').suffix or '.png'}"
        with open(first_path, "wb") as fout:
            fout.write(await image.read())
        if last_image is not None:
            last_path = in_dir / f"last{Path(getattr(last_image, 'filename', '') or 'last.png').suffix or '.png'}"
            with open(last_path, "wb") as fout:
                fout.write(await last_image.read())

    out_path = OUTPUTS_DIR / f"{pred_id}.mp4"

    _log(
        f"[{pred_id}] /predictions received: prompt={prompt!r:.80s} "
        f"size={w}x{h} frames={num_frames} fps={fps} "
        f"first={first_path.name} last={last_path.name if last_path else None}"
    )

    try:
        final_h, final_w = await run_backend_async(
            pred_id=pred_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            first_image_path=first_path,
            last_image_path=last_path,
            image_prompt_paths=image_prompt_paths,
            num_frames=num_frames,
            fps=fps,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            seed=seed,
            out_path=out_path,
        )
    except Exception as e:
        _log(f"[{pred_id}] FAILED: {e}")
        return {
            "id": pred_id,
            "status": "failed",
            "error": str(e),
            "created_at": created_at,
            "completed_at": _iso_now(),
        }

    _log(f"[{pred_id}] succeeded -> {out_path.name} ({final_w}x{final_h})")
    base = str(request.base_url).rstrip("/")
    return {
        "id": pred_id,
        "status": "succeeded",
        "output": f"{base}/files/{pred_id}.mp4",
        "urls": {"get": f"{base}/predictions/{pred_id}"},
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames,
            "frames_per_second": fps,
            "resolution": resolved_res,
            "height": final_h,
            "width": final_w,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
            "seed": seed,
        },
        "created_at": created_at,
        "completed_at": _iso_now(),
    }


@app.get("/files/{name}")
def get_file(name: str) -> FileResponse:
    path = OUTPUTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/predictions/{pred_id}")
def get_prediction(pred_id: str, request: Request) -> dict:
    out_file = OUTPUTS_DIR / f"{pred_id}.mp4"
    if not out_file.is_file():
        raise HTTPException(status_code=404, detail=f"prediction {pred_id} not found")
    base = str(request.base_url).rstrip("/")
    return {
        "id": pred_id,
        "status": "succeeded",
        "output": f"{base}/files/{pred_id}.mp4",
        "urls": {"get": f"{base}/predictions/{pred_id}"},
    }
