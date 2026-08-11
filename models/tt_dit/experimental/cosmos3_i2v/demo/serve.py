# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent HTTP server for interactive Cosmos3 generation (I2V / T2V / T2I).

Builds the pipeline once at startup, then serves many generations off the warm
trunk — the trace is captured on the first run for the launch shape, so every
later gen at that shape skips the ~30s cold step-0. The device is single-tenant:
one background worker owns the mesh and drains a queue, so requests enqueue and
poll rather than blocking an HTTP thread on the device.

    GET  /              → single-page UI (mode, prompt, seed, steps, image upload)
    POST /generate      → multipart or urlencoded form → {"job_id": ...}
    GET  /status/<id>   → job JSON: status, phase, step X/Y, elapsed, media URL
    GET  /jobs          → JSON list of all jobs
    GET  /media/<name>  → MP4/PNG (HTTP Range supported)

Every gen routes through `run_cosmos3`, so the mode follows the conditioning:
an uploaded image → image2video; no image → text2video; frames==1 → text2image.
Overriding height/width/frames away from the launch shape re-traces (cold step 0);
prompt/seed/steps/image stay warm.

Launch via ttq so the broker holds the device lock for the server's lifetime:

    ttq -H g03blx04 run-bg -t 86400 "cd ~/tt-metal && \\
      TT_DIT_CACHE_DIR=~/.tt-dit-cache TT_COSMOS3_SDPA_HIFI2=1 \\
      TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
      python -m models.tt_dit.experimental.cosmos3_i2v.demo.serve \\
        --image ~/ref.jpg --port 8080 --mesh-shape 4x8 \\
        --frames 189 --height 720 --width 1280 --steps 35"

    ssh -N -L 8080:localhost:8080 g03blx04     # from your Mac
    open http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_default
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cosmos3 interactive HTTP generator")
    p.add_argument("--image", type=Path, default=None, help="Default reference image for image2video.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--out-dir", type=Path, default=Path.home() / "cosmos3_serve")
    p.add_argument("--steps", type=int, default=35, help="UniPC denoise steps (NVIDIA I2V default).")
    p.add_argument("--frames", type=int, default=189, help="Frame count; must be 4k+1 for video.")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--mesh-shape", default="auto")
    p.add_argument("--weight-dtype", default="auto", choices=["auto", "bfloat16", "bfloat8_b"])
    p.add_argument("--pipeline", default="native-cfg", choices=["native", "native-cfg"])
    p.add_argument("--num-links", type=int, default=None)
    p.add_argument("--flow-shift", type=float, default=6.0)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--vae-encoder-t-chunk-size", type=int, default=4)
    # Full-T decode (None/0) OOMs at 720p/189f (bank_manager alloc); 1 fits and is fastest.
    p.add_argument("--vae-decoder-t-chunk-size", type=int, default=1)
    p.add_argument("--cfg-serial-dispatch", action="store_true")
    return p.parse_args(argv)


class Job:
    """Mutable generation record shared between the worker and status readers."""

    def __init__(self, job_id: str, params: dict) -> None:
        self.id = job_id
        self.params = params
        self.status = "queued"  # queued → running → done | error
        self.phase = "queued"
        self.step = 0
        self.total_steps = int(params["steps"])
        self.error: str | None = None
        self.media: str | None = None
        self.gen_s: float | None = None
        self.wall_s: float | None = None
        self.t_submit = time.time()
        self.t_start: float | None = None

    def snapshot(self) -> dict:
        elapsed = None
        if self.t_start is not None:
            end = self.t_start + self.wall_s if self.wall_s is not None else time.time()
            elapsed = round(end - self.t_start, 1)
        return {
            "id": self.id,
            "status": self.status,
            "phase": self.phase,
            "step": self.step,
            "total_steps": self.total_steps,
            "elapsed_s": elapsed,
            "gen_s": self.gen_s,
            "wall_s": self.wall_s,
            "media": self.media,
            "error": self.error,
            "prompt": self.params["prompt"],
            "mode": self.params["mode"],
            "seed": self.params["seed"],
        }


class Server:
    """Owns the warm pipeline, the job queue, and the single device worker."""

    def __init__(self, args, mesh, pipe, default_image) -> None:
        self.args = args
        self.mesh = mesh
        self.pipe = pipe
        self.default_image = default_image
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, name="cosmos3-worker", daemon=True)
        self.worker.start()

    def submit(self, params: dict) -> str:
        job = Job(uuid.uuid4().hex[:12], params)
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.queue.put(job.id)
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def all_snapshots(self) -> list[dict]:
        with self.lock:
            return [self.jobs[i].snapshot() for i in reversed(self.order[-50:])]

    def _worker_loop(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._run(job)
            except Exception as e:  # a failed gen must not kill the worker
                job.status = "error"
                job.phase = "error"
                job.error = f"{type(e).__name__}: {e}"
                print(f"[serve] {job.id} FAILED: {job.error}", flush=True)

    def _run(self, job: Job) -> None:
        import torch
        from diffusers.utils import export_to_video

        from models.tt_dit.experimental.cosmos3_i2v.pipelines.cosmos3_mode import run_cosmos3
        from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_native import _make_release_callback

        p = job.params
        job.status = "running"
        job.phase = "denoise"
        job.t_start = time.time()
        release = _make_release_callback(p["steps"])

        def progress(pipe_ref, step, timestep, callback_kwargs):
            job.step = step + 1  # step is 0-indexed; report 1-based to humans
            if step >= p["steps"] - 1:
                job.phase = "vae_decode"
            return release(pipe_ref, step, timestep, callback_kwargs)

        generator = torch.Generator(device="cpu").manual_seed(p["seed"]) if p["seed"] is not None else None
        call_kwargs = dict(
            height=p["height"],
            width=p["width"],
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance_scale"],
            output_type="pil",
            callback_on_step_end=progress,
            callback_on_step_end_tensor_inputs=[],
        )
        if generator is not None:
            call_kwargs["generator"] = generator

        t_gen0 = time.time()
        result = run_cosmos3(
            self.pipe,
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            image=p["image"],
            num_frames=p["frames"],
            **call_kwargs,
        )
        job.gen_s = round(time.time() - t_gen0, 1)

        job.phase = "encoding"
        if p["frames"] == 1:
            out_path = self.args.out_dir / f"{job.id}.png"
            result.video[0].save(out_path)
        else:
            out_path = self.args.out_dir / f"{job.id}.mp4"
            export_to_video(result.video, str(out_path), fps=self.args.fps)

        job.media = f"/media/{out_path.name}"
        job.wall_s = round(time.time() - job.t_start, 1)
        job.status = "done"
        job.phase = "done"
        print(
            f"[serve] {job.id} done mode={p['mode']} wall={job.wall_s}s gen={job.gen_s}s prompt={p['prompt']!r}",
            flush=True,
        )


def _resolve_request(server: Server, fields: dict, image_bytes: bytes | None):
    """Build a generation params dict from a parsed request, deriving the mode."""
    from PIL import Image

    args = server.args
    mode = (fields.get("mode") or "image2video").strip()
    prompt = (fields.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    def _int(name, dflt):
        v = (fields.get(name) or "").strip()
        return int(v) if v else dflt

    def _float(name, dflt):
        v = (fields.get(name) or "").strip()
        return float(v) if v else dflt

    frames = 1 if mode == "text2image" else _int("frames", args.frames)
    if mode != "text2image" and (frames - 1) % 4 != 0:
        raise ValueError(f"frames must be 4k+1 for video (got {frames})")

    width = _int("width", args.width)
    height = _int("height", args.height)

    image = None
    if mode == "image2video":
        from io import BytesIO

        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((width, height))
        elif server.default_image is not None:
            image = server.default_image.resize((width, height))
        else:
            raise ValueError("image2video needs an uploaded image or a launch --image")

    seed_str = (fields.get("seed") or "").strip()
    negative = (fields.get("negative_prompt") or "").strip() or None

    return {
        "mode": mode,
        "prompt": prompt,
        "negative_prompt": negative,
        "image": image,
        "frames": frames,
        "width": width,
        "height": height,
        "steps": _int("steps", args.steps),
        "guidance_scale": _float("guidance_scale", args.guidance_scale),
        "seed": int(seed_str) if seed_str else None,
    }


def build(args):
    import torch
    from PIL import Image

    import ttnn
    from models.tt_dit.experimental.cosmos3_i2v.demo.generate import (
        close_mesh,
        open_mesh,
        resolve_mesh_shape,
        resolve_weight_dtype,
    )

    available = ttnn.get_num_devices()
    mesh_shape = resolve_mesh_shape(args.mesh_shape, available)
    weight_dtype = resolve_weight_dtype(args.weight_dtype, mesh_shape)
    print(f"[serve] mesh={mesh_shape}/{available} weight_dtype={weight_dtype} pipeline={args.pipeline}", flush=True)

    default_image = None
    if args.image is not None:
        if not args.image.exists():
            raise SystemExit(f"image not found: {args.image}")
        default_image = Image.open(args.image).convert("RGB")

    mesh = open_mesh(mesh_shape)
    try:
        t0 = time.time()
        if args.pipeline == "native":
            from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_native import (
                build_cosmos3_i2v_native_pipeline,
            )

            pipe = build_cosmos3_i2v_native_pipeline(
                mesh,
                dtype=torch.bfloat16,
                use_tt_vae=True,
                num_links=args.num_links,
                flow_shift=args.flow_shift,
                trunk_weight_dtype=weight_dtype,
                vae_decoder_t_chunk_size=args.vae_decoder_t_chunk_size,
                vae_encoder_t_chunk_size=args.vae_encoder_t_chunk_size,
            )
        else:
            from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_native_cfg import (
                build_cosmos3_i2v_native_cfg_pipeline,
            )

            pipe = build_cosmos3_i2v_native_cfg_pipeline(
                mesh,
                dtype=torch.bfloat16,
                use_tt_vae=True,
                num_links=args.num_links,
                flow_shift=args.flow_shift,
                trunk_weight_dtype=weight_dtype,
                vae_decoder_t_chunk_size=args.vae_decoder_t_chunk_size,
                vae_encoder_t_chunk_size=args.vae_encoder_t_chunk_size,
                serial_dispatch=args.cfg_serial_dispatch,
            )
        print(f"[serve] pipeline built in {time.time() - t0:.1f}s", flush=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        return mesh, pipe, default_image, close_mesh
    except Exception:
        close_mesh(mesh)
        raise


_INDEX_HTML = """<!doctype html><html><head><title>Cosmos3</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:820px;margin:28px auto;padding:0 16px;color:#1a1a1a}
label{font-size:14px;font-weight:600;display:block;margin:12px 0 4px}
textarea,input,select{width:100%;font-size:14px;padding:7px;box-sizing:border-box}
textarea{height:80px}
.row{display:flex;gap:12px}.row>div{flex:1}
button{margin-top:14px;padding:10px 20px;font-size:15px;cursor:pointer}
.jobs{margin-top:28px;border-top:1px solid #ccc;padding-top:12px}
.job{padding:8px 0;border-bottom:1px solid #eee;font-size:14px}
.t{color:#666;font-size:12px}
.bar{height:5px;background:#eee;border-radius:3px;overflow:hidden;margin-top:4px}
.fill{height:100%;background:#2a7;width:0}
</style></head><body>
<h2>Cosmos3 generator</h2>
<form id="f">
<label>Mode</label>
<select name="mode" id="mode">
<option value="image2video">image → video</option>
<option value="text2video">text → video</option>
<option value="text2image">text → image</option>
</select>
<label>Prompt</label><textarea name="prompt" required autofocus></textarea>
<label>Negative prompt (optional — blank = mode default)</label><textarea name="negative_prompt"></textarea>
<div id="imgrow"><label>Reference image (image→video; blank = launch default)</label>
<input type="file" name="image" accept="image/*"></div>
<div class="row">
<div><label>Seed</label><input name="seed" placeholder="random"></div>
<div><label>Steps</label><input name="steps" value="__STEPS__"></div>
<div><label>Frames</label><input name="frames" id="frames" value="__FRAMES__"></div>
</div>
<div class="row">
<div><label>Width</label><input name="width" value="__WIDTH__"></div>
<div><label>Height</label><input name="height" value="__HEIGHT__"></div>
</div>
<button type="submit">Generate</button>
</form>
<p class="t">warm shape: __WIDTH__×__HEIGHT__ · __FRAMES__f — changing size/frames re-traces (slow first run)</p>
<div class="jobs"><h3>Jobs</h3><div id="jobs">(loading…)</div></div>
<script>
const mode=document.getElementById('mode'), imgrow=document.getElementById('imgrow'), frames=document.getElementById('frames');
mode.onchange=()=>{imgrow.style.display=mode.value==='image2video'?'':'none';
  if(mode.value==='text2image'){frames.value='1';frames.disabled=true}else{frames.disabled=false;if(frames.value==='1')frames.value='__FRAMES__'}};
document.getElementById('f').onsubmit=async(e)=>{e.preventDefault();
  const r=await fetch('/generate',{method:'POST',body:new FormData(e.target)});
  const j=await r.json(); if(j.error){alert(j.error);return} poll();};
function media(j){if(j.status!=='done'||!j.media)return '';
  return j.media.endsWith('.png')?`<br><img src="${j.media}" style="max-width:320px">`
    :`<br><video src="${j.media}" controls style="max-width:420px"></video>`;}
async function poll(){const r=await fetch('/jobs');const js=await r.json();
  document.getElementById('jobs').innerHTML=js.map(j=>{
    const pct=j.total_steps?Math.round(100*j.step/j.total_steps):0;
    const st=j.status==='running'?`${j.phase} ${j.step}/${j.total_steps}`:j.status;
    const tm=j.wall_s?`wall ${j.wall_s}s`:(j.elapsed_s!=null?`${j.elapsed_s}s`:'');
    return `<div class="job"><b>${j.mode}</b> — ${st} <span class="t">${tm} ${j.error||''}</span>
      <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
      <div class="t">${j.prompt}</div>${media(j)}</div>`;}).join('')||'(none yet)';}
poll();setInterval(poll,1500);
</script>
</body></html>
"""


def make_handler(server: Server):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # noqa: ARG002
            sys.stderr.write(f"[serve] {self.address_string()} - {fmt % a}\n")

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _html(self, body, status=200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _index(self):
            a = server.args
            return (
                _INDEX_HTML.replace("__STEPS__", str(a.steps))
                .replace("__FRAMES__", str(a.frames))
                .replace("__WIDTH__", str(a.width))
                .replace("__HEIGHT__", str(a.height))
            )

        def _serve_media(self, name):
            path = server.args.out_dir / name
            if path.name != name or not path.exists() or path.suffix not in (".mp4", ".png"):
                self.send_error(404)
                return
            ctype = "video/mp4" if path.suffix == ".mp4" else "image/png"
            data = path.read_bytes()
            total = len(data)
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                lo_s, _, hi_s = rng[6:].partition("-")
                lo = int(lo_s) if lo_s else 0
                hi = int(hi_s) if hi_s else total - 1
                hi = min(hi, total - 1)
                chunk = data[lo : hi + 1]
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{total}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(total))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self._html(self._index())
            elif self.path == "/jobs":
                self._json(server.all_snapshots())
            elif self.path.startswith("/status/"):
                job = server.get(self.path[len("/status/") :])
                if job is None:
                    self.send_error(404)
                else:
                    self._json(job.snapshot())
            elif self.path.startswith("/media/"):
                self._serve_media(Path(self.path).name)
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if self.path != "/generate":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "")
            try:
                fields, image_bytes = _parse_form(ctype, body)
                params = _resolve_request(server, fields, image_bytes)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, status=400)
                return
            job_id = server.submit(params)
            self._json({"job_id": job_id})

    return H


def _parse_form(content_type: str, body: bytes) -> tuple[dict, bytes | None]:
    """Parse urlencoded or multipart/form-data into (text fields, image bytes)."""
    if content_type.startswith("multipart/form-data"):
        # Reconstruct a MIME message so the stdlib email parser handles the boundary
        # split; cgi.FieldStorage is gone in 3.13, email.parser is the supported route.
        msg = BytesParser(policy=email_default).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
        )
        fields: dict[str, str] = {}
        image_bytes = None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name is None:
                continue
            if part.get_filename():
                payload = part.get_payload(decode=True)
                if payload:
                    image_bytes = payload
            else:
                fields[name] = part.get_content().strip()
        return fields, image_bytes

    import urllib.parse

    parsed = urllib.parse.parse_qs(body.decode("utf-8"))
    return {k: v[0] for k, v in parsed.items()}, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mesh, pipe, default_image, close_mesh = build(args)
    server = Server(args, mesh, pipe, default_image)

    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        close_mesh(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
