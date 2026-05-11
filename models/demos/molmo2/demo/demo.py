# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-8B demo: text / image / video generation on T3K.

Mirrors the structure of models/demos/qwen3_vl/demo/demo.py:
  - pytest.mark.parametrize over JSON prompt files
  - model loaded once per test; warm-up prefill on first batch
  - greedy (argmax) or temperature sampling
  - per-iteration throughput logging
  - profiler timers for prefill / compile / decode

Supported prompt JSON format (same as input_prompts argument):
  Text-only:    [[ {"role": "user", "content": "..."} ]]
  Image+text:   [[ {"role": "user", "content": [{"type":"image","image":"path_or_url"}, {"type":"text","text":"..."}]} ]]
  Video+text:   [[ {"role": "user", "content": [{"type":"video","video":"path_or_url"}, {"type":"text","text":"..."}]} ]]

  Each outer list entry is one user / prompt.

Run:
  export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "text_only"
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "image"
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "video"

  # Custom prompt file:
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py --input_prompts my.json -v

CLI overrides (passed via pytest addopts or --flag):
  --input_prompts PATH    JSON file to load
  --max_new_tokens N      tokens to generate (default from parametrize)
  --batch_size N          users per batch
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn

# ---------------------------------------------------------------------------
# Coordinate output helper (pointing / tracking use cases)
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(
    r"<(?:points?|tracks?)[^>]*\s+coords=\"([0-9\t:;,. ]+)\"[^>]*/?>",
    re.IGNORECASE,
)


def _format_response(text: str) -> str:
    """Pretty-print coordinate tags if present; otherwise return text unchanged."""
    coords = _COORD_RE.findall(text)
    if not coords:
        return text
    clean = _COORD_RE.sub("", text).strip()
    coord_lines = "\n  ".join(c.strip() for c in coords)
    prefix = f"{clean}\n" if clean else ""
    return f"{prefix}  [coords] {coord_lines}"


# ---------------------------------------------------------------------------
# Coordinate overlay helpers
# ---------------------------------------------------------------------------

# Radius of dots drawn on annotated images (pixels)
_DOT_RADIUS = 200
_DOT_COLOUR = (255, 60, 60)  # bright red


def _draw_dot(draw, cx: int, cy: int, r: int = _DOT_RADIUS, fill=_DOT_COLOUR) -> None:
    """Draw a filled circle centred at (cx, cy)."""
    from PIL import ImageDraw  # local import — PIL available at runtime

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=(0, 0, 0))


def annotate_image_with_points(
    image,
    coords_str: str,
    output_path,
) -> None:
    """Draw point coordinates onto *image* and save to *output_path*.

    Args:
        image: PIL.Image or path-like.
        coords_str: space-separated triplets ``image_idx x_norm y_norm ...``
                    where x_norm, y_norm ∈ [0, 1000].
        output_path: destination PNG path.
    """
    from PIL import Image as _Image
    from PIL import ImageDraw as _ImageDraw

    if not isinstance(image, _Image.Image):
        image = _Image.open(image).convert("RGB")
    else:
        image = image.copy()

    W, H = image.size
    draw = _ImageDraw.Draw(image)

    # Molmo coordinate format: image_idx  x_3-4digit  y_3-4digit
    # x and y are always 3–4 digit numbers (scaled × 1000), e.g. "1 070 576".
    # Using a regex ensures we skip any spurious 1-2 digit tokens.
    _POINTS_RE = re.compile(r"[0-9]+\s+([0-9]{3,4})\s+([0-9]{3,4})")
    for m in _POINTS_RE.finditer(coords_str):
        try:
            x_norm = float(m.group(1))
            y_norm = float(m.group(2))
            cx = int(x_norm / 1000 * W)
            cy = int(y_norm / 1000 * H)
            _draw_dot(draw, cx, cy)
        except ValueError:
            pass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))


def annotate_video_frame_with_points(
    frame,
    x_norm: float,
    y_norm: float,
    output_path,
) -> None:
    """Draw a single point on a video frame and save to *output_path*.

    Args:
        frame: PIL.Image or path-like.
        x_norm, y_norm: normalised coordinates in [0, 1000].
        output_path: destination PNG path.
    """
    from PIL import Image as _Image
    from PIL import ImageDraw as _ImageDraw

    if not isinstance(frame, _Image.Image):
        frame = _Image.open(frame).convert("RGB")
    else:
        frame = frame.copy()

    W, H = frame.size
    draw = _ImageDraw.Draw(frame)
    cx = int(x_norm / 1000 * W)
    cy = int(y_norm / 1000 * H)
    _draw_dot(draw, cx, cy)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.save(str(output_path))


def annotate_video_with_points(video_path: str, coords_str: str, output_path: str) -> None:
    """Read *video_path*, draw coordinate dots on matching frames, write *output_path* (MP4).

    Args:
        video_path:  source video file.
        coords_str:  space-separated triplets ``frame_time x_norm y_norm ...``
                     where x_norm, y_norm ∈ [0, 1000] and frame_time is in seconds.
        output_path: destination MP4 path.
    """
    import av
    import numpy as np
    from PIL import Image as _Image
    from PIL import ImageDraw as _ImageDraw

    # Molmo video coord format: frame_time obj_idx x_norm y_norm [obj_idx x_norm y_norm ...]
    # Multiple objects at the same timestamp share one frame_time:
    #   "18.0 1 415 650 2 555 669" → two dots both at t=18.0
    # Tracking repeats across timestamps separated by ';':
    #   "0.0 1 500 500;0.5 1 520 510;..."
    # frame_time always has a decimal point (e.g. "0.0", "18.0") to distinguish
    # it from obj_idx (a small integer like "1", "2").
    coord_list: list[tuple[float, float, float]] = []  # (t, x_norm, y_norm)
    tokens = re.split(r"[;,\s]+", coords_str.strip())
    tokens = [t for t in tokens if t]
    i = 0
    current_t: float | None = None
    while i < len(tokens):
        tok = tokens[i]
        try:
            val = float(tok)
        except ValueError:
            i += 1
            continue
        # frame_time: has decimal point OR value >= 10 (large timestamps)
        if "." in tok or val >= 10:
            current_t = val
            i += 1
        elif current_t is not None and i + 2 < len(tokens):
            # obj_idx (small int) followed by 3-4 digit x, y
            try:
                x_str, y_str = tokens[i + 1], tokens[i + 2]
                x_norm, y_norm = float(x_str), float(y_str)
                if len(x_str) >= 3 and len(y_str) >= 3:
                    coord_list.append((current_t, x_norm, y_norm))
                    i += 3
                    continue
            except (ValueError, IndexError):
                pass
            i += 1
        else:
            i += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    in_container = av.open(video_path)
    in_stream = in_container.streams.video[0]
    fps = float(in_stream.average_rate)
    W = in_stream.width
    H = in_stream.height

    out_container = av.open(output_path, mode="w")
    out_stream = out_container.add_stream("libx264", rate=int(round(fps)))
    out_stream.width = W
    out_stream.height = H
    out_stream.pix_fmt = "yuv420p"

    # Group coords by timestamp so multi-object frames draw all their dots
    from collections import defaultdict

    coords_by_t: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for t_coord, x_norm, y_norm in coord_list:
        coords_by_t[t_coord].append((x_norm, y_norm))
    sorted_times = sorted(coords_by_t)

    for frame_idx, packet in enumerate(in_container.decode(video=0)):
        t = frame_idx / fps
        arr = packet.to_ndarray(format="rgb24")
        img = _Image.fromarray(arr)
        if sorted_times:
            draw = _ImageDraw.Draw(img)
            # Nearest-timestamp policy: dot always visible, not just on 1-frame window
            nearest_t = min(sorted_times, key=lambda ts: abs(t - ts))
            for x_norm, y_norm in coords_by_t[nearest_t]:
                cx = int(x_norm / 1000 * W)
                cy = int(y_norm / 1000 * H)
                _draw_dot(draw, cx, cy)
        out_frame = av.VideoFrame.from_ndarray(np.array(img), format="rgb24")
        for pkt in out_stream.encode(out_frame):
            out_container.mux(pkt)

    for pkt in out_stream.encode():
        out_container.mux(pkt)
    in_container.close()
    out_container.close()


def save_annotated_outputs(prompts: list, responses: list, output_dir) -> None:
    """Overlay model coordinates onto source images/video frames and save to *output_dir*.

    For each user whose response contains coordinate tags, extracts the source
    image(s) or video path from the prompt, draws the predicted points, and
    writes annotated PNG(s) to output_dir.

    Silently skips text-only prompts (no visual input → nothing to annotate).
    """
    import re as _re

    from PIL import Image as _Image

    # Closing quote is optional — model may truncate at max_new_tokens before it
    _COORD_FULL_RE = _re.compile(
        r'<(?:points?|tracks?)[^>]*\s+coords="([0-9\t:;,. ]+)',
        _re.IGNORECASE,
    )

    for u, (conv, response) in enumerate(zip(prompts, responses)):
        all_coords = _COORD_FULL_RE.findall(response)
        if not all_coords:
            continue  # no coordinates — skip

        # Collect images / video paths from the conversation
        images, video_path = [], None
        for msg in conv:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        images.append(item.get("image", ""))
                    elif item.get("type") == "video":
                        video_path = item.get("video", "")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        coords_str = " ".join(all_coords)

        if images:
            for img_idx, img_src in enumerate(images):
                try:
                    img = _Image.open(img_src).convert("RGB")
                    out = output_dir / f"user{u}_image{img_idx}_annotated.png"
                    annotate_image_with_points(img, coords_str, out)
                    logger.info(f"  [overlay] saved {out}")
                except Exception as exc:
                    logger.warning(f"  [overlay] could not annotate {img_src}: {exc}")

        elif video_path:
            # Produce an annotated MP4 with dots burned onto every matching frame
            try:
                out = output_dir / f"user{u}_annotated.mp4"
                annotate_video_with_points(video_path, coords_str, str(out))
                logger.info(f"  [overlay] saved annotated video {out}")
            except Exception as exc:
                logger.warning(f"  [overlay] could not annotate video {video_path}: {exc}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HF_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/" "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    )
)
SAMPLE_DIR = Path(__file__).parent / "sample_inputs"

# ---------------------------------------------------------------------------
# Pytest CLI options
# ---------------------------------------------------------------------------


# pytest_addoption is in conftest.py (needed for repo-root pytest invocations).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_inputs(path: str, batch_size: int) -> list:
    with open(path) as f:
        data = json.load(f)
    # Each entry in the JSON is a list of messages for one user
    prompts = [entry if isinstance(entry, list) else [entry] for entry in data]
    if len(prompts) < batch_size:
        # Repeat to fill batch
        prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]
    return prompts[:batch_size]


def build_inputs(processor, prompts: list):
    """Convert a list of message-lists into HF processor inputs.

    Handles text-only, image, and video prompts by inspecting content type.
    Images / videos are loaded from local paths or URLs using PIL / decord.
    """
    sys.path.insert(0, str(HF_PATH))

    texts = processor.apply_chat_template(
        [p for p in prompts],  # list of message-lists
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(texts, str):
        texts = [texts]

    # Collect images and videos referenced in the prompts
    images, videos = [], []
    has_image, has_video = False, False
    for conversation in prompts:
        for msg in conversation:
            for item in msg.get("content") if isinstance(msg.get("content"), list) else []:
                if item.get("type") == "image":
                    src = item.get("image", "")
                    images.append(_load_image(src))
                    has_image = True
                elif item.get("type") == "video":
                    src = item.get("video", "")
                    videos.append(src)  # pass path to processor (uses decord)
                    has_video = True

    kwargs = {"text": texts, "return_tensors": "pt", "padding": True}
    if has_image:
        kwargs["images"] = images
    if has_video:
        kwargs["videos"] = videos[0] if len(videos) == 1 else videos

    return processor(**kwargs)


def _load_image(src: str) -> Image.Image:
    if src.startswith("http"):
        from io import BytesIO

        import requests

        r = requests.get(src, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return Image.open(src).convert("RGB")


def batch_vision_inputs(proc_out):
    """Return (pv_batched, pool_idx) or (None, None) for text-only inputs."""
    pv = proc_out.get("pixel_values")
    if pv is not None:
        return pv.float().unsqueeze(0), proc_out["image_token_pooling"].unsqueeze(0)
    pv_vid = proc_out.get("pixel_values_videos")
    if pv_vid is not None:
        return pv_vid.float().unsqueeze(0), proc_out["video_token_pooling"].unsqueeze(0)
    return None, None


# ---------------------------------------------------------------------------
# Model / mesh fixtures (module-scoped — loaded once per test)
# ---------------------------------------------------------------------------

_MESH_SHAPE = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE if isinstance(_MESH_SHAPE, tuple) else (1, _MESH_SHAPE)
    is_single = rows * cols == 1
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    logger.info(f"Opened {rows}×{cols} mesh: {device.get_num_devices()} devices")
    yield device
    ttnn.close_mesh_device(device)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def tt_model(mesh_device):
    """Load full Molmo2 TTNN model (weights + config)."""
    from transformers import AutoModelForImageTextToText

    from models.demos.molmo2.tt.model import TtMolmo2Model
    from models.demos.molmo2.tt.model_config import Molmo2Config
    from models.tt_transformers.tt.ccl import TT_CCL

    sys.path.insert(0, str(HF_PATH))
    logger.info("Loading Molmo2-8B weights (bfloat16)...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        str(HF_PATH),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    state_dict = hf_model.state_dict()
    del hf_model

    cfg = Molmo2Config(mesh_device=mesh_device)
    ccl = TT_CCL(mesh_device)
    from pathlib import Path

    weight_cache = Path("/tmp/molmo2_weight_cache")
    weight_cache.mkdir(parents=True, exist_ok=True)
    model = TtMolmo2Model(
        mesh_device=mesh_device,
        tt_ccl=ccl,
        state_dict=state_dict,
        weight_cache_path=weight_cache,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )
    del state_dict
    logger.info("Model ready")
    return model, cfg


@pytest.fixture(scope="module")
def processor():
    sys.path.insert(0, str(HF_PATH))
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(HF_PATH), trust_remote_code=True)


# ---------------------------------------------------------------------------
# Core demo function
# ---------------------------------------------------------------------------


def run_demo(
    mesh_device,
    tt_model,
    processor,
    request,
    input_prompts_path: str,
    max_new_tokens: int,
    batch_size: int,
    temperature: float = 0.0,
    stop_at_eos: bool = True,
):
    model, cfg = tt_model
    eos = processor.tokenizer.eos_token_id

    # CLI overrides
    input_prompts_path = request.config.getoption("--input_prompts") or input_prompts_path
    max_new_tokens = request.config.getoption("--max_new_tokens") or max_new_tokens
    batch_size = request.config.getoption("--batch_size") or batch_size

    logger.info(f"Loading prompts from {input_prompts_path} (batch={batch_size})")
    prompts = load_inputs(input_prompts_path, batch_size)

    logger.info("Preprocessing inputs...")
    t0 = time.time()
    proc_out = build_inputs(processor, prompts)
    input_ids = proc_out["input_ids"]  # [B, S]
    token_type_ids = proc_out.get("token_type_ids")
    pv_batched, pool_idx = batch_vision_inputs(proc_out)
    S = input_ids.shape[1]
    logger.info(f"  Inputs ready: batch={batch_size} seq_len={S}  preprocess={time.time()-t0:.2f}s")

    # ---- Warm-up (compile) pass — not timed ----
    # Run a minimal text-only prefill to trigger TTNN JIT compilation before measuring.
    # This avoids counting first-run kernel compile time in the benchmark.
    if not getattr(model, "_demo_warmed_up", False):
        logger.info("Warm-up compile pass (not timed)...")
        t_warmup = time.time()
        warmup_ids = input_ids[:1, :32]  # first 32 tokens, text path only
        _ = model.forward_prefill(input_ids=warmup_ids, pixel_values=None, user_id=0)
        # Capture decode trace (covers JIT compile + trace capture)
        model.warmup_decode_trace(prefill_seq_len=32)
        model._demo_warmed_up = True
        logger.info(f"  Warm-up done in {time.time()-t_warmup:.2f}s (not counted in timing)")

    # ---- Timed prefill ----
    logger.info("Prefill (filling KV cache)...")
    t_prefill = time.time()
    logits = model.forward_prefill(
        input_ids=input_ids[:1],
        pixel_values=pv_batched[:1] if pv_batched is not None else None,
        pooled_patches_idx=pool_idx[:1] if pool_idx is not None else None,
        token_type_ids=token_type_ids[:1] if token_type_ids is not None else None,
        user_id=0,
    )
    prefill_ms = (time.time() - t_prefill) * 1000
    logger.info(f"  Prefill: {prefill_ms:.0f}ms  ({S} tokens, {S / (prefill_ms / 1000):.0f} tok/s)")

    # Additional users (batch > 1)
    for user_id in range(1, batch_size):
        model.forward_prefill(
            input_ids=input_ids[user_id : user_id + 1],
            pixel_values=pv_batched[user_id : user_id + 1] if pv_batched is not None else None,
            pooled_patches_idx=pool_idx[user_id : user_id + 1] if pool_idx is not None else None,
            token_type_ids=token_type_ids[user_id : user_id + 1] if token_type_ids is not None else None,
            user_id=user_id,
        )

    # ---- Timed decode loop ----
    logger.info("Starting decode loop (TTNN)...")
    all_outputs = [[] for _ in range(batch_size)]
    user_done = [False] * batch_size

    # First token from prefill logits (user 0)
    next_tokens = torch.argmax(logits[:, -1:], dim=-1).squeeze(-1)  # [1]
    next_tokens = next_tokens.expand(batch_size)
    for u in range(batch_size):
        all_outputs[u].append(int(next_tokens[u].item()))

    current_pos = S

    t_decode_start = time.time()
    for iteration in range(max_new_tokens - 1):
        if all(user_done):
            break

        t_iter = time.time()

        new_next = []
        for u in range(batch_size):
            tok = int(next_tokens[u].item())
            if user_done[u]:
                new_next.append(eos)
            elif batch_size == 1 and model._decode_trace_id is not None and temperature == 0:
                # On-device argmax: returns int directly, no D2H of full logits
                new_next.append(model._execute_decode_trace(tok, current_pos))
            else:
                logits_u = model.forward_decode_step(tok, current_pos)
                if temperature == 0:
                    new_next.append(int(logits_u[0].argmax().item()))
                else:
                    probs = torch.softmax(logits_u[0].float() / temperature, dim=-1)
                    new_next.append(int(torch.multinomial(probs, 1).item()))

        next_tokens = torch.tensor(new_next)
        current_pos += 1
        iter_ms = (time.time() - t_iter) * 1000
        tps = 1000 / iter_ms

        for u in range(batch_size):
            tok = int(next_tokens[u].item())
            if not user_done[u]:
                if stop_at_eos and tok == eos:
                    user_done[u] = True
                else:
                    all_outputs[u].append(tok)

        if iteration < 3 or iteration % 10 == 0:
            logger.info(f"  iter {iteration+1}: {iter_ms:.0f}ms  {tps:.1f} tok/s/user")

    decode_elapsed = time.time() - t_decode_start
    total_tokens = sum(len(o) for o in all_outputs)
    overall_tps = total_tokens / decode_elapsed if decode_elapsed > 0 else 0

    # ---- Print outputs ----
    logger.info(
        f"\nDecode complete: {total_tokens} tokens in {decode_elapsed:.1f}s " f"({overall_tps:.1f} tok/s aggregate)\n"
    )

    responses = []
    for u, output_ids in enumerate(all_outputs):
        text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        responses.append(text)
        prompt_text = (
            prompts[u][-1]["content"]
            if isinstance(prompts[u][-1]["content"], str)
            else next((c["text"] for c in prompts[u][-1]["content"] if c.get("type") == "text"), "")
        )
        logger.info(f"\n=== USER {u} ===\nPROMPT:   {prompt_text[:120]}\nRESPONSE: {_format_response(text).strip()}\n")

    # Overlay coordinates on source images / video frames if any response has coords
    # One sub-folder per demo so outputs don't overwrite each other
    stem = Path(input_prompts_path).stem  # e.g. "point_video_demo"
    output_dir = Path(input_prompts_path).parent / "annotated_outputs" / stem
    save_annotated_outputs(prompts, responses, output_dir)

    return responses


# ---------------------------------------------------------------------------
# Test parametrization (mirrors qwen3_vl/demo/demo.py structure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_prompts, max_new_tokens, batch_size",
    [
        (  # text-only, batch=1
            str(SAMPLE_DIR / "text_only.json"),
            128,
            1,
        ),
        (  # text-only, batch=2
            str(SAMPLE_DIR / "text_only.json"),
            128,
            2,
        ),
        (  # image + text, batch=1
            str(SAMPLE_DIR / "image_demo.json"),
            200,
            1,
        ),
        (  # video + text, batch=1
            str(SAMPLE_DIR / "video_demo.json"),
            64,
            1,
        ),
        (  # 2 images + text, batch=1 — verifies multi-image via native image path
            str(SAMPLE_DIR / "multi_image_2_demo.json"),
            200,
            1,
        ),
        (  # 20 images + text, batch=1 — stress test near max=23 (36864//(8*196))
            str(SAMPLE_DIR / "multi_image_20_demo.json"),
            200,
            1,
        ),
        (  # single image + pointing prompt — model outputs <points coords="..."/>
            str(SAMPLE_DIR / "point_image_demo.json"),
            200,
            1,
        ),
        (  # video + pointing prompt — model outputs per-frame <points coords="..."/>
            str(SAMPLE_DIR / "point_video_demo.json"),
            200,
            1,
        ),
        (  # video + tracking prompt — model outputs <tracks coords="..."/>
            str(SAMPLE_DIR / "track_video_demo.json"),
            200,
            1,
        ),
        (  # 2 images + multi-image pointing — model outputs per-image <points coords="..."/>
            str(SAMPLE_DIR / "point_multi_image_demo.json"),
            200,
            1,
        ),
    ],
    ids=[
        "text_only-batch1",
        "text_only-batch2",
        "image-batch1",
        "video-batch1",
        "multi-image-2-batch1",
        "multi-image-20-batch1",
        "image-point-batch1",
        "video-point-batch1",
        "video-track-batch1",
        "multi-image-point-batch1",
    ],
)
def test_demo(
    mesh_device,
    tt_model,
    processor,
    request,
    input_prompts,
    max_new_tokens,
    batch_size,
):
    """Run Molmo2-8B end-to-end demo on T3K for text / image / video inputs."""
    responses = run_demo(
        mesh_device=mesh_device,
        tt_model=tt_model,
        processor=processor,
        request=request,
        input_prompts_path=input_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        temperature=0.0,
        stop_at_eos=True,
    )

    assert responses, "No output generated"
    for u, r in enumerate(responses):
        assert len(r.strip()) > 0, f"User {u}: empty response"
        logger.info(f"User {u} response length: {len(r)} chars")
