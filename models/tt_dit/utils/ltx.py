# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import os
from io import BytesIO

import torch

# I2V conditioning-image H.264 CRF: round-trip through the codec the VAE/DiT were trained on
# before encoding (a pristine image gives OOD latents). Mirrors ltx_pipelines DEFAULT_IMAGE_CRF.
DEFAULT_IMAGE_CRF = 33

# LTX-2 VAE compression ratios (pixel -> latent). Used throughout to map pixel
# dims to the latent token grid. NOTE: the TILE size used for SP padding (also 32)
# is a separate concept — do NOT replace `32 * sp_factor` padding math with these.
TEMPORAL_COMPRESSION = 8
SPATIAL_COMPRESSION = 32

DEFAULT_LTX_PROMPT = (
    "A confident rapper in a black leather jacket and gold chain leans toward camera "
    "in a dimly lit studio. The camera holds a dynamic medium shot, slightly below eye "
    "level for powerful framing. Purple and blue LED strips create moody rim lighting "
    "while a key light illuminates his face from the right. He delivers lyrics with "
    'precise articulation: "I rise up from the basement, now Im elevated / Success is '
    'what Im chasin, never been debated." His hands gesture naturally - right hand '
    'emphasizing the beat near his chest, left hand spreading outward on "elevated." '
    "Facial expressions shift from focused intensity to confident smirk. Head nods "
    "match the rhythm while maintaining eye contact with lens. Deep bass and crisp "
    "hi-hats underscore his clear vocal delivery. Subtle breath control visible "
    "between bars. Shot with 50mm lens at f/1.8, shallow depth of field blurring the "
    "urban studio background. Color grading emphasizes cool tones with warm skin "
    "highlights. Handheld stabilization adds subtle energy without excessive movement. "
    "Natural motion blur on hand gestures, synchronized audio-visual performance."
)

# Distinct prompts for traced distilled steady-state (gen #1 / #2) so the encoder is measured
# instead of hitting the embed cache written by gen #0. Mirror of main #52968.
STEADY_STATE_LTX_PROMPT = (
    "A grey tabby cat sits on a windowsill in afternoon light, tail curled around its paws. "
    "The camera holds a steady medium shot as the cat blinks slowly and turns its head toward "
    "the window. Dust drifts through the sunbeam behind it. Shot with a 50mm lens at f/2.0, "
    "shallow depth of field, natural warm color grade. "
    "Audio: faint birdsong through glass, a quiet purr, soft room tone."
)

STEADY_STATE_REPLAY_LTX_PROMPT = (
    "A young woman with shoulder-length wavy brown hair sits on a wooden stool, "
    "cradling an acoustic guitar. The camera holds a steady medium close-up, "
    "framing her face and guitar neck. Warm key light illuminates her left side "
    "while soft fill light prevents harsh shadows. She strums gently, looking "
    "directly at camera with genuine warmth. Her mouth opens clearly as she sings "
    '"Doo-be-doo, doo-be-day, oh what a sunny day" with precise lip sync and '
    "natural facial expressions. Her head moves subtly with the rhythm. Simple "
    "chord progression underlies her melodic voice. Shot with 50mm lens at f/2.0, "
    "shallow depth of field, warm color grade emphasizing skin tones."
)

# Second prompt for the traced steady-state pass, so its encode is a real one rather than a hit on
# the prompt-embedding cache that the first pass just populated.
STEADY_STATE_LTX_PROMPT = (
    "A grey tabby cat sits on a windowsill in afternoon light, tail curled around its paws. "
    "The camera holds a steady medium shot as the cat blinks slowly and turns its head toward "
    "the window. Dust drifts through the sunbeam behind it. Shot with a 50mm lens at f/2.0, "
    "shallow depth of field, natural warm color grade. "
    "Audio: faint birdsong through glass, a quiet purr, soft room tone."
)

# Third prompt, for the pass after the encode trace is captured: a second unseen prompt is what makes
# that pass replay every trace rather than capture one.
STEADY_STATE_REPLAY_LTX_PROMPT = (
    "A red paper boat drifts across a still pond at dusk, ripples spreading behind it. "
    "The camera holds a low steady shot near the waterline as the light fades. "
    "Audio: gentle water laps, distant crickets, soft evening air."
)


def ceil_to(x: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``x``."""
    return -(-x // multiple) * multiple


def latent_grid(num_frames: int, height: int, width: int) -> tuple[int, int, int]:
    """Map pixel dims to the LTX latent token grid ``(latent_frames, latent_h, latent_w)``."""
    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    return latent_frames, height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION


def pad_hw_replicate(x_BCFHW: torch.Tensor, h_mult: int, w_mult: int) -> tuple[torch.Tensor, int, int]:
    """Replicate-pad a ``(B, C, F, H, W)`` tensor's H/W up to multiples of ``h_mult``/``w_mult``.

    The sharded VAE convs seam the latent at the 2x4 mesh boundaries when H/W don't divide
    evenly across the mesh (the uneven-dim halo runs a crop-masking path); padding to even
    shards avoids it. Returns ``(padded, H, W)`` — the original H/W so the caller can crop the
    replicated margin back off after the op.
    """
    B, C, frames, H, W = x_BCFHW.shape
    pad_h, pad_w = (-H) % h_mult, (-W) % w_mult
    if pad_h or pad_w:
        x_BCFHW = torch.nn.functional.pad(
            x_BCFHW.reshape(B * C, frames, H, W), (0, pad_w, 0, pad_h), mode="replicate"
        ).reshape(B, C, frames, H + pad_h, W + pad_w)
    return x_BCFHW, H, W


def default_ltx_checkpoint(filename: str) -> str:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser(f"~/.cache/ltx-checkpoints/{filename}")
    if os.path.exists(local):
        return local
    return f"Lightricks/LTX-2.3:{filename}"


def default_ltx_gemma() -> str:
    return os.environ.get("GEMMA_PATH") or "google/gemma-3-12b-it-qat-q4_0-unquantized"


# Relative paths under an LTX-2.5 split checkpoint root (HF hub layout).
LTX25_TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
LTX25_DISTILLED_TRANSFORMER = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
# Two video VAEs ship for 2.5 and they are not interchangeable files: the ``-conv-`` one holds the
# convolutional decoder (gated on HF), and the plain one holds the diffusion decoder (DiffVAE),
# which is what the distilled recipe actually decodes with.
LTX25_VIDEO_VAE_CONV = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
LTX25_VIDEO_VAE_DIFF = "vae/ltx-2.5-video-vae-bf16.safetensors"
LTX25_VIDEO_VAE = LTX25_VIDEO_VAE_CONV  # alias used by the distilled pipeline
LTX25_AUDIO_VAE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
LTX25_SPATIAL_UPSAMPLER = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

_MLPERF_LTX25_HUB = "/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5"


def default_ltx25_root() -> str | None:
    """Resolve an LTX-2.5 split-checkpoint root, or ``None`` if none is present.

    Order: ``LTX25_ROOT`` → ``~/.cache/ltx-checkpoints/ltx-2.5`` → MLPerf HF hub snapshot.
    """
    explicit = os.environ.get("LTX25_ROOT")
    if explicit:
        root = os.path.expanduser(explicit)
        if os.path.isdir(root):
            return root
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5")
    if os.path.isdir(local):
        return local
    refs = os.path.join(_MLPERF_LTX25_HUB, "refs", "main")
    if os.path.isfile(refs):
        with open(refs) as f:
            snap = f.read().strip()
        snap_root = os.path.join(_MLPERF_LTX25_HUB, "snapshots", snap)
        if os.path.isdir(snap_root):
            return snap_root
    return None


def default_ltx25_path(rel: str) -> str | None:
    """Join ``rel`` onto ``default_ltx25_root()``; ``None`` if the root (or file) is missing."""
    root = default_ltx25_root()
    if root is None:
        return None
    path = os.path.join(root, rel)
    return path if os.path.exists(path) else None


def default_ltx25_video_vae(*, diffusion: bool = False) -> str | None:
    """Video VAE for 2.5 decode.

    With ``diffusion``, resolve the DiffVAE file and nothing else: its decoder weights exist in no
    other checkpoint, so falling back would silently decode with a different model.

    Otherwise prefer the split ``*-video-vae-conv-bf16`` file. If it is not on disk yet (gated HF /
    incomplete MLPerf mirror), fall back to a local 2.3 monolith — PORT notes the conv VAE
    arch is identical, so this unblocks generate until the 2.5 conv file is available.
    """
    if diffusion:
        return default_ltx25_path(LTX25_VIDEO_VAE_DIFF)
    conv = default_ltx25_path(LTX25_VIDEO_VAE_CONV)
    if conv:
        return conv
    for name in ("ltx-2.3-22b-distilled-1.1.safetensors", "ltx-2.3-22b-dev.safetensors"):
        local = os.path.expanduser(f"~/.cache/ltx-checkpoints/{name}")
        if os.path.exists(local):
            return local
        hub = default_ltx_checkpoint(name)
        if os.path.exists(hub):
            return hub
    return None


def decode_breakdown_rows(root, wall_s: float, *, depth: int, min_frac: float = 0.01):
    """Indented (label, seconds) rows for the perf table, taken from a ``decode_tree`` root.

    Siblings pool by exact label the way ``decode_tree.render_tree`` does, so ``attention`` (one span per
    band) is one row while ``stage5 block 0..7`` stay distinct. Children under ``min_frac`` of the
    wall-clock decode are omitted, and each level's unattributed remainder is shown as ``other`` only
    when it clears the same bar, so a level may sum to slightly under its parent. Tree spans carry two
    device syncs each, so the root can run longer than the wall clock it explains; the delta is
    reported on the first row instead of hidden.
    """
    from collections import OrderedDict

    rows = []
    root_s = root.incl_ms / 1000.0
    rows.append((f"  · tree total (sync-inflated, {root_s - wall_s:+.2f} s vs wall)", root_s))

    def walk(nodes, parent_s, prefix, level):
        if level > depth:
            return
        groups = OrderedDict()
        for n in nodes:
            for c in n.children:
                groups.setdefault(c.label.strip(), []).append(c)
        shown = [(lbl, sum(c.incl_ms for c in cs) / 1000.0, cs) for lbl, cs in groups.items()]
        shown = [(lbl, secs, cs) for lbl, secs, cs in shown if secs >= min_frac * wall_s]
        other = parent_s - sum(secs for _, secs, _ in shown)
        for i, (lbl, secs, cs) in enumerate(shown):
            last = i == len(shown) - 1 and other < min_frac * wall_s
            rows.append((f"{prefix}{'└─ ' if last else '├─ '}{lbl}", secs))
            walk(cs, secs, prefix + ("   " if last else "│  "), level + 1)
        if shown and other >= min_frac * wall_s:
            rows.append((f"{prefix}└─ · other", other))

    walk([root], root_s, "  ", 1)
    return rows


def decode_category_rows(root, *, min_frac: float = 0.01):
    """(category, seconds, share) from ``decode_tree.category_totals``: exclusive self-time, so the
    rows partition the root and answer "where did the decode actually go" without double counting a
    kv-allgather that runs inside an attention span. Only meaningful with DIFFVAE_BLOCK_PROF, since
    without the deep spans nearly everything is charged to a stage's own remainder."""
    from .decode_tree import category_totals

    totals, _ = category_totals(root)
    total_ms = root.incl_ms or 1.0
    rows = []
    for cat, ms in totals.items():
        if ms / total_ms >= min_frac:
            rows.append((cat, ms / 1000.0, 100.0 * ms / total_ms))
    return rows


def print_ltx_timing_table(
    pipeline, *, label, num_frames, height, width, mesh_shape, sp_axis, tp_axis, topology, output_path, prompt
):
    """Per-stage wall-clock table for one gen.

    With ``LTX_PERF_BREAKDOWN=N`` (N >= 1) and a decode tree recorded for this gen, the VAE decode row
    expands into N levels of sub-rows from ``pipeline.last_decode_tree``; with DIFFVAE_BLOCK_PROF also
    on, a second block lists exclusive decode time by category. run_ltx25_pipeline.sh sets both under
    PROFILE=1. A traced decode records no tree, so the table then silently stays flat.
    """
    timings = getattr(pipeline, "last_timings", None)
    if not timings:
        return

    from . import decode_tree

    depth = int(os.environ.get("LTX_PERF_BREAKDOWN", "0") or 0)
    root = getattr(pipeline, "last_decode_tree", None) if depth > 0 else None

    mesh = tuple(mesh_shape)
    topo = str(topology).split(".")[-1]
    prompt_short = prompt if len(prompt) <= 60 else prompt[:57] + "..."
    meta = [
        f"Resolution   {height}x{width} · {num_frames} frames",
        f"Mesh         {mesh} · sp={mesh[sp_axis]} tp={mesh[tp_axis]} · {topo}",
        f"Output       {output_path}",
        f"Prompt       {prompt_short}",
    ]
    rows = []
    for name, secs in timings:
        rows.append((name, f"{secs:.2f} s"))
        if root is not None and name == "VAE decode":
            rows.extend((lbl, f"{s:.2f} s") for lbl, s in decode_breakdown_rows(root, secs, depth=depth))
    total_row = ("Total", f"{sum(s for _, s in timings):.2f} s")

    cat_rows = []
    if root is not None and decode_tree.DEEP:
        cat_rows = [(f"  decode · {cat}", f"{s:.2f} s  {pct:4.1f}%") for cat, s, pct in decode_category_rows(root)]

    all_rows = rows + [total_row] + cat_rows
    lw = max([len(n) for n, _ in all_rows] + [len("Stage")])
    rw = max([len(t) for _, t in all_rows] + [len("Time")])
    full = max(lw + rw + 5, max(len(m) for m in meta) + 1)
    lw = full - rw - 5

    def line(name, t):
        return "│ " + name.ljust(lw) + " │ " + t.rjust(rw) + " │"

    sep = "├" + "─" * (lw + 2) + "┼" + "─" * (rw + 2) + "┤"
    out = ["", "┌" + "─" * full + "┐", "│" + f"{label} — PERFORMANCE".center(full) + "│"]
    for m in meta:
        out.append("│ " + m.ljust(full - 1) + "│")
    out.append("├" + "─" * (lw + 2) + "┬" + "─" * (rw + 2) + "┤")
    out.append(line("Stage", "Time"))
    out.append(sep)
    for name, t in rows:
        out.append(line(name, t))
    out.append(sep)
    out.append(line(*total_row))
    if cat_rows:
        out.append(sep)
        out.append(line("VAE decode by category (exclusive, BLOCK_PROF)", ""))
        for name, t in cat_rows:
            out.append(line(name, t))
    out.append("└" + "─" * (lw + 2) + "┴" + "─" * (rw + 2) + "┘")
    print("\n".join(out))


def crf_codec_roundtrip(arr, crf: int):
    """Encode/decode an RGB ``(H,W,3)`` uint8 image through libx264 at the given CRF, cropped
    to even dims. Port of ``ltx_pipelines.utils.media_io`` encode/decode_single_frame."""
    import av  # lazy import (matches utils/video.py); only needed for I2V conditioning
    import numpy as np

    # libx264 requires even dimensions; crop to a multiple of 2 like the reference.
    height = arr.shape[0] // 2 * 2
    width = arr.shape[1] // 2 * 2
    arr = np.ascontiguousarray(arr[:height, :width])

    with BytesIO() as buf:
        container = av.open(buf, mode="w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
            stream.height = height
            stream.width = width
            av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
        finally:
            container.close()
        video_bytes = buf.getvalue()

    with BytesIO(video_bytes) as buf:
        container = av.open(buf)
        try:
            vstream = next(s for s in container.streams if s.type == "video")
            frame = next(container.decode(vstream))
        finally:
            container.close()
    return frame.to_ndarray(format="rgb24")


def load_conditioning_image(image_path: str, height: int, width: int, crf: int = DEFAULT_IMAGE_CRF) -> torch.Tensor:
    """Decode -> CRF round-trip -> resize+center-crop -> normalize to [-1,1]. Returns
    ``(1,3,1,H,W)`` float32. Port of ``load_image_and_preprocess``; ``crf=0`` skips the codec."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)  # (H, W, 3) uint8
    if crf and crf > 0:
        arr = crf_codec_roundtrip(arr, crf)
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()  # (H, W, 3)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    _, _, src_h, src_w = tensor.shape
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    tensor = tensor.unsqueeze(2)  # (1, 3, 1, H, W)
    return tensor / 127.5 - 1.0
