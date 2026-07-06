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


def print_ltx_timing_table(
    pipeline, *, label, num_frames, height, width, mesh_shape, sp_axis, tp_axis, topology, output_path, prompt
):
    timings = getattr(pipeline, "last_timings", None)
    if not timings:
        return

    mesh = tuple(mesh_shape)
    topo = str(topology).split(".")[-1]
    prompt_short = prompt if len(prompt) <= 60 else prompt[:57] + "..."
    meta = [
        f"Resolution   {height}x{width} · {num_frames} frames",
        f"Mesh         {mesh} · sp={mesh[sp_axis]} tp={mesh[tp_axis]} · {topo}",
        f"Output       {output_path}",
        f"Prompt       {prompt_short}",
    ]
    rows = [(name, f"{secs:.2f} s") for name, secs in timings]
    rows.append(("Total", f"{sum(s for _, s in timings):.2f} s"))

    lw = max([len(n) for n, _ in rows] + [len("Stage")])
    rw = max([len(t) for _, t in rows] + [len("Time")])
    full = max(lw + rw + 5, max(len(m) for m in meta) + 1)
    lw = full - rw - 5

    out = ["", "┌" + "─" * full + "┐", "│" + f"{label} — PERFORMANCE".center(full) + "│"]
    for m in meta:
        out.append("│ " + m.ljust(full - 1) + "│")
    out.append("├" + "─" * (lw + 2) + "┬" + "─" * (rw + 2) + "┤")
    out.append("│ " + "Stage".ljust(lw) + " │ " + "Time".rjust(rw) + " │")
    out.append("├" + "─" * (lw + 2) + "┼" + "─" * (rw + 2) + "┤")
    for name, t in rows[:-1]:
        out.append("│ " + name.ljust(lw) + " │ " + t.rjust(rw) + " │")
    out.append("├" + "─" * (lw + 2) + "┼" + "─" * (rw + 2) + "┤")
    out.append("│ " + rows[-1][0].ljust(lw) + " │ " + rows[-1][1].rjust(rw) + " │")
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
