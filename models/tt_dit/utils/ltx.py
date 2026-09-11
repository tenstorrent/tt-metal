# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass
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

LTX_FAST_SP_FACTOR = 8
LTX_FAST_LOWEST_BUCKET = "lowest"
LTX_FAST_LOWEST_BUCKET_VIDEO_N = {"s1": 8704, "s2": 34560}
LTX_FAST_LOWEST_BUCKET_AUDIO_N = 512
LTX_FAST_FPS_VALUES = (24, 25, 48, 50)
LTX_FAST_DURATION_VALUES = (6, 8, 10, 12, 14, 16, 18, 20)
LTX_FAST_CANVASES = {
    "720p-landscape": (704, 1280),
    "720p-portrait": (1280, 704),
    "1080p-landscape": (1088, 1920),
    "1080p-portrait": (1920, 1088),
    "1440p-landscape": (1440, 2560),
    "1440p-portrait": (2560, 1440),
    "4k-landscape": (2176, 3840),
    "4k-portrait": (3840, 2176),
}
LTX_FAST_LOWEST_BUCKET_DURATIONS_BY_FPS = {
    24: (6, 8, 10, 12),
    25: (6, 8, 10, 12),
    48: (6,),
    50: (6,),
}


@dataclass(frozen=True)
class LTXFastBucketRoute:
    bucket: str
    duration_seconds: int
    latent_frames: int
    stage_video_n_real: dict[str, int]
    stage_video_n: dict[str, int]

    def trace_key(self, stage: str) -> tuple[str, str]:
        if stage not in self.stage_video_n:
            raise ValueError(f"unknown LTX stage {stage!r}")
        return stage, self.bucket


def ltx_fast_aligned_num_frames(fps: int, duration_seconds: int) -> int:
    """Return the first VAE-compatible frame count covering the requested duration."""
    target_frames = fps * duration_seconds
    return math.ceil((target_frames - 1) / TEMPORAL_COMPRESSION) * TEMPORAL_COMPRESSION + 1


def ltx_fast_nominal_configs() -> tuple[tuple[str, int, int], ...]:
    """The complete 8-canvas × 4-fps × 8-duration product grid."""
    return tuple(
        (canvas, fps, duration)
        for canvas in LTX_FAST_CANVASES
        for fps in LTX_FAST_FPS_VALUES
        for duration in LTX_FAST_DURATION_VALUES
    )


def ltx_fast_lowest_bucket_configs() -> tuple[tuple[str, int, int], ...]:
    """The 20 product configs served by the first Fast trace tier."""
    return tuple(
        (canvas, fps, duration)
        for canvas in ("720p-landscape", "720p-portrait")
        for fps, durations in LTX_FAST_LOWEST_BUCKET_DURATIONS_BY_FPS.items()
        for duration in durations
    )


def ltx_fast_lowest_bucket_n(stage: str, n_real: int) -> int:
    """Return the stage bucket at the inclusive upper boundary."""
    if stage not in LTX_FAST_LOWEST_BUCKET_VIDEO_N:
        raise ValueError(f"unknown LTX stage {stage!r}")
    n_bucket = LTX_FAST_LOWEST_BUCKET_VIDEO_N[stage]
    if n_real < 1:
        raise ValueError(f"logical N must be positive, got {n_real}")
    if n_real > n_bucket:
        raise ValueError(f"{stage} logical N={n_real} exceeds lowest bucket N={n_bucket}")
    return n_bucket


def route_ltx_fast_lowest_bucket(
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: int,
    sp_factor: int,
    mode: str = "av",
    image_conditioned: bool = False,
) -> LTXFastBucketRoute:
    """Route an exact T2V AV request to the first Fast trace bucket.

    The boundary is inclusive: a real sequence equal to the bucket size is valid.
    Requests outside the finalized 20-config grid fail instead of aliasing a trace.
    """
    if sp_factor != LTX_FAST_SP_FACTOR:
        raise ValueError(f"LTX Fast lowest bucket requires SP=8, got SP={sp_factor}")
    if mode != "av":
        raise ValueError(f"LTX Fast lowest bucket supports AV mode only, got mode={mode!r}")
    if image_conditioned:
        raise ValueError("LTX Fast lowest bucket supports T2V only; I2V requires a separate trace class")

    canvas = next(
        (name for name in ("720p-landscape", "720p-portrait") if LTX_FAST_CANVASES[name] == (height, width)),
        None,
    )
    if canvas is None:
        raise ValueError(
            "LTX Fast lowest bucket supports only 720p canvases "
            f"{LTX_FAST_CANVASES['720p-landscape']} and {LTX_FAST_CANVASES['720p-portrait']}; "
            f"got {(height, width)}"
        )

    durations = LTX_FAST_LOWEST_BUCKET_DURATIONS_BY_FPS.get(fps)
    duration = next(
        (seconds for seconds in durations or () if ltx_fast_aligned_num_frames(fps, seconds) == num_frames),
        None,
    )
    if duration is None:
        raise ValueError(
            "unsupported LTX Fast lowest-bucket timing: "
            f"{num_frames} frames at {fps} fps; supported (fps, seconds) pairs are "
            f"{[(f, d) for f, ds in LTX_FAST_LOWEST_BUCKET_DURATIONS_BY_FPS.items() for d in ds]}"
        )

    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    stage_n_real = {
        "s1": latent_frames * (height // 2 // SPATIAL_COMPRESSION) * (width // 2 // SPATIAL_COMPRESSION),
        "s2": latent_frames * (height // SPATIAL_COMPRESSION) * (width // SPATIAL_COMPRESSION),
    }
    stage_n = {stage: ltx_fast_lowest_bucket_n(stage, n_real) for stage, n_real in stage_n_real.items()}

    return LTXFastBucketRoute(
        bucket=LTX_FAST_LOWEST_BUCKET,
        duration_seconds=duration,
        latent_frames=latent_frames,
        stage_video_n_real=stage_n_real,
        stage_video_n=stage_n,
    )


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
