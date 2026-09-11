# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass
from io import BytesIO

import torch

from .patchifiers import AudioLatentShape, VideoPixelShape

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


# =============================================================================
# Trace bucket ladder (sequence-length bucketing for trace reuse)
# =============================================================================
#
# A ttnn trace bakes every activation shape, so without bucketing each (canvas, fps, duration)
# would need its own captured denoise trace per stage. Instead the video token sequence is
# zero-padded up to the smallest ladder rung that fits and the real length is handed to ring
# SDPA as a one-element device tensor (``logical_n``), read on every replay. One capture per rung
# then serves every config that lands on it. The pad tail costs memory everywhere and compute in
# the row-independent matmuls (QKV/out/FFN/AdaLN), but ring SDPA masks and skips it, so attention
# -- the only O(N^2) piece -- pays nothing for the padding.
#
# The ladder is shared by both distilled stages: s1 and s2 run the same ``inner_step``, so a trace
# is keyed purely by its padded (video_N, audio_N) shapes, not by which stage is replaying it.

# SP=8 on the 4x8 Galaxy: every rung must divide by ttnn.TILE_SIZE * sp_factor so the SP shard is
# tile-aligned. Kept numeric here so this module stays importable without ttnn.
LTX_BUCKET_SP_FACTOR = 8
LTX_BUCKET_ALIGN = 32 * LTX_BUCKET_SP_FACTOR

LTX_FPS_VALUES = (24, 25, 48, 50)
LTX_DURATION_VALUES = (6, 8, 10, 12, 14, 16, 18, 20)
LTX_CANVASES = {
    "720p-landscape": (704, 1280),
    "720p-portrait": (1280, 704),
    "1080p-landscape": (1088, 1920),
    "1080p-portrait": (1920, 1088),
    "1440p-landscape": (1440, 2560),
    "1440p-portrait": (2560, 1440),
    "4k-landscape": (2176, 3840),
    "4k-portrait": (3840, 2176),
}
# The canvases a single Galaxy commits to serving from resident traces. 1440p/4k stay in the table
# (their token counts are still computable) but are rejected by ``route_ltx_request``.
LTX_SERVED_CANVASES = ("720p-landscape", "720p-portrait", "1080p-landscape", "1080p-portrait")

# ~1.4x geometric spacing, 256-aligned. Stage-2 real N over the served grid spans 16,720 (720p 24fps
# 6s) to 257,040 (1080p 50fps 20s); stage-1 is ~1/4 of that (4,180 to 64,260). Eleven rungs cover
# all 64 served configs at ~19% mean pad waste. Drop rungs from the top to shrink the envelope;
# never insert a rung that is not a multiple of LTX_BUCKET_ALIGN.
LTX_BUCKET_LADDER = (8704, 12288, 17408, 24576, 34560, 48384, 67840, 94976, 133120, 186368, 261120)

# Audio real N over the served grid spans 151..505 latent frames; one bucket covers all of them.
LTX_AUDIO_N_BUCKET = 512


def validate_bucket_ladder(ladder: tuple[int, ...], align: int = LTX_BUCKET_ALIGN) -> None:
    """A ladder must be non-empty, strictly increasing, and every rung a positive multiple of ``align``."""
    if not ladder:
        raise ValueError("bucket ladder must not be empty")
    if align <= 0:
        raise ValueError(f"bucket alignment must be positive, got {align}")
    previous = 0
    for rung in ladder:
        if rung <= previous:
            raise ValueError(f"bucket ladder must be strictly increasing, got {ladder}")
        if rung % align != 0:
            raise ValueError(f"bucket rung {rung} is not a multiple of {align}")
        previous = rung


def select_bucket(n: int, ladder: tuple[int, ...] = LTX_BUCKET_LADDER) -> int:
    """Return the smallest rung that holds ``n`` tokens (inclusive); raise above the top rung."""
    if n < 1:
        raise ValueError(f"logical N must be positive, got {n}")
    for rung in ladder:
        if n <= rung:
            return rung
    raise ValueError(f"logical N={n} exceeds the top bucket rung {ladder[-1]}")


def ltx_aligned_num_frames(fps: int, duration_seconds: int) -> int:
    """First VAE-compatible frame count (8k+1) covering ``duration_seconds`` at ``fps``."""
    target_frames = fps * duration_seconds
    return math.ceil((target_frames - 1) / TEMPORAL_COMPRESSION) * TEMPORAL_COMPRESSION + 1


def ltx_canvas_name(height: int, width: int) -> str | None:
    """Name of the canvas with exactly this ``(height, width)``, or None."""
    return next((name for name, hw in LTX_CANVASES.items() if hw == (height, width)), None)


def ltx_stage_video_n_real(num_frames: int, height: int, width: int) -> dict[str, int]:
    """Real (unpadded) video token count per distilled stage: s1 denoises at half resolution."""
    lf, lh, lw = latent_grid(num_frames, height, width)
    _, s1_lh, s1_lw = latent_grid(num_frames, height // 2, width // 2)
    return {"s1": lf * s1_lh * s1_lw, "s2": lf * lh * lw}


def ltx_audio_n_real(num_frames: int, fps: int, height: int, width: int) -> int:
    return AudioLatentShape.from_video_pixel_shape(
        VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=fps)
    ).frames


def ltx_served_configs(
    canvases: tuple[str, ...] = LTX_SERVED_CANVASES,
    fps_values: tuple[int, ...] = LTX_FPS_VALUES,
    durations: tuple[int, ...] = LTX_DURATION_VALUES,
) -> tuple[tuple[str, int, int], ...]:
    """The ``(canvas, fps, duration_seconds)`` product grid a deployment serves."""
    for canvas in canvases:
        if canvas not in LTX_CANVASES:
            raise ValueError(f"unknown LTX canvas {canvas!r}; known: {tuple(LTX_CANVASES)}")
    return tuple((canvas, fps, duration) for canvas in canvases for fps in fps_values for duration in durations)


@dataclass(frozen=True)
class LTXBucketRoute:
    """Where one request lands on the ladder: per-stage real lengths and the rung each stage replays."""

    canvas: str
    fps: int
    num_frames: int
    latent_frames: int
    stage_video_n_real: dict[str, int]
    stage_rung: dict[str, int]
    audio_n_real: int
    audio_n: int

    def trace_key(self, stage: str) -> int:
        if stage not in self.stage_rung:
            raise ValueError(f"unknown LTX stage {stage!r}; expected one of {tuple(self.stage_rung)}")
        return self.stage_rung[stage]

    def video_n(self, stage: str) -> int:
        return self.trace_key(stage)

    def video_n_real(self, stage: str) -> int:
        if stage not in self.stage_video_n_real:
            raise ValueError(f"unknown LTX stage {stage!r}; expected one of {tuple(self.stage_video_n_real)}")
        return self.stage_video_n_real[stage]

    @property
    def rungs(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.stage_rung.values())))


def route_ltx_request(
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: int,
    sp_factor: int,
    mode: str = "av",
    image_conditioned: bool = False,
    ladder: tuple[int, ...] = LTX_BUCKET_LADDER,
    served_canvases: tuple[str, ...] | None = LTX_SERVED_CANVASES,
    audio_n_bucket: int = LTX_AUDIO_N_BUCKET,
) -> LTXBucketRoute:
    """Route an exact T2V AV request onto the trace ladder.

    Rejects (instead of silently aliasing a trace) anything outside the served envelope: a canvas
    not in ``served_canvases``, a frame count that is not ``8k+1``, I2V (separate trace class:
    per-token timestep modulation), non-AV mode, and a mesh whose SP factor does not match the
    ladder alignment. ``served_canvases=None`` accepts any 64-aligned ``(height, width)`` (the
    canvas is then named ``"{height}x{width}"``); the ladder bound still applies.
    """
    if sp_factor != LTX_BUCKET_SP_FACTOR:
        raise ValueError(f"LTX trace buckets are laid out for SP={LTX_BUCKET_SP_FACTOR}, got SP={sp_factor}")
    if mode != "av":
        raise ValueError(f"LTX trace buckets support AV mode only, got mode={mode!r}")
    if image_conditioned:
        raise ValueError("LTX trace buckets support T2V only; I2V requires a separate trace class")
    validate_bucket_ladder(ladder, 32 * sp_factor)

    canvas = ltx_canvas_name(height, width)
    if served_canvases is None:
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"height/width must be multiples of 64, got {(height, width)}")
        canvas = canvas or f"{height}x{width}"
    elif canvas is None or canvas not in served_canvases:
        served = {name: LTX_CANVASES[name] for name in served_canvases}
        raise ValueError(f"canvas {(height, width)} is not served from resident traces; served: {served}")
    if num_frames < 1 or (num_frames - 1) % TEMPORAL_COMPRESSION != 0:
        raise ValueError(f"num_frames must be 8k+1 for the LTX VAE, got {num_frames}")
    if fps not in LTX_FPS_VALUES:
        raise ValueError(f"fps {fps} is not served; supported: {LTX_FPS_VALUES}")

    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    stage_n_real = ltx_stage_video_n_real(num_frames, height, width)
    stage_rung = {stage: select_bucket(n_real, ladder) for stage, n_real in stage_n_real.items()}
    audio_n_real = ltx_audio_n_real(num_frames, fps, height, width)
    if audio_n_real > audio_n_bucket:
        raise ValueError(f"audio N={audio_n_real} exceeds the audio bucket {audio_n_bucket}")

    return LTXBucketRoute(
        canvas=canvas,
        fps=fps,
        num_frames=num_frames,
        latent_frames=latent_frames,
        stage_video_n_real=stage_n_real,
        stage_rung=stage_rung,
        audio_n_real=audio_n_real,
        audio_n=audio_n_bucket,
    )


def route_ltx_config(
    canvas: str, fps: int, duration_seconds: int, *, sp_factor: int = LTX_BUCKET_SP_FACTOR, **kwargs
) -> LTXBucketRoute:
    """``route_ltx_request`` for a ``(canvas, fps, duration)`` grid entry."""
    height, width = LTX_CANVASES[canvas]
    return route_ltx_request(
        num_frames=ltx_aligned_num_frames(fps, duration_seconds),
        height=height,
        width=width,
        fps=fps,
        sp_factor=sp_factor,
        **kwargs,
    )


def rungs_for_configs(
    configs: tuple[tuple[str, int, int], ...],
    *,
    sp_factor: int = LTX_BUCKET_SP_FACTOR,
    ladder: tuple[int, ...] = LTX_BUCKET_LADDER,
    served_canvases: tuple[str, ...] = LTX_SERVED_CANVASES,
) -> tuple[int, ...]:
    """The sorted set of rungs (== denoise traces) needed to serve ``configs`` on this ladder."""
    rungs: set[int] = set()
    for canvas, fps, duration in configs:
        route = route_ltx_config(
            canvas, fps, duration, sp_factor=sp_factor, ladder=ladder, served_canvases=served_canvases
        )
        rungs.update(route.stage_rung.values())
    return tuple(sorted(rungs))


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
