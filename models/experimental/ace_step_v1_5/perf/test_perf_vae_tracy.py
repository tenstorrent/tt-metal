# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy-oriented performance harness for ACE-Step **VAE decode** (Priority 3).

Profiles the once-per-``generate()`` chunked Oobleck decode path:

    ``TtOobleckVaeDecoder.decode_tiled`` (``ttnn_impl/oobleck_vae_decoder.py``)
    → ``TtOobleckDecoder`` (``ttnn_impl/vae/*``)

This matches ``AceStepE2EModel.decode_vae`` in ``ttnn_impl/e2e_model_tt.py``: latents stay on
device after the DiT denoise loop, then ``decode_tiled`` slices along time with overlap-add using
``ACE_STEP_VAE_CHUNK_LATENTS`` / ``ACE_STEP_VAE_OVERLAP_LATENTS``.

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/experimental/ace_step_v1_5/perf/test_perf_vae_tracy.py::test_perf_ace_step_vae_tracy_profile \\
        -v -s

CSV / Tracy artifacts land under ``generated/profiler/reports/<timestamp>/`` — see
``docs/source/tt-metalium/tools/tracy_profiler.rst``.

Production defaults (``math_perf_env``): LoFi conv compute + BF16 activations/weights + L1 interleaved
buffers on conv / Snake / overlap-add glue — same path as PCC (no env toggle).

``bfloat8_b`` conv/Snake **compute** is **on by default** (inter-op buffers stay BF16 ``ROW_MAJOR``).
Set ``ACE_STEP_VAE_BFLOAT8_ACTIVATIONS=0`` to fall back to full BF16 compute.

Large ``1×1`` conv im2col matmuls (``61440×128``, ``30720×128``, ``7680×256``) use tuned 1D
reuse (**tall-M** ``MultiCast1D``, ``mcast_in0=0``) by default — including under Tracy — so
``ttnn.conv1d`` does not probe 640 M-tiles as 640 cores. Set ``ACE_STEP_VAE_LARGE_M_MATMUL=0``
to force the default conv path.

**TILE layout contracts (residual unit, no env toggle):**

- **conv1(k=7)→snake2:** ``return_sharded=True`` → HEIGHT_SHARDED L1 TILE when
  ``ACE_STEP_VAE_K7_SHARDED_OUTPUT=1``, else **DRAM TILE** (automatic ``return_tile`` fallback).
  Tracy targets: ``TilizeDeviceOperation|1920×512|DRAM→L1`` on snake2 input.
- **snake2→conv2(k=1):** ``snake2(..., return_tile=True)`` → **L1 TILE** out; ``TtConv1d`` accepts
  TILE L1 in0 on ``ttnn.linear`` (no Tilize) or untilizes once before ``ttnn.conv1d``.
  Tracy targets: ``UntilizeDeviceOperation`` on snake2 output + ``TilizeDeviceOperation`` on conv2 input.


**Important:** do **not** set ``ACE_STEP_USE_TRACE=1`` for this test (device profiler flush and TTNN
trace capture are incompatible — same constraint as the DiT / conditioning Tracy harnesses).

Optional environment variables:

- ``ACE_STEP_CKPT_DIR``: checkpoint root (default: Hugging Face ACE-Step cache).
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control (same as
  other ACE perf tests). Only the ``vae/`` bundle is required for the production path.
- ``ACE_STEP_VAE_PERF_TINY=1``: random tiny Oobleck decoder (same shape envelope as
  ``test_vae_decoder_pcc.py``) — fast smoke without hub weights; not representative of production
  channel counts.
- ``ACE_STEP_VAE_PERF_DURATION_SEC`` (default ``10.0``): audio duration → ``frames = 25 × sec``.
  Default exceeds ``ACE_STEP_VAE_CHUNK_LATENTS`` so the timed pass exercises multi-tile overlap-add.
- ``ACE_STEP_VAE_PERF_FRAMES``: override latent frame count directly (wins over duration).
- ``ACE_STEP_VAE_CHUNK_LATENTS`` (default ``32``): max latent time per decode tile — same env as E2E.
- ``ACE_STEP_VAE_OVERLAP_LATENTS`` (default ``4``): latent overlap between tiles — same env as E2E.
- ``ACE_STEP_VAE_PERF_MODE`` (default ``tiled``):
  - ``tiled``: ``decode_tiled`` (production once-per-generate path).
  - ``single``: one-shot ``TtOobleckVaeDecoder.__call__`` when ``frames <= chunk_size`` (debug only).
- ``ACE_STEP_VAE_PERF_ITERS`` (default ``4``): timed perf-pass iterations.
- ``ACE_STEP_PERF_WARMUP`` (default ``1``): extra decode iterations before the timed pass.
- ``ACE_STEP_PERF_SEED`` (default ``0``): RNG seed for synthetic latents.
- ``ACE_STEP_TRACY_EACH_VAE_ITER``: set to ``1`` for one Tracy signpost per perf iteration.
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf iterations (default ``1``).
- ``ACE_STEP_PROFILER_FLUSH_EVERY_LAYER`` (default ``1`` when profiling): drain profiler rings after
  each VAE layer / decode tile so Tracy merge finds ops in ``cpp_device_perf_report.csv``.
- ``ACE_STEP_PERF_MAX_SECONDS``: optional wall-time budget on the timed perf pass.

If Tracy's merge step fails with ``Device data missing``, run without ``-p`` for host-only timelines, or
post-process with ``python tools/tracy/process_ops_logs.py --date``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import Profiler
from models.experimental.ace_step_v1_5.demo.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.experimental.ace_step_v1_5.torch_ref.vae.oobleck_decoder import OobleckDecoder
from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
)
from models.experimental.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder
from models.experimental.ace_step_v1_5.ttnn_impl.vae import TtOobleckDecoder

_PRODUCTION_LATENT_CHANNELS = 64


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def _resolve_vae_dir(ckpt_dir: Path) -> Path:
    return ckpt_dir / "vae"


def _vae_checkpoint_ready(vae_dir: Path) -> bool:
    if not (vae_dir / "config.json").is_file():
        return False
    for name in ("model.safetensors", "diffusion_pytorch_model.safetensors", "model.fp16.safetensors"):
        if (vae_dir / name).is_file():
            return True
    return bool(list(vae_dir.glob("*.safetensors")))


def _tracy_signpost(label: str) -> None:
    if _is_ci():
        return
    try:
        from tracy import signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        signpost(label)
    except Exception:
        pass


def _resolve_latent_frames(*, tiny: bool) -> int:
    if "ACE_STEP_VAE_PERF_FRAMES" in os.environ:
        return max(1, int(os.environ["ACE_STEP_VAE_PERF_FRAMES"]))
    default_duration = "1.0" if tiny else "10.0"
    duration = float(os.environ.get("ACE_STEP_VAE_PERF_DURATION_SEC", default_duration))
    return max(1, int(round(duration * 25.0)))


def _resolve_chunk_overlap() -> tuple[int, int]:
    chunk = int(os.environ.get("ACE_STEP_VAE_CHUNK_LATENTS", "32"))
    overlap = int(os.environ.get("ACE_STEP_VAE_OVERLAP_LATENTS", "4"))
    return chunk, overlap


def _effective_overlap(chunk_size: int, overlap: int) -> int:
    min_overlap = 4
    effective = int(overlap)
    while chunk_size - 2 * effective <= 0 and effective > min_overlap:
        effective //= 2
    if effective < min_overlap and overlap >= min_overlap:
        effective = min_overlap
    return effective


def _estimate_num_tiles(*, latent_frames: int, chunk_size: int, overlap: int) -> int:
    if latent_frames <= chunk_size:
        return 1
    effective = _effective_overlap(chunk_size, overlap)
    stride = chunk_size - 2 * effective
    if stride <= 0:
        return 1
    return int(math.ceil(latent_frames / stride))


@dataclass(frozen=True)
class _TinyDecoderConfig:
    """Reduced Oobleck config (matches ``test_vae_decoder_pcc.py`` / trace test)."""

    decoder_channels: int = 32
    decoder_input_channels: int = 16
    audio_channels: int = 2
    upsampling_ratios: tuple = (2, 2)
    channel_multiples: tuple = (1, 2)


def _build_tiny_vae(*, device) -> tuple[TtOobleckVaeDecoder, int]:
    cfg = _TinyDecoderConfig()
    torch.manual_seed(11)
    torch_dec = OobleckDecoder(
        channels=cfg.decoder_channels,
        input_channels=cfg.decoder_input_channels,
        audio_channels=cfg.audio_channels,
        upsampling_ratios=cfg.upsampling_ratios,
        channel_multiples=cfg.channel_multiples,
    ).eval()
    full_sd = {f"decoder.{k}": v for k, v in torch_dec.state_dict().items()}
    tt_dec = TtOobleckDecoder(
        state_dict=full_sd,
        device=device,
        decoder_prefix="decoder.",
        channels=cfg.decoder_channels,
        input_channels=cfg.decoder_input_channels,
        audio_channels=cfg.audio_channels,
        upsampling_ratios=cfg.upsampling_ratios,
        channel_multiples=cfg.channel_multiples,
    )
    return TtOobleckVaeDecoder(tt_dec), cfg.decoder_input_channels


def _make_device_latents(
    *,
    device,
    batch: int,
    frames: int,
    c_lat: int,
    seed: int,
) -> ttnn.Tensor:
    """Synthetic denoised latents on device — TILE float32 like ``run_ttnn_denoise_loop`` output."""
    rng = np.random.default_rng(int(seed))
    lat_np = rng.standard_normal((batch, frames, c_lat), dtype=np.float32)
    return ttnn.as_tensor(
        lat_np,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_vae_tracy_harness(
    device,
    *,
    vae: TtOobleckVaeDecoder,
    latent_frames: int,
    c_lat: int,
    variant_label: str,
) -> None:
    perf_mode = os.environ.get("ACE_STEP_VAE_PERF_MODE", "tiled").strip().lower()
    if perf_mode not in ("tiled", "single"):
        pytest.fail(f"Unknown ACE_STEP_VAE_PERF_MODE={perf_mode!r}; use 'tiled' or 'single'.")

    iters = max(1, int(os.environ.get("ACE_STEP_VAE_PERF_ITERS", "4")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))
    trace_each_iter = os.environ.get("ACE_STEP_TRACY_EACH_VAE_ITER", "").lower() in ("1", "true", "yes")
    try:
        flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        flush_every = 1

    chunk, overlap = _resolve_chunk_overlap()
    num_tiles = _estimate_num_tiles(latent_frames=latent_frames, chunk_size=chunk, overlap=overlap)

    if perf_mode == "single" and latent_frames > chunk:
        pytest.skip(
            f"ACE_STEP_VAE_PERF_MODE=single requires frames <= chunk ({latent_frames} > {chunk}). "
            "Use tiled mode or shorten ACE_STEP_VAE_PERF_FRAMES."
        )

    profiler = Profiler()
    profiler.clear()
    is_ci = _is_ci()

    def _decode_once(lat_tt: ttnn.Tensor) -> ttnn.Tensor:
        if perf_mode == "tiled":
            return vae.decode_tiled(lat_tt, chunk_size=chunk, overlap=overlap)
        return vae(lat_tt)

    # --- INIT -------------------------------------------------------------------------
    profiler.disable()
    profiler.start("ace_step_vae_init", force_enable=True)
    _tracy_signpost("VAE_INIT")

    lat_tt = _make_device_latents(device=device, batch=1, frames=latent_frames, c_lat=c_lat, seed=seed)

    profiler.end("ace_step_vae_init", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    # --- COMPILE PASS -----------------------------------------------------------------
    profiler.start("ace_step_vae_compile_pass", force_enable=True)
    _tracy_signpost("VAE_COMPILE_PASS")
    wav_tt = _decode_once(lat_tt)
    try:
        ttnn.deallocate(wav_tt)
    except Exception:
        pass
    profiler.end("ace_step_vae_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    # --- WARMUP -----------------------------------------------------------------------
    profiler.start("ace_step_vae_warmup", force_enable=True)
    _tracy_signpost("VAE_WARMUP")
    for _ in range(warmup):
        wav_tt = _decode_once(lat_tt)
        try:
            ttnn.deallocate(wav_tt)
        except Exception:
            pass
    profiler.end("ace_step_vae_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    # --- PERF PASS --------------------------------------------------------------------
    profiler.enable()
    profiler.start("ace_step_vae_perf_pass")
    _tracy_signpost("VAE_PERF_PASS")

    for iter_idx in range(iters):
        if trace_each_iter and not is_ci:
            _tracy_signpost(f"VAE_ITER_{iter_idx}")
        wav_tt = _decode_once(lat_tt)
        try:
            ttnn.deallocate(wav_tt)
        except Exception:
            pass
        if flush_every > 0 and (iter_idx + 1) % flush_every == 0:
            ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_vae_perf_pass")
    ace_step_flush_device_profiler(device)

    try:
        ttnn.deallocate(lat_tt)
    except Exception:
        pass

    profiler.print()
    init_wall = profiler.get("ace_step_vae_init")
    compile_wall = profiler.get("ace_step_vae_compile_pass")
    warmup_wall = profiler.get("ace_step_vae_warmup")
    perf_wall = profiler.get("ace_step_vae_perf_pass")
    per_iter_ms = (perf_wall * 1000.0 / max(1, iters)) if iters else 0.0

    logger.info(
        "AceStep VAE Tracy harness (mode={}, variant={}, frames={}, c_lat={}, "
        "chunk={}, overlap={}, est_tiles={}, iters={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/iter)",
        perf_mode,
        variant_label,
        latent_frames,
        c_lat,
        chunk,
        overlap,
        num_tiles,
        iters,
        init_wall,
        compile_wall,
        warmup,
        warmup_wall,
        perf_wall,
        per_iter_ms,
    )

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert perf_wall <= float(budget), f"VAE perf pass exceeded ACE_STEP_PERF_MAX_SECONDS={budget}"


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_vae_tracy_profile(device, request):
    """Profile chunked Oobleck VAE decode (once per generate) with Tracy signposts."""
    ace_step_enable_tracy_profiler_env()

    def _final_profiler_flush() -> None:
        for _ in range(2):
            ace_step_flush_device_profiler(device)

    request.addfinalizer(_final_profiler_flush)

    use_tiny = os.environ.get("ACE_STEP_VAE_PERF_TINY", "").lower() in ("1", "true", "yes")
    latent_frames = _resolve_latent_frames(tiny=use_tiny)

    if use_tiny:
        vae, c_lat = _build_tiny_vae(device=device)
        ace_step_flush_device_profiler(device)
        _run_vae_tracy_harness(
            device,
            vae=vae,
            latent_frames=latent_frames,
            c_lat=c_lat,
            variant_label="tiny",
        )
        return

    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    vae_dir = _resolve_vae_dir(ckpt_root)

    if not _vae_checkpoint_ready(vae_dir):
        if _perf_download_disabled():
            pytest.skip(
                f"Missing VAE checkpoint under {vae_dir}. "
                "Unset ACE_STEP_PERF_NO_DOWNLOAD to fetch, or set ACE_STEP_VAE_PERF_TINY=1."
            )
        pytest.importorskip("huggingface_hub")
        ckpt_root.mkdir(parents=True, exist_ok=True)
        logger.info("ACE-Step VAE perf: fetching vae bundle …")
        try:
            _ensure_variant("vae", ckpt_root)
        except Exception as exc:
            pytest.skip(f"Hugging Face download failed: {exc}")
        if not _vae_checkpoint_ready(vae_dir):
            pytest.fail(f"Download finished but VAE weights still missing under {vae_dir}")

    act_dtype = ttnn.bfloat16
    vae = TtOobleckVaeDecoder.from_hf_vae_dir(
        str(vae_dir),
        device=device,
        latent_frames=latent_frames,
        batch_size=1,
        activation_dtype=act_dtype,
        weights_dtype=act_dtype,
    )
    ace_step_flush_device_profiler(device)
    _run_vae_tracy_harness(
        device,
        vae=vae,
        latent_frames=latent_frames,
        c_lat=_PRODUCTION_LATENT_CHANNELS,
        variant_label="production",
    )
