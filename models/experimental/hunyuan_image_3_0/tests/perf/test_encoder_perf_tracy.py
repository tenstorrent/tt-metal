# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for Hunyuan VAE encoder across input spatial sizes.

The encoder input is ``[B, C, T, H, W]`` with fixed ``T=PIXEL_T`` (4). For I2I cond
encode the variable dimension is spatial ``H×W`` (not transformer ``seq_len``). We
sweep from the smallest to the largest VAE resolution in the model's
``ResolutionGroup`` (defaults: 512² → 2048² when ``image_base_size=1024``).

**Recommended for optimization / CSV analysis** — one encoder, one resolution, ``start``/``stop``
signposts (default 1024²):

    HUNYUAN_MODEL_DIR=/path/to/HunyuanImage-3.0 \\
    python_env/bin/python -m tracy -p -r -v --op-support-count 10000 -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_encoder_perf_tracy.py \\
      -k test_encoder_perf_tracy_one -s --timeout=0

Multi-size sweep (all resolutions in one capture):

    HUNYUAN_MODEL_DIR=/path/to/HunyuanImage-3.0 \\
    python_env/bin/python -m tracy -p -r -v --op-support-count 25000 -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_encoder_perf_tracy.py \\
      -k sweep -s --timeout=0

The full size sweep emits ~15k device ops per chip on a 2×2 mesh (default profiler
cap is 1000). Without a higher ``--op-support-count``, Tracy's ``-r`` post-process
fails with "Device data missing: Op … not present in cpp_device_perf_report.csv".
The test module also sets ``TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`` if unset.

Parametrized single-size (filter with ``-k 1024x1024`` etc.):

    HUNYUAN_MODEL_DIR=/path/to/HunyuanImage-3.0 \\
    python_env/bin/python -m tracy -p -r -v --op-support-count 10000 -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_encoder_perf_tracy.py \\
      -k "test_encoder_perf_tracy_single_size and 1024x1024" -s --timeout=0

Environment overrides:

    HUNYUAN_MODEL_DIR=/path/to/HunyuanImage-3.0   # checkpoint (else HF hub / auto-download)
    HY_ENCODER_PERF_H=1024 HY_ENCODER_PERF_W=1024  # one-shot test resolution (default 1024²)
    HY_ENCODER_PERF_BASE_SIZE=1024   # ResolutionGroup base (default 1024)
    HY_ENCODER_PERF_SIZES=512,1024,2048   # comma-separated squares; overrides group sweep
    HY_ENCODER_PERF_STEP=512         # square sweep step (default 512²,1024²,1536²,2048²)
    HY_ENCODER_PERF_FULL_RESO=1      # use every ResolutionGroup aspect ratio (33+ sizes)
    HY_ENCODER_PERF_ITERS=3          # timed forward passes per signpost window
    HY_ENCODER_W_SPATIAL=1           # H/W spatial on 2×2 (axis0=H, axis1=W); conv halos + dist GN
    HY_ENCODER_PERF_WARMUP=1         # warmup passes before each timed window
    HY_ENCODER_PERF_MIN_ONLY=1       # only smallest size (quick smoke)
    HY_ENCODER_PERF_MAX_ONLY=1       # only largest size
"""

from __future__ import annotations

import os

# Full encoder sweep on 2×2 mesh exceeds the default profiler program cap (~1000).
os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "25000")

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hunyuan_image_3_0.ref.tokenizer.resolution import ResolutionGroup
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import IN_CHANNELS, PIXEL_T
from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights
from models.experimental.hunyuan_image_3_0.tests.pcc.vae_helpers import (
    pad_encoder_channels_bcthw,
    upload_bcthw,
    upload_bcthw_spatial,
)
from models.experimental.hunyuan_image_3_0.tt.vae.encoder import VAEEncoderTTNN


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _encoder_input_volume(pixel_h: int, pixel_w: int) -> int:
    """``T × H × W`` elements per batch row (proxy for spatial 'seq' volume)."""
    return PIXEL_T * pixel_h * pixel_w


def encoder_perf_spatial_sizes() -> list[tuple[int, int]]:
    """Return ``(H, W)`` pairs sorted smallest → largest by input volume."""
    custom = os.environ.get("HY_ENCODER_PERF_SIZES")
    if custom:
        sizes = []
        for part in custom.split(","):
            part = part.strip()
            if not part:
                continue
            if "x" in part.lower():
                h_s, w_s = part.lower().split("x", 1)
                sizes.append((int(h_s), int(w_s)))
            else:
                s = int(part)
                sizes.append((s, s))
        return sorted(set(sizes), key=lambda hw: _encoder_input_volume(hw[0], hw[1]))

    base = _env_int("HY_ENCODER_PERF_BASE_SIZE", 1024)
    if os.environ.get("HY_ENCODER_PERF_FULL_RESO") == "1":
        group = ResolutionGroup(base_size=base, align=16)
        sizes = sorted({(r.h, r.w) for r in group.data}, key=lambda hw: _encoder_input_volume(hw[0], hw[1]))
    else:
        # Square sweep: 512² … 2048² by default (step 512 → 4 points).
        lo = base // 2
        hi = base * 2
        step = _env_int("HY_ENCODER_PERF_STEP", 512)
        sizes = [(s, s) for s in range(lo, hi + 1, step)]

    if os.environ.get("HY_ENCODER_PERF_MIN_ONLY") == "1":
        return [sizes[0]]
    if os.environ.get("HY_ENCODER_PERF_MAX_ONLY") == "1":
        return [sizes[-1]]
    return sizes


def encoder_perf_one_size() -> tuple[int, int]:
    """Return ``(H, W)`` for the one-shot encoder perf test (default ``1024×1024``)."""
    h = os.environ.get("HY_ENCODER_PERF_H")
    w = os.environ.get("HY_ENCODER_PERF_W")
    if h is not None and w is not None:
        return int(h), int(w)

    custom = os.environ.get("HY_ENCODER_PERF_SIZES")
    if custom:
        part = custom.split(",")[0].strip()
        if part:
            if "x" in part.lower():
                h_s, w_s = part.lower().split("x", 1)
                return int(h_s), int(w_s)
            s = int(part)
            return s, s

    if os.environ.get("HY_ENCODER_PERF_MAX_ONLY") == "1":
        return encoder_perf_spatial_sizes()[-1]
    if os.environ.get("HY_ENCODER_PERF_MIN_ONLY") == "1":
        return encoder_perf_spatial_sizes()[0]

    base = _env_int("HY_ENCODER_PERF_BASE_SIZE", 1024)
    return base, base


def _make_encoder_input(pixel_h: int, pixel_w: int) -> torch.Tensor:
    x = torch.randn(1, IN_CHANNELS, PIXEL_T, pixel_h, pixel_w, dtype=torch.float32)
    return pad_encoder_channels_bcthw(x)


def _run_encoder_forward(
    mesh_device: ttnn.MeshDevice,
    encoder: VAEEncoderTTNN,
    x_bcthw: torch.Tensor,
    *,
    w_spatial: bool = False,
) -> None:
    if w_spatial:
        x_bthwc = upload_bcthw_spatial(mesh_device, x_bcthw)
    else:
        x_bthwc = upload_bcthw(mesh_device, x_bcthw)
    out = encoder(x_bthwc)
    ttnn.deallocate(x_bthwc, force=False)
    ttnn.deallocate(out, force=False)


def _profile_encoder_forward(
    mesh_device: ttnn.MeshDevice,
    pixel_h: int,
    pixel_w: int,
    *,
    warmup: int,
    iters: int,
    start_signpost: str = "start",
    stop_signpost: str = "stop",
) -> None:
    """Build one encoder, warm up, then time ``iters`` forwards inside signposts."""
    from tracy import signpost

    tag = f"{pixel_h}x{pixel_w}"
    vol = _encoder_input_volume(pixel_h, pixel_w)
    logger.info(
        f"[encoder perf] {tag}  T*H*W={vol}  shape=[1,{IN_CHANNELS},{PIXEL_T},{pixel_h},{pixel_w}]  "
        f"warmup={warmup} iters={iters}"
    )

    mesh_device.enable_program_cache()

    from models.experimental.hunyuan_image_3_0.tt.vae.spatial import encoder_w_spatial_enabled
    from models.tt_dit.parallel.manager import CCLManager

    w_spatial = encoder_w_spatial_enabled()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear) if w_spatial else None
    encoder = VAEEncoderTTNN(
        mesh_device,
        pixel_t=PIXEL_T,
        pixel_h=pixel_h,
        pixel_w=pixel_w,
        ccl_manager=ccl,
        h_mesh_axis=0 if w_spatial else None,
        w_mesh_axis=1 if w_spatial else None,
    )
    x = _make_encoder_input(pixel_h, pixel_w)

    for _ in range(warmup):
        _run_encoder_forward(mesh_device, encoder, x, w_spatial=w_spatial)
    ttnn.synchronize_device(mesh_device)

    signpost(start_signpost)
    for _ in range(iters):
        _run_encoder_forward(mesh_device, encoder, x, w_spatial=w_spatial)
    signpost(stop_signpost)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"[encoder perf] done {tag}")


@pytest.fixture(scope="session", autouse=True)
def _hunyuan_checkpoint():
    """Resolve VAE weights before device tests (env override, HF cache, or download)."""
    return ensure_base_weights()


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _sweep_ids() -> list[str]:
    return [f"{h}x{w}" for h, w in encoder_perf_spatial_sizes()]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_encoder_perf_tracy_one(mesh_device):
    """One encoder instance at one resolution — best for Tracy CSV / conv blocking work.

    Default resolution is ``1024×1024`` (``HY_ENCODER_PERF_BASE_SIZE``). Override with
    ``HY_ENCODER_PERF_H`` / ``HY_ENCODER_PERF_W`` or a single ``HY_ENCODER_PERF_SIZES`` entry.
    Uses plain ``start``/``stop`` signposts for the standard ops report analyzer.
    """
    pixel_h, pixel_w = encoder_perf_one_size()
    warmup = _env_int("HY_ENCODER_PERF_WARMUP", 2)
    iters = _env_int("HY_ENCODER_PERF_ITERS", 5)
    _profile_encoder_forward(mesh_device, pixel_h, pixel_w, warmup=warmup, iters=iters)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_encoder_perf_tracy_sweep(mesh_device):
    """One Tracy capture: all encoder input sizes from smallest spatial volume to largest."""
    sizes = encoder_perf_spatial_sizes()
    warmup = _env_int("HY_ENCODER_PERF_WARMUP", 1)
    iters = _env_int("HY_ENCODER_PERF_ITERS", 3)

    logger.info(
        f"VAE encoder Tracy sweep: {len(sizes)} sizes, "
        f"volume T*H*W from {_encoder_input_volume(*sizes[0])} to {_encoder_input_volume(*sizes[-1])}, "
        f"warmup={warmup} iters={iters}"
    )

    for pixel_h, pixel_w in sizes:
        tag = f"{pixel_h}x{pixel_w}"
        _profile_encoder_forward(
            mesh_device,
            pixel_h,
            pixel_w,
            warmup=warmup,
            iters=iters,
            start_signpost=f"start_enc_{tag}",
            stop_signpost=f"stop_enc_{tag}",
        )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("pixel_h,pixel_w", encoder_perf_spatial_sizes(), ids=_sweep_ids())
def test_encoder_perf_tracy_single_size(mesh_device, pixel_h, pixel_w):
    """Profile one encoder resolution (filter with ``-k 1024x1024`` etc.)."""
    warmup = _env_int("HY_ENCODER_PERF_WARMUP", 1)
    iters = _env_int("HY_ENCODER_PERF_ITERS", 5)
    _profile_encoder_forward(mesh_device, pixel_h, pixel_w, warmup=warmup, iters=iters)
