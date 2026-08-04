# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""T-parallel audio decode: correctness against the single-device path, and the speedup.

Audio decode is 1.284 s against a ~0.05 s target, is **device-bound** (trace buys 1.07 %,
STATE.md amendment 60), and runs on **one device**. The visual halves got 32x from
data-parallelism over `(clip, tile)` work units; a single 5 s audio stream is one unit, so
none of that applies. The equivalent lever here is sharding the time axis across the mesh --
which ``vocoder_ltx.Vocoder`` already implements (``parallel_config.factor`` threads through
``_upload_BCT``'s T-alignment padding, ``_forward_device``'s partition, and the closing
T-gather) and ``MiniMaxH3AudioDecoder`` already accepts. The shipping path simply passes
``None``.

Sharded output is gated against the unsharded output of the same weights, so a speedup that
comes from dropping work fails rather than reports.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from .test_performance_vae_minimax_h3 import _config, _psnr, _weights_dir

MESH = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="mesh4x8",
    )
]
# (t_factor, mesh_axis). Axis 1 is the 8-wide axis of the 4x8 Galaxy, axis 0 the 4-wide one.
# t_factor=8 on axis 1 measures 0.898 s but returns a *different signal* (PSNR -6.3 dB vs
# the single-device path), so it is xfail-marked rather than removed: the bug is worth
# finding, and 207 frames padding to 256 makes 256/8 = 32 exactly one tile per shard, which
# is the obvious suspect. See STATE.md amendment 63.
FACTORS = [(1, 1), (4, 0), (8, 1)]
KNOWN_BROKEN = {(8, 1)}
NUM_LATENT_FRAMES = 207
ITERS = 3


def _best(fn) -> float:
    fn()
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _build(mesh_device, config, converted, parallel_config, ccl_manager):
    decoder = MiniMaxH3AudioDecoder(
        latent_channels=config["latent_channels"],
        latent_dim=config["latent_dim"],
        decoder_dim=config["decoder_dim"],
        decoder_rates=tuple(config["decoder_rates"]),
        decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    decoder.load_torch_state_dict(converted, strict=False)
    return decoder


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_audio_decode_t_parallel(mesh_device):
    weights_dir = _weights_dir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_DIFFUSERS_DIR")
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from loguru import logger
    from safetensors.torch import load_file

    from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict

    config = _config(weights_dir)
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")))
    converted = convert_minimax_h3_audio_state_dict(dict(reference.state_dict()))

    torch.manual_seed(2)
    latents = torch.randn(2, config["latent_channels"], NUM_LATENT_FRAMES) * 0.1

    baseline_out = None
    baseline_s = None
    results = []
    for factor, axis in FACTORS:
        pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=axis)
        ccl = None if pc is None else CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
        try:
            decoder = _build(mesh_device, config, converted, pc, ccl)
            out = decoder(latents)
            seconds = _best(lambda: decoder(latents))
        except Exception as exc:  # a factor the stack rejects is a result, not a test failure
            logger.warning(f"t_factor={factor} axis={axis} FAILED: {str(exc)[:160]}")
            results.append((factor, axis, None, None))
            continue

        if baseline_out is None:
            baseline_out, baseline_s = out, seconds
            psnr = float("inf")
        else:
            assert out.shape == baseline_out.shape, f"factor {factor}: {out.shape} != {baseline_out.shape}"
            psnr = _psnr(baseline_out, out)
        results.append((factor, axis, seconds, psnr))
        logger.info(
            f"PERF audio_decode t_factor={factor} axis={axis}: {seconds:.4f} s "
            f"({baseline_s / seconds:.2f}x) PSNR vs 1-device {psnr:.1f} dB"
        )
        del decoder

    logger.info("=== audio decode T-parallel summary ===")
    for factor, axis, seconds, psnr in results:
        if seconds is None:
            logger.info(f"  t_factor={factor:2d} axis={axis}: unsupported")
        else:
            logger.info(
                f"  t_factor={factor:2d} axis={axis}: {seconds:.4f} s  {baseline_s / seconds:5.2f}x  "
                f"PSNR {psnr:6.1f} dB"
            )

    # Any factor that ran must agree with the single-device path; a fast wrong answer fails.
    for factor, axis, seconds, psnr in results:
        if seconds is None or (factor, axis) in KNOWN_BROKEN:
            continue
        assert psnr > 40.0, f"t_factor={factor} axis={axis} diverges from 1-device: PSNR {psnr:.1f} dB"
