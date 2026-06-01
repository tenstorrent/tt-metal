# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end audio-decode PSNR: on-device path vs host reference.

Feeds the SAME audio latent through ``LTXPipeline.decode_audio_device`` (Stage
A mel-VAE → Stage B vocoder → Stage C BWE, all on device, real LTX-2.3 22B
distilled checkpoint weights) and ``decode_audio_reference`` (CPU torch), then
checks waveform PSNR ≥ 28 dB — the acceptance bar for the on-device audio path.

This is the first test of the *full* chain with production weights end to end;
the component tests in ``test_{audio_decoder,vocoder,bwe}_ltx.py`` only check
each stage in isolation with random weights. With real weights the Stage B
magnitude divergence seen under random init does not reproduce.

Requires the distilled checkpoint (``LTX_CHECKPOINT`` or the local cache path).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")


def _resolve_checkpoint() -> str | None:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit):
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors")
    return local if os.path.exists(local) else None


def _psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """PSNR in dB, peak referenced to the reference signal's max amplitude."""
    ref = ref.float()
    test = test.float()
    mse = torch.mean((ref - test) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = ref.abs().max().item()
    if peak == 0.0:
        return float("inf")
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        ((1, 1), {}),
        # Production 2×4: Stage A replicated + Stage B/C T-shard factor 4 (larger axis)
        # + channel-TP factor 2 (smaller axis). Validates the sharded path end-to-end
        # with real weights.
        ((2, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    ids=["1x1", "2x4_prod"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_e2e_psnr(mesh_device: ttnn.MeshDevice):
    ckpt = _resolve_checkpoint()
    if ckpt is None:
        pytest.skip("distilled checkpoint not found (set LTX_CHECKPOINT)")

    torch.manual_seed(0)

    # Construct the pipeline shell WITHOUT triggering the heavy module load
    # (checkpoint_name=None skips _instantiate_modules); the audio path only
    # needs mesh_device / parallel_config / vae_ccl_manager / checkpoint_name.
    mesh_shape = tuple(mesh_device.shape)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=mesh_shape[1], mesh_axis=1),
        tensor_parallel=ParallelFactor(factor=mesh_shape[0], mesh_axis=0),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        checkpoint_name=None,
    )
    # Point the audio methods at the real checkpoint (bypasses the heavy
    # transformer/VAE instantiation the ctor would otherwise run), then build the
    # audio decoder shells the ctor skipped (checkpoint_name=None at construction).
    pipeline.checkpoint_name = ckpt
    pipeline._new_audio_decoder()

    # Representative audio latent (1, audio_N, 128). The decode path is
    # resolution-independent, so a synthetic latent exercises the full chain.
    audio_N = 256
    audio_latent = torch.randn(1, audio_N, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    ref_audio = pipeline.decode_audio_reference(audio_latent, num_frames, fps=fps)
    assert ref_audio is not None, "host reference audio decode returned None"

    # fallback=False: a device failure must surface as a test failure, not be
    # masked by the host-reference fallback (which would make PSNR trivially inf).
    dev_audio = pipeline.decode_audio_device(audio_latent, num_frames, fps=fps, fallback=False)
    assert dev_audio is not None, "on-device audio decode returned None"
    assert dev_audio.sampling_rate == ref_audio.sampling_rate

    ref_w = ref_audio.waveform.float()
    dev_w = dev_audio.waveform.float()
    # Align lengths defensively (both paths trim identically, but guard anyway).
    n = min(ref_w.shape[-1], dev_w.shape[-1])
    ref_w, dev_w = ref_w[..., :n], dev_w[..., :n]

    psnr = _psnr(ref_w, dev_w)
    logger.info(
        f"Audio decode PSNR: {psnr:.2f} dB  (ref {tuple(ref_audio.waveform.shape)} "
        f"@ {ref_audio.sampling_rate}Hz, dev {tuple(dev_audio.waveform.shape)})"
    )
    logger.info(
        f"  ref stats min={ref_w.min():.4f} max={ref_w.max():.4f} std={ref_w.std():.4f}; "
        f"dev stats min={dev_w.min():.4f} max={dev_w.max():.4f} std={dev_w.std():.4f}"
    )
    assert psnr >= 28.0, f"audio decode PSNR {psnr:.2f} dB < 28 dB"
    logger.info("PASSED: on-device audio decode PSNR ≥ 28 dB vs host reference")
