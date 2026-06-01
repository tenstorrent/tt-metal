# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for LTX audio decode on device.

Consolidates:
- test_audio_decode_perf_ltx.py
- test_audio_decode_e2e_psnr_ltx.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

_WARM_ITERS = 3


def _resolve_dev_checkpoint() -> str | None:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit):
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    return local if os.path.exists(local) else None


def _resolve_distilled_checkpoint() -> str | None:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit):
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors")
    return local if os.path.exists(local) else None


def _psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """PSNR in dB, peak referenced to the reference signal max amplitude."""
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
    "mesh_device, sp_axis, tp_axis, device_params",
    [
        ((2, 4), 1, 0, {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
        ((1, 8), 1, 0, {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    ids=["bh_2x4_sp1tp0_f4", "bh_1x8_sp1tp0_f8"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_perf(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    ckpt = _resolve_dev_checkpoint()
    if ckpt is None:
        pytest.skip("checkpoint not found (set LTX_CHECKPOINT)")

    torch.manual_seed(0)
    mesh_shape = tuple(mesh_device.shape)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        checkpoint_name=None,
    )
    pipeline.checkpoint_name = ckpt
    pipeline._new_audio_decoder()

    audio_N = 256
    audio_latent = torch.randn(1, audio_N, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    t0 = time.time()
    dev_audio = pipeline.decode_audio_device(audio_latent, num_frames, fps=fps, fallback=False)
    t_cold = time.time() - t0
    assert dev_audio is not None
    logger.info(f"COLD audio decode (build+compile+forward): {t_cold:.2f}s on mesh {mesh_shape} sp_axis={sp_axis}")

    warm = []
    for i in range(_WARM_ITERS):
        t0 = time.time()
        pipeline.decode_audio_device(audio_latent, num_frames, fps=fps, fallback=False)
        dt = time.time() - t0
        warm.append(dt)
        logger.info(f"WARM iter {i}: {dt:.2f}s")

    z = pipeline.tt_audio_decoder.z_channels
    audio_spatial = audio_latent.reshape(1, audio_N, z, audio_latent.shape[2] // z).permute(0, 2, 1, 3).float()
    a_times, bc_times = [], []
    for _ in range(_WARM_ITERS):
        t0 = time.time()
        mel = pipeline.tt_audio_decoder(audio_spatial)
        a_times.append(time.time() - t0)
        t0 = time.time()
        pipeline.tt_vocoder_with_bwe(mel)
        bc_times.append(time.time() - t0)

    logger.info(
        f"AUDIO DECODE PERF (mesh {mesh_shape}, T-shard factor={parallel_config.sequence_parallel.factor}): "
        f"cold={t_cold:.2f}s  warm_min={min(warm):.2f}s  warm_mean={sum(warm)/len(warm):.2f}s  "
        f"(out {tuple(dev_audio.waveform.shape)} @ {dev_audio.sampling_rate}Hz)"
    )
    logger.info(
        f"  STAGE BREAKDOWN (warm_min): StageA mel-VAE (UNSHARDED) = {min(a_times):.2f}s  |  "
        f"StageB/C vocoder+BWE (sharded f{parallel_config.sequence_parallel.factor}) = {min(bc_times):.2f}s"
    )

    vw = pipeline.tt_vocoder_with_bwe
    mel = pipeline.tt_audio_decoder(audio_spatial)
    main_t, mel_t, bwe_t, res_t = [], [], [], []
    for _ in range(_WARM_ITERS):
        t0 = time.time()
        x = vw.vocoder(mel.float())
        main_t.append(time.time() - t0)
        rem = x.shape[-1] % vw.hop_length
        if rem:
            x = torch.nn.functional.pad(x, (0, vw.hop_length - rem))
        t0 = time.time()
        mel_bwe = vw._compute_mel_device(x)
        mel_t.append(time.time() - t0)
        mel_for_bwe = mel_bwe.transpose(2, 3).contiguous()
        t0 = time.time()
        vw.bwe_generator(mel_for_bwe)
        bwe_t.append(time.time() - t0)
        t0 = time.time()
        vw._resample_device(x)
        res_t.append(time.time() - t0)
    logger.info(
        f"  B/C SUB-BREAKDOWN (warm_min): main_vocoder={min(main_t):.2f}s  mel_stft={min(mel_t):.2f}s  "
        f"bwe_generator={min(bwe_t):.2f}s  resampler={min(res_t):.2f}s"
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        ((1, 1), {}),
        ((2, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    ids=["1x1", "2x4_prod"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_e2e_psnr(mesh_device: ttnn.MeshDevice):
    ckpt = _resolve_distilled_checkpoint()
    if ckpt is None:
        pytest.skip("distilled checkpoint not found (set LTX_CHECKPOINT)")

    torch.manual_seed(0)
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
    pipeline.checkpoint_name = ckpt
    pipeline._new_audio_decoder()

    audio_N = 256
    audio_latent = torch.randn(1, audio_N, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    ref_audio = pipeline.decode_audio_reference(audio_latent, num_frames, fps=fps)
    assert ref_audio is not None, "host reference audio decode returned None"

    dev_audio = pipeline.decode_audio_device(audio_latent, num_frames, fps=fps, fallback=False)
    assert dev_audio is not None, "on-device audio decode returned None"
    assert dev_audio.sampling_rate == ref_audio.sampling_rate

    ref_w = ref_audio.waveform.float()
    dev_w = dev_audio.waveform.float()
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
