# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Warm wall-clock perf of the on-device audio decode on a 2×4 mesh.

Times ``LTXPipeline.decode_audio_device`` (Stage A mel-VAE on device replicated
+ Stage B/C vocoder+BWE T-sharded across the mesh). The first call builds the
TT modules and compiles kernels (cold); subsequent calls are warm forward-only.
Reports cold and warm-min so the warm number isolates the per-decode cost from
one-time build/compile.

NOT a tuned number: the Stage A conv3d shapes miss the blocking table and Stage
A runs unsharded. This measures the current state, not the ceiling.
"""

from __future__ import annotations

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


def _resolve_checkpoint() -> str | None:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit):
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    return local if os.path.exists(local) else None


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
    ckpt = _resolve_checkpoint()
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
    # Build the audio decoder shells the ctor skipped (checkpoint_name=None at
    # construction); decode_audio_device only loads weights into them.
    pipeline._new_audio_decoder()

    audio_N = 256
    audio_latent = torch.randn(1, audio_N, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    # Cold: builds modules + compiles kernels + forward.
    t0 = time.time()
    dev_audio = pipeline.decode_audio_device(audio_latent, num_frames, fps=fps, fallback=False)
    t_cold = time.time() - t0
    assert dev_audio is not None
    logger.info(f"COLD audio decode (build+compile+forward): {t_cold:.2f}s on mesh {mesh_shape} sp_axis={sp_axis}")

    # Warm: modules + kernels cached; per-decode forward only.
    warm = []
    for i in range(_WARM_ITERS):
        t0 = time.time()
        pipeline.decode_audio_device(audio_latent, num_frames, fps=fps, fallback=False)
        dt = time.time() - t0
        warm.append(dt)
        logger.info(f"WARM iter {i}: {dt:.2f}s")

    # Per-stage breakdown (warm): Stage A mel-VAE (unsharded) vs Stage B/C
    # vocoder+BWE (sharded). Mirrors decode_audio_device's reshape so the
    # split is apples-to-apples with the full-decode number above.
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

    # Sub-breakdown WITHIN Stage B/C: localize the floor. Replicates
    # LTXVocoderWithBWE._bwe_from_waveform's call sequence.
    vw = pipeline.tt_vocoder_with_bwe
    mel = pipeline.tt_audio_decoder(audio_spatial)
    main_t, mel_t, bwe_t, res_t = [], [], [], []
    for _ in range(_WARM_ITERS):
        t0 = time.time()
        x = vw.vocoder(mel.float())  # main vocoder → low-rate waveform
        main_t.append(time.time() - t0)
        rem = x.shape[-1] % vw.hop_length
        if rem:
            x = torch.nn.functional.pad(x, (0, vw.hop_length - rem))
        t0 = time.time()
        mel_bwe = vw._compute_mel_device(x)  # mel STFT (host reshape + device)
        mel_t.append(time.time() - t0)
        mel_for_bwe = mel_bwe.transpose(2, 3).contiguous()
        t0 = time.time()
        vw.bwe_generator(mel_for_bwe)  # 2nd vocoder
        bwe_t.append(time.time() - t0)
        t0 = time.time()
        vw._resample_device(x)  # Hann resampler
        res_t.append(time.time() - t0)
    logger.info(
        f"  B/C SUB-BREAKDOWN (warm_min): main_vocoder={min(main_t):.2f}s  mel_stft={min(mel_t):.2f}s  "
        f"bwe_generator={min(bwe_t):.2f}s  resampler={min(res_t):.2f}s"
    )
