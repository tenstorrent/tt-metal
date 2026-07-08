# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Audio decode integration tests for LTX-2.

Post-denoise decode only (mel-VAE → vocoder → BWE):
- on-device decode runs on representative meshes
- warm-path decode stays functional after cold compile/build
- on-device decode quality remains close to host reference (PSNR)

Denoise-path audio regressions live in sibling files (see LTX2.md).
"""

import math
import os
import time
from glob import glob

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.utils.test import line_params, ring_params

_WARM_ITERS = 3


# ---------------------------------------------------------------------------
# Mesh / device params (mirrors test_pipeline_ltx_fast_av.py)
# ---------------------------------------------------------------------------
def _with_audio_dev_l1(base: dict) -> dict:
    return {**base, "l1_small_size": 32768}


_line_params = _with_audio_dev_l1(line_params)
_ring_params = _with_audio_dev_l1(ring_params)
_ring_trace_params = {**_ring_params, "trace_region_size": 300_000_000}

_AUDIO_FAST_AV_MESH_PARAMS_FULL = [
    pytest.param((2, 2), (2, 2), 0, 1, 2, False, _line_params, ttnn.Topology.Linear, True, id="2x2sp0tp1"),
    pytest.param((2, 4), (2, 4), 0, 1, 1, True, _line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
    pytest.param((2, 4), (2, 4), 1, 0, 2, True, _line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
    pytest.param((4, 8), (4, 8), 1, 0, 4, False, _ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
    pytest.param((4, 8), (4, 8), 1, 0, 2, False, _line_params, ttnn.Topology.Linear, False, id="bh_4x8sp1tp0_linear"),
    pytest.param((4, 8), (4, 8), 1, 0, 2, False, _ring_trace_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0_ring"),
    pytest.param((4, 32), (4, 32), 1, 0, 2, False, _ring_params, ttnn.Topology.Ring, False, id="bh_4x32sp1tp0"),
]


def _audio_mesh_params():
    full = os.environ.get("LTX_AUDIO_FULL_MATRIX", "0").lower() in ("1", "true", "yes")
    return _AUDIO_FAST_AV_MESH_PARAMS_FULL if full else _AUDIO_FAST_AV_MESH_PARAMS_FULL[1:3]


_AUDIO_FAST_AV_MESH_PARAMS = _audio_mesh_params()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_checkpoint(candidates: list[str], expected_filenames: tuple[str, ...]) -> str | None:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit) and os.path.basename(explicit) in expected_filenames:
        return explicit
    for path in candidates:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded
    # Reuse default HF cache location if present; avoid network dependency.
    for name in expected_filenames:
        pattern = os.path.expanduser(f"~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/{name}")
        hits = glob(pattern)
        if hits:
            return hits[0]
    return None


def _build_pipeline(
    mesh_device: ttnn.MeshDevice,
    *,
    sp_axis: int,
    tp_axis: int,
    checkpoint: str,
    num_links: int,
    topology: ttnn.Topology,
) -> tuple[LTXPipeline, DiTParallelConfig]:
    mesh_shape = tuple(mesh_device.shape)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=topology)
    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        checkpoint_name=None,
    )
    pipeline.checkpoint_name = checkpoint
    pipeline._new_audio_decoder()
    return pipeline, parallel_config


def _psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref = ref.float()
    test = test.float()
    mse = torch.mean((ref - test) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = ref.abs().max().item()
    if peak == 0.0:
        return float("inf")
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def _decode_audio_device_compat(pipeline: LTXPipeline, audio_latent: torch.Tensor, num_frames: int, fps: float):
    if hasattr(pipeline, "decode_audio_device"):
        return pipeline.decode_audio_device(audio_latent, num_frames, fps=fps, fallback=False)
    return pipeline.decode_audio(audio_latent, num_frames, fps=fps)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "dynamic_load",
        "device_params",
        "topology",
        "is_fsdp",
    ),
    _AUDIO_FAST_AV_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_perf(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
):
    ckpt = _resolve_checkpoint(
        ["~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"],
        ("ltx-2.3-22b-dev.safetensors",),
    )
    if ckpt is None:
        pytest.skip("checkpoint not found (set LTX_CHECKPOINT)")

    _ = (dynamic_load, is_fsdp)  # parity with fast_av config schema
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    torch.manual_seed(0)
    pipeline, parallel_config = _build_pipeline(
        mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        checkpoint=ckpt,
        num_links=num_links,
        topology=topology,
    )
    mesh_shape = tuple(mesh_device.shape)

    audio_latent = torch.randn(1, 256, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    t0 = time.time()
    dev_audio = _decode_audio_device_compat(pipeline, audio_latent, num_frames, fps=fps)
    t_cold = time.time() - t0
    assert dev_audio is not None

    warm_times = []
    for _ in range(_WARM_ITERS):
        t0 = time.time()
        out = _decode_audio_device_compat(pipeline, audio_latent, num_frames, fps=fps)
        warm_times.append(time.time() - t0)
        assert out is not None

    logger.info(
        f"Audio decode perf (mesh {mesh_shape}, T-shard={parallel_config.sequence_parallel.factor}): "
        f"cold={t_cold:.2f}s warm_min={min(warm_times):.2f}s warm_mean={sum(warm_times)/len(warm_times):.2f}s "
        f"wave={tuple(dev_audio.waveform.shape)} sr={dev_audio.sampling_rate}"
    )


@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "dynamic_load",
        "device_params",
        "topology",
        "is_fsdp",
    ),
    _AUDIO_FAST_AV_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_e2e_psnr(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
):
    ckpt = _resolve_checkpoint(
        ["~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"],
        ("ltx-2.3-22b-distilled-1.1.safetensors",),
    )
    if ckpt is None:
        pytest.skip("distilled checkpoint not found (set LTX_CHECKPOINT)")

    _ = (dynamic_load, is_fsdp)  # parity with fast_av config schema
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    torch.manual_seed(0)
    mesh_shape = tuple(mesh_device.shape)
    pipeline, _ = _build_pipeline(
        mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        checkpoint=ckpt,
        num_links=num_links,
        topology=topology,
    )

    audio_latent = torch.randn(1, 256, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    if not hasattr(pipeline, "decode_audio_reference"):
        pytest.skip("decode_audio_reference() unavailable in this pipeline revision")

    ref_audio = pipeline.decode_audio_reference(audio_latent, num_frames, fps=fps)
    assert ref_audio is not None, "host reference audio decode returned None"

    dev_audio = _decode_audio_device_compat(pipeline, audio_latent, num_frames, fps=fps)
    assert dev_audio is not None, "on-device audio decode returned None"
    assert dev_audio.sampling_rate == ref_audio.sampling_rate

    n = min(ref_audio.waveform.shape[-1], dev_audio.waveform.shape[-1])
    ref_w = ref_audio.waveform[..., :n].float()
    dev_w = dev_audio.waveform[..., :n].float()
    psnr = _psnr(ref_w, dev_w)

    logger.info(
        f"Audio decode PSNR (mesh {mesh_shape}): {psnr:.2f} dB "
        f"(ref {tuple(ref_audio.waveform.shape)} -> dev {tuple(dev_audio.waveform.shape)})"
    )
    assert psnr >= 28.0, f"audio decode PSNR {psnr:.2f} dB < 28 dB"
