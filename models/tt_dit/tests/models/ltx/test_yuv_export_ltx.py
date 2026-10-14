# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device YUV export at LTX's clip length: a clip longer than rgb_to_yuv's 32-frame wide_rows
limit reads back the same bytes as the same clip converted 32 frames at a time."""

import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....pipelines.ltx.pipeline_ltx_distilled import yuv_export_enabled
from ....utils.yuv_d2h import fast_device_to_host_yuv

MESH_4X8 = [pytest.param((4, 8), {}, id="4x8")]
WIDE_ROWS_MAX_T = 32


def _vae_config(mesh_shape, sp_axis, tp_axis):
    """The VAE layout LTXPipeline.create_pipeline builds: H on the tp axis, W on the sp axis."""
    return VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
        width_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
    )


SP1TP0_4X8 = _vae_config((4, 8), sp_axis=1, tp_axis=0)


@pytest.mark.parametrize(
    ("output_path", "env_value", "vae_config", "distributed", "expected"),
    [
        pytest.param("out.mp4", None, SP1TP0_4X8, False, True, id="default_on"),
        pytest.param("out.mp4", "1", SP1TP0_4X8, False, True, id="env_1_on"),
        pytest.param("out.mp4", "0", SP1TP0_4X8, False, False, id="env_0_off"),
        pytest.param(None, None, SP1TP0_4X8, False, False, id="no_output_path_off"),
        pytest.param(None, "1", SP1TP0_4X8, False, False, id="no_output_path_env_1_off"),
        pytest.param("out.mp4", None, _vae_config((2, 2), 0, 1), False, False, id="2x2sp0tp1_off"),
        pytest.param("out.mp4", None, _vae_config((2, 4), 0, 1), False, False, id="2x4sp0tp1_off"),
        pytest.param("out.mp4", "1", _vae_config((2, 4), 0, 1), False, False, id="2x4sp0tp1_env_1_off"),
        pytest.param("out.mp4", None, _vae_config((2, 4), 1, 0), False, True, id="2x4sp1tp0_on"),
        pytest.param("out.mp4", None, _vae_config((1, 1), 0, 1), False, True, id="1x1sp0tp1_on"),
        pytest.param("out.mp4", None, _vae_config((1, 4), 0, 1), False, False, id="1x4sp0tp1_off"),
        pytest.param("out.mp4", None, _vae_config((4, 32), 1, 0), True, False, id="multi_host_default_off"),
        pytest.param("out.mp4", "1", _vae_config((4, 32), 1, 0), True, True, id="multi_host_env_1_on"),
    ],
)
def test_yuv_export_gate(output_path, env_value, vae_config, distributed, expected):
    assert yuv_export_enabled(output_path, env_value, vae_config, distributed) is expected


def _timed(mesh_device, fn, n=3):
    out = fn()
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("height", "width", "frames"), [pytest.param(1088, 1920, 145, id="1088x1920x145")])
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_yuv_export_matches_32_frame_chunks(mesh_device, height, width, frames):
    torch.manual_seed(0)
    video = torch.rand(1, 3, frames, height, width, dtype=torch.bfloat16) * 2 - 1
    tt_video = ttnn.from_torch(
        video,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(3, 4)),
    )
    _, _, _, h_per, w_per = tt_video.shape

    def chunked():
        finish = []
        for t0 in range(0, frames, WIDE_ROWS_MAX_T):
            t1 = min(t0 + WIDE_ROWS_MAX_T, frames)
            part = ttnn.slice(tt_video, [0, 0, t0, 0, 0], [1, 3, t1, h_per, w_per])
            finish.append(fast_device_to_host_yuv(part, mesh_device, defer=True))
        return np.concatenate([f() for f in finish], axis=0)

    whole, ms_whole = _timed(mesh_device, lambda: fast_device_to_host_yuv(tt_video, mesh_device))
    parts, ms_chunked = _timed(mesh_device, chunked)
    logger.info(f"YUVEXPORT ({height},{width},{frames}): whole {ms_whole:.1f} ms, 32-frame chunks {ms_chunked:.1f} ms")

    assert whole.shape == (frames, height * width * 3 // 2), f"planar shape {whole.shape}"
    n_diff = int((whole != parts).sum())
    assert n_diff == 0, f"{n_diff} bytes differ between the whole clip and its 32-frame chunks"
