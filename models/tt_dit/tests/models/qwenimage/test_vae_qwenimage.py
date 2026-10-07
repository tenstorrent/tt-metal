# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The Qwen-Image VAE decoder and encoder against diffusers."""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_wan_2d import WanVaeDecoder2DAdapter, WanVaeEncoder2DAdapter
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params_req_exact_devices

if TYPE_CHECKING:
    from collections.abc import Callable

CHECKPOINT = "Qwen/Qwen-Image"
ITERATIONS = 5


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((2, 2), (1, 2), id="2x2_1x2"),
        pytest.param((2, 4), (1, 4), id="2x4_1x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({**line_params_req_exact_devices, "l1_small_size": 32768, "trace_region_size": 20000000}, id="line")],
    indirect=True,
)
@pytest.mark.parametrize(("height", "width"), [(1024, 1024)])
def test_vae_decoder(*, mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], height: int, width: int) -> None:
    """The decoder against diffusers, untraced and traced, and its decode times."""
    torch.manual_seed(0)

    # The submesh and VAE parallel config of the pipeline's last submesh.
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig.from_axes(submesh_device, h_axis=1, w_axis=0)

    torch_vae = WanVaeDecoder2DAdapter(
        checkpoint_name=CHECKPOINT, parallel_config=parallel_config, ccl_manager=ccl_manager, use_torch=True
    )
    tt_vae = WanVaeDecoder2DAdapter(
        checkpoint_name=CHECKPOINT, parallel_config=parallel_config, ccl_manager=ccl_manager, use_torch=False
    )

    latents = torch.randn(1, height // 8, width // 8, 16)

    logger.info("running torch VAE decode...")
    torch_out = torch_vae.decode(latents, traced=False)

    for traced in [False, True]:
        times, out = _time_decode(lambda traced=traced: tt_vae.decode(latents, traced=traced), submesh_device)
        logger.info(f"traced={traced}: median {statistics.median(times):.4f} s, all {_fmt(times)}")
        assert out.shape == torch_out.shape, f"{out.shape} != {torch_out.shape}"
        assert_quality(torch_out, out, pcc=0.99, relative_rmse=0.1)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((2, 2), (1, 2), id="2x2_1x2"),
        pytest.param((2, 4), (1, 4), id="2x4_1x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({**line_params_req_exact_devices, "l1_small_size": 32768, "trace_region_size": 20000000}, id="line")],
    indirect=True,
)
@pytest.mark.parametrize(("height", "width"), [(1024, 1024)])
def test_vae_encoder(*, mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], height: int, width: int) -> None:
    """The encoder against diffusers, untraced and traced."""
    torch.manual_seed(0)

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig.from_axes(submesh_device, h_axis=1, w_axis=0)

    torch_vae = WanVaeEncoder2DAdapter(
        checkpoint_name=CHECKPOINT, parallel_config=parallel_config, ccl_manager=ccl_manager, use_torch=True
    )
    tt_vae = WanVaeEncoder2DAdapter(
        checkpoint_name=CHECKPOINT, parallel_config=parallel_config, ccl_manager=ccl_manager, use_torch=False
    )

    images = torch.rand(1, 3, height, width) * 2 - 1

    logger.info("running torch VAE encode...")
    torch_out = torch_vae.encode(images, traced=False)

    for traced in [False, True]:
        out = tt_vae.encode(images, traced=traced)
        assert out.shape == torch_out.shape, f"{out.shape} != {torch_out.shape}"
        assert_quality(torch_out, out, pcc=0.99, relative_rmse=0.1)


def _time_decode(decode: Callable[[], torch.Tensor], device: ttnn.MeshDevice) -> tuple[list[float], torch.Tensor]:
    # The first call compiles the programs or captures the trace, so it is not timed.
    out = decode()
    times = []
    for _ in range(ITERATIONS):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = decode()
        times.append(time.perf_counter() - start)
    return times, out


def _fmt(times: list[float]) -> str:
    return " ".join(f"{t:.4f}" for t in times)
