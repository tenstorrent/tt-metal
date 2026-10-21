# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.block_matmul must compute a @ w on its core rectangles, split over W's tile columns."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

pytestmark = run_for_blackhole()

_ROWS, _K = 640, 1024
_MESHES = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="SP8xTP4",
    )
]


def _cores(x1, y1):
    return ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))


def _operands(mesh_device, n, weight_dtype):
    generator = torch.Generator().manual_seed(n)
    a = torch.randn(1, _ROWS, _K, generator=generator)
    w = torch.randn(_K, n, generator=generator) / _K**0.5
    to_device = lambda host, dtype: ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    a_tt, w_tt = to_device(a, ttnn.bfloat16), to_device(w, weight_dtype)
    # The reference from the values the device holds.
    a_host = ttnn.to_torch(ttnn.get_device_tensors(a_tt)[0]).float()
    w_host = ttnn.to_torch(ttnn.get_device_tensors(w_tt)[0]).float()
    return a_tt, w_tt, a_host @ w_host


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
@pytest.mark.parametrize(
    "n, weight_dtype, cores, k_block",
    [
        # The input projection's shape of work: the 11 x 10 matmul grid, a bfloat8_b weight in 8-tile K blocks.
        pytest.param(1184, ttnn.bfloat8_b, _cores(10, 9), 8, id="matmul-grid-bfp8"),
        # The output gate's: a narrower block, BF16 weight, 4-tile K blocks (rows and columns that do not divide).
        pytest.param(1184, ttnn.bfloat16, _cores(7, 9), 4, id="gate-block-bf16"),
    ],
)
def test_block_matmul_matches_torch(mesh_device, n, weight_dtype, cores, k_block):
    a, w, expected = _operands(mesh_device, n, weight_dtype)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    output = ttnn.experimental.kda.block_matmul(
        a, w, cores=cores, k_block=k_block, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=compute_config
    )
    for index, local in enumerate(ttnn.get_device_tensors(output)):
        assert_accurate(
            expected, ttnn.to_torch(local).float(), name=f"device {index}", pcc_threshold=0.9999, nrmse_threshold=0.01
        )


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
def test_block_matmul_rejects_a_k_block_that_does_not_divide_k(mesh_device, expect_error):
    a, w, _ = _operands(mesh_device, 1184, ttnn.bfloat16)
    with expect_error(RuntimeError, "must split into 3-tile blocks"):
        ttnn.experimental.kda.block_matmul(a, w, cores=_cores(10, 9), k_block=3)
