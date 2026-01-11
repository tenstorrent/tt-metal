# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_SDPA
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 8, 2),
        (1, 1, 32, 8),
    ],
    ids=["32x8x2", "1x32x8"],
)
def test_reduce_sum_single_device(device, shape):
    """
    Single device sum reduction test.

    Test configuration:
    - Input: bfloat16, L1 INTERLEAVED, TILE layout
    - Reduce dim: W (width, dim=-1)
    - Output: bfloat16, L1 INTERLEAVED
    - HiFi4 compute kernel (math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    """
    torch.manual_seed(1234)

    # Create input tensor
    torch_input = torch.rand(shape, dtype=torch.bfloat16)

    # Reference: sum along width dimension
    torch_output = torch_input.sum(dim=-1, keepdim=True)

    # Prepare input - L1 INTERLEAVED, TILE layout
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Run sum reduction along width dimension
    tt_output = ttnn.sum(
        tt_input,
        dim=-1,
        keepdim=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_SDPA,
    )

    # Get output and compare
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 8, 2),
        (1, 1, 32, 8),
    ],
    ids=["32x8x2", "1x32x8"],
)
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_reduce_sum_mesh_device(mesh_device, shape, enable_trace, device_params):
    """
    Mesh device sum reduction test.

    Test configuration:
    - Input: bfloat16, L1 INTERLEAVED, TILE layout
    - Reduce dim: W (width, dim=-1)
    - Output: bfloat16, L1 INTERLEAVED
    - HiFi4 compute kernel (math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    """
    torch_input = random_torch_tensor(ttnn.bfloat16, shape)
    torch_output = torch_input.sum(dim=-1, keepdim=True)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        tt_output = ttnn.sum(
            tt_input,
            dim=-1,
            keepdim=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_SDPA,
        )
        return tt_output

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, pcc=0.99)

    run_test(mesh_device, run_op, check_op, enable_trace)
