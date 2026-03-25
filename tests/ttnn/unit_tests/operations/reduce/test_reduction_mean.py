# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics, assert_with_pcc, assert_allclose
from models.common.utility_functions import torch_random
from tests.ttnn.unit_tests.operations.reduce.numeric_check import (
    collect_and_dump_numeric_metrics,
)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean(device, batch_size, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_mean[batch_size={batch_size},h={h},w={w},dim={dim},keepdim={keepdim}]"
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        test_name=test_name,
        csv_filename="test_reduction_mean.csv",
        test_params=None,
    )
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999896,
        rtol=0.117540088,
        atol=0.0019931875,
        frobenius_threshold=0.00504272596,
    )


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_scaling(device, shape, dim, keepdim):
    """Use assert_allclose with ones() to test that mean's scaling factor is
    computed correctly.
    """
    torch_input_tensor = torch.ones(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch.bfloat16)
    torch_output_tensor = torch_output_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_mean_scaling[shape={shape},dim={dim},keepdim={keepdim}]"
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        test_name=test_name,
        csv_filename="test_reduction_mean.csv",
        test_params=None,
    )
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.9999,
        rtol=0.003985375,
        atol=0.003985375,
        frobenius_threshold=0.003984376,
    )


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("scalar", [2.0])
def test_mean_scaling_factor(device, shape, dim, scalar):
    torch_input_tensor = torch.ones(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, dtype=torch.bfloat16)
    torch_output_tensor = torch_output_tensor * scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_mean_scaling_factor[shape={shape},dim={dim},scalar={scalar}]"
    collect_and_dump_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        test_name=test_name,
        csv_filename="test_reduction_mean.csv",
        test_params=None,
    )
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.9999,
        rtol=0.003985375,
        atol=0.00796975,
        frobenius_threshold=0.003984376,
    )


@pytest.mark.parametrize("mem_config", [None, ttnn.DRAM_MEMORY_CONFIG, "block"])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_shard(device, mem_config, keepdim):
    if mem_config is None and not keepdim:
        pytest.skip("Skipping because reshape does not work in this scenario. Issue #35145")
    torch_input_tensor = torch.randn(1, 1024, 160, dtype=torch.bfloat16)
    block_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1024, 160),
        core_grid=ttnn.CoreGrid(x=5, y=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=block_sharded_config,
    )

    memory_config = block_sharded_config if mem_config == "block" else mem_config
    output_tensor = ttnn.mean(
        input_tensor,
        dim=-1,
        keepdim=keepdim,
        memory_config=memory_config,
    )
    tt_output_torch = ttnn.to_torch(output_tensor)
    torch_output = torch.mean(torch_input_tensor, -1, keepdim)
    # Collect numeric metrics and dump to CSV using reusable function
    test_name = f"test_mean_shard[mem_config={mem_config},keepdim={keepdim}]"
    collect_and_dump_numeric_metrics(
        torch_output,
        tt_output_torch,
        test_name=test_name,
        csv_filename="test_reduction_mean.csv",
        test_params=None,
    )
    assert_numeric_metrics(
        torch_output,
        tt_output_torch,
        pcc_threshold=0.999896,
        rtol=0.609610324,
        atol=0.0019931875,
        frobenius_threshold=0.0048248305,
    )
