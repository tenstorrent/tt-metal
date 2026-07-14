# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics, assert_with_ulp
from models.common.utility_functions import torch_random

TEST_PADDING_VALUE = -42


def _skip_fp32_nc_reduce(dtype, dim):
    # The accurate SFPU fp32 path covers H/W reductions only; batch/channel (dim 0/1) reductions use
    # the NC path with a bf16-rounded 1/N scaler (~1e-3 error), so fp32 there is not ULP-exact.
    axes = [dim] if isinstance(dim, int) else list(dim)
    if dtype == ttnn.float32 and any(a in (0, 1) for a in axes):
        pytest.skip("fp32 batch/channel-dim mean uses the bf16-scaler NC path; accurate SFPU covers H/W only")


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_mean(device, batch_size, h, w, dim, keepdim, dtype):
    torch.manual_seed(0)

    if dtype == ttnn.float32:
        # FLOAT32 defaults to the accurate SFPU path, so it gets far tighter thresholds than the FPU BF16 path.
        torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.float32)
        torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim)
    else:
        torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
        torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    assert output_tensor.memory_config() == input_tensor.memory_config()
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    if dtype == ttnn.float32:
        assert_numeric_metrics(
            torch_output_tensor,
            output_tensor,
            pcc_threshold=0.9999,
            rtol=0.01,
            atol=1e-4,
            frobenius_threshold=0.001,
            check_ulp=False,
        )
    else:
        assert_numeric_metrics(
            torch_output_tensor,
            output_tensor,
            pcc_threshold=0.999,
            rtol=0.118,
            atol=0.002,
            frobenius_threshold=0.005,
            check_ulp=False if dim == -2 else True,
        )


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_mean_scaling(device, shape, dim, keepdim, dtype):
    """Ones input → uniform mean; check the exact result via ULP (both dtypes)."""
    _skip_fp32_nc_reduce(dtype, dim)
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input_tensor = torch.ones(shape, dtype=torch_dtype)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("scalar", [2.0])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_mean_scaling_factor(device, shape, dim, scalar, dtype):
    _skip_fp32_nc_reduce(dtype, dim)
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input_tensor = torch.ones(shape, dtype=torch_dtype)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, dtype=torch_dtype)
    torch_output_tensor = torch_output_tensor * scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.mean(input_tensor, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)


@pytest.mark.parametrize("mem_config", [None, ttnn.DRAM_MEMORY_CONFIG, "block", "height"])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_mean_shard(device, mem_config, keepdim, dtype):
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    if mem_config == "height":
        # Height 100 is intentionally non-tile-aligned (not a multiple of 32).
        # Physical height pads to 128, so shard height 32 across 4 cores is valid.
        # After reducing dim=-1 with keepdim=False the output shape is (1, 100),
        # which exercises reshape_tiled's shard spec recomputation for HEIGHT_SHARDED.
        torch_input_tensor = torch.randn(1, 100, 160, dtype=torch_dtype)
        sharded_config = ttnn.create_sharded_memory_config(
            shape=(32, 160),
            core_grid=ttnn.CoreGrid(x=1, y=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        torch_input_tensor = torch.randn(1, 1024, 160, dtype=torch_dtype)
        sharded_config = ttnn.create_sharded_memory_config(
            shape=(1, 1024, 160),
            core_grid=ttnn.CoreGrid(x=5, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_config,
    )

    if mem_config in ("block", "height"):
        memory_config = sharded_config
    else:
        memory_config = mem_config

    output_tensor = ttnn.mean(
        input_tensor,
        dim=-1,
        keepdim=keepdim,
        memory_config=memory_config,
    )
    tt_output_torch = ttnn.to_torch(output_tensor)
    torch_output = torch.mean(torch_input_tensor, -1, keepdim)
    # test for equivalance; FLOAT32 runs the accurate SFPU path, so its abs tolerance is far tighter.
    assert_numeric_metrics(
        torch_output,
        tt_output_torch,
        pcc_threshold=0.999,
        rtol=0.610,
        atol=1e-4 if dtype == ttnn.float32 else 0.002,
        frobenius_threshold=0.0055,
    )

    output_mem_config = output_tensor.memory_config()
    if mem_config == ttnn.DRAM_MEMORY_CONFIG:
        assert output_mem_config == mem_config
    else:
        assert output_mem_config.buffer_type == ttnn.BufferType.L1
        assert output_mem_config.is_sharded()
