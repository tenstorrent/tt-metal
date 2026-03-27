# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import os

import pytest
import torch
import ttnn

# Make the operation module importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "operation"))
from minimal_binary_op import MinimalBinaryConfig, minimal_binary_op

from tests.ttnn.utils_for_testing import assert_with_ulp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _torch_op(op_type: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b) if op_type == "add" else torch.mul(a, b)


def _to_torch_dtype(ttnn_dtype: ttnn.DataType) -> torch.dtype:
    return torch.float32 if ttnn_dtype == ttnn.float32 else torch.bfloat16


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------

SHAPES = [
    pytest.param([32, 32], id="1tile"),
    pytest.param([32, 128], id="1x4tiles"),
    pytest.param([32, 64], id="small_2d"),
    pytest.param([2, 4, 32, 64], id="4d"),
    pytest.param([1, 4096, 4096], id="large_2d", marks=pytest.mark.slow),
]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
]

OPS = ["add", "mul"]

CONFIGS = [
    pytest.param(MinimalBinaryConfig(block_size=1, sub_block_size=1), id="block1_sub1"),
    pytest.param(MinimalBinaryConfig(block_size=4, sub_block_size=1), id="block4_sub1"),
    pytest.param(MinimalBinaryConfig(block_size=4, sub_block_size=2), id="block4_sub2"),
    pytest.param(MinimalBinaryConfig(block_size=8, sub_block_size=4), id="block8_sub4"),
    pytest.param(MinimalBinaryConfig(use_dual_reader=False), id="noc_balanced"),
    pytest.param(MinimalBinaryConfig(use_flushed_writes=True), id="flushed_writes"),
    pytest.param(
        MinimalBinaryConfig(block_size=4, sub_block_size=2, use_dual_reader=False, use_flushed_writes=True),
        id="block4_sub2_balanced_flushed",
    ),
    #    pytest.param(
    #        MinimalBinaryConfig(block_size=8, sub_block_size=8, use_flushed_writes=True), id="block8_sub8_flushed",
    #    )
]


# ---------------------------------------------------------------------------
# Main correctness test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", OPS)
@pytest.mark.parametrize("config", CONFIGS)
def test_minimal_binary(device, shape, dtype, op_type, config):
    torch.manual_seed(1634706)

    # Skip invalid sub_block_size for float32
    if dtype == ttnn.float32 and config.sub_block_size > 2:
        pytest.skip("sub_block_size > 2 not supported for float32")
    # Skip block_size=8/sub=4 for float32 (sub_block_size would be > 2)
    if dtype == ttnn.float32 and config.block_size > 4:
        pytest.skip("block_size > 4 not supported for float32 with sub_block_size constraint")

    torch_dtype = _to_torch_dtype(dtype)
    a_torch = torch.rand(shape, dtype=torch_dtype)
    b_torch = torch.rand(shape, dtype=torch_dtype)

    # a_torch = torch.full(shape, fill_value=3.0, dtype=torch_dtype)
    # b_torch = torch.full(shape, fill_value=4.0, dtype=torch_dtype)

    # Avoid near-zero values for mul (to keep ULP errors small)
    # if op_type == "mul":
    #    a_torch = a_torch + 0.1
    #    b_torch = b_torch + 0.1

    # Compute golden in the same dtype as the device to match rounding
    golden = _torch_op(op_type, a_torch, b_torch)

    a = ttnn.from_torch(
        a_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b = ttnn.from_torch(
        b_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = minimal_binary_op(a, b, op_type, config)
    result_torch = ttnn.to_torch(result)

    # Crop padded output back to original shape for comparison
    # slices = tuple(slice(0, s) for s in shape)
    # result_torch = result_torch[slices]

    # print(f"golden = \n{golden}")
    # print(f"output = \n{result_torch}")

    # ULP of finite number should not be more than 2 regardless of data type
    # If a * b is off by more than 2 ULP then this means that the output is fundamentally useless
    # Only exception is for subnormals; where 0 is expected instead
    assert_with_ulp(golden, result_torch, ulp_threshold=2)


# ---------------------------------------------------------------------------
# Smoke test: ensure all config variants run without crash on a tiny tensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_type", OPS)
@pytest.mark.parametrize("config", CONFIGS)
def test_minimal_binary_smoke_bf16(device, op_type, config):
    """Quick smoke test with small bfloat16 tensor for all config combos."""
    shape = [32, 64]
    dtype = ttnn.bfloat16

    a_torch = torch.rand(shape, dtype=torch.bfloat16) + 0.1
    b_torch = torch.rand(shape, dtype=torch.bfloat16) + 0.1

    golden = _torch_op(op_type, a_torch, b_torch).float()

    a = ttnn.from_torch(
        a_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b = ttnn.from_torch(
        b_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = minimal_binary_op(a, b, op_type, config)
    result_torch = ttnn.to_torch(result).float()

    slices = tuple(slice(0, s) for s in shape)
    assert_with_ulp(golden, result_torch[slices], ulp_threshold=2 * 65536)
