# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Bandwidth benchmark for minimal_binary_op.

Tensors: 16384×16384 bf16 or fp32, DRAM interleaved.
Sweeps all MinimalBinaryConfig variants × 2 dtypes × 2 ops.

Bandwidth accounts for 3 DRAM tensor-copies: read A, read B, write C.

Usage:
    # Profile + aggregate bandwidth table in one step:
    python tracy_tools/analyze_tracy_perf.py \\
        --run "pytest custom_op/minimal_binary/bench/test_minimal_binary_perf.py" \\
        --aggregate

    # Or separately:
    python -m tracy -r -p -v -m pytest custom_op/minimal_binary/bench/test_minimal_binary_perf.py
    python tracy_tools/analyze_tracy_perf.py --aggregate

Signpost format: {dtype}/{H}-{W}x3-{config_id}-{op_type}
  analyze_tracy_perf.py parses this as: sizes=[H, W], factor=3
  -> num_elements = 3 * H * W
  -> bw_gb_s     = 3 * H * W * dtype_bytes / duration_s
"""

import sys
import os

import pytest
import ttnn
from tracy import signpost

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "operation"))
from minimal_binary_op import MinimalBinaryConfig, minimal_binary_op


NUM_WARMUP = 3
NUM_ITERATIONS = 10

SHAPE = (16384, 16384)

CONFIGS = [
    (MinimalBinaryConfig(block_size=1, sub_block_size=1), "block1_sub1"),
    # (MinimalBinaryConfig(block_size=4, sub_block_size=1),                                          "block4_sub1"),
    # (MinimalBinaryConfig(block_size=4, sub_block_size=2),                                          "block4_sub2"),
    (MinimalBinaryConfig(block_size=8, sub_block_size=4), "block8_sub4"),
    (MinimalBinaryConfig(block_size=8, sub_block_size=8), "block8_sub8"),
    (MinimalBinaryConfig(block_size=8, sub_block_size=8, use_flushed_writes=True), "block8_sub8_flushed"),
    (MinimalBinaryConfig(block_size=16, sub_block_size=8), "block16_sub8"),
    (MinimalBinaryConfig(block_size=16, sub_block_size=8, use_flushed_writes=True), "block16_sub8_flushed"),
]

DTYPES = [
    (ttnn.bfloat16, "bf16"),
    (ttnn.float32, "fp32"),
]

OPS = ["add", "mul"]

_TTNN_OP = {"add": ttnn.add, "mul": ttnn.mul}


@pytest.mark.parametrize("config,config_id", CONFIGS)
@pytest.mark.parametrize("op_type", OPS)
@pytest.mark.parametrize("dtype,dtype_str", DTYPES)
@pytest.mark.parametrize("device_id", ttnn.get_device_ids())
def test_minimal_binary_perf(device_id, dtype, dtype_str, op_type, config, config_id):
    """Sweep MinimalBinaryConfig variants and report DRAM bandwidth via Tracy signposts."""

    # fp32 constraints (same as correctness tests)
    if dtype == ttnn.float32 and config.sub_block_size > 2:
        pytest.skip("sub_block_size > 2 not supported for float32")
    if dtype == ttnn.float32 and config.block_size > 4:
        pytest.skip("block_size > 4 not supported for float32")

    device = ttnn.open_device(device_id=device_id)
    H, W = SHAPE

    # Allocate directly on device — ttnn.rand overhead excluded from measurement.
    # device is a positional arg in ttnn.rand.
    a = ttnn.rand([H, W], device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.rand([H, W], device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Warmup: compile kernels and prime DRAM caches before measurement.
    for _ in range(NUM_WARMUP):
        c = minimal_binary_op(a, b, op_type, config)
        ttnn.deallocate(c)
    ttnn.synchronize_device(device)

    # Signpost format: {dtype}/{H}-{W}x{factor}-{config_id}-{op_type}
    # factor=3 encodes the 3 DRAM tensor-copies (read A + read B + write C).
    # analyze_tracy_perf.py computes: num_elements = 3 * H * W
    #   bw_gb_s = 3 * H * W * dtype_bytes / duration_s
    sp = f"{dtype_str}/{H}-{W}x3-{config_id}-{op_type}"
    signpost(header=f"{sp}-start")
    for _ in range(NUM_ITERATIONS):
        c = minimal_binary_op(a, b, op_type, config)
        ttnn.deallocate(c)
    ttnn.synchronize_device(device)
    signpost(header=f"{sp}-end")

    ttnn.deallocate(a)
    ttnn.deallocate(b)
    ttnn.close_device(device)


@pytest.mark.parametrize("op_type", OPS)
@pytest.mark.parametrize("dtype,dtype_str", DTYPES)
@pytest.mark.parametrize("device_id", ttnn.get_device_ids())
def test_ttnn_binary_perf(device_id, dtype, dtype_str, op_type):
    """Reference: ttnn.add / ttnn.mul (binary_ng kernels) for the same shape and dtype."""
    device = ttnn.open_device(device_id=device_id)
    H, W = SHAPE

    a = ttnn.rand([H, W], device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.rand([H, W], device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    op = _TTNN_OP[op_type]

    for _ in range(NUM_WARMUP):
        c = op(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(c)
    ttnn.synchronize_device(device)

    sp = f"{dtype_str}/{H}-{W}x3-ttnn_{op_type}"
    signpost(header=f"{sp}-start")
    for _ in range(NUM_ITERATIONS):
        c = op(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(c)
    ttnn.synchronize_device(device)
    signpost(header=f"{sp}-end")

    ttnn.deallocate(a)
    ttnn.deallocate(b)
    ttnn.close_device(device)
