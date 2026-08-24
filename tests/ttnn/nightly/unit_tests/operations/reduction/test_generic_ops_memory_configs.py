# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Generic reductions over sharded and dtype/layout memory configurations.
# Split from test_reduction_ops.py: corner-case tests, not exhaustive sweeps.

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose_and_pcc, is_wormhole_b0
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import (
    TTNN_REDUCTION_WRAPPERS,
)

# Module-scoped device: these tests all run with the default device config, so the
# device is opened once per file (one device context per test group) instead of
# once per test case.
pytestmark = pytest.mark.use_module_device


# explicit_output_mem_config is exercised only for sum and var; pairing it with
# the op (instead of crossing) removes the always-skipped combinations.
OP_EXPLICIT_CFG = [
    ("mean", False),
    ("sum", False),
    ("sum", True),
    ("max", False),
    ("min", False),
    ("std", False),
    ("var", False),
    ("var", True),
]


@pytest.mark.parametrize(
    "shapes",
    [
        ([2, 1, 256, 2048], [1, 1, 128, 256], 2, 4),
        ([4, 4, 64, 128], [2, 2, 32, 64], 2, 4),
        ([4, 4, 64, 128], [2, 2, 32, 64], 0, 0),
    ],
)
@pytest.mark.parametrize("keepdim", [True])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("op, explicit_output_mem_config", OP_EXPLICIT_CFG)
def test_generic_ops_ndim_shard(device, shapes, keepdim, layout, op, explicit_output_mem_config):
    torch.manual_seed(0)
    dim = -2
    input_shape, shard_shape, end_x, end_y = shapes

    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
        ),
    )

    torch_input_tensor = torch.rand(input_shape)

    # torch.max/min don't accept a tuple for dim; use amax/amin which do.
    torch_op_name = {"max": "amax", "min": "amin"}.get(op, op)
    torch_op = getattr(torch, torch_op_name)
    torch_output_tensor = torch_op(torch_input_tensor, dim=dim, keepdim=keepdim)

    # Use the op directly (not the ttnn_<op> determinism wrapper): the wrapper's
    # second execution exhausts device memory for the larger sharded shapes here.
    ttnn_op = getattr(ttnn, op)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )
    if explicit_output_mem_config:
        op_output_tensor = ttnn_op(input_tensor, dim=dim, keepdim=keepdim, memory_config=memory_config)
    else:
        op_output_tensor = ttnn_op(input_tensor, dim=dim, keepdim=keepdim)

    # Verify output is sharded with correct properties (doc: "Output sharding will mirror the input")
    output_mem_config = op_output_tensor.memory_config()
    assert output_mem_config.is_sharded(), f"op={op}: expected output to be sharded"
    assert (
        output_mem_config.buffer_type == ttnn.BufferType.L1
    ), f"op={op}: expected L1 buffer type, got {output_mem_config.buffer_type}"
    output_nd_spec = output_mem_config.nd_shard_spec
    assert output_nd_spec is not None, f"op={op}: expected output to have nd_shard_spec"

    # Expected output shard shape: same as input shard shape, but the reduced dim
    # becomes 1 when keepdim=True, or is removed when keepdim=False.
    # Output is always TILE layout (per nanobind doc), so the last two shard
    # dimensions are tile-aligned (multiples of 32) regardless of input layout.
    expected_output_shard_shape = list(shard_shape)
    normalized_dim = dim if dim >= 0 else dim + len(input_shape)
    if keepdim:
        expected_output_shard_shape[normalized_dim] = 1
    else:
        del expected_output_shard_shape[normalized_dim]
    # Align the last two shard dims up to tile boundaries, since the output is
    # always TILE layout. E.g. a reduced dim of logical size 1 becomes 32 (one full tile).
    rank = len(expected_output_shard_shape)
    for i in range(max(0, rank - 2), rank):
        # The formula rounds up to the nearest multiple of TILE_SIZE:
        #   1. Add (TILE_SIZE - 1) so that integer division rounds up instead of down
        #   2. Integer-divide by TILE_SIZE to get the number of tiles needed
        #   3. Multiply back by TILE_SIZE to convert from tile count to element count
        expected_output_shard_shape[i] = (
            (expected_output_shard_shape[i] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        ) * ttnn.TILE_SIZE
    actual_output_shard_shape = list(output_nd_spec.shard_shape)
    assert actual_output_shard_shape == expected_output_shard_shape, (
        f"op={op}: expected output shard shape {expected_output_shard_shape}, " f"got {actual_output_shard_shape}"
    )

    output_tensor = ttnn.to_torch(op_output_tensor)

    atol = rtol = 0.01
    pcc = 0.99
    passing, output_pcc = comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"op={op} {output_pcc}, torch: {torch_output_tensor}, ttnn: {output_tensor}"


# Test that generic reduction ops work correctly with Width, Height, and Block sharding.
@pytest.mark.parametrize(
    "input_shape, shard_2d_shape, end_x, end_y, memory_layout, dim",
    [
        # HEIGHT_SHARDED: each core gets a horizontal slice (some rows, full width)
        ([8, 8, 32, 32], [1024, 32], 1, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, -2),
        ([4, 4, 64, 64], [512, 64], 0, 1, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, -2),
        # WIDTH_SHARDED: each core gets a vertical slice (full height, some columns)
        ([8, 8, 32, 128], [2048, 32], 3, 0, ttnn.TensorMemoryLayout.WIDTH_SHARDED, -2),
        ([4, 4, 64, 256], [1024, 32], 7, 0, ttnn.TensorMemoryLayout.WIDTH_SHARDED, -2),
        # BLOCK_SHARDED: both height and width split across a 2D grid
        ([4, 4, 64, 64], [512, 32], 1, 1, ttnn.TensorMemoryLayout.BLOCK_SHARDED, -2),
        # Also test W reduction case to validate shard-shape recomputation when
        # the reduced output width becomes one tile.
        ([4, 4, 64, 64], [512, 32], 1, 1, ttnn.TensorMemoryLayout.BLOCK_SHARDED, -1),
    ],
)
@pytest.mark.parametrize("keepdim", [True])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("op, explicit_output_mem_config", OP_EXPLICIT_CFG)
def test_generic_ops_wh_block_shard(
    device,
    input_shape,
    shard_2d_shape,
    end_x,
    end_y,
    memory_layout,
    dim,
    keepdim,
    layout,
    op,
    explicit_output_mem_config,
):
    torch.manual_seed(0)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
        shard_2d_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        memory_layout=memory_layout,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    torch_input_tensor = torch.rand(input_shape)

    # torch.max/min don't accept a tuple for dim; use amax/amin which do.
    torch_op_name = {"max": "amax", "min": "amin"}.get(op, op)
    torch_op = getattr(torch, torch_op_name)
    torch_output_tensor = torch_op(torch_input_tensor, dim=dim, keepdim=keepdim)

    ttnn_op = TTNN_REDUCTION_WRAPPERS[op]
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )
    if explicit_output_mem_config:
        op_output_tensor = ttnn_op(input_tensor, dim=dim, keepdim=keepdim, memory_config=memory_config)
    else:
        op_output_tensor = ttnn_op(input_tensor, dim=dim, keepdim=keepdim)

    # Verify output is sharded with correct properties (doc: "Output sharding will mirror the input")
    output_mem_config = op_output_tensor.memory_config()
    assert output_mem_config.is_sharded(), f"op={op}: expected output to be sharded"
    assert (
        output_mem_config.buffer_type == ttnn.BufferType.L1
    ), f"op={op}: expected L1 buffer type, got {output_mem_config.buffer_type}"
    assert (
        output_mem_config.memory_layout == memory_layout
    ), f"op={op}: expected memory layout {memory_layout}, got {output_mem_config.memory_layout}"
    output_shard_spec = output_mem_config.shard_spec
    assert output_shard_spec is not None, f"op={op}: expected output to have shard_spec"

    def round_up_to_tile(dim_size):
        """Round up to the nearest multiple of TILE_SIZE (e.g. 1 -> 32, 33 -> 64).

        Adding (TILE_SIZE - 1) before integer-dividing by TILE_SIZE effectively
        computes ceil(dim_size / TILE_SIZE), i.e. the number of tiles needed.
        Multiplying back by TILE_SIZE converts from tile count to element count.
        """
        return ((dim_size + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

    # Compute expected 2D output shard shape.
    # Legacy sharding flattens the tensor to 2D: [product(dims[:-1]), dims[-1]].
    # After reducing dim with keepdim=True, that product changes.
    # Output is always TILE layout, so the last two dims must be tile-padded
    # BEFORE flattening to 2D (physical_shape pads then flattens, not the
    # other way around).
    output_shape = list(input_shape)
    normalized_dim = dim if dim >= 0 else dim + len(input_shape)
    if keepdim:
        output_shape[normalized_dim] = 1
    else:
        del output_shape[normalized_dim]

    # Tile-pad the last two dims before flattening, matching physical_shape().
    rank = len(output_shape)
    padded_shape = list(output_shape)
    for i in range(max(0, rank - 2), rank):
        padded_shape[i] = round_up_to_tile(padded_shape[i])

    output_2d_height = 1
    for d in padded_shape[:-1]:
        output_2d_height *= d
    output_2d_width = padded_shape[-1]
    num_cores = (end_x + 1) * (end_y + 1)

    if memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # Height is split across cores, width stays full
        expected_shard_h = (output_2d_height + num_cores - 1) // num_cores
        expected_shard_w = output_2d_width
    elif memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        # Height split across grid rows, width split across grid columns
        num_rows = end_y + 1
        num_cols = end_x + 1
        expected_shard_h = (output_2d_height + num_rows - 1) // num_rows
        expected_shard_w = (output_2d_width + num_cols - 1) // num_cols
    else:
        # Width is split across cores, height stays full
        expected_shard_h = output_2d_height
        expected_shard_w = (output_2d_width + num_cores - 1) // num_cores

    # The output is always TILE layout, so each per-core shard must contain
    # whole tiles. Round up both shard dimensions to tile boundaries.
    expected_shard_h = round_up_to_tile(expected_shard_h)
    expected_shard_w = round_up_to_tile(expected_shard_w)

    actual_shard_shape = list(output_shard_spec.shape)
    assert actual_shard_shape == [expected_shard_h, expected_shard_w], (
        f"op={op}: expected output shard shape [{expected_shard_h}, {expected_shard_w}], " f"got {actual_shard_shape}"
    )

    output_tensor = ttnn.to_torch(op_output_tensor)

    atol = rtol = 0.01
    pcc = 0.99
    passing, output_pcc = comp_allclose_and_pcc(torch_output_tensor, output_tensor, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"op={op} {output_pcc}, torch: {torch_output_tensor}, ttnn: {output_tensor}"


# Test that generic reduction ops produce correct results, preserve dtype, and emit the
# layout documented in nanobind across all supported dtype/layout combinations.
@pytest.mark.parametrize("op", ["sum", "mean", "max", "min", "std", "var"])
# bfloat8_b only exists in TILE layout, so dtype and layout are paired.
@pytest.mark.parametrize(
    "dtype, layout",
    [
        (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.float32, ttnn.TILE_LAYOUT),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ],
)
def test_generic_ops_dtypes_layouts(device, op, dtype, layout):
    """
    Test generic reduction ops across all documented dtype/layout combinations.
    Validates numerical correctness against PyTorch, verifies output dtype matches
    input dtype, and verifies output layout is TILE as documented in nanobind.

    Unset output_layout: dense RM sum/mean keep ROW_MAJOR; else TILE.
    """
    shape = (4, 2, 64, 64)
    dim = -1

    # torch has no bfloat8_b; use float32 as highest-precision reference.
    torch_dtype_map = {
        ttnn.float32: torch.float32,
        ttnn.bfloat16: torch.bfloat16,
        ttnn.bfloat8_b: torch.float32,
    }
    torch_dtype = torch_dtype_map[dtype]

    torch.manual_seed(0)
    torch_tensor = torch.randn(shape, dtype=torch_dtype)
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device)

    # torch.max/min with a single int dim return a namedtuple; use amax/amin instead.
    torch_op_name = {"max": "amax", "min": "amin"}.get(op, op)
    torch_op = getattr(torch, torch_op_name)
    torch_result = torch_op(torch_tensor, dim=dim)

    ttnn_op = TTNN_REDUCTION_WRAPPERS[op]
    ttnn_result = ttnn_op(ttnn_tensor, dim=dim)

    # Validate output dtype matches input dtype
    assert ttnn_result.dtype == dtype, f"Expected output dtype {dtype}, got {ttnn_result.dtype}"

    # Dense RM sum/mean keep ROW_MAJOR; everything else returns TILE.
    if op in ("sum", "mean") and layout == ttnn.ROW_MAJOR_LAYOUT and dtype in (ttnn.float32, ttnn.bfloat16):
        assert ttnn_result.layout == ttnn.ROW_MAJOR_LAYOUT
    else:
        assert ttnn_result.layout == ttnn.TILE_LAYOUT

    ttnn_result_torch = ttnn.to_torch(ttnn.from_device(ttnn_result))

    rtol = 0.01
    if dtype == ttnn.bfloat8_b:
        # BFLOAT8_B has lower precision.
        atol = 0.25
        pcc = 0.997
    elif op == "sum" and is_wormhole_b0():
        # Due to hardware bug (#38306), Wormhole B0 uses lower precision.
        atol = 0.04
        pcc = 0.999
    else:
        atol = 0.01
        pcc = 0.999

    passing, output_pcc = comp_allclose_and_pcc(torch_result, ttnn_result_torch, pcc=pcc, rtol=rtol, atol=atol)
    assert passing, f"{output_pcc}, torch: {torch_result}, ttnn: {ttnn_result_torch}"
