# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import (
    height_sharded_l1_config,
    int8_with_min_value,
)

# 4D shapes cover tile-aligned and unaligned H (40); the rank 3 shape goes through the unsqueeze/squeeze to 4D.
_DIMS_4D = [(1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (0, 1)]
_DIMS_3D = [(0, 1), (0, 2), (1, 2)]
_SHAPE_DIMS = [(shape, dims) for shape in ([2, 3, 64, 96], [1, 2, 32, 64], [1, 2, 40, 64]) for dims in _DIMS_4D]
_SHAPE_DIMS += [([3, 64, 96], dims) for dims in _DIMS_3D]


@pytest.mark.parametrize(
    "shape, dims", _SHAPE_DIMS, ids=[f"{'x'.join(map(str, shape))}-{dims[0]}{dims[1]}" for shape, dims in _SHAPE_DIMS]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_transpose_block_float_tile(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_input = ttnn.to_torch(input_tensor)

    output_tensor = ttnn.transpose(input_tensor, dims[0], dims[1])
    assert output_tensor.dtype == dtype
    output = ttnn.to_torch(output_tensor)

    # Transposing the last dim moves values between shared-exponent blocks, so those results are requantized
    # and only match the transposed quantized input within the format's precision. The other pairs keep every
    # block intact and must be exact.
    expected = quantized_input.transpose(dims[0], dims[1]).contiguous()

    assert list(output.shape) == list(expected.shape)
    if len(shape) - 1 in dims:
        assert_with_pcc(expected, output, 0.99)
    else:
        assert_equal(expected, output)


# INT8 -128 regression (#67429): the Int8 CB format is sign-magnitude, so unpack/pack turned -128 into 0.
# Factories reached by ttnn.transpose of the last two dims:
#   tile_interleaved, tile_interleaved_unaligned: TransposeWHProgramFactory (Wormhole and Blackhole).
#   tile_height_sharded: TransposeWHShardedProgramFactory (Wormhole and Blackhole).
#   row_major_interleaved: ttnn::transpose -> prim::permute -> MultiCoreBlockedGeneric (Wormhole and Blackhole).
#   row_major_height_sharded, row_major_height_sharded_staged: Wormhole reaches TransposeWHShardedRMProgramFactory;
#     288 rows are 9 tile rows, and the factory stages its output only when it has more than 8, so the staged case
#     covers that path. Blackhole moves sharded 8-bit row-major input to interleaved L1 first (pack_untilize_dest
#     truncation in the sharded RM factory) and reaches prim::permute MultiCoreBlockedGeneric. Both architectures
#     therefore run a factory that needed the fix.
@pytest.mark.parametrize(
    "shape, layout, num_shard_cores",
    [
        ([1, 1, 64, 64], ttnn.TILE_LAYOUT, None),
        ([1, 1, 50, 70], ttnn.TILE_LAYOUT, None),
        ([1, 2, 64, 64], ttnn.TILE_LAYOUT, 2),
        ([1, 1, 64, 64], ttnn.ROW_MAJOR_LAYOUT, None),
        ([1, 2, 64, 64], ttnn.ROW_MAJOR_LAYOUT, 2),
        ([1, 2, 288, 64], ttnn.ROW_MAJOR_LAYOUT, 2),
    ],
    ids=[
        "tile_interleaved",
        "tile_interleaved_unaligned",
        "tile_height_sharded",
        "row_major_interleaved",
        "row_major_height_sharded",
        "row_major_height_sharded_staged",
    ],
)
def test_transpose_wh_int8_min_value(device, shape, layout, num_shard_cores):
    x = int8_with_min_value(shape)
    in_config = out_config = ttnn.DRAM_MEMORY_CONFIG
    if num_shard_cores is not None:
        in_config = height_sharded_l1_config(shape, num_shard_cores)
        out_config = height_sharded_l1_config(shape[:2] + [shape[3], shape[2]], num_shard_cores)
    t = ttnn.from_torch(x, dtype=ttnn.int8, layout=layout, device=device, memory_config=in_config)

    got = ttnn.to_torch(ttnn.from_device(ttnn.transpose(t, -2, -1, memory_config=out_config)).to(ttnn.ROW_MAJOR_LAYOUT))

    assert torch.equal(got, x.transpose(-2, -1))
