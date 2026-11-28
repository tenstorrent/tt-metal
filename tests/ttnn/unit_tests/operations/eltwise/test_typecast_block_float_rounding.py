# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def _source_values(kind, shape):
    torch.manual_seed(0)
    if kind == "randn":
        return torch.randn(shape)
    if kind == "wide_range":
        # Values that share a block span many exponents, so most of them are shifted far below the block exponent.
        return torch.randn(shape) * torch.pow(2.0, torch.randint(-20, 21, shape).float())
    if kind == "ties":
        # Multiples of 1/16 below 16: in a block whose largest value is in [8, 16), bfloat8_b's step is 1/8 and
        # bfloat4_b's is 2, so the odd multiples of 1/16 (and of 1) are exact ties.
        return (torch.randint(-255, 256, shape).float() / 16.0).clamp(-15.9375, 15.9375)
    if kind == "round_up_to_max":
        # Every value rounds up to 2, past the largest mantissa, so it has to clamp. 1 + 127/128 is exact in bfloat16.
        return torch.full(shape, 1.9921875) * torch.where(torch.rand(shape) < 0.5, -1.0, 1.0)
    if kind == "zeros_and_tiny":
        x = torch.randn(shape)
        x[torch.rand(shape) < 0.3] = 0.0
        x[torch.rand(shape) < 0.2] *= 1e-12
        return x
    raise ValueError(kind)


@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (ttnn.float32, ttnn.bfloat8_b),
        (ttnn.float32, ttnn.bfloat4_b),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
        (ttnn.bfloat8_b, ttnn.bfloat4_b),
    ],
)
@pytest.mark.parametrize("kind", ["randn", "wide_range", "ties", "round_up_to_max", "zeros_and_tiny"])
@pytest.mark.parametrize("shape", [(128, 128), (2, 64, 96)])
def test_typecast_to_block_float_matches_from_torch(device, input_dtype, output_dtype, kind, shape):
    # Quantize to the input dtype on the host first, so device and host start from the same values.
    x = ttnn.to_torch(ttnn.from_torch(_source_values(kind, shape), dtype=input_dtype, layout=ttnn.TILE_LAYOUT)).float()
    t = ttnn.from_torch(x, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.typecast(t, output_dtype)
    assert result.dtype == output_dtype
    expected = ttnn.to_torch(ttnn.from_torch(x, dtype=output_dtype, layout=ttnn.TILE_LAYOUT)).float()
    actual = ttnn.to_torch(result).float()
    assert torch.equal(actual, expected), f"{int((actual != expected).sum())} of {actual.numel()} values differ"


@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_typecast_to_block_float_sharded_matches_from_torch(device, input_dtype, output_dtype):
    torch.manual_seed(0)
    x = ttnn.to_torch(ttnn.from_torch(torch.randn(1, 1, 128, 128), dtype=input_dtype, layout=ttnn.TILE_LAYOUT)).float()
    shard_config = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.CoreGrid(y=4, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    t = ttnn.from_torch(x, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_config)
    expected = ttnn.to_torch(ttnn.from_torch(x, dtype=output_dtype, layout=ttnn.TILE_LAYOUT)).float()
    assert torch.equal(ttnn.to_torch(ttnn.typecast(t, output_dtype)).float(), expected)


@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_typecast_to_block_float_unaligned_width_matches_from_torch(device, output_dtype):
    # A width that is not a multiple of 16 leaves implicit padding in the last block of each row.
    torch.manual_seed(0)
    x = torch.randn(64, 50)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    expected = ttnn.to_torch(ttnn.from_torch(x, dtype=output_dtype, layout=ttnn.TILE_LAYOUT)).float()
    assert torch.equal(ttnn.to_torch(ttnn.typecast(t, output_dtype)).float(), expected)
