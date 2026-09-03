# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import ttnn


def make_tile_constant_tensor(shape, offset):
    values = torch.arange(math.prod(shape[:-2]), dtype=torch.bfloat16).reshape(*shape[:-2], 1, 1)
    return values.expand(shape) + offset


@pytest.mark.parametrize(
    "ttnn_op, torch_op",
    [
        (ttnn.add, torch.add),
        (ttnn.subtract, torch.subtract),
        (ttnn.multiply, torch.multiply),
    ],
)
@pytest.mark.parametrize("broadcast_lhs", [True, False])
@pytest.mark.parametrize("rank", [7, 8])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_binary_ng_broadcast_dim_minus_6_at_high_rank(device, ttnn_op, torch_op, broadcast_lhs, rank, layout):
    # Dimensions -7 and outward match. Dimension -6 broadcasts while an outer
    # dimension is non-unit, so the collapsed nD input index must repeat.
    outer_dims = [2] * (rank - 6)
    broadcast_shape = (*outer_dims, 1, 2, 2, 2, 32, 32)
    full_shape = (*outer_dims, 3, 2, 2, 2, 32, 32)

    lhs_shape, rhs_shape = (broadcast_shape, full_shape) if broadcast_lhs else (full_shape, broadcast_shape)
    torch_lhs = make_tile_constant_tensor(lhs_shape, 1)
    torch_rhs = make_tile_constant_tensor(rhs_shape, 10)

    lhs = ttnn.from_torch(torch_lhs, dtype=ttnn.bfloat16, layout=layout, device=device)
    rhs = ttnn.from_torch(torch_rhs, dtype=ttnn.bfloat16, layout=layout, device=device)
    actual = ttnn.to_torch(ttnn_op(lhs, rhs))
    expected = torch_op(torch_lhs, torch_rhs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "lhs_hw, rhs_hw",
    [
        ((1, 32), (32, 32)),
        ((32, 1), (32, 32)),
        ((1, 1), (32, 32)),
        ((1, 32), (32, 1)),
    ],
)
@pytest.mark.parametrize("broadcast_lhs", [True, False])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_binary_ng_broadcast_dim_minus_6_with_subtile_broadcast(device, lhs_hw, rhs_hw, broadcast_lhs, layout):
    lhs_nd, rhs_nd = (1, 3) if broadcast_lhs else (3, 1)
    lhs_shape = (2, lhs_nd, 1, 1, 1, *lhs_hw)
    rhs_shape = (2, rhs_nd, 1, 1, 1, *rhs_hw)
    torch_lhs = make_tile_constant_tensor(lhs_shape, 1)
    torch_rhs = make_tile_constant_tensor(rhs_shape, 10)

    lhs = ttnn.from_torch(torch_lhs, dtype=ttnn.bfloat16, layout=layout, device=device)
    rhs = ttnn.from_torch(torch_rhs, dtype=ttnn.bfloat16, layout=layout, device=device)
    actual = ttnn.to_torch(ttnn.add(lhs, rhs))
    expected = torch.add(torch_lhs, torch_rhs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("broadcast_lhs", [True, False])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_binary_ng_broadcast_dim_minus_6_across_nd_slices(device, broadcast_lhs, layout):
    # One tile per collapsed nD slice and many more slices than cores, so each core
    # walks several nD iterations and repeatedly crosses a dim -6 broadcast boundary.
    broadcast_shape = (48, 1, 1, 1, 1, 32, 32)
    full_shape = (48, 4, 1, 1, 1, 32, 32)

    lhs_shape, rhs_shape = (broadcast_shape, full_shape) if broadcast_lhs else (full_shape, broadcast_shape)
    torch_lhs = make_tile_constant_tensor(lhs_shape, 1)
    torch_rhs = make_tile_constant_tensor(rhs_shape, 10)

    lhs = ttnn.from_torch(torch_lhs, dtype=ttnn.bfloat16, layout=layout, device=device)
    rhs = ttnn.from_torch(torch_rhs, dtype=ttnn.bfloat16, layout=layout, device=device)
    actual = ttnn.to_torch(ttnn.add(lhs, rhs))
    expected = torch.add(torch_lhs, torch_rhs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("broadcast_first", [True, False])
def test_binary_ng_broadcast_dim_minus_6_program_cache(device, broadcast_first):
    full_shape = (2, 3, 1, 1, 1, 32, 32)
    broadcast_shape = (2, 1, 1, 1, 1, 32, 32)
    shape_pairs = [(broadcast_shape, full_shape), (full_shape, full_shape)]
    if not broadcast_first:
        shape_pairs.reverse()

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        for lhs_shape, rhs_shape in shape_pairs:
            torch_lhs = make_tile_constant_tensor(lhs_shape, 1)
            torch_rhs = make_tile_constant_tensor(rhs_shape, 10)
            lhs = ttnn.from_torch(torch_lhs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            rhs = ttnn.from_torch(torch_rhs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            with device.cache_entries_counter.measure():
                output = ttnn.add(lhs, rhs)
            actual = ttnn.to_torch(output)
            expected = torch.add(torch_lhs, torch_rhs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        assert device.cache_entries_counter.total == 1
    finally:
        device.disable_and_clear_program_cache()


@pytest.mark.parametrize("variant", ["tts", "tst"])
@pytest.mark.parametrize("broadcast_operand", ["predicate", "tensor"])
def test_binary_ng_where_scalar_broadcast_dim_minus_6_at_high_rank(device, variant, broadcast_operand):
    # ttnn.where tensor-scalar is implemented on binary_ng, not the ternary readers.
    full_shape = (2, 3, 2, 2, 2, 32, 32)
    broadcast_shape = (2, 1, 2, 2, 2, 32, 32)
    predicate_shape = broadcast_shape if broadcast_operand == "predicate" else full_shape
    tensor_shape = broadcast_shape if broadcast_operand == "tensor" else full_shape
    torch_predicate = make_tile_constant_tensor(predicate_shape, 0).remainder(2)
    torch_tensor = make_tile_constant_tensor(tensor_shape, 1)

    predicate = ttnn.from_torch(torch_predicate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if variant == "tts":
        actual = ttnn.to_torch(ttnn.where(predicate, tensor, -1.0))
        expected = torch.where(torch_predicate != 0, torch_tensor, -1.0)
    else:
        actual = ttnn.to_torch(ttnn.where(predicate, 1.0, tensor))
        expected = torch.where(torch_predicate != 0, 1.0, torch_tensor)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
