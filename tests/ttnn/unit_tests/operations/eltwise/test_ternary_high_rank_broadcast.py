# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import ttnn


def make_tile_constant_tensor(shape, offset=0):
    values = torch.arange(math.prod(shape[:-2])).remainder(127).to(torch.bfloat16).reshape(*shape[:-2], 1, 1)
    return values.expand(shape) + offset


def make_predicate(shape):
    return make_tile_constant_tensor(shape).remainder(2)


def run_where(device, predicate_shape, true_shape, false_shape):
    torch_predicate = make_predicate(predicate_shape)
    torch_true = make_tile_constant_tensor(true_shape, 1)
    torch_false = -make_tile_constant_tensor(false_shape, 1)

    predicate = ttnn.from_torch(torch_predicate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    value_true = ttnn.from_torch(torch_true, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    value_false = ttnn.from_torch(torch_false, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    actual = ttnn.to_torch(ttnn.where(predicate, value_true, value_false))
    expected = torch.where(torch_predicate != 0, torch_true, torch_false)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def run_lerp(device, input_shape, end_shape, weight=0.5):
    torch_input = make_tile_constant_tensor(input_shape, 1)
    torch_end = -make_tile_constant_tensor(end_shape, 1)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_end = ttnn.from_torch(torch_end, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    actual = ttnn.to_torch(ttnn.lerp(tt_input, tt_end, weight))
    expected = torch.lerp(torch_input.float(), torch_end.float(), weight).to(torch.bfloat16)
    # Consecutive nD slices are at least 8 apart in these tensors, so this stays far tighter than the
    # error a mis-mapped dim -6 index produces while tolerating bfloat16 rounding in the LLK.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


# Per-operand (H, W) that steer get_broadcast_type() to each subtile-broadcast reader.
TTT_SUBTILE_PATTERNS = {
    "row_bcast": {"predicate": (1, 32), "true": (32, 32), "false": (32, 32)},
    "col_bcast": {"predicate": (32, 1), "true": (32, 32), "false": (32, 32)},
    "scalar_bcast": {"predicate": (1, 1), "true": (32, 32), "false": (32, 32)},
    "row_col_bcast": {"predicate": (32, 1), "true": (1, 32), "false": (32, 32)},
}

TS_SUBTILE_PATTERNS = {
    "outer_bcast": {"input": (32, 32), "end": (32, 32)},
    "row_bcast": {"input": (32, 32), "end": (1, 32)},
    "col_bcast": {"input": (32, 32), "end": (32, 1)},
    "scalar_b_bcast": {"input": (32, 32), "end": (1, 1)},
}


@pytest.mark.parametrize("rank", [7, 8])
@pytest.mark.parametrize("broadcast_operand", ["predicate", "true", "false"])
def test_where_broadcast_dim_minus_6_at_high_rank(device, rank, broadcast_operand):
    outer_dims = [2] * (rank - 6)
    full_shape = (*outer_dims, 3, 2, 2, 2, 32, 32)
    broadcast_shape = (*outer_dims, 1, 2, 2, 2, 32, 32)
    shapes = {"predicate": full_shape, "true": full_shape, "false": full_shape}
    shapes[broadcast_operand] = broadcast_shape

    run_where(device, shapes["predicate"], shapes["true"], shapes["false"])


@pytest.mark.parametrize("subtile_pattern", list(TTT_SUBTILE_PATTERNS))
@pytest.mark.parametrize("broadcast_operand", ["predicate", "true", "false"])
def test_where_broadcast_dim_minus_6_with_subtile_broadcast(device, subtile_pattern, broadcast_operand):
    # Each subtile pattern selects a different TTT reader, and every one of them has to map the
    # dim -6 index rather than indexing with the output's.
    subtile_shapes = TTT_SUBTILE_PATTERNS[subtile_pattern]
    shapes = {
        operand: (2, 1 if operand == broadcast_operand else 3, 2, 2, 2, *subtile_shape)
        for operand, subtile_shape in subtile_shapes.items()
    }

    run_where(device, shapes["predicate"], shapes["true"], shapes["false"])


@pytest.mark.parametrize("subtile_pattern", list(TS_SUBTILE_PATTERNS))
@pytest.mark.parametrize("broadcast_operand", ["input", "end"])
def test_lerp_broadcast_dim_minus_6_with_subtile_broadcast(device, subtile_pattern, broadcast_operand):
    # ttnn.lerp with a scalar weight is the tensor-scalar variant, which has its own set of readers.
    subtile_shapes = TS_SUBTILE_PATTERNS[subtile_pattern]
    shapes = {
        operand: (2, 1 if operand == broadcast_operand else 3, 2, 2, 2, *subtile_shape)
        for operand, subtile_shape in subtile_shapes.items()
    }

    run_lerp(device, shapes["input"], shapes["end"])


@pytest.mark.parametrize("broadcast_operand", ["predicate", "true", "false"])
def test_where_broadcast_dim_minus_6_across_many_nd_slices(device, broadcast_operand):
    # More nD slices per core than D tiles per slice, with D > 1, so each
    # operand exercises both non-zero repeat and advance nD shifts.
    full_shape = (2, 100, 2, 1, 1, 32, 32)
    broadcast_shape = (2, 1, 2, 1, 1, 32, 32)
    shapes = {"predicate": full_shape, "true": full_shape, "false": full_shape}
    shapes[broadcast_operand] = broadcast_shape

    run_where(device, shapes["predicate"], shapes["true"], shapes["false"])


@pytest.mark.parametrize("broadcast_first", [True, False])
def test_where_broadcast_dim_minus_6_program_cache_factor_update(device, broadcast_first):
    # These predicate shapes have equal volumes and H/W, so they share one cache
    # entry. Their nD factors differ because the broadcast moves from dim -6 to -5.
    full_shape = (2, 3, 3, 1, 1, 32, 32)
    broadcast_dim_minus_6 = (2, 1, 3, 1, 1, 32, 32)
    broadcast_dim_minus_5 = (2, 3, 1, 1, 1, 32, 32)
    predicate_shapes = [broadcast_dim_minus_6, broadcast_dim_minus_5]
    if not broadcast_first:
        predicate_shapes.reverse()

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        for predicate_shape in predicate_shapes:
            run_where(device, predicate_shape, full_shape, full_shape)

        assert device.num_program_cache_entries() == 1
    finally:
        device.disable_and_clear_program_cache()
