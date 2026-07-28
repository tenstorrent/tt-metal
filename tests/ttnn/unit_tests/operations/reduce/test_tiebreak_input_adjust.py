# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Validation for the greedy tie-break input adjustment used by TTSampling.

``ttnn.sampling`` breaks exact-value ties by array position, which depends on
all-gather/device order and therefore varies run-to-run and slot-to-slot.
``TTSampling._adjust_values_for_tiebreak`` corrects the sampling *input* instead:
for ARGMAX users (k == 1) it boosts the single lowest-GLOBAL-INDEX candidate among
the tied maxima, so argmax selects it deterministically. Users with k > 1 must be
left bit-identical.

These tests drive the real method (bound to a stub carrying only the two
attributes it reads) so they fail if the implementation drifts, and they run on a
restricted sub-core grid, which is where the model actually hits this.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.sampling.tt_sampling import TTSampling

NUM_USERS = 32
NUM_CANDIDATES = 64
MAX_VALUE = 1.0  # exactly representable in bf16
FILLER_VALUE = 0.25  # strictly below MAX_VALUE, exactly representable in bf16

# Restricted grid that does not start at (0, 0): the compact "full grid" placement
# would not exercise the sub-device path the model runs on.
SUB_CORE_GRIDS = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 3))})


def _build_inputs(device, greedy_users):
    """Return (values_tt, global_indices_tt, greedy_col_tt, values_torch, indices_torch, tie_positions).

    Every user row gets three exact ties at MAX_VALUE. The global indices are a
    reversed permutation, so the lowest global index among the tied maxima is NOT
    the first tied array position -- a position-based tie-break picks a different
    element than an index-based one, which is exactly what we need to distinguish.
    """
    shape = (1, 1, NUM_USERS, NUM_CANDIDATES)
    values = torch.full(shape, FILLER_VALUE, dtype=torch.float32)

    # Global indices descend across the row, so array position 0 holds the HIGHEST
    # global index and the last tied position holds the lowest.
    indices = torch.arange(NUM_CANDIDATES - 1, -1, -1, dtype=torch.int32).expand(shape).contiguous()

    tie_positions = {}
    for user in range(NUM_USERS):
        # Vary the tie positions per user so a single hardcoded winner cannot pass.
        positions = [(user + offset) % NUM_CANDIDATES for offset in (0, 7, 23)]
        values[0, 0, user, positions] = MAX_VALUE
        tie_positions[user] = positions

    greedy_col = torch.zeros((1, 1, NUM_USERS, 1), dtype=torch.float32)
    for user in greedy_users:
        greedy_col[0, 0, user, 0] = 1.0

    values_tt = ttnn.from_torch(values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    indices_tt = ttnn.from_torch(indices, device=device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
    greedy_col_tt = ttnn.from_torch(greedy_col, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    return values_tt, indices_tt, greedy_col_tt, values, indices, tie_positions


def _adjust(values_tt, indices_tt, greedy_col_tt, sub_core_grids):
    """Call the real TTSampling method with a stub holding the attributes it reads."""
    stub = SimpleNamespace(sub_core_grids=sub_core_grids, _greedy_col=greedy_col_tt)
    return TTSampling._adjust_values_for_tiebreak(stub, values_tt, indices_tt)


def _expected_winner_position(indices_row, tie_positions):
    """Array position of the tied maximum holding the lowest global index."""
    return min(tie_positions, key=lambda pos: int(indices_row[pos]))


@pytest.mark.parametrize("sub_core_grids", [SUB_CORE_GRIDS, None], ids=["sub_core_grid", "full_grid"])
def test_tiebreak_boosts_lowest_global_index_for_greedy_users(device, sub_core_grids):
    """Greedy (k == 1) rows: exactly the lowest-global-index tied max becomes the strict argmax."""
    greedy_users = list(range(0, NUM_USERS, 2))  # even users are greedy
    values_tt, indices_tt, greedy_col_tt, values, indices, tie_positions = _build_inputs(device, greedy_users)

    adjusted = ttnn.to_torch(_adjust(values_tt, indices_tt, greedy_col_tt, sub_core_grids)).float()
    original = values.to(torch.bfloat16).float()

    for user in greedy_users:
        expected_pos = _expected_winner_position(indices[0, 0, user], tie_positions[user])

        # The winner is the strict argmax after the boost...
        assert int(torch.argmax(adjusted[0, 0, user])) == expected_pos, (
            f"user {user}: expected argmax at position {expected_pos} "
            f"(global index {int(indices[0, 0, user, expected_pos])}), "
            f"got position {int(torch.argmax(adjusted[0, 0, user]))}"
        )

        # ...and it is the ONLY element that changed.
        changed = (adjusted[0, 0, user] != original[0, 0, user]).nonzero().flatten().tolist()
        assert changed == [expected_pos], f"user {user}: expected only position {expected_pos} to change, got {changed}"


@pytest.mark.parametrize("sub_core_grids", [SUB_CORE_GRIDS, None], ids=["sub_core_grid", "full_grid"])
def test_tiebreak_leaves_random_users_bit_identical(device, sub_core_grids):
    """Random (k > 1) rows must be byte-for-byte unchanged, so their sampling is untouched."""
    greedy_users = list(range(0, NUM_USERS, 2))
    random_users = [user for user in range(NUM_USERS) if user not in greedy_users]
    values_tt, indices_tt, greedy_col_tt, values, _, _ = _build_inputs(device, greedy_users)

    adjusted = ttnn.to_torch(_adjust(values_tt, indices_tt, greedy_col_tt, sub_core_grids))
    original = values.to(torch.bfloat16)

    for user in random_users:
        assert torch.equal(adjusted[0, 0, user], original[0, 0, user]), f"user {user}: k>1 row was modified"


def test_tiebreak_is_repeatable(device):
    """Repeated calls on the same input produce identical adjusted values."""
    greedy_users = list(range(NUM_USERS))
    values_tt, indices_tt, greedy_col_tt, _, _, _ = _build_inputs(device, greedy_users)

    first = ttnn.to_torch(_adjust(values_tt, indices_tt, greedy_col_tt, SUB_CORE_GRIDS))
    second = ttnn.to_torch(_adjust(values_tt, indices_tt, greedy_col_tt, SUB_CORE_GRIDS))

    assert torch.equal(first, second), "tie-break adjustment is not repeatable"


def test_sampling_picks_lowest_global_index_after_adjust(device):
    """End-to-end: ttnn.sampling with k == 1 selects the lowest-global-index tied maximum.

    Without the adjustment the pick is whichever tied maximum happens to sit first in
    the array, which is placement-dependent; with it, the pick is the lowest global index.
    """
    greedy_users = list(range(NUM_USERS))
    values_tt, indices_tt, greedy_col_tt, _, indices, tie_positions = _build_inputs(device, greedy_users)

    adjusted_tt = _adjust(values_tt, indices_tt, greedy_col_tt, SUB_CORE_GRIDS)

    # ttnn.sampling reads the index tensor in ROW_MAJOR, matching the model's call.
    indices_rm_tt = ttnn.from_torch(indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_tt = ttnn.from_torch(
        torch.ones(NUM_USERS, dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    p_tt = ttnn.from_torch(torch.zeros(NUM_USERS), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp_tt = ttnn.from_torch(torch.ones(NUM_USERS), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    sampled = ttnn.to_torch(
        ttnn.sampling(adjusted_tt, indices_rm_tt, k=k_tt, p=p_tt, temp=temp_tt, seed=1234)
    ).flatten()

    for user in range(NUM_USERS):
        expected_pos = _expected_winner_position(indices[0, 0, user], tie_positions[user])
        expected_index = int(indices[0, 0, user, expected_pos])
        assert (
            int(sampled[user]) == expected_index
        ), f"user {user}: expected sampled global index {expected_index}, got {int(sampled[user])}"
