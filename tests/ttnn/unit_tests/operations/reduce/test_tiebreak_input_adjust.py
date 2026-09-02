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
from models.common.sampling.tt_sampling import TIEBREAK_INDEX_SENTINEL, TTSampling

NUM_USERS = 32
NUM_CANDIDATES = 64
# Global vocabulary indices are in the 100k range, like a real gathered candidate set. Deliberately
# far above 2**11: the FPU reduce truncates its source registers to a 10-bit mantissa, so indices this
# large come back rounded from a float32 ttnn.min and match no real index. Only the int32 reduce is
# exact here, and these magnitudes are what make the test notice if the index half leaves int32.
INDEX_BASE = 100_000
# TTSampling.forward builds the gathered global-index tensor as uint32, so that is what these tests
# feed by default; uint32 min/max are on the same inexact FPU reduce path as float32, and the method is
# responsible for casting to int32 itself. INDEX_DTYPES also covers int32 in case a caller pre-casts.
MODEL_INDEX_DTYPE = ttnn.uint32
INDEX_DTYPES = [ttnn.uint32, ttnn.int32]
INDEX_DTYPE_IDS = ["uint32_indices", "int32_indices"]

# (tied maximum, filler). All exactly representable in bf16, filler strictly below the maximum.
# The magnitudes matter: the boost has to be at least one bf16 ULP of the tied maximum, and bf16
# spacing is 2.0 at 256 and 8.0 at 1024, so a fixed +1.0 boost would round away and leave the tie
# in place for the large-magnitude rows. Negative and zero maxima are covered for the same reason.
MAGNITUDES = [
    (1.0, 0.25),
    (256.0, 128.0),
    (1024.0, 512.0),
    (-256.0, -512.0),
    (0.0, -1.0),
]
MAGNITUDE_IDS = ["max_1", "max_256", "max_1024", "max_neg256", "max_0"]

DEFAULT_MAX_VALUE, DEFAULT_FILLER_VALUE = MAGNITUDES[0]

# Restricted grid that does not start at (0, 0): the compact "full grid" placement
# would not exercise the sub-device path the model runs on.
SUB_CORE_GRIDS = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 3))})


def _build_inputs(
    device,
    greedy_users,
    max_value=DEFAULT_MAX_VALUE,
    filler_value=DEFAULT_FILLER_VALUE,
    index_dtype=MODEL_INDEX_DTYPE,
):
    """Return (values_tt, global_indices_tt, greedy_col_tt, values_torch, indices_torch, tie_positions).

    Every user row gets three exact ties at max_value. The global indices are a
    reversed permutation, so the lowest global index among the tied maxima is NOT
    the first tied array position -- a position-based tie-break picks a different
    element than an index-based one, which is exactly what we need to distinguish.
    """
    shape = (1, 1, NUM_USERS, NUM_CANDIDATES)
    values = torch.full(shape, filler_value, dtype=torch.float32)

    # Global indices descend across the row, so array position 0 holds the HIGHEST
    # global index and the last tied position holds the lowest.
    indices = (
        torch.arange(INDEX_BASE + NUM_CANDIDATES - 1, INDEX_BASE - 1, -1, dtype=torch.int32).expand(shape).contiguous()
    )

    tie_positions = {}
    for user in range(NUM_USERS):
        # Vary the tie positions per user so a single hardcoded winner cannot pass.
        positions = [(user + offset) % NUM_CANDIDATES for offset in (0, 7, 23)]
        values[0, 0, user, positions] = max_value
        tie_positions[user] = positions

    greedy_col = torch.zeros((1, 1, NUM_USERS, 1), dtype=torch.float32)
    for user in greedy_users:
        greedy_col[0, 0, user, 0] = 1.0

    values_tt = ttnn.from_torch(values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    indices_tt = ttnn.from_torch(indices, device=device, dtype=index_dtype, layout=ttnn.TILE_LAYOUT)
    greedy_col_tt = ttnn.from_torch(greedy_col, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return values_tt, indices_tt, greedy_col_tt, values, indices, tie_positions


def _adjust(values_tt, indices_tt, greedy_col_tt, sub_core_grids):
    """Call the real TTSampling method with a stub holding the attributes it reads."""
    stub = SimpleNamespace(sub_core_grids=sub_core_grids, _greedy_col=greedy_col_tt)
    return TTSampling._adjust_values_for_tiebreak(stub, values_tt, indices_tt)


def _expected_winner_position(indices_row, tie_positions):
    """Array position of the tied maximum holding the lowest global index."""
    return min(tie_positions, key=lambda pos: int(indices_row[pos]))


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
def test_masked_index_min_reduce_is_exact(device, index_dtype):
    """The index half of the tie-break must be exact at real vocabulary magnitudes.

    ``ttnn.min`` on a float32 or uint32 tensor runs on the FPU, which truncates its source registers to
    a 10-bit mantissa, so an index above 2**11 comes back rounded to a value that equals no real index:
    the winner mask ends up empty and the whole adjustment degrades into a silent no-op. This pins the
    int32 reduce (and the int32 broadcast equality) the implementation depends on, so a regression
    there fails here and names itself instead of resurfacing as an unexplained tie downstream.

    Runs first in this file on purpose: under ``-x`` it is the failure you want to see.
    """
    shape = (1, 1, NUM_USERS, NUM_CANDIDATES)
    indices = (
        torch.arange(INDEX_BASE + NUM_CANDIDATES - 1, INDEX_BASE - 1, -1, dtype=torch.int32).expand(shape).contiguous()
    )

    # Exactly one surviving candidate per row, at a different position in each row so a placement-
    # dependent reduce cannot pass by accident.
    kept_positions = [(user * 5) % NUM_CANDIDATES for user in range(NUM_USERS)]
    not_max = torch.ones(shape, dtype=torch.float32)
    for user, position in enumerate(kept_positions):
        not_max[0, 0, user, position] = 0.0

    indices_tt = ttnn.from_torch(indices, device=device, dtype=index_dtype, layout=ttnn.TILE_LAYOUT)
    not_max_tt = ttnn.from_torch(not_max, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # The same op sequence _adjust_values_for_tiebreak uses for its index half.
    idx_tt = ttnn.typecast(indices_tt, ttnn.int32, sub_core_grids=SUB_CORE_GRIDS)
    offset_tt = ttnn.typecast(
        ttnn.multiply(not_max_tt, TIEBREAK_INDEX_SENTINEL, sub_core_grids=SUB_CORE_GRIDS),
        ttnn.int32,
        sub_core_grids=SUB_CORE_GRIDS,
    )
    masked_tt = ttnn.add(idx_tt, offset_tt, sub_core_grids=SUB_CORE_GRIDS)
    row_min_tt = ttnn.min(masked_tt, dim=3, keepdim=True, sub_core_grids=SUB_CORE_GRIDS)
    row_min = ttnn.to_torch(row_min_tt).flatten()
    selected = ttnn.to_torch(ttnn.eq(idx_tt, row_min_tt, sub_core_grids=SUB_CORE_GRIDS))

    for user, position in enumerate(kept_positions):
        expected_index = int(indices[0, 0, user, position])
        assert int(row_min[user]) == expected_index, (
            f"user {user}: masked int32 row min returned {int(row_min[user])}, expected the one "
            f"unmasked global index {expected_index}"
        )
        hits = selected[0, 0, user].nonzero().flatten().tolist()
        assert hits == [position], f"user {user}: int32 broadcast equality selected {hits}, expected [{position}]"


@pytest.mark.parametrize("max_value, filler_value", MAGNITUDES, ids=MAGNITUDE_IDS)
@pytest.mark.parametrize("sub_core_grids", [SUB_CORE_GRIDS, None], ids=["sub_core_grid", "full_grid"])
def test_tiebreak_boosts_lowest_global_index_for_greedy_users(device, sub_core_grids, max_value, filler_value):
    """Greedy (k == 1) rows: exactly the lowest-global-index tied max becomes the strict argmax.

    Parametrised over the tied maximum's magnitude because the boost must exceed one bf16 ULP at
    that magnitude: a fixed +1.0 boost is silently lost at 256 and above, leaving the tie in place.
    """
    greedy_users = list(range(0, NUM_USERS, 2))  # even users are greedy
    values_tt, indices_tt, greedy_col_tt, values, indices, tie_positions = _build_inputs(
        device, greedy_users, max_value, filler_value
    )

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


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
@pytest.mark.parametrize("sub_core_grids", [SUB_CORE_GRIDS, None], ids=["sub_core_grid", "full_grid"])
def test_tiebreak_leaves_random_users_bit_identical(device, sub_core_grids, index_dtype):
    """Random (k > 1) rows must be byte-for-byte unchanged, so their sampling is untouched."""
    greedy_users = list(range(0, NUM_USERS, 2))
    random_users = [user for user in range(NUM_USERS) if user not in greedy_users]
    values_tt, indices_tt, greedy_col_tt, values, _, _ = _build_inputs(device, greedy_users, index_dtype=index_dtype)

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


@pytest.mark.parametrize("max_value, filler_value", MAGNITUDES, ids=MAGNITUDE_IDS)
def test_sampling_picks_lowest_global_index_after_adjust(device, max_value, filler_value):
    """End-to-end: ttnn.sampling with k == 1 selects the lowest-global-index tied maximum.

    Without the adjustment the pick is whichever tied maximum happens to sit first in
    the array, which is placement-dependent; with it, the pick is the lowest global index.
    """
    greedy_users = list(range(NUM_USERS))
    values_tt, indices_tt, greedy_col_tt, _, indices, tie_positions = _build_inputs(
        device, greedy_users, max_value, filler_value
    )

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
