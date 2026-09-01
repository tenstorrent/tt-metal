# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only contract tests for compute_layer_split's model-aware boundary handling.

GLM-5.2 (DSA cross-layer indexer reuse) requires every pipeline rank to START on a ``full`` layer —
a rank beginning on a ``shared`` layer has no prior top-k to reuse and the reuse-seed assertion in
tt_prefill_transformer aborts. So the runner must snap the default even split onto, and validate an
explicit PREFILL_PP_LAYER_COUNTS against, the adapter's ``layer_split_boundaries`` (the full layers).

The ``mtp_levels`` tests cover the #53533 MTP4 re-split: the tail rank runs K extra MTP blocks, so the
split balances num_layers + K layer-equivalents instead of the trunk alone. That is a four-galaxy
property, so it is pinned here rather than on hardware in reach. No device is opened here."""

import pytest

from models.demos.common.prefill.runners.prefill_runner import compute_layer_split
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_2 import GLM52Adapter

GLM52_LAYERS = 78


@pytest.fixture
def full_starts():
    starts = GLM52Adapter().layer_split_boundaries(GLM52_LAYERS)
    assert starts is not None and 0 in starts, "GLM-5.2 must expose full-layer start boundaries incl. layer 0"
    return starts


@pytest.mark.parametrize("num_ranks", [1, 2, 4, 8])
def test_default_split_snaps_to_full_layers(monkeypatch, full_starts, num_ranks):
    """The bare default (no PREFILL_PP_LAYER_COUNTS) must yield a split whose every rank starts on a
    full layer — i.e. multi-rank launches work out of the box."""
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    ranges = compute_layer_split(GLM52_LAYERS, num_ranks, full_starts)
    assert len(ranges) == num_ranks
    assert sum(count for _, count in ranges) == GLM52_LAYERS
    assert all(first in full_starts for first, _ in ranges)


def test_explicit_valid_split_ok(monkeypatch, full_starts):
    monkeypatch.setenv("PREFILL_PP_LAYER_COUNTS", "18,20,20,20")
    ranges = compute_layer_split(GLM52_LAYERS, 4, full_starts)
    assert [first for first, _ in ranges] == [0, 18, 38, 58]


def test_explicit_invalid_boundary_rejected(monkeypatch, full_starts, expect_error):
    # 20,20,19,19 -> boundaries 20/40/59, all shared layers -> must be rejected early with guidance.
    monkeypatch.setenv("PREFILL_PP_LAYER_COUNTS", "20,20,19,19")
    with expect_error(ValueError, "not a valid boundary"):
        compute_layer_split(GLM52_LAYERS, 4, full_starts)


def test_dense_unconstrained_even_split(monkeypatch):
    # valid_starts=None (dense) -> plain even split, remainder to earlier ranks, no snapping/validation.
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    ranges = compute_layer_split(61, 4, None)
    assert [count for _, count in ranges] == [16, 15, 15, 15]
    assert [first for first, _ in ranges] == [0, 16, 31, 46]


# --- MTP4 re-split (#53533): the tail rank also runs K MTP levels -------------------------------


def test_mtp4_resplit_matches_target(monkeypatch, full_starts):
    """The re-split #53533 asks for: 22/20/20/20 layer-equivalents at starts 0/22/42/62. The TRUNK
    counts are 22/20/20/16 — the tail's last 4 equivalents are its 4 MTP levels, not trunk layers."""
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    ranges = compute_layer_split(GLM52_LAYERS, 4, full_starts, 4)
    assert [first for first, _ in ranges] == [0, 22, 42, 62]
    assert [count for _, count in ranges] == [22, 20, 20, 16]
    equivalents = [c + (4 if r == 3 else 0) for r, (_, c) in enumerate(ranges)]
    assert equivalents == [22, 20, 20, 20]
    assert sum(count for _, count in ranges) == GLM52_LAYERS


def test_mtp_resplit_lowers_peak_stage_load(monkeypatch, full_starts):
    """Why the re-split exists: split the trunk alone and the tail carries its full share PLUS all K
    MTP blocks, with every other rank stalled behind it. Peak per-stage work must strictly improve."""
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    k = 4
    naive = [count for _, count in compute_layer_split(GLM52_LAYERS, 4, full_starts)]
    naive[-1] += k  # 18/20/20/20 trunk + 4 bolted on the tail -> peak 24
    aware = [count for _, count in compute_layer_split(GLM52_LAYERS, 4, full_starts, k)]
    aware[-1] += k  # 22/20/20/16 trunk + the same 4 -> peak 22
    assert max(aware) < max(naive), f"re-split did not improve balance: {aware} vs {naive}"
    assert sum(aware) == sum(naive) == GLM52_LAYERS + k


@pytest.mark.parametrize("num_ranks", [1, 2, 4, 8])
def test_mtp_split_still_lands_on_full_layers(monkeypatch, full_starts, num_ranks):
    """Re-balancing must not push a rank start onto a ``shared`` layer — the constraint the snapping
    exists for. Trunk counts still sum to num_layers (MTP slots are not trunk layers)."""
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    ranges = compute_layer_split(GLM52_LAYERS, num_ranks, full_starts, 4)
    assert len(ranges) == num_ranks
    assert sum(count for _, count in ranges) == GLM52_LAYERS
    assert all(first in full_starts for first, _ in ranges)


@pytest.mark.parametrize("num_ranks", [1, 2, 4, 8])
def test_mtp_levels_zero_is_the_old_split(monkeypatch, full_starts, num_ranks):
    """Regression guard: every non-MTP model passes K=0 and must get byte-identical ranges."""
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    assert compute_layer_split(GLM52_LAYERS, num_ranks, full_starts, 0) == compute_layer_split(
        GLM52_LAYERS, num_ranks, full_starts
    )


def test_single_rank_mtp_split_unchanged(monkeypatch, full_starts):
    """The configuration MTP is being brought up in: one rank owns every trunk layer and the MTP
    tail, so K must not carve anything out of it."""
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    assert compute_layer_split(GLM52_LAYERS, 1, full_starts, 4) == [(0, GLM52_LAYERS)]


def test_explicit_counts_are_trunk_only(monkeypatch, full_starts):
    """PREFILL_PP_LAYER_COUNTS stays the hand-tuning escape hatch: it lists TRUNK counts summing to
    num_layers, and K neither changes it nor invalidates it."""
    monkeypatch.setenv("PREFILL_PP_LAYER_COUNTS", "22,20,20,16")
    ranges = compute_layer_split(GLM52_LAYERS, 4, full_starts, 4)
    assert [first for first, _ in ranges] == [0, 22, 42, 62]
    assert [count for _, count in ranges] == [22, 20, 20, 16]


def test_mtp_starved_tail_rejected(monkeypatch, expect_error):
    """K large enough to consume the tail's whole share leaves a rank with no trunk stage at all —
    reject it by name rather than emitting a zero/negative count downstream."""
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    with expect_error(ValueError, "no trunk stage"):
        compute_layer_split(4, 4, None, 4)
