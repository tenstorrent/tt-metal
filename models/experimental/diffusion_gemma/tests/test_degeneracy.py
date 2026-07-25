# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Degenerate-canvas detection and the commit-path policy (#48291).

Block diffusion commits a whole 256-token canvas at once, so degeneration is directly visible in
the committed tensor -- the observed GPQA failure emitted a canvas of ``\\ \\ \\ ...`` and then a
solid wall of one token id. These CPU tests pin the measurement, the calibration bounds against
realistic prose, and the three policies.
"""

import torch

from models.experimental.diffusion_gemma.tt import degeneracy as DG

CANVAS = 256


def _prose_like(seed: int = 0, length: int = CANVAS) -> torch.Tensor:
    """A canvas with a realistic id distribution: Zipf-ish, with repeated common tokens."""
    generator = torch.Generator().manual_seed(seed)
    weights = 1.0 / torch.arange(1, 4001, dtype=torch.float32)
    ids = torch.multinomial(weights, length, replacement=True, generator=generator)
    return ids.reshape(1, length)


def test_wall_of_one_token_is_maximally_degenerate():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), 1, dtype=torch.long))
    assert stats["top_frac"] == 1.0
    assert stats["distinct"] == 1
    assert stats["max_run"] == CANVAS
    assert DG.is_degenerate(stats)


def test_short_cycle_is_caught_even_with_a_small_max_run():
    """The `\\ \\ \\ \\` shape: two ids alternating, so no long run but a huge top_frac."""
    ids = torch.tensor([[7, 9] * (CANVAS // 2)], dtype=torch.long)
    stats = DG.block_degeneracy(ids)
    assert stats["max_run"] == 1
    assert stats["top_frac"] == 0.5
    assert DG.is_degenerate(stats), "a 2-cycle must be caught by top_frac even with max_run == 1"


def test_realistic_prose_is_not_flagged():
    """The bounds must be far from healthy text or the gate is useless."""
    for seed in range(8):
        stats = DG.block_degeneracy(_prose_like(seed))
        assert not DG.is_degenerate(stats), f"seed {seed} false-positived: {stats}"
        assert stats["top_frac"] < DG.DEFAULT_TOP_FRAC
        assert stats["max_run"] < DG.DEFAULT_MAX_RUN


def test_long_run_of_one_token_inside_otherwise_healthy_text():
    ids = _prose_like(3).clone()
    ids[0, 100 : 100 + DG.DEFAULT_MAX_RUN] = 12345
    stats = DG.block_degeneracy(ids)
    assert stats["max_run"] >= DG.DEFAULT_MAX_RUN
    assert DG.is_degenerate(stats)


def test_longest_run_edges():
    assert DG.longest_run(torch.tensor([], dtype=torch.long)) == 0
    assert DG.longest_run(torch.tensor([5], dtype=torch.long)) == 1
    assert DG.longest_run(torch.tensor([1, 1, 2, 3, 3, 3, 4], dtype=torch.long)) == 3
    assert DG.longest_run(torch.tensor([1, 2, 3], dtype=torch.long)) == 1


def test_default_policy_is_warn_and_never_raises(monkeypatch):
    """The default must measure without changing behaviour: stats out, no exception."""
    monkeypatch.delenv("DG_DEGENERACY_POLICY", raising=False)
    assert DG.resolve_policy() == "warn"
    stats = DG.check_committed_block(torch.full((1, CANVAS), 42, dtype=torch.long))
    assert stats["top_frac"] == 1.0


def test_off_policy_measures_nothing(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "off")
    assert DG.check_committed_block(torch.full((1, CANVAS), 1, dtype=torch.long)) == {}


def test_warn_policy_returns_stats_without_raising(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "warn")
    stats = DG.check_committed_block(torch.full((1, CANVAS), 1, dtype=torch.long), block_idx=3)
    assert stats["top_frac"] == 1.0


def test_stop_policy_raises_and_carries_the_evidence(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    tokens = torch.full((1, CANVAS), 42, dtype=torch.long)
    with expect_error(DG.DegenerateBlockError, match="degenerate committed canvas at block 5"):
        DG.check_committed_block(tokens, block_idx=5)


def test_stop_policy_leaves_healthy_blocks_alone(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    stats = DG.check_committed_block(_prose_like(1), block_idx=0)
    assert stats["distinct"] > 1


def test_invalid_policy_fails_loudly(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "halt")
    with expect_error(ValueError, match="DG_DEGENERACY_POLICY"):
        DG.resolve_policy()


def test_thresholds_are_overridable_and_validated(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    monkeypatch.setenv("DG_DEGENERACY_TOP_FRAC", "0.02")
    with expect_error(DG.DegenerateBlockError):
        DG.check_committed_block(_prose_like(0))
    monkeypatch.setenv("DG_DEGENERACY_TOP_FRAC", "1.5")
    with expect_error(ValueError, match="DG_DEGENERACY_TOP_FRAC"):
        DG.check_committed_block(_prose_like(0))


def test_empty_canvas_is_not_degenerate():
    stats = DG.block_degeneracy(torch.zeros((1, 0), dtype=torch.long))
    assert stats["tokens"] == 0
    assert not DG.is_degenerate(stats)


# The distinction calibration forced: on the real device run, a finished answer fills the canvas
# with <eos> (id 1) and scores top_frac 1.0 / max_run 256 -- numerically identical to the
# degenerate walls, opposite meaning. The GPQA degenerate canvases collapse onto CONTENT ids
# instead (id 621 " \\" and id 236770 "1"), which is what keeps the two separable.
EOS_ID = 1
CONTENT_ID = 236770


def test_all_eos_canvas_is_termination_not_degeneration():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), EOS_ID, dtype=torch.long))
    assert DG.is_degenerate(stats), "with no stop set declared it still looks degenerate"
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), "a terminating canvas must not be flagged"


def test_wall_of_a_content_token_is_still_degenerate_with_a_stop_set():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), CONTENT_ID, dtype=torch.long))
    assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID, 0])


def test_stop_policy_lets_a_terminating_block_through(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    stats = DG.check_committed_block(
        torch.full((1, CANVAS), EOS_ID, dtype=torch.long), block_idx=9, stop_token_ids=[EOS_ID]
    )
    assert stats["top_id"] == EOS_ID


def test_stop_policy_still_catches_the_content_wall(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    with expect_error(DG.DegenerateBlockError, match="degenerate committed canvas"):
        DG.check_committed_block(
            torch.full((1, CANVAS), CONTENT_ID, dtype=torch.long), block_idx=7, stop_token_ids=[EOS_ID]
        )


def test_measured_healthy_block_shapes_are_not_flagged():
    """Real per-block statistics from the traced device sweep (host Gumbel, seed 0).

    Healthy blocks of a LaTeX-heavy physics answer: 54-106 distinct ids, top_frac 0.06-0.08,
    max_run 1-2. The degenerate ones from the same run: 1-16 distinct, top_frac 0.94-1.00,
    max_run 240-256. The bounds must sit in that gap.
    """
    healthy = [
        {"top_frac": 0.0703, "max_run": 1, "top_id": 621},
        {"top_frac": 0.0820, "max_run": 1, "top_id": CONTENT_ID},
        {"top_frac": 0.0664, "max_run": 2, "top_id": CONTENT_ID},
        {"top_frac": 0.0625, "max_run": 2, "top_id": 621},
    ]
    degenerate = [
        {"top_frac": 0.9492, "max_run": 243, "top_id": CONTENT_ID},
        {"top_frac": 0.9375, "max_run": 240, "top_id": CONTENT_ID},
    ]
    for stats in healthy:
        assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats
    for stats in degenerate:
        assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats


def test_scalar_stop_token_id_is_accepted():
    """Sessions initialised from a bare `eos_token_id` pass a scalar, not a collection."""
    stats = DG.block_degeneracy(torch.full((1, CANVAS), EOS_ID, dtype=torch.long))
    assert not DG.is_degenerate(stats, stop_token_ids=EOS_ID)
    assert DG.is_degenerate(stats, stop_token_ids=999)
