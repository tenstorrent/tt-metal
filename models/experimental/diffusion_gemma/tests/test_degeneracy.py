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


def test_default_policy_is_off_and_measures_nothing(monkeypatch):
    monkeypatch.delenv("DG_DEGENERACY_POLICY", raising=False)
    assert DG.resolve_policy() == "off"
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
