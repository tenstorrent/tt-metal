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


def test_default_policy_is_stop(monkeypatch, expect_error):
    """The default must actually prevent the emission, not just log it."""
    monkeypatch.delenv("DG_DEGENERACY_POLICY", raising=False)
    assert DG.resolve_policy() == "stop"
    with expect_error(DG.DegenerateBlockError):
        DG.check_committed_block(torch.full((1, CANVAS), 42, dtype=torch.long))


def test_warn_policy_still_available(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "warn")
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
        # the two most extreme HEALTHY canvases across all 136 measured: the bounds must clear both
        {"top_frac": 0.1836, "max_run": 18, "top_id": 236772},
        {"top_frac": 0.1719, "max_run": 1, "top_id": 236772},
    ]
    degenerate = [
        {"top_frac": 0.9492, "max_run": 243, "top_id": CONTENT_ID},
        {"top_frac": 0.9375, "max_run": 240, "top_id": CONTENT_ID},
        # GPQA doc7 under the host Gumbel default -- the case that proves the root-cause fix
        # alone is not sufficient and the guard has to be the default.
        {"top_frac": 0.8516, "max_run": 86, "top_id": CONTENT_ID},
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


# ── the terminal-padding shape (2026-07-28) ──────────────────────────────────────────────────
# What the tests above missed: the canvases the served run actually rejected were not walls of
# <eos>, they were ANSWERS followed by <eos> padding. A block that finishes at position 149 pads
# the remaining 107 -- top_frac 0.58 and max_run 107 over the whole canvas, both past the gate,
# while the content region is ordinary prose. On tt-shield run 30285823000 that ended 110 of 198
# requests and threw away the block holding the answer. Shapes below are taken from that run's
# `ending request at block N` lines.


def _answer_then_padding(content_len: int, *, seed: int = 0, pad_id: int = EOS_ID) -> torch.Tensor:
    """A finished answer: healthy prose in the first ``content_len`` positions, then padding."""
    ids = _prose_like(seed, length=content_len).flatten()
    pad = torch.full((CANVAS - content_len,), pad_id, dtype=torch.long)
    return torch.cat([ids, pad]).reshape(1, CANVAS)


def test_answer_followed_by_eos_padding_is_not_degenerate():
    tokens = _answer_then_padding(149)
    whole = DG.block_degeneracy(tokens)
    assert whole["top_frac"] >= DG.DEFAULT_TOP_FRAC or whole["max_run"] >= DG.DEFAULT_MAX_RUN
    assert DG.is_degenerate(whole), "the whole-canvas view is what mis-rejected it"

    stats = DG.block_degeneracy(tokens, stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == CANVAS - 149
    assert stats["content_tokens"] == 149
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats


def test_the_five_shapes_the_served_run_rejected(monkeypatch):
    """Real (distinct, top_frac, max_run) tuples the 07-27 server log printed as degenerate."""
    monkeypatch.delenv("DG_DEGENERACY_TOP_FRAC", raising=False)
    for content_len, recorded_top_frac in ((18, 0.930), (98, 0.617), (68, 0.734), (73, 0.715), (107, 0.582)):
        tokens = _answer_then_padding(content_len, seed=content_len)
        whole = DG.block_degeneracy(tokens)
        assert abs(whole["top_frac"] - recorded_top_frac) < 0.02, (content_len, whole["top_frac"])
        assert DG.is_degenerate(whole), "reproduces the pre-fix rejection"
        stats = DG.block_degeneracy(tokens, stop_token_ids=[EOS_ID])
        assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), (content_len, stats)


def test_padding_of_any_declared_stop_id_is_stripped():
    """<end_of_turn> tails must count as padding too, not just <eos>."""
    tokens = _answer_then_padding(120, pad_id=106)
    stats = DG.block_degeneracy(tokens, stop_token_ids=[EOS_ID, 106, 50])
    assert stats["stop_tail"] == CANVAS - 120
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID, 106, 50])


def test_mixed_stop_ids_in_the_tail_are_one_run():
    ids = _prose_like(5, length=200).flatten()
    tail = torch.tensor([EOS_ID] * 30 + [106] * 26, dtype=torch.long)
    stats = DG.block_degeneracy(torch.cat([ids, tail]).reshape(1, CANVAS), stop_token_ids=[EOS_ID, 106])
    assert stats["stop_tail"] == 56, "the tail is padding regardless of which stop id fills it"
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID, 106])


def test_content_collapse_with_an_eos_tail_is_still_degenerate():
    """The 6 real trips on content ids: a short prefix, then a wall of content, then padding."""
    collapsed = torch.cat(
        [
            _prose_like(9, length=39).flatten(),
            torch.full((161,), CONTENT_ID, dtype=torch.long),
            torch.full((56,), EOS_ID, dtype=torch.long),
        ]
    ).reshape(1, CANVAS)
    stats = DG.block_degeneracy(collapsed, stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == 56
    assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats


def test_full_content_wall_with_a_stop_set_is_still_degenerate():
    """`尼` x 256 -- the 07-24 shape. Stripping must not empty the content region here."""
    stats = DG.block_degeneracy(torch.full((1, CANVAS), 239054, dtype=torch.long), stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == 0
    assert stats["content_tokens"] == CANVAS
    assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID])


def test_all_stop_token_canvas_has_no_content_region():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), EOS_ID, dtype=torch.long), stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == CANVAS
    assert stats["content_tokens"] == 0
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID])


def test_terminal_stop_run_only_counts_the_tail():
    ids = torch.tensor([5, EOS_ID, EOS_ID, 7, EOS_ID, EOS_ID, EOS_ID], dtype=torch.long)
    assert DG.terminal_stop_run(ids, {EOS_ID}) == 3, "a mid-canvas stop token is not padding"
    assert DG.terminal_stop_run(torch.tensor([5, 7], dtype=torch.long), {EOS_ID}) == 0
    assert DG.terminal_stop_run(torch.tensor([], dtype=torch.long), {EOS_ID}) == 0
    assert DG.terminal_stop_run(ids, set()) == 0, "no declared stop set means nothing is padding"


def test_whole_canvas_rule_is_unchanged_without_stop_ids():
    """Callers that declare no stop set keep the old behaviour rather than a weakened gate."""
    tokens = _answer_then_padding(149)
    stats = DG.block_degeneracy(tokens)
    assert "content_tokens" not in stats
    assert DG.is_degenerate(stats)


def test_stop_policy_commits_a_finished_answer(monkeypatch):
    """End to end through the policy: the shape that lost 110 answers must now pass."""
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    stats = DG.check_committed_block(_answer_then_padding(149), block_idx=6, stop_token_ids=[EOS_ID])
    assert stats["content_tokens"] == 149


def test_describe_reports_the_content_region():
    stats = DG.block_degeneracy(
        torch.cat(
            [
                torch.full((200,), CONTENT_ID, dtype=torch.long),
                torch.full((56,), EOS_ID, dtype=torch.long),
            ]
        ).reshape(1, CANVAS),
        stop_token_ids=[EOS_ID],
    )
    message = DG.describe(stats, block_idx=3)
    assert "content 200/256" in message and "stop tail 56" in message
