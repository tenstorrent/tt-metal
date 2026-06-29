# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Baseline shape-crash fallback: the perf-test generator caps the sequence length small (for
profiler speed), but a model whose matmul program configs are pinned to its native shape crashes
that capped baseline (e.g. sentence_bert: block_h=12 needs Mt=96, but seq=128 gives Mt=32). The
baseline capture must detect that shape/program-config crash and retry at the model's native seq."""

from agent.before_loop import _SHAPE_CONFIG_CRASH_RE, _seq_retry_candidates

SBERT_CRASH = "TT_FATAL: block_h (12) must equal ceil(Mt / num_cores_r) (4); Mt=32, num_cores_r=8 (assert.hpp:104)"


def test_shape_config_crash_is_detected():
    assert _SHAPE_CONFIG_CRASH_RE.search(SBERT_CRASH)


def test_generic_crash_is_not_treated_as_shape_crash():
    assert not _SHAPE_CONFIG_CRASH_RE.search("RuntimeError: device hung waiting for fabric")


def test_native_seq_derived_from_crash_is_tried_first():
    # block_h(12) * num_cores_r(8) = 96 wanted tiles; cur Mt=32 at seq=128 -> native = 128*96/32 = 384.
    cands = _seq_retry_candidates(SBERT_CRASH, 128)
    assert cands[0] == 384


def test_unparseable_shape_crash_falls_back_to_ladder_including_native():
    cands = _seq_retry_candidates("block_h program_config mismatch", 128)
    assert 384 in cands and all(s > 128 for s in cands)


def test_no_candidates_below_current_seq():
    # already at a large seq -> never retry smaller (the cap is the problem, not the cure).
    assert all(s > 512 for s in _seq_retry_candidates(SBERT_CRASH, 512))
