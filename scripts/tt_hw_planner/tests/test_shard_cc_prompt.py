# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Piece F: the cc bring-up prompt must teach the 'shard' rung (get_shard_plan -> copy tt_transformers
-> shard weights + collective -> run_component mode='shard' -> record_result mode='shard'), without
disturbing the existing rung guidance."""
from pathlib import Path

from scripts.tt_hw_planner._cli_helpers.bringup_cc import _bringup_cc_prompt


def test_prompt_has_shard_rung():
    p = _bringup_cc_prompt("some/model", Path("/tmp/demo"), 0.99)
    assert "shard:" in p
    assert "get_shard_plan" in p
    assert "ShardTensorToMesh" in p
    assert "all_reduce" in p and "all_gather" in p
    assert "mode='shard'" in p
    assert "tt_transformers" in p


def test_prompt_preserves_existing_rungs():
    p = _bringup_cc_prompt("some/model", Path("/tmp/demo"), 0.99)
    for rung in ("emit / repair", "resolve_loader", "fix_harness", "decompose", "fallback", "mark_manual"):
        assert rung in p
    assert "STOP only when can_stop=true" in p


def test_prompt_guards_native_snapshot():
    p = _bringup_cc_prompt("some/model", Path("/tmp/demo"), 0.99)
    assert "NEVER edit the single-device .last_good_native" in p
