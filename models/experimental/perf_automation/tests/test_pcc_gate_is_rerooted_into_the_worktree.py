# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An absolute --pcc-test into the operator's main checkout must be re-rooted into the run's worktree.

optimize edits and measures an isolated worktree. A gate left pointing at the operator's main
checkout would pytest the unedited copy on every lever -- a correctness gate that cannot fail. On
2026-09-23 the lead agent refused the run for exactly this, and the supervisor replayed the same
flags for three ~17-minute attempts. --perf-test already got re-rooted; the PCC gate now does too,
matched on the longest existing path tail so a sibling demo sharing the filename cannot hijack it.
"""

import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_GATE = "tests/e2e/test_e2e_pipeline.py"
_TEST_SRC = "PCC_THRESHOLD = 0.95\n\n\ndef test_gate3_e2e_pcc():\n    pass\n"


def _checkout(root: Path, demo: str) -> Path:
    model_root = root / "models" / "demos" / demo
    (model_root / "tests" / "e2e").mkdir(parents=True)
    (model_root / _GATE).write_text(_TEST_SRC)
    return model_root


def test_absolute_gate_from_another_checkout_lands_in_this_tree(tmp_path):
    from agent import model_files as mf

    main = tmp_path / "main"
    worktree = tmp_path / "worktree"
    demo = "nvidia_nemotron_3_5_lightning_30b_a3b_bf16"
    _checkout(main, demo)
    wt_model = _checkout(worktree, demo)
    # A sibling demo with the SAME filename exists in both trees (the real repo has nemotron_3_nano).
    _checkout(main, "nvidia_nemotron_3_nano_30b_a3b_bf16")
    _checkout(worktree, "nvidia_nemotron_3_nano_30b_a3b_bf16")

    node = f"{main / 'models' / 'demos' / demo / _GATE}::test_gate3_e2e_pcc"
    node_rel, thr, pcc_abs = mf.resolve_pcc_node(wt_model, node, worktree)
    assert pcc_abs == (wt_model / _GATE).resolve(), "gate must point at THIS run's tree"
    assert node_rel == f"{_GATE}::test_gate3_e2e_pcc", "no '..' escape back to the main checkout"
    assert thr == pytest.approx(0.95)


def test_relative_and_in_tree_absolute_gates_are_unchanged(tmp_path):
    from agent import model_files as mf

    worktree = tmp_path / "worktree"
    demo = "nvidia_nemotron_3_5_lightning_30b_a3b_bf16"
    wt_model = _checkout(worktree, demo)
    rel = f"models/demos/{demo}/{_GATE}::test_gate3_e2e_pcc"
    node_rel, _, pcc_abs = mf.resolve_pcc_node(wt_model, rel, worktree)
    assert pcc_abs == (wt_model / _GATE).resolve() and node_rel == f"{_GATE}::test_gate3_e2e_pcc"
    absolute = f"{wt_model / _GATE}::test_gate3_e2e_pcc"
    node_rel2, _, pcc_abs2 = mf.resolve_pcc_node(wt_model, absolute, worktree)
    assert (node_rel2, pcc_abs2) == (node_rel, pcc_abs)


def test_a_gate_with_no_counterpart_in_the_tree_keeps_the_old_behaviour(tmp_path):
    """Re-rooting is opportunistic: when nothing in the tree matches any tail of the path, the
    resolver behaves exactly as before (the existing out-of-tree file, reached via '..'), and a
    path that exists nowhere is still the same error."""
    from agent import model_files as mf

    worktree = tmp_path / "worktree"
    demo = "nvidia_nemotron_3_5_lightning_30b_a3b_bf16"
    wt_model = _checkout(worktree, demo)
    elsewhere = tmp_path / "elsewhere" / "models" / "demos" / "other_model" / _GATE
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(_TEST_SRC)
    node_rel, _, pcc_abs = mf.resolve_pcc_node(wt_model, f"{elsewhere}::test_gate3_e2e_pcc", worktree)
    assert pcc_abs == elsewhere.resolve() and node_rel.startswith("..")
    with pytest.raises(mf.ModelFilesError):  # allow-pytest.raises: no expect_error fixture
        mf.resolve_pcc_node(wt_model, f"{tmp_path / 'missing.py'}::test_x", worktree)
