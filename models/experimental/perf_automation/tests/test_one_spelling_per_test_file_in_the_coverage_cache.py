# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The coverage cache must find a test file however the caller spells its node.

2026-09-23, nvidia_nemotron_3_5_lightning: before_loop measured 29,436 op invocations per decode
step and recorded that under the operator's ABSOLUTE --perf-test path. The orchestrator's coverage
step keyed the same test as the tt-root-relative node WITH its ::case; the profiling server keyed it
as the tt-root-relative file alone. Three slots, three fingerprints (the absolute one hashed the
operator's checkout, the others this worktree). The server's lookup missed, _capacity_scaled_osl saw
no signal, and the loop's first profile ran the declared OSL=128 under tracy: 233 GB, killed by the
run's own memory cap, while the baseline it was to be compared with had run at the 2-step cap.

Every key and fingerprint now goes through one canonical spelling: case dropped, an absolute path
re-rooted into this tree by the same helper --pcc-test uses, expressed relative to the repo root.
No variable name, stage name or model name is written here.
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
_PERF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PERF))

_REL = "models/somewhere/a_demo/tests/e2e/test_it.py"
_CASE = "test_it"


@pytest.fixture()
def run():
    spec = importlib.util.spec_from_file_location("_run_under_test_spelling", str(_PERF / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_run_under_test_spelling"] = m
    spec.loader.exec_module(m)
    return m


def _two_checkouts(tmp_path):
    """This run's worktree and the operator's own checkout, both holding the same test file."""
    worktree = tmp_path / "worktree"
    operator = tmp_path / "operator"
    for root in (worktree, operator):
        f = root / _REL
        f.parent.mkdir(parents=True)
        f.write_text("def test_it():\n    pass\n")
        (root / "models" / "experimental" / "perf_automation" / "cc_optimize").mkdir(parents=True)
    return worktree, operator


def test_all_three_spellings_share_one_key_and_one_fingerprint(run, tmp_path):
    worktree, operator = _two_checkouts(tmp_path)
    spellings = (str(operator / _REL), f"{_REL}::{_CASE}", _REL, f"{operator / _REL}::{_CASE}")
    keys = {run._cache_node(s, worktree) for s in spellings}
    assert keys == {_REL}, keys
    fps = {run._coverage_fingerprint(s, worktree) for s in spellings}
    assert len(fps) == 1 and "" not in fps, fps


def test_the_ops_per_step_signal_written_by_one_caller_is_read_by_the_others(run, tmp_path):
    """The incident, end to end: written under the absolute spelling, read under the relative ones."""
    worktree, operator = _two_checkouts(tmp_path)
    run._coverage_cache_put(worktree, str(operator / _REL), _CASE, 7, ops_per_step=29436)
    assert run.coverage_cache_get_ops_per_step(worktree, f"{_REL}::{_CASE}", _CASE) == 29436
    assert run.coverage_cache_get_ops_per_step(worktree, _REL, _CASE) == 29436
    assert run._coverage_cache_get(worktree, _REL, _CASE) == 7
    cache = json.loads(run._coverage_cache_path(worktree).read_text())
    assert list(cache) == [f"{_REL}|{_CASE}"], "exactly one slot for one test file"


def test_the_capacity_cap_now_reaches_the_profiling_servers_spelling(run, tmp_path):
    """What the server actually calls, with the numbers from the incident: 29,436 ops/step at a
    declared OSL of 128 must come back capped, not None."""
    from models.experimental.perf_automation.agent.measure import _capacity_scaled_osl

    worktree, operator = _two_checkouts(tmp_path)
    run._coverage_cache_put(worktree, str(operator / _REL), _CASE, 6, ops_per_step=29436)
    scaled = _capacity_scaled_osl(None, worktree, _REL, _CASE, 128)
    assert scaled is not None and int(scaled[0]) < 128, scaled


def test_a_depth_cap_recorded_under_one_spelling_is_found_under_another(run, tmp_path):
    worktree, operator = _two_checkouts(tmp_path)
    cap = {"SOME_DEPTH_VAR": "2"}
    run._depth_cache_put(worktree, str(operator / _REL), cap)
    assert run._depth_cache_get(worktree, f"{_REL}::{_CASE}") == cap
    assert run._depth_cache_get(worktree, _REL) == cap


def test_a_file_with_no_counterpart_in_the_tree_keeps_one_case_free_spelling(run, tmp_path):
    worktree, _ = _two_checkouts(tmp_path)
    elsewhere = tmp_path / "elsewhere" / "models" / "other" / "tests" / "test_other.py"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("def test_other():\n    pass\n")
    assert run._cache_node(f"{elsewhere}::test_other", worktree) == run._cache_node(str(elsewhere), worktree)
    assert run._cache_node(str(elsewhere), worktree) == elsewhere.as_posix()


def test_a_stale_fingerprint_still_invalidates_across_spellings(run, tmp_path):
    """Canonical keys must not weaken invalidation: editing the file beside the test invalidates
    the entry for every spelling, as before."""
    worktree, operator = _two_checkouts(tmp_path)
    run._coverage_cache_put(worktree, str(operator / _REL), _CASE, 7, ops_per_step=100)
    later = time.time() + 100
    os.utime(worktree / _REL, (later, later))
    assert run.coverage_cache_get_ops_per_step(worktree, _REL, _CASE) is None
    assert run.coverage_cache_get_ops_per_step(worktree, str(operator / _REL), _CASE) is None


def test_no_raw_node_is_used_as_a_cache_key(run):
    src = (_PERF / "cc_optimize" / "run.py").read_text()
    assert 'f"{node}|' not in src and 'f"depth|{node}"' not in src
