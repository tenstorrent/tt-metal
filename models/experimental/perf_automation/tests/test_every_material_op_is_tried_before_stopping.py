"""The optimizer must see every op with a material gap, and must not stop while one is untried.

Two hard caps decided what the gate could work on:

    agent/roofline.py:280       "open_ops": open_ops[:10]
    cc_optimize/perf_mcp.py     for o in (rep.get("open_ops") or [])[:8]

`open_ops` is documented as "the ttnn-reachable work still on the table, biggest gap first" -- a
summary for the report. It became the gate's work queue, and the display limit came with it. It is a
SLICE, not a priority scheme: the tail is discarded on every call, so a low-ranked op can never
resurface no matter how many rounds remain or how many higher-ranked ops get cleared.

gemma-3-12b-it run 21: the roofline found 28 open ops, the gate received 10, and can_stop fired once
those ten had full checklists. Below the cut, never attempted:

    PagedUpdateCacheDeviceOperation           gap 4.18 ms   on 1 core
    SdpaDecodeDeviceOperation                 gap 4.08 ms
    NLPCreateQKVHeadsDecodeDeviceOperation    gap 3.27 ms   on 1 core
    ...15 more

A one-core op is the clearest grid candidate there is, and the run declared itself done without
looking at it.

So: no cap on what the gate can see, and stopping requires that every material-gap op has been
ATTEMPTED at least once. Running out of rounds still ends the run -- that bound lives in the loop
(`while rounds < max_rounds`), not here -- but the gate may no longer declare completion while known
work is untouched.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.agent import roofline as R  # noqa: E402


def _profile(n_ops: int):
    """A profile with n_ops distinct matmuls, descending gaps, none at its floor."""
    tops = []
    for i in range(n_ops):
        m = 32 * (n_ops - i)
        tops.append(
            {
                "op_code": "MatmulDeviceOperation %d x 3840 x 15360" % m,
                "shape": "%dx3840 @ 3840x15360" % m,
                "device_ms": 50.0 - i,
                "count": 1,
                "bytes": 1e9,
                "cores": 8,
                "fidelity": "lofi",
                "grid": "partial",
                "memory": "dram_interleaved",
            }
        )
    return {"device_ms": sum(t["device_ms"] for t in tops), "buckets": [{"id": "matmul", "top_ops": tops}]}


HW = {"dram_bw_gbps": 512.0, "worker_cores": 110, "mesh_chips": 1, "peak_tflops_per_core": {"lofi": 4.0, "hifi4": 1.0}}


# ---------------------------------------------------------------- the cap


def test_open_ops_is_not_truncated(mcp_free=None):
    """28 open ops must be reported as 28, not 10."""
    rep = R.residual_report(_profile(28), HW)
    assert rep["n_open"] == len(rep["open_ops"]), (rep["n_open"], len(rep["open_ops"]))


def test_a_small_profile_is_unaffected(mcp_free=None):
    """Whatever the count, the list and the count agree. (A 4-op profile reports few or no open ops:
    dispatch_floor_per_op self-calibrates to the smallest op, so with a handful they all sit at their
    floor. That is the existing heuristic, not the cap -- what matters here is that nothing is cut.)"""
    rep = R.residual_report(_profile(4), HW)
    assert len(rep["open_ops"]) == rep["n_open"]


def test_open_ops_stays_sorted_biggest_gap_first(mcp_free=None):
    """Removing the cap must not remove the ordering -- the gate still works the largest gap first."""
    ops = R.residual_report(_profile(20), HW)["open_ops"]
    gaps = [o.get("eff_gap_ms") or o.get("gap_ms") or 0 for o in ops]
    assert gaps == sorted(gaps, reverse=True), gaps


def test_the_ops_below_the_old_cut_are_present(mcp_free=None):
    """The 11th-biggest gap and beyond were unreachable. They must now appear."""
    ops = R.residual_report(_profile(28), HW)["open_ops"]
    assert len(ops) > 10
    assert ops[10].get("op_code")


# ---------------------------------------------------------------- stopping


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def test_an_untried_material_op_blocks_stopping(mcp):
    """The rule: known work, never attempted -> not done."""
    blocking = [{"op": "PagedUpdateCacheDeviceOperation", "gap_ms": 4.18}]
    assert mcp._untried_material_ops(blocking, attempts=[]) == ["PagedUpdateCacheDeviceOperation"]


def test_an_op_with_one_attempt_no_longer_blocks(mcp):
    """ATTEMPTED, not won. A measured dead end clears the op -- that is the existing contract."""
    blocking = [{"op": "PagedUpdateCacheDeviceOperation", "gap_ms": 4.18}]
    attempts = [{"op_signature": "PagedUpdateCacheDeviceOperation", "kernel_kind": "grid", "measured_ms": 4.7}]
    assert mcp._untried_material_ops(blocking, attempts) == []


def test_a_wedged_attempt_counts_as_tried(mcp):
    """A candidate that crashed the device was attempted; demanding a clean one would loop forever."""
    blocking = [{"op": "X", "gap_ms": 4.0}]
    attempts = [{"op_signature": "X", "kernel_kind": "grid", "wedged": True}]
    assert mcp._untried_material_ops(blocking, attempts) == []


def test_shapes_are_matched_not_just_classes(mcp):
    """An attempt on one matmul must not clear a different-shape matmul -- the existing _op_match rule."""
    blocking = [{"op": "MatmulDeviceOperation 32 x 3840 x 15360", "gap_ms": 13.8}]
    attempts = [{"op_signature": "MatmulDeviceOperation 512 x 3840 x 15360", "kernel_kind": "grid"}]
    assert mcp._untried_material_ops(blocking, attempts) == ["MatmulDeviceOperation 32 x 3840 x 15360"]


def test_nothing_blocking_means_nothing_untried(mcp):
    assert mcp._untried_material_ops([], attempts=[]) == []
