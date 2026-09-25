# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS: precision-aware matmul gap.

The gap-first loop ranked ops by `gap_ms` = measured − floor at the op's CURRENT fidelity. A matmul
already near its HiFi4 compute floor therefore read as near-done and never reached its fidelity/dtype
rung — even though the matmul engine runs ~4x faster at LoFi, a LOWER floor and a real reachable win
(the matmul-sweep already finds those, PCC-gated). This added an `achievable_*` floor at the best
(LoFi) precision so the loop can PRIORITIZE that headroom; the config is still PCC-gated when applied.

Invariants pinned here (all model-agnostic):
  p1  a COMPUTE-bound matmul at its current-fidelity floor surfaces via achievable_gap and can OUTRANK a
      non-matmul with a larger CURRENT gap.
  p2  achievable_gap_ms >= gap_ms always (a lower floor cannot shrink the gap).
  p3  a DISPATCH-bound matmul gets NO precision win (the launch floor is precision-invariant) -> no
      false surfacing.
  p4  NON-matmuls are byte-identical to the old behaviour: achievable_gap_ms is None, gap_ms/at_floor
      unchanged, eff_gap_ms == gap_ms.
  p5  the achievable floor uses the HIGHEST peak (LoFi) -> the lowest floor the hardware allows.
  p6  gap_ms (the reported/at-floor number) is left untouched -> the report stays the honest current state.
  p7  the perf_mcp blocking gate reads eff_gap for keep-material + rank (source-level, like siblings).
  p8  no model name / byte / arch value baked into the new logic.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
from agent import roofline as rf  # noqa: E402

_ENV = {
    "peak_tflops_per_core": {"lofi": 4.0, "hifi2": 2.0, "hifi3": 1.33, "hifi4": 1.0},
    "worker_cores": 64,
    "dram_bw_gbps": 288.0,
}


def _mm(m, k, n, fidelity="hifi4", over=1.05):
    """A matmul whose measured time is `over`x its floor at `fidelity`."""
    flops = rf.matmul_flops(m, k, n)
    floor = rf.ideal_ms_compute(flops, fidelity, _ENV)
    return {
        "op_code": "matmul",
        "shape": f"{m}x{k} @ {k}x{n}",
        "count": 1,
        "fidelity": fidelity,
        "device_ms": round(floor * over, 4),
        "bytes": 0,
        "memory": "dram_interleaved",
    }


def _report(*buckets):
    prof = {"device_ms": 1000.0, "buckets": [{"id": bid, "top_ops": ops} for bid, ops in buckets]}
    return rf.residual_report(prof, _ENV)


# --------------------------------------------------------------------------- p1
def test_p1_compute_bound_matmul_surfaces_and_outranks_bigger_current_gap():
    mm = _mm(2048, 8192, 8192)  # big compute-bound matmul, ~5% above HiFi4 floor
    tiny = {"op_code": "Reshape", "count": 1, "device_ms": 0.05, "bytes": 1000, "memory": "dram_interleaved"}
    dm = {"op_code": "AllGather", "count": 1, "device_ms": 3.0, "bytes": int(288e6), "memory": "dram_interleaved"}
    rep = _report(("matmul", [mm]), ("eltwise", [tiny]), ("ccl", [dm]))
    rows = {r["op_code"]: r for r in rep["rows"]}
    # the matmul's CURRENT gap is tiny (near its HiFi4 floor) but its achievable gap is large
    assert rows["matmul"]["gap_ms"] < rows["AllGather"]["gap_ms"], "precondition: current matmul gap is the smaller"
    assert rows["matmul"]["eff_gap_ms"] > rows["AllGather"]["eff_gap_ms"], "precision headroom must lift it above"
    assert rep["open_ops"][0]["op_code"] == "matmul", "matmul must now lead the work order"


# --------------------------------------------------------------------------- p2 / p5 / p6
def test_p2_achievable_never_below_current_and_p6_gap_untouched():
    for shp in [(64, 4096, 1024), (512, 512, 512), (2048, 8192, 8192), (32, 1024, 1024)]:
        mm = _mm(*shp)
        gap_before = None
        # measure the pure annotate (dispatch=None) so the compute floor is the binding one
        op = dict(mm)
        rf.annotate_op(op, _ENV, None)
        assert op["achievable_gap_ms"] >= op["gap_ms"] - 1e-9, f"{shp}: achievable < current"
        # gap_ms equals the current-fidelity floor gap, i.e. unchanged from pre-feature math
        flops = rf.matmul_flops(*shp)
        cur_floor = rf.ideal_ms_compute(flops, "hifi4", _ENV)
        assert op["gap_ms"] == round(max(0.0, op["device_ms"] - cur_floor), 4), f"{shp}: gap_ms drifted"
        # achievable floor uses LoFi (p5): the lowest floor
        assert op["achievable_ideal_ms"] == round(rf.ideal_ms_compute(flops, "lofi", _ENV), 4), f"{shp}: not LoFi"


# --------------------------------------------------------------------------- p3
def test_p3_dispatch_bound_matmul_gets_no_precision_win():
    # a matmul so small its launch floor dominates -> lowering fidelity buys nothing
    mm = _mm(32, 64, 64)
    tiny = {"op_code": "Reshape", "count": 1, "device_ms": mm["device_ms"], "bytes": 10, "memory": "dram_interleaved"}
    rep = _report(("matmul", [mm]), ("eltwise", [tiny]))
    row = next(r for r in rep["rows"] if r["op_code"] == "matmul")
    # dispatch floor (precision-invariant) == the small measured time -> achievable == current (both ~0)
    assert abs((row["achievable_gap_ms"] or 0.0) - (row["gap_ms"] or 0.0)) < 1e-6, "dispatch-bound got a phantom win"


# --------------------------------------------------------------------------- p4
def test_p4_non_matmul_is_byte_identical_to_old_behaviour():
    for oc in ("AllGather", "ReduceScatter", "LayerNorm", "Reshape", "Embedding"):
        op = {"op_code": oc, "count": 1, "device_ms": 10.0, "bytes": int(288e6), "memory": "dram_interleaved"}
        rf.annotate_op(op, _ENV, 0.05)
        assert op["achievable_gap_ms"] is None, f"{oc}: non-matmul must have no achievable gap"
        assert op["achievable_ideal_ms"] is None, f"{oc}: non-matmul must have no achievable ideal"
    # and in a report, eff_gap_ms falls back to gap_ms exactly
    dm = {"op_code": "AllGather", "count": 1, "device_ms": 5.0, "bytes": int(288e6), "memory": "dram_interleaved"}
    rep = _report(("ccl", [dm]))
    row = rep["rows"][0]
    assert row["achievable_gap_ms"] is None and row["eff_gap_ms"] == row["gap_ms"]


# --------------------------------------------------------------------------- p3b: no-op profile never crashes
def test_p3b_unmodeled_and_empty_profiles_do_not_crash():
    # op with no shape/bytes -> unmodeled -> achievable None, no exception
    op = {"op_code": "matmul", "shape": "?x? @ ?x?", "count": 1, "device_ms": 1.0}
    rf.annotate_op(op, _ENV, None)
    assert op["achievable_gap_ms"] is None
    assert rf.residual_report({"device_ms": 0.0, "buckets": []}, _ENV)["open_ops"] == []


# --------------------------------------------------------------------------- p7 (source-level wiring)
def test_p7_blocking_gate_uses_eff_gap():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("blocking, cleared = [], []")
    j = src.index("can_stop = not blocking", i)
    body = src[i:j]
    assert 'o.get("eff_gap_ms")' in body, "blocking builder must read the precision-aware eff_gap"
    assert "max(gap, eff_gap) < material" in body, "keep the op if EITHER gap is material"
    assert 'b.get("eff_gap_ms")' in body, "blocking must be ranked by eff_gap (fallback gap_ms)"


# --------------------------------------------------------------------------- p8 (no hardcoding)
def test_p8_no_model_or_arch_value_hardcoded_in_new_logic():
    src = (_PA / "agent" / "roofline.py").read_text()
    i = src.index("def annotate_op(")
    j = src.index("\ndef ", i + 1)
    body = "\n".join(ln for ln in src[i:j].splitlines() if not ln.lstrip().startswith("#"))
    # the ONLY literal the new logic may name is the fidelity tier "lofi" (a hardware peak key, not a
    # model/arch constant). No byte counts, no core counts, no model names.
    for junk in ("xtts", "133582652", "2086199391", "288", "1.868", "4096"):
        assert junk not in body.lower(), f"model/arch value hardcoded in annotate_op logic: {junk}"
    assert '"lofi"' in body, "the achievable floor must key off the LoFi hardware peak"
