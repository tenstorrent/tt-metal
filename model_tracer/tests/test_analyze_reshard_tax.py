# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for model_tracer/analyze_reshard_tax.py (read-only reshard-tax analysis, #50943).

Runs the analyzer against small synthetic ops_perf_results.csv fixtures and asserts the
ranked output detects fallback triples + lone reshards, sums device time correctly, and
excludes ops that never fell back. Runnable via pytest or directly (python this_file.py).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_TRACER = os.path.dirname(_HERE)
sys.path.insert(0, _MODEL_TRACER)

import analyze_reshard_tax as art  # noqa: E402

FIXTURES = os.path.join(_HERE, "fixtures")
SLICE_CSV = os.path.join(FIXTURES, "ops_perf_results_slice_fallback.csv")
TRANSPOSE_CSV = os.path.join(FIXTURES, "ops_perf_results_transpose_model.csv")
MASTER = os.path.join(FIXTURES, "ttnn_operations_master_mini.json")


def _analyze_single(csv_path, model_name, master_ops=None):
    return art.analyze([(csv_path, model_name)], master_ops, debug=False)


def test_slice_fallback_triple_detected_and_summed():
    """The slice fallback triple (S2I=800 + slice=2000 + I2S=900) => 3700 ns tax."""
    groups, total_events, _ = _analyze_single(SLICE_CSV, "modelA")

    # 2 fallback triples (slice, pad) + 1 lone reshard = 3 events.
    assert total_events == 3, f"expected 3 events, got {total_events}"

    by_op = {g["op"]: g for g in groups}
    assert "slice" in by_op, f"slice missing from ranking: {list(by_op)}"
    slice_g = by_op["slice"]
    assert slice_g["total_tax_ns"] == 3700.0, slice_g["total_tax_ns"]
    assert slice_g["occurrences"] == 1
    assert slice_g["recognized"] is True
    # The enclosed op runs on interleaved staging tensors; the sharded layout is recovered
    # from the staging ops (S2I input / I2S output) -> HEIGHT_SHARDED here.
    assert "SHARDED" in slice_g["shard_layout"], slice_g["shard_layout"]


def test_pad_and_lone_reshard_present():
    groups, _, _ = _analyze_single(SLICE_CSV, "modelA")
    by_op = {g["op"]: g for g in groups}

    # pad triple: S2I=850 + pad=2500 + I2S=950 = 4300 ns
    assert "pad" in by_op
    assert by_op["pad"]["total_tax_ns"] == 4300.0

    # lone ReshardDeviceOperation = 1200 ns, attributed to its own op code (unrecognized gap op).
    reshard_groups = [g for g in groups if any("reshard" in c.lower() for c in g["op_codes"])]
    assert reshard_groups, "lone reshard not captured"
    assert reshard_groups[0]["total_tax_ns"] == 1200.0


def test_non_fallback_ops_absent():
    """Matmul / Softmax / LayerNorm never staged and must not appear in the ranking."""
    groups, _, _ = _analyze_single(SLICE_CSV, "modelA")
    ops = {g["op"] for g in groups}
    for clean in ("matmul", "softmax", "layernorm"):
        assert clean not in ops, f"{clean} should not be ranked; got {ops}"


def test_ranking_ordered_by_tax_desc():
    groups, _, _ = _analyze_single(SLICE_CSV, "modelA")
    taxes = [g["total_tax_ns"] for g in groups]
    assert taxes == sorted(taxes, reverse=True), taxes
    # pad (4300) outranks slice (3700) outranks reshard (1200).
    assert groups[0]["op"] == "pad"


def test_quasar_dm_trisc_split_summed():
    """The transpose CSV has no combined FW column; DM+TRISC must be added (500+300 etc)."""
    groups, total_events, _ = _analyze_single(TRANSPOSE_CSV, "modelB")
    assert total_events == 2  # transpose triple + slice triple
    by_op = {g["op"]: g for g in groups}
    assert "transpose" in by_op
    # S2I=(500+300) + transpose=(1500+800) + I2S=(600+350) = 800+2300+950 = 4050
    assert by_op["transpose"]["total_tax_ns"] == 4050.0


def test_cross_model_aggregation():
    """slice appears in both CSVs with the same (op, layout, shape) => merged group, 2 models."""
    per_csv = [("modelA", art.detect_fallback_events(art.order_rows(art.load_ops_perf_csv(SLICE_CSV))))]
    per_csv.append(
        ("modelB", art.detect_fallback_events(art.order_rows(art.load_ops_perf_csv(TRANSPOSE_CSV))))
    )
    groups = art.aggregate_events(per_csv, None)
    by_op = {g["op"]: g for g in groups}
    assert "slice" in by_op
    slice_g = by_op["slice"]
    # slice shape signature differs (224x224 input both), both models feed the same group.
    assert set(slice_g["models"].keys()) == {"modelA", "modelB"}, slice_g["models"]
    assert slice_g["occurrences"] == 2


def test_master_enrichment_flags_known_config():
    master = art.load_master_file(MASTER)
    master_ops = master.get("operations", {})
    groups, _, _ = art.analyze([(SLICE_CSV, "modelA")], master_ops, debug=False)
    by_op = {g["op"]: g for g in groups}
    assert by_op["slice"]["known_config"] is True
    # Matmul-style non-gap op is absent, and reshard is not in master => not flagged.


def test_absent_master_degrades_gracefully():
    groups, total_events, _ = art.analyze([(SLICE_CSV, "modelA")], None, debug=False)
    assert total_events == 3
    assert all(g["known_config"] is False for g in groups)


def test_empty_csv_does_not_crash(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("OP CODE,DEVICE FW DURATION [ns]\n")
    groups, total_events, paths = art.analyze([(str(empty), "empty")], None)
    assert total_events == 0
    assert groups == []


def _run_all():
    import tempfile

    passed = 0
    failed = 0
    tests = [
        test_slice_fallback_triple_detected_and_summed,
        test_pad_and_lone_reshard_present,
        test_non_fallback_ops_absent,
        test_ranking_ordered_by_tax_desc,
        test_quasar_dm_trisc_split_summed,
        test_cross_model_aggregation,
        test_master_enrichment_flags_known_config,
        test_absent_master_degrades_gracefully,
    ]
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1

    # tmp_path-dependent test run manually
    try:
        with tempfile.TemporaryDirectory() as d:

            class _P:
                def __init__(self, base):
                    self.base = base

                def __truediv__(self, name):
                    return _F(os.path.join(self.base, name))

            class _F:
                def __init__(self, p):
                    self.p = p

                def write_text(self, s):
                    with open(self.p, "w") as f:
                        f.write(s)

                def __str__(self):
                    return self.p

            test_empty_csv_does_not_crash(_P(d))
            print("✅ test_empty_csv_does_not_crash")
            passed += 1
    except AssertionError as e:
        print(f"❌ test_empty_csv_does_not_crash: {e}")
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
