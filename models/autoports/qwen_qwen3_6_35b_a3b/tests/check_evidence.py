# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the functional stage's recorded hard gates."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOC = ROOT / "doc/functional_decoder"


def read(name):
    return json.loads((DOC / name).read_text())


def main():
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())
    assert contract["hf_advertised_context"] == contract["supported_context"] == 262144
    checks = []
    for kind, idx in (("linear", 0), ("full", 3)):
        long = read(f"{kind}_long.json")
        prefills = [r for r in long["results"] if r["mode"] == "prefill"]
        assert {r["length"] for r in prefills} == {262143, 262144}
        assert all(r["pcc"] >= 0.995 and min(r["selected_row_pcc"].values()) >= 0.995 for r in prefills)
        decode = [r for r in long["results"] if r["mode"] == "traced_decode"][0]
        assert decode["position"] == 262143 and decode["pcc"] >= 0.995 and decode["deterministic"]
        if idx == 3:
            tails = long["all_trailing_64_rows_pcc"]
            assert set(tails) == {"262143", "262144"}
            assert all(len(rows) == 64 and min(rows.values()) >= 0.995 for rows in tails.values())
        boundary = read(f"partition_{idx}.json")
        lengths = {r["length"] for r in boundary["results"] if r["mode"] == "prefill"}
        assert {1, 31, 32, 33, 255, 256, 257, 513, 7} <= lengths
        assert all(r["pcc"] >= 0.995 for r in boundary["results"])
        assert any(r["mode"] == "chunk_partition_parity" for r in boundary["results"])
        resume_file = "linear_battery.json" if idx == 0 else "full_resume_fixed.json"
        resume = [r for r in read(resume_file)["results"] if r["mode"] == "resume"]
        assert {r["start"] for r in resume} == {32, 33, 256, 512, 528}
        assert all(min(r["per_row_pcc"]) >= 0.995 and min(r["dropped_history_per_row_pcc"]) < 0.995 for r in resume)
        trace = read(f"trace_reuse_{idx}.json")
        assert trace["tracking"] and trace["program_cache_included"] and trace["positive_control"] == "rejected"
        assert trace["captures"] == 1 and trace["request_releases"] == 0
        assert trace["changed_input_changed_output"] and trace["repeated_request_bit_equal"]
        assert len({r["trace_id"] for r in trace["results"]}) == 1
        assert all(r["pcc"] >= 0.995 for r in trace["results"])
        batch = read("linear_mixed_batch.json" if idx == 0 else "full_shared_pool.json")
        rows = batch["results"]
        assert len([r for r in rows if r["mode"] == "batch_prefill"]) == 32
        assert len([r for r in rows if r["mode"] == "batch_traced_decode"]) == 96
        assert all(r.get("pcc", 1) >= 0.995 for r in rows)
        if idx == 3:
            assert len([r for r in rows if r["mode"] == "shared_pool_cache"]) == 64
            assert any(r["mode"] == "unowned_guard_pages" and r["untouched"] for r in rows)
        for phase in ("prefill", "decode"):
            report = DOC / f"tracy/layer_{idx}/{phase}_perf_report.txt"
            assert report.stat().st_size > 300
            assert (DOC / f"tracy/layer_{idx}/{phase}_perf_report.csv").exists() or (
                DOC / f"tracy/layer_{idx}/{phase}_perf_report.csv.gz"
            ).exists()
        checks.append(
            dict(
                kind=kind,
                full_context=True,
                pcc=True,
                boundaries=True,
                resume_controls=True,
                trace_reuse=True,
                batch_32=True,
            )
        )
    xml = ET.parse(DOC / "synthetic_pytest.xml").getroot()
    suites = list(xml.iter("testsuite"))
    assert sum(int(s.get("tests", 0)) for s in suites) == 2
    assert all(int(s.get("errors", 0)) == int(s.get("failures", 0)) == int(s.get("skipped", 0)) == 0 for s in suites)
    assert read("fallback_audit.json")["python_audit_pass"]
    assert read("watcher_audit.json")["clean"]
    capacities = read("page_extent_regression.json")["results"]
    assert {r["capacity"] for r in capacities} == {1, 31, 32, 33, 255, 256, 257, 511, 512, 513}
    assert all(min(r["prefill_pcc"], r["decode_pcc"]) >= 0.995 for r in capacities)
    assert all(r["unowned_pages_untouched"] and r["future_rows_untouched"] for r in capacities)
    padding = read("padded_trace_regression.json")
    assert padding["captures"] == 1 and padding["table_entries"] == 8 and padding["logical_capacity"] == 33
    assert padding["sentinel_masked_bit_equal"] and padding["repeated_request_bit_equal"]
    assert padding["changed_request_changed_output"] and padding["tracking"]
    assert len(padding["results"]) == 6
    assert all(min(r["prefill_pcc"], r["decode_pcc"]) >= 0.995 and r["guards_untouched"] for r in padding["results"])
    perf = read("perf_summary.json")
    assert read("profiler_completeness.json")["valid"]
    assert len(perf["measurements"]) == 4
    assert all(r["latency_us"] > 0 for r in perf["measurements"])
    result = dict(status="PASS", checks=checks, synthetic_tests=2, watcher=True, runtime_audit=True, performance=True)
    (DOC / "hard_gates.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FUNCTIONAL_DECODER_HARD_GATES_PASS")


if __name__ == "__main__":
    main()
