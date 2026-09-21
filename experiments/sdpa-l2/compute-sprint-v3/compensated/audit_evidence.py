# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standard-library evidence/source audit; never opens a device."""
import argparse
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[3]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("files", nargs="*", type=Path)
args = p.parse_args()
files = args.files or sorted([*here.glob("[eg]-replay-*.json"), *here.glob("[eg]-valid-*.json")])
total = 0
failures = []
max_row_delta = (0.0, "")
for path in files:
    data = json.loads(path.read_text())
    for record in data if isinstance(data, list) else [data]:
        total += 1
        label = f"{path.name}:{record.get('case', 'single')}"
        for name in ("trace_equal", "source_stable", "original_host_immutable",
                     "original_device_immutable", "prepared_device_immutable"):
            assert record[name] is True, (label, name)
        assert record["mandatory_trace_replays_per_kernel"] == 2
        assert record["preprocessing_mismatches"] == [0, 0, 0]
        if record["arguments"]["mode"] == "distinct":
            assert record["canonical_adapter_equal"] is True
        a, b = record["baseline_metadata"], record["candidate_metadata"]
        for name in ("fidelity", "fp32_dst", "input_slots", "cb_specs", "cb_bytes"):
            assert a[name] == b[name], (label, name)
        assert a["fidelity"] == "MathFidelity.LoFi" and a["fp32_dst"] is False
        assert a["input_slots"] == 2
        strip = lambda d: {k: v for k, v in d.items() if k != "SDPA_SPRINT_CANDIDATE_HEADER"}
        assert strip(a["defines"]) == strip(b["defines"])
        for source, sha in record["source_sha256"].items():
            assert hashlib.sha256((root / source).read_bytes()).hexdigest() == sha, (label, source)
        if "suite_source_sha256" in record:
            assert hashlib.sha256((here / "qualify.py").read_bytes()).hexdigest() == record["suite_source_sha256"]
        if "distinct_driver_sha256" in record:
            source = root / record.get("distinct_driver_source", str((here / "distinct_perf.py").relative_to(root)))
            assert hashlib.sha256(source.read_bytes()).hexdigest() == record["distinct_driver_sha256"]
        comp = record["numerical_comparison"]
        if not comp["acceptance"]["pass"]:
            failures.append(dict(case=label, acceptance=comp["acceptance"]))
        am, bm = comp["baseline_metrics"], comp["candidate_metrics"]
        if am["row_l2_pct_max"] is not None and bm["row_l2_pct_max"] is not None:
            delta = bm["row_l2_pct_max"] - am["row_l2_pct_max"]
            if delta > max_row_delta[0]:
                max_row_delta = (delta, label)
print(json.dumps(dict(records=total, invariant_checks="PASS", numerical_pass=total-len(failures),
                      numerical_failures=failures, maximum_worst_row_increase_pp=max_row_delta,
                      note="Row changes require review; this script does not invent a row acceptance threshold."), indent=2))
