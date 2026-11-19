# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject stale final artifacts and verify the integrated stage evidence inventory."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "doc/optimized_decoder"
sha = hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest()
checks = []
for kind in ("linear", "full"):
    names = [
        f"{kind}_{suffix}.json" for suffix in ("final", "watcher", "paired", "trajectory128", "long", "profile_final")
    ]
    names += [f"default_{kind}_s{n}.json" for n in (128, 1023, 1024, 1025, 8192)]
    names += [f"default_{kind}_batch32.json"]
    for name in names:
        d = json.loads((ART / name).read_text())
        measured_sha = d.get("implementation_sha256", d.get("implementations", {}).get("optimized_decoder"))
        assert measured_sha == sha, name
        assert not d.get("synthetic_weights", False) and not d.get("diagnostic_only", False), name
        assert d.get("implementation", "OptimizedDecoder") == "OptimizedDecoder", name
        scores = [v for r in d["records"] for k, v in r.items() if k in ("pcc", "contribution_pcc", "traced_pcc")]
        assert scores and min(scores) >= 0.995, (name, scores)
        checks.append(dict(artifact=name, min_pcc=min(scores), source_matches=True))
    for mode in ("prefill", "decode"):
        folder = ART / f"tracy_final/{kind}"
        d = json.loads((folder / f"{mode}_provenance.json").read_text())
        assert d["implementation_sha256"] == sha
        rows = list(csv.DictReader((folder / f"{mode}_perf_report.csv").open()))
        matmuls = [r for r in rows if "Matmul" in r["OP Code"]]
        assert len(matmuls) == (5 if mode == "decode" else 4)
        assert all(r["Input 1 Datatype"] == "BFLOAT8_B" for r in matmuls), matmuls
        assert all("LoFi" in r["Math Fidelity"] for r in matmuls[-(3 if mode == "decode" else 2) :])
        assert all("HiFi2" in r["Math Fidelity"] for r in matmuls[:2])
        assert (folder / f"{mode}_perf_report.txt").stat().st_size > 0
contract = json.loads((ROOT / "doc/context_contract.json").read_text())["optimized_decoder"]
assert contract["implementation_sha256"] == sha and contract["supported_context"] == 262144
perf = json.loads((ART / "perf_summary.json").read_text())
assert perf["implementation_sha256"] == sha
assert all(v["roofline_fraction"] >= 0.70 and v["batch32_roofline_fraction"] >= 0.70 for v in perf["layers"].values())
positive = json.loads((ART / "trace_alloc_tracking/positive_control.json").read_text())
assert positive["implementation_sha256"] == sha
assert {"positive_control": "raised"} in positive["records"]
(ART / "final_audit.json").write_text(
    json.dumps(
        dict(
            implementation_sha256=sha,
            verdict="pass",
            checks=checks,
            actual_matmul_precision_verified=True,
            context=262144,
            roofline_threshold_verified=True,
            trace_tracker_positive_control=True,
        ),
        indent=2,
    )
    + "\n"
)
print("OPTIMIZED_EVIDENCE_AUDIT_PASS", sha)
