# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail closed on missing/stale stage evidence and generate the final compact tables."""

import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/multichip_decoder"
sha = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
# Static companion to the executable device_only guard in every contract runner.
tree = ast.parse((ROOT / "tt/multichip_decoder.py").read_text())
for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
    for method in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != "from_state_dict"]:
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                assert not name.startswith("torch."), (method.name, name)
                assert name not in {"ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor"}, (
                    method.name,
                    name,
                )
accuracy = []
required = [
    *(
        f"{name}_{layer}"
        for layer in [0, 3]
        for name in ["final_contracts", "final_stress", "final_long", "watcher", "profile", "final_perf"]
    ),
    "final_stack_1",
    "final_stack_32",
    "final_stack_perf_1",
    "final_stack_perf_32",
]
for name in required:
    data = json.loads((D / f"{name}.json").read_text())
    assert data["source_sha256"] == sha, (name, "stale source")
    assert not data.get("baseline", False), name
    for row in data["records"]:
        for key, value in row.items():
            if "pcc" in key:
                assert value >= 0.995, (name, row)
                accuracy.append({"artifact": f"{name}.json", "label": row["label"], "metric": key, "value": value})
    if name.startswith(("final_contracts", "final_stress", "final_long", "final_stack")) and not name.startswith(
        "final_stack_perf"
    ):
        assert data["tracking"] == "1" and data["skip_program_cache_tracking"] == "0", name
    if name.startswith(("final_contracts", "final_stress")):
        assert data["trace"] and data["tracking"] == "1", name
    if name.startswith("final_contracts"):
        labels = {r["label"] for r in data["records"]}
        for n in [1, 2, 3, 4, 31, 32, 33, 1023, 1024, 1025, 2051]:
            assert f"prefill_{n}" in labels
            assert all(f"decode_{n}_{step}" in labels for step in range(8))
            states = ["conv", "recurrent"] if data["layer"] == 0 else ["key", "value"]
            assert all(f"{state}_{n}" in labels for state in states)
        for row in data["records"]:
            if row["label"].startswith(("prefill_", "decode_", "refresh_")):
                assert "all_ranks_contribution_pcc" in row
        shapes = data["state_shapes"]
        if data["layer"] == 0:
            assert shapes == {"conv": [1, 1, 3, 2048], "recurrent": [1, 8, 128, 128]}
        else:
            assert shapes["key"][1:] == shapes["value"][1:] == [1, 32, 256]
        assert any(r["label"] == "trace_allocation_positive_control" and r["passed"] for r in data["records"])
        assert sum(r.get("refresh_replays", 0) for r in data["records"]) == 33
    if name.startswith("final_stress"):
        assert data["steps"] == 128
        assert len([r for r in data["records"] if r["label"].startswith("decode_")]) == 128
    if name.startswith("final_stack"):
        assert data["shared_collective_workspace"]
        assert len(data["records"]) == data["batch"] * 6
    if name.startswith("final_stack_perf"):
        assert data["tracking"] == "0" and data["benchmark"] and data["decode_stack_ms"] > 0
    if name.startswith("final_long"):
        assert data["prefill_lengths"] == [262139, 262143, 262144]
        assert data["decode_position"] == 262143 and data["deterministic_replays"] == 3
    if name.startswith("watcher"):
        text = (D / f"{name}.log").read_text()
        assert "PROBE_PASS" in text and "Traceback" not in text and "TT_FATAL" not in text, name
    if name.startswith("profile"):
        layer = name.split("_")[-1]
        for phase in ["prefill", "decode"]:
            path = D / "tracy" / layer / f"{phase}_perf_report.txt"
            assert path.stat().st_size > 1000, path
            assert (D / "tracy" / layer / f"{phase}_perf_report.csv").stat().st_size > 1000
            provenance = json.loads((D / "tracy" / layer / f"{phase}_provenance.json").read_text())
            assert provenance["implementation_sha256"] == sha

performance = []
for layer in [0, 3]:

    def timing(name):
        x = json.loads((D / f"{name}_{layer}.json").read_text())
        return next(r for r in x["records"] if r["label"] == "timing_1024")

    before, after = timing("perf_baseline"), timing("final_perf")
    for mode in ["prefill", "decode"]:
        b, a = before[f"{mode}_ms"], after[f"{mode}_ms"]
        performance.append(
            {"layer": layer, "mode": mode, "baseline_ms": b, "tp4_ms": a, "speedup": b / a, "efficiency": b / a / 4}
        )
for batch in [1, 32]:
    before = json.loads((D / f"baseline_stack_perf_{batch}.json").read_text())["decode_stack_ms"]
    after = json.loads((D / f"final_stack_perf_{batch}.json").read_text())["decode_stack_ms"]
    performance.append(
        {
            "layer": "stack_0_3",
            "mode": f"decode_batch{batch}",
            "baseline_ms": before,
            "tp4_ms": after,
            "speedup": before / after,
            "efficiency": before / after / 4,
        }
    )
for filename, rows in [("accuracy.csv", accuracy), ("performance.csv", performance)]:
    with (D / filename).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
context = json.loads((ROOT / "doc/context_contract.json").read_text())
assert context["supported_context"] == context["hf_advertised_context"] == 262144
memory = json.loads((D / "memory_capacity_plan.json").read_text())
assert memory["total_budget_per_chip"] < memory["device_dram_bytes"]
(D / "final_audit.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "runtime_fallback_audit": "static scan plus dynamic torch/host-transfer guards passed",
            "source_sha256": sha,
            "artifacts": required,
            "minimum_pcc": min(r["value"] for r in accuracy),
            "performance": performance,
            "supported_context": 262144,
            "memory_budget_per_chip": memory["total_budget_per_chip"],
        },
        indent=2,
    )
    + "\n"
)
print("MULTICHIP_AUDIT_PASS", sha)
