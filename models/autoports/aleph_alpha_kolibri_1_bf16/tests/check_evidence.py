# SPDX-License-Identifier: Apache-2.0
"""Check stage evidence freshness and numerical gates without opening a device."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "doc/functional_decoder"
source_hash = hashlib.sha256((ROOT / "tt/functional_decoder.py").read_bytes()).hexdigest()
records = []


def verify_source(recorded_hash, dependency):
    current = (ROOT / dependency).read_bytes()
    if recorded_hash == hashlib.sha256(current).hexdigest():
        return
    # Preserve the exact executed bytes. Only inspected commit-hook import
    # formatting is accepted here; arbitrary stale dependencies still fail.
    assert dependency in ("tests/reference.py", "tests/run_context.py"), dependency
    entries = json.loads((EVIDENCE / "commit_formatting.json").read_text())
    entry = next(r for r in entries if r["path"] == dependency)
    original = (ROOT / entry["snapshot"]).read_bytes()
    assert recorded_hash == entry["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert entry["formatted_sha256"] == hashlib.sha256(current).hexdigest()
    old_tree, new_tree = ast.parse(original), ast.parse(current)
    if dependency == "tests/run_context.py":
        for tree in (old_tree, new_tree):
            tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    assert ast.dump(old_tree) == ast.dump(new_tree), dependency


def read(name, tracking=True):
    path = EVIDENCE / name
    value = json.loads(path.read_text())
    assert value["provenance"]["source_sha256"]["tt/functional_decoder.py"] == source_hash, f"Stale decoder: {name}"
    for dependency in ("tests/reference.py", "tests/config.json"):
        verify_source(value["provenance"]["source_sha256"][dependency], dependency)
    environment = value["provenance"]["environment"]
    assert environment["TT_METAL_TRACE_ALLOC_TRACKING"] == ("1" if tracking else "0"), name
    assert environment["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] != "1", name
    return value


for layer in (0, 4):
    for kind in ("coverage", "synthetic", "watcher"):
        name = f"{kind}_{layer}.json"
        data = read(name)
        assert data["real_weights"] == (kind != "synthetic"), name
        assert data["runtime_audit"] == "clean" and data["deterministic"], name
        assert data["trace"]["captures"] == 1 and data["trace"]["replays"] == 33, name
        assert data["trace"]["request_releases"] == 0, name
        minimum = min(r["pcc"] for r in data["rows"])
        assert minimum >= 0.995, (name, minimum)
        if kind == "watcher":
            assert data["provenance"]["environment"]["TT_METAL_WATCHER"] == "10", name
            assert data["provenance"]["environment"]["TT_METAL_DEVICE_PROFILER"] is None, name
            logs = list((EVIDENCE / f"watcher/layer_{layer}/generated/watcher").glob("*.log"))
            assert logs, name
            for log in logs:
                lowered = log.read_text().lower()
                assert not any(
                    term in lowered for term in ("fatal", "exception", "corrupt", "overflow", "invalid", "assert")
                ), log
        records.append(dict(artifact=name, rows=len(data["rows"]), minimum_pcc=minimum))
    data = read(f"context_{layer}.json")
    verify_source(data["provenance"]["source_sha256"]["tests/run_context.py"], "tests/run_context.py")
    assert data["largest_tested_prefill"] == data["largest_tested_decode"] == 1048576
    assert data["full_layer_tokens_executed"] == 1048576
    assert min(r["pcc"] for r in data["rows"]) >= 0.995
    assert any(r.get("logical_length") == 1048559 for r in data["rows"])
    assert any(r.get("context") == 1048576 and r["case"] == "full_context_traced_decode" for r in data["rows"])
    records.append(dict(artifact=f"context_{layer}.json", minimum_pcc=min(r["pcc"] for r in data["rows"])))
    data = read(f"profile_{layer}.json", tracking=False)
    assert all(r["pcc"] >= 0.995 for r in data["pcc"])
    assert data["repetitions"] == 3
    assert data["provenance"]["environment"]["TT_METAL_WATCHER"] is None
    for mode in ("prefill", "decode"):
        directory = EVIDENCE / f"tracy/layer_{layer}"
        assert (directory / f"{mode}_perf_report.csv").is_file()
        assert (directory / f"{mode}_ops.csv").is_file()
        table = directory / f"{mode}_perf_report.txt"
        assert table.is_file() and table.stat().st_size > 2000

for layer, batch in ((0, 32), (4, 32), (0, 31), (4, 13), (0, 2), (4, 2)):
    name = f"batch{batch}_{layer}_final.json"
    data = read(name)
    assert data["deterministic"]
    assert (
        min(data["prefill_pcc"], data["decode_pcc"], data["changed_batch_pcc"], *data["changed_batch_per_row_pcc"])
        >= 0.995
    ), name
    records.append(dict(artifact=name, minimum_changed_lane_pcc=min(data["changed_batch_per_row_pcc"])))

result = dict(
    status="all_machine_checks_passed_awaiting_independent_stage_review", decoder_sha256=source_hash, checks=records
)
(EVIDENCE / "evidence_check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
