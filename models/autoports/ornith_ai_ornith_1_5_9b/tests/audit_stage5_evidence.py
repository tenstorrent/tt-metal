# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify final-default stage-five provenance and the unchanged acceptance gates.

The stage owner writes final_manifest.json only after final selection. This audit
checks that its required artifacts came from successful TP4 jobs using the exact
current production sources, and emits a compact reviewer-facing evidence table.
"""

import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/optimized_multichip_decoder"


def pcc_values(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("pcc") and isinstance(item, (int, float)):
                yield path + "/" + key, item
            else:
                yield from pcc_values(item, path + "/" + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from pcc_values(item, path + f"/{index}")


def main():
    manifest = json.loads((D / "final_manifest.json").read_text())
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in manifest["production_sources"]}
    assert sources == manifest["source_sha256"], "Manifest is stale"
    jobs = {j["label"]: j for j in map(json.loads, (D / "device_jobs.jsonl").read_text().splitlines())}
    rows = []
    categories = set()
    batches = set()
    kinds = {category: set() for category in ("contracts", "stress", "max_context", "pool")}
    for entry in manifest["artifacts"]:
        job = jobs[entry["job"]]
        assert job["exit_code"] == 0 and job["regime"] == "serial", entry
        assert job["devices"] == [0, 1, 2, 3] and "--baseline" not in job["command"], entry
        for name, digest in sources.items():
            assert job["source_sha256"][name] == digest, (entry, name, "stale source")
        data = json.loads((D / entry["path"]).read_text())
        assert data["source_sha256"] == sources["tt/multichip_decoder.py"], entry
        assert not data.get("baseline", False), entry
        scores = list(pcc_values(data.get("records", [])))
        assert scores and min(score for _, score in scores) >= 0.995, entry
        if entry.get("tracking"):
            assert data["tracking"] == "1" and data["skip_program_cache_tracking"] == "0", entry
        category = entry["category"]
        categories.add(category)
        if category in kinds:
            kinds[category].add(data["layer"])
        if category == "batch_matrix":
            batches.add(data["batch"])
            assert data["state_checks"] and data["steps"] >= 16, entry
        if category == "contracts":
            assert data["runtime_fallback_guard"] and data["trace"], entry
            assert any(
                r.get("label") == "trace_allocation_positive_control" and r["passed"] for r in data["records"]
            ), entry
        if category == "stress":
            assert data["steps"] >= 128, entry
        if category == "long_answer":
            assert data["steps"] >= 1024 and data["batch"] == 32 and data["state_checks"], entry
        if category == "max_context":
            assert max(data["prefill_lengths"]) == 262144 and data["decode_position"] == 262143, entry
        if category == "watcher":
            assert job.get("env", {}).get("TT_METAL_WATCHER"), entry
            capture = job["watcher_capture"]
            watcher_bytes = (D / capture["directory"] / "watcher.log").read_bytes()
            assert watcher_bytes and len(watcher_bytes) == capture["bytes"], entry
            assert hashlib.sha256(watcher_bytes).hexdigest() == capture["sha256"], entry
            assert b"Watcher detected" not in watcher_bytes, entry
        text = (D / (entry["job"] + ".log")).read_text(errors="replace")
        assert "Traceback" not in text and "TT_FATAL" not in text, entry
        rows.append(
            {
                "artifact": entry["path"],
                "category": category,
                "pcc_checks": len(scores),
                "minimum_pcc": min(score for _, score in scores),
                "job": entry["job"],
                "source_matches": True,
            }
        )
    assert batches == set(range(1, 33)), batches
    assert all(values == {0, 3} for values in kinds.values()), kinds
    assert {
        "contracts",
        "stress",
        "max_context",
        "pool",
        "watcher",
        "batch_matrix",
        "long_answer",
        "performance",
        "profile",
    } <= categories

    direct_waves = set()
    fused_conv_waves = set()
    for entry in manifest["direct_kernel_artifacts"]:
        job = jobs[entry["job"]]
        assert job["exit_code"] == 0 and job["devices"] == [0, 1, 2, 3], entry
        assert all(job["source_sha256"][name] == digest for name, digest in sources.items()), entry
        data = json.loads((D / entry["path"]).read_text())
        assert data["complete"] and data["watcher"], entry
        assert {record["batch"] for record in data["records"]} == set(range(1, 33)), entry
        if "waves" in data["options"]:
            # The optional integrated conv/gate path has independently owned
            # production and identical-gate controls, not the legacy conv formula.
            expected_waves = {int(value) for value in data["options"]["waves"].split(",")}
            assert expected_waves == {7, 104}, entry
            for wave in expected_waves:
                wave_records = [r for r in data["records"] if r["wave"] == wave]
                assert {r["batch"] for r in wave_records} == set(range(1, 33)), entry
                assert len(wave_records) == 32 * data["options"]["steps"] * 4, entry
            for key in (
                "history_vs_production",
                "conv_vs_production",
                "state_same_gates_vs_production",
                "output_same_gates_vs_production",
            ):
                assert all(record[key]["bitwise_equal"] for record in data["records"]), (entry, key)
            fused_conv_waves.update(expected_waves)
        else:
            assert all(record["state_vs_ttnn"]["bitwise_equal"] for record in data["records"]), entry
            direct_waves.add(data["options"]["wave"])
        scores = list(pcc_values(data["records"]))
        assert scores and min(score for _, score in scores) >= 0.995, entry
        capture = job["watcher_capture"]
        watcher_bytes = (D / capture["directory"] / "watcher.log").read_bytes()
        assert watcher_bytes and len(watcher_bytes) == capture["bytes"], entry
        assert hashlib.sha256(watcher_bytes).hexdigest() == capture["sha256"], entry
        assert b"Watcher detected" not in watcher_bytes, entry
        rows.append(
            {
                "artifact": entry["path"],
                "category": "direct_kernel_watcher",
                "pcc_checks": len(scores),
                "minimum_pcc": min(score for _, score in scores),
                "job": entry["job"],
                "source_matches": True,
            }
        )
    assert direct_waves == {7, 104}, direct_waves
    if "tt/conv_gates.py" in sources:
        assert fused_conv_waves == {7, 104}, "Missing direct coverage of selected convolution/gate fusion"

    # Setup is allowed to prepare weights and immutable request offsets. Every
    # forward/helper method must remain free of torch execution and host transfers.
    for filename in manifest["production_sources"]:
        if not filename.endswith(".py"):
            continue
        tree = ast.parse((ROOT / filename).read_text())
        setup_ranges = [
            (n.lineno, n.end_lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in {"from_state_dict", "allocate_batch_state"}
        ]
        for method in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            if any(start <= method.lineno <= end for start, end in setup_ranges):
                continue
            for node in ast.walk(method):
                if not isinstance(node, ast.Call):
                    continue
                name = ast.unparse(node.func)
                assert not name.startswith("torch."), (filename, method.name, name)
                assert name not in {"ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor"}, (
                    filename,
                    method.name,
                    name,
                )
    context = json.loads((ROOT / "doc/context_contract.json").read_text())
    assert context["supported_context"] == context["hf_advertised_context"] == 262144
    for filename in manifest["profile_provenance"]:
        provenance = json.loads((D / filename).read_text())
        assert provenance["implementation_sha256"] == sources["tt/multichip_decoder.py"]
        assert provenance["devices"] == [0, 1, 2, 3] and provenance["decode_traced"] and provenance["warmed"]
        folder = (D / filename).parent
        phase = Path(filename).name.removesuffix("_provenance.json")
        assert (folder / f"{phase}_perf_report.csv").stat().st_size > 1000
        assert (folder / f"{phase}_perf_report.txt").stat().st_size > 1000
    with (D / "final_accuracy.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "passed",
        "source_sha256": sources,
        "artifacts": rows,
        "minimum_pcc": min(row["minimum_pcc"] for row in rows),
        "runtime_fallback_audit": "Static source audit and dynamic TorchDispatch/host-transfer guards passed",
        "supported_context": 262144,
        "final_default_only": True,
    }
    (D / "final_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("STAGE5_EVIDENCE_AUDIT_PASS", len(rows), report["minimum_pcc"])


if __name__ == "__main__":
    main()
