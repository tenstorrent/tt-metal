# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only narrow closure of ordered projection v1/v2 failures and v3 pass."""
import ast
import hashlib
import json
import re
from pathlib import Path

r = Path(__file__).resolve().parents[1]
d = r / "doc/functional_decoder"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identity(p):
    return {"path": str(p.relative_to(r)), "sha256": sha(p), "bytes": p.stat().st_size}


def executable(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


files = [
    "tt/ordered_projection.py",
    "tt/kernels/ordered_projection_reader.cpp",
    "tt/kernels/ordered_projection_compute.cpp",
    "tt/kernels/ordered_projection_writer.cpp",
    "tests/probe_ordered_projection.py",
]
report = {
    "scope": "Standalone real-weight ordered indexer operator only; no full-context QSA or stage acceptance inferred",
    "stage_acceptance": "not_evaluated",
    "runs": [],
}
for suffix, exitcode in [("cpu_all_v1", 0), ("device_v1", 1), ("device_v2", 1), ("device_v3", 0)]:
    stem = "ordered_projection_" + suffix
    p = d / "provenance" / f"{stem}.json"
    prov = json.loads(p.read_text())
    result = d / f"{stem}.json"
    j = json.loads(result.read_text())
    log = Path(prov["cwd"]) / prov["log"]
    assert prov["status"] == "finished" and prov["exit_code"] == exitcode and prov["log_sha256"] == sha(log)
    sources = []
    for name in files:
        old = prov["source_sha256"][name]
        current = r / name
        if sha(current) == old:
            sources.append({"executed": name, **identity(current), "binding": "current_exact"})
        else:
            version = "v1" if suffix in ("cpu_all_v1", "device_v1") else "v2" if suffix == "device_v2" else "v3"
            saved = (
                r
                / "tests/diagnostic_sources"
                / (f"ordered_projection_device_{version}_" + name.replace("/", "_") + ".txt")
                if version != "v3"
                else r / "tests/diagnostic_sources/ordered_projection.py.validated_v3.txt"
            )
            assert sha(saved) == old, (suffix, name)
            sources.append({"executed": name, **identity(saved), "binding": "exact_historical"})
            if version == "v3":
                assert name == "tt/ordered_projection.py" and executable(saved) == executable(current)
    row = {
        "name": stem,
        "artifact": identity(result),
        "provenance": identity(p),
        "log": identity(log),
        "exit_code": exitcode,
        "reported_passed": j["passed"],
        "cpu_cases": len(j["cpu"]),
        "sources": sources,
    }
    assert all(x["bf16_exact"] and x["finite"] and x["different_elements"] == 0 for x in j["cpu"])
    if suffix == "cpu_all_v1":
        assert not j["device_executed"] and not j["passed"]
        row["classification"] = "CPU-only order proof; passed false intentionally, no device run"
    if suffix in ("device_v1", "device_v2"):
        assert not j["passed"] and j["device"][0]["different_elements"] == 2
        row["classification"] = "Preserved strict direct-pack failure before capture"
    if suffix == "device_v2":
        fp = j["device"][0]["fp32_diagnostic"]
        assert fp["different_fp32_bits"] == 0 and fp["maximum_absolute_fp32_error"] == 0
        for key in [
            "device_fp32_host_integer_rne_vs_ordered",
            "explicit_device_typecast_vs_ordered",
            "explicit_device_typecast_vs_integer_rne",
        ]:
            assert fp[key]["bf16_exact"]
    if suffix == "device_v3":
        assert (
            j["passed"]
            and j["runtime_audit"]
            and j["repeated_input_exact"]
            and j["preallocated_output_address_preserved"]
            and j["capture_output_readbacks_before_replay"] == 0
            and j["prepared_program_cache_entries"] == j["final_program_cache_entries"] == 33
        )
        warm = [x for x in j["device"] if x["phase"] == "warm"]
        replay = [x for x in j["device"] if x["phase"] == "trace_replay"]
        assert len(warm) == 32 and len(replay) == 352 and len(j["cpu"]) == 320
        assert {x["count"] for x in warm} == set(range(1, 33)) and all(x["bf16_exact"] and x["finite"] for x in warm)
        assert all(
            x["runtime_audit"] and all(x[k]["bf16_exact"] and x[k]["finite"] for k in ("literal", "ordered"))
            for x in replay
        )
        audit = r / "tests/probe_linear_contract.py"
        assert sha(audit) == prov["source_sha256"]["tests/probe_linear_contract.py"]
        sources.append(
            {"executed": "tests/probe_linear_contract.py", **identity(audit), "binding": "current_exact_audit_helper"}
        )
        row.update(
            classification="All-Q operator exact eager/trace pass, not full-context decoder pass",
            warm_cases=len(warm),
            trace_replays=len(replay),
        )
    if suffix.startswith("device"):
        w = d / "watcher" / stem / "generated/watcher/watcher.log"
        text = w.read_text()
        pattern = r"(?i)\b(fatal|invalid|overflow|sanitize|assert|corrupt|hang)\b"
        hits = [line for line in text.splitlines() if re.search(pattern, line)]
        assert not hits, (stem, hits)
        assert "attach device" in text and "Dump #" in text
        row["watcher"] = {**identity(w), "lines": len(text.splitlines()), "pattern": pattern, "findings": hits}
    report["runs"].append(row)
report["all_checks_passed"] = True
(d / "ordered_projection_operator_closure_v1.json").write_text(json.dumps(report, indent=2) + "\n")
print("closure passed", [(x["name"], x.get("watcher", {}).get("lines")) for x in report["runs"]])
