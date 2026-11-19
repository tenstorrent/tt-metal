# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only final Watcher closure: exact layers0/1/3, never stage acceptance.

A successful run, actual completed Watcher dumps and clean diagnostics are all
required. Missing/pending records fail; output is written only after all three
layers close. Historical linear records may opt into the existing exact QSA
constructor dependency proof. Numerical/control acceptance remains separate.
"""

import argparse
import json
import re
from pathlib import Path

from .validate_trace_evidence import argument_path, digest, require, source_closure
from .verify_format19_dependency import FormatSourceClosure
from .verify_linear_qsa_dependency import verify_record

# These signatures cover WatcherDeviceReader's NOC/address/CB sanitization,
# assert, stack-overflow and mailbox corruption reports, plus host fatal errors.
FAILURE_PATTERN = r"(?i)\b(?:fatal|invalid|overflow(?:ed|s)?|sanitize|sanitization|assert(?:ion)?|corrupt(?:ed|ion)?|hang|traceback)\b|Watcher (?:detected|stopped|unexpected)|\bNOC error\b|\bTT_(?:FATAL|THROW|ASSERT)\b"
PROFILER_ENV = (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_STREAMING_PROFILER",
    "TT_METAL_STREAMING_PROFILER_DRAM_MB",
    "TT_METAL_STREAMING_PROFILER_FIFO_MB",
    "TT_METAL_STREAMING_PROFILER_OPS_CSV",
    "TT_METAL_STREAMING_PROFILER_ZONE_CSV",
    "TT_METAL_STREAMING_PROFILER_TRACY",
    "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY",
)


def scan(text, label):
    require(bool(text.strip()), f"Empty {label}")
    findings = [
        {"line": i, "text": line[:500]}
        for i, line in enumerate(text.splitlines(), 1)
        if re.search(FAILURE_PATTERN, line)
    ]
    require(not findings, f"Diagnostic findings in {label}: {findings[:8]}")
    return {"lines": len(text.splitlines()), "findings": findings}


def verify_watcher_run(root, binding):
    formatting = FormatSourceClosure(root, binding.get("format_source_closure"))
    layer = binding["layer"]
    require(type(layer) is int and layer in (0, 1, 3), "Unexpected layer")
    artifact = root / binding["artifact"]
    provenance_path = root / binding["provenance"]
    report = json.loads(artifact.read_text())
    record = json.loads(provenance_path.read_text())
    require(record.get("status") == "finished" and record.get("exit_code") == 0, "Unfinished or unsuccessful run")
    require(report.get("passed") is True, "Run did not pass its declared gate")
    require(
        type(report.get("layer_idx", report.get("layer"))) is int
        and report.get("layer_idx", report.get("layer")) == layer,
        "Wrong report layer",
    )
    require(argument_path(record, "--output") == artifact.resolve(), "Unbound result artifact")
    environment = record["environment"]
    interval = environment.get("TT_METAL_WATCHER")
    require(
        isinstance(interval, str) and re.fullmatch(r"\d+(?:\.\d+)?", interval) and float(interval) > 0,
        "Watcher was not enabled with a positive interval",
    )
    for name in PROFILER_ENV:
        require(name in environment and environment[name] in (None, "", "0"), f"Profiler not explicitly absent: {name}")
    log = Path(record["cwd"]) / record["log"]
    text = log.read_text()
    require(digest(log) == record["log_sha256"], "Command log changed")
    require(text.rstrip().endswith("RUN_RECORDED_FINISHED exit_code=0"), "Command log lacks final completion")
    command_scan = scan(text, "command log")
    argv = record["argv"]
    require(argv.count("-m") == 1, "Missing unique executed module")
    module = argv[argv.index("-m") + 1]
    prefix = "models.autoports.qwen_qwen3_8_flash_next.tests."
    require(module.startswith(prefix) and "." not in module[len(prefix) :], "Wrong model-local probe module")
    harness = f"tests/{module[len(prefix):]}.py"
    required = set(binding["required_sources"]) | {harness, "tests/run_recorded.py"}
    for name in required:
        if name != "tt/qsa.py":
            formatting.verify(name, record["source_sha256"].get(name), "Required harness/source changed")
    closure = binding.get("source_closure")
    if closure is not None:
        require(closure == "linear_qsa_constructor_v1" and layer in (0, 1), "Invalid linear-only source closure")
        proof = verify_record(root, binding["artifact"], binding["provenance"], binding.get("format_source_closure"))
        sources = {name: record["source_sha256"][name] for name in sorted(required)}
    else:
        sources = source_closure(root, record, required, binding.get("format_source_closure"))
        proof = None
    logs_root = Path(environment["TT_METAL_LOGS_PATH"])
    if not logs_root.is_absolute():
        logs_root = Path(record["cwd"]) / logs_root
    require(logs_root.resolve().is_relative_to(root.resolve()), "Watcher root outside retained model evidence")
    watcher = logs_root / "generated/watcher/watcher.log"
    watcher_text = watcher.read_text()
    watcher_scan = scan(watcher_text, "watcher log")
    attached = re.findall(r"^At .* attach device (\d+)\s*$", watcher_text, re.M)
    detached = re.findall(r"^At .* detach device (\d+)\s*$", watcher_text, re.M)
    starts = re.findall(r"^Dump #(\d+) at ", watcher_text, re.M)
    completed = re.findall(r"^Dump #(\d+) completed at ", watcher_text, re.M)
    require(attached and sorted(attached) == sorted(detached), "Missing device attach/detach coverage")
    require(starts and starts == completed and len(starts) == len(set(starts)), "Missing/incomplete Watcher dumps")
    require(re.search(r"^Device \d+ worker core", watcher_text, re.M), "No actual worker-core observations")
    return {
        "layer": layer,
        "artifact": binding["artifact"],
        "artifact_sha256": digest(artifact),
        "provenance": binding["provenance"],
        "provenance_sha256": digest(provenance_path),
        "command_log": str(log),
        "command_log_sha256": digest(log),
        "command_scan": command_scan,
        "watcher_log": str(watcher.relative_to(root)),
        "watcher_log_sha256": digest(watcher),
        "watcher_bytes": watcher.stat().st_size,
        "watcher_scan": watcher_scan,
        "completed_dumps": len(completed),
        "attached_devices": attached,
        "watcher_interval": interval,
        "profiler_absent": True,
        "required_source_sha256": sources,
        "linear_qsa_dependency_proof": proof,
        "format_source_closure": formatting.proof,
        "declared_run_passed": True,
        "literal_hf_strict_passed": report.get("literal_hf_strict_passed"),
        "clean": True,
    }


def close_watcher(root, manifest):
    bindings = manifest["runs"]
    require(len(bindings) == 3 and sorted(row["layer"] for row in bindings) == [0, 1, 3], "Require exactly layers0/1/3")
    require(
        len({row["artifact"] for row in bindings}) == 3 and len({row["provenance"] for row in bindings}) == 3,
        "Duplicate run bindings",
    )
    results = [verify_watcher_run(root, row) for row in bindings]
    require(len({row["watcher_log"] for row in results}) == 3, "Reused Watcher logs cannot cover independent runs")
    return {
        "all_clean": True,
        "layers": [0, 1, 3],
        "run_count": 3,
        "all_sources_closed": True,
        "all_profiler_absent": True,
        "all_runs_completed_successfully": True,
        "failure_pattern": FAILURE_PATTERN,
        "scope": "Representative successful final decoder runs; clean Watcher/source evidence only, no numerical-control or stage acceptance waiver",
        "stage_acceptance": "not_evaluated",
        "runs": results,
        "validator_sha256": digest(Path(__file__)),
        "linear_dependency_validator_sha256": digest(Path(__file__).with_name("verify_linear_qsa_dependency.py")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = close_watcher(Path(__file__).resolve().parents[1], json.loads(args.manifest.read_text()))
    report["manifest_sha256"] = digest(args.manifest)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("all_clean", "layers", "run_count", "stage_acceptance")}))


if __name__ == "__main__":
    main()
