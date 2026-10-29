#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Kernel-only NVFP4 worktrees and an accuracy-first hardware evaluator.

The agent calls prepare/evaluate/compare; this program does not invoke an LLM.
All hardware access goes through run_safe_pytest.sh. Runtime, Python environment,
tests, and compiler stay in the trusted checkout; sparse workers contain kernels.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
KERNELS = Path("ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4/device/kernels")
OPERATIONS = Path("tests/ttnn/unit_tests/operations")
MANIFEST = ".nvfp4-worktree.json"
TRUSTED_FILES = [
    Path("scripts/nvfp4_harness.py"),
    Path("scripts/nvfp4_harness_plugin.py"),
    Path("scripts/run_safe_pytest.sh"),
    Path("conftest.py"),
    OPERATIONS / "test_nvfp4_accuracy.py",
    OPERATIONS / "test_dequantize_nvfp4.py",
    OPERATIONS / "test_nvfp4_perf.py",
    Path("ttnn/ttnn/_experimental/nvfp4.py"),
]


class EvaluationError(RuntimeError):
    pass


def git(*args, cwd=REPO):
    return subprocess.check_output(["git", "-C", str(cwd), *map(str, args)], text=True).strip()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def hashes(root, paths):
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def prepare(path, ref):
    path = path.resolve()
    if path.exists():
        raise EvaluationError(f"Worktree path already exists: {path}")
    base = git("rev-parse", ref)
    git("worktree", "add", "--detach", "--no-checkout", path, base)
    git("sparse-checkout", "set", "--no-cone", f"/{KERNELS}/", cwd=path)
    git("checkout", base, cwd=path)
    # Share existing instruction files; never copy instructions, venv or binaries.
    for name in ("CLAUDE.md", "AGENTS.md", "AGENTS.override.md"):
        if (REPO / name).exists():
            (path / name).symlink_to(REPO / name)
    write_json(path / MANIFEST, {"runtime_root": str(REPO), "base_revision": base})
    return {"worktree": str(path), "base_revision": base, "kernel_directory": str(path / KERNELS)}


def validate_candidate(candidate):
    candidate = candidate.resolve(strict=True)
    manifest = read_json(candidate / MANIFEST)
    if Path(manifest["runtime_root"]).resolve() != REPO:
        raise EvaluationError("Candidate belongs to another runtime checkout")
    if (
        Path(git("rev-parse", "--git-common-dir", cwd=candidate)).resolve()
        != Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    ):
        raise EvaluationError("Candidate is not a worktree of the trusted repository")
    base = manifest["base_revision"]
    changed = git("diff", "--name-only", base, cwd=candidate).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", cwd=candidate).splitlines()
    for name in changed + untracked:
        path = Path(name)
        if name in (MANIFEST, "CLAUDE.md", "AGENTS.md", "AGENTS.override.md"):
            continue
        if not path.is_relative_to(KERNELS) or path.suffix not in (".cpp", ".h", ".hpp"):
            raise EvaluationError(f"Candidate changed an out-of-scope file: {name}")
        if (candidate / path).is_symlink():
            raise EvaluationError(f"Candidate kernel cannot be a symlink: {name}")
    for path in (candidate / KERNELS).rglob("*"):
        if path.is_symlink():
            raise EvaluationError(f"Candidate kernel cannot be a symlink: {path}")
    paths = sorted(path.relative_to(candidate) for path in (candidate / KERNELS).rglob("*") if path.is_file())
    source_hashes = hashes(candidate, paths)
    return candidate, manifest, source_hashes


def validate_junit(path, expected_skips=0):
    if not path.exists():
        raise EvaluationError(f"Missing pytest result artifact: {path}")
    tree = ET.parse(path)
    cases = tree.findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    if (
        not cases
        or len(cases) <= skipped
        or skipped != expected_skips
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error"))
    ):
        raise EvaluationError(f"Failed, errored, unexpectedly skipped, or empty pytest run: {path}")


def run_stage(candidate, output, name, flags, tests, extra_env, timeout):
    stage = output / name
    stage.mkdir()
    env = os.environ.copy()
    # Measurements are hardware-only. Keep debug/profiler instrumentation from
    # leaking from the caller into a timing run; --dev sets its own debug flags.
    for key in list(env):
        if key.startswith("TT_METAL_WATCHER") or key in (
            "TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS",
            "TT_METAL_LLK_ASSERTS",
            "TT_METAL_PROFILE_PERF_COUNTERS",
            "TT_METAL_SLOW_DISPATCH_MODE",
            "TT_METAL_SIMULATOR",
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_PROFILER_DIR",
        ):
            env.pop(key)
    env.update(
        TT_METAL_HOME=str(REPO),
        TT_METAL_RUNTIME_ROOT=str(REPO),
        TT_METAL_KERNEL_PATH=str(candidate),
        NVFP4_KERNEL_ROOT=str(candidate),
        TT_METAL_CACHE=str(output.parent / "jit-cache" / hashlib.sha256(str(candidate).encode()).hexdigest()[:16]),
        TT_METAL_PROFILER_DIR=str(stage / "profiler"),
        TT_METAL_LOGGER_LEVEL="critical",
        PYTHONPATH=str(REPO) + os.pathsep + str(REPO / "ttnn"),
        PATH=str(REPO / "python_env/bin") + os.pathsep + env.get("PATH", ""),
    )
    env.update(extra_env)
    junit = stage / "pytest.xml"
    command = [
        str(REPO / "scripts/run_safe_pytest.sh"),
        *flags,
        *[str(REPO / test) for test in tests],
        "-q",
        "-p",
        "scripts.nvfp4_harness_plugin",
        f"--junitxml={junit}",
    ]
    write_json(stage / "command.json", {"argv": command, "candidate": str(candidate), "timeout_seconds": timeout})
    print(f"NVFP4 {name}: {candidate.name}", flush=True)
    with (stage / "run.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            # The safe runner's dirty marker remains for its next invocation.
            raise EvaluationError(f"Stage timed out: {name}; inspect {stage / 'run.log'}") from exc
    if code:
        raise EvaluationError(f"Stage {name} failed with exit code {code}; inspect {stage / 'run.log'}")
    # The six intentional UINT8/BF16-DEST exclusions are the only allowed skips.
    validate_junit(junit, expected_skips=6 if name == "accuracy" else 0)
    return stage


def profile_metrics(stage, repetitions=1):
    files = list((stage / "profiler").glob("reports/*/ops_perf_results*.csv"))
    if len(files) != 1:
        raise EvaluationError(f"Expected one fresh profiler report, found {len(files)} in {stage}")
    with files[0].open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["OP CODE"] == "Nvfp4DeviceOperation"]
    metadata = read_json(stage / "workload.json")
    if len(rows) != metadata["calls"]:
        raise EvaluationError(f"Expected {metadata['calls']} profiled operations, got {len(rows)}")
    raw_files = list((stage / "profiler").rglob("profile_log_device.csv"))
    if not raw_files:
        raise EvaluationError("Profiler raw device cycles/clock metadata are missing")
    # Tracy's raw log records the actual profiler frequency in its first line.
    with raw_files[0].open() as stream:
        first = stream.readline()
    clock = re.search(r"CHIP_FREQ\[MHz\]:\s*([\d.]+)", first, re.IGNORECASE)
    if not clock:
        raise EvaluationError(f"Cannot determine device clock from {raw_files[0]}: {first.strip()}")
    mhz = float(clock.group(1))
    samples, math_samples = [], []
    for row in rows[metadata["warmup_calls"] :]:
        if int(row["CORE COUNT"]) != metadata["cores"]:
            raise EvaluationError("Profiled core count differs from workload metadata")
        divisor = metadata["maximum_output_tiles_per_core"]
        samples.append(float(row["DEVICE KERNEL DURATION [ns]"]) * mhz / 1000 / divisor)
        math_samples.append(float(row["DEVICE TRISC1 KERNEL DURATION [ns]"]) * mhz / 1000 / divisor / repetitions)
    if not all(math.isfinite(value) and value > 0 for value in samples + math_samples):
        raise EvaluationError("Invalid timing samples")
    return {
        "profile_csv": str(files[0]),
        "clock_mhz": mhz,
        "workload": metadata,
        "pipeline_cycles_per_output_tile": statistics.median(samples),
        "pipeline_samples": samples,
        "pipeline_mad": statistics.median(abs(v - statistics.median(samples)) for v in samples),
        "math_cycles_per_output_tile": statistics.median(math_samples),
        "math_samples": math_samples,
        "math_repetitions": repetitions,
    }


def save_disassembly(candidate, output, stage, metrics):
    with Path(metrics["profile_csv"]).open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["OP CODE"] == "Nvfp4DeviceOperation"]
    kernel_hashes = set()
    for row in rows:
        kernel_hashes.update(re.findall(r"dequantize_fast/(\d+)/", row.get("COMPUTE KERNEL HASH", "")))
    if len(kernel_hashes) != 1:
        raise EvaluationError("Expected one compute kernel hash in the performance report")
    cache = output.parent / "jit-cache" / hashlib.sha256(str(candidate).encode()).hexdigest()[:16]
    elfs = list(cache.glob(f"*/kernels/dequantize_fast/{kernel_hashes.pop()}/trisc1/trisc1.elf"))
    if not elfs:
        raise EvaluationError("Could not locate the measured math ELF")
    elf = max(elfs, key=lambda path: path.stat().st_mtime_ns)
    source = candidate / KERNELS / "compute/dequantize_fast.cpp"
    generated = elf.parent.parent / "chlkc_math.cpp"
    if str(source) not in generated.read_text():
        raise EvaluationError(f"Measured compute kernel did not compile the candidate source: {generated}")
    assembly = stage / "trisc1.asm"
    with assembly.open("w") as stream:
        subprocess.run(
            [str(REPO / "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"), "-d", str(elf)],
            stdout=stream,
            check=True,
        )
    metrics.update(resolved_compute_source=str(source), math_elf=str(elf), disassembly=str(assembly))


def math_overlay(candidate, output, repeats):
    """Diagnostic only: replay a face in DEST; amortize other pipeline work."""
    overlay = output / "math-overlay"
    compute = overlay / KERNELS / "compute"
    compute.mkdir(parents=True)
    source = (candidate / KERNELS / "compute/dequantize_fast.cpp").read_text()
    anchor = "                MATH((sfpu::nvfp4_decode_column_face"
    if source.count(anchor) != 1:
        raise EvaluationError("Math diagnostic cannot instrument this candidate's compute structure")
    start = source.index(anchor)
    end = source.index("));", start) + 3
    call = source[start:end]
    replacement = f"#pragma GCC unroll 0\n                for (uint32_t repeat = 0; repeat < {repeats}; ++repeat) {{\n{call}\n                }}"
    (compute / "dequantize_fast.cpp").write_text(source[:start] + replacement + source[end:])
    for path in (candidate / KERNELS).rglob("*"):
        if path.is_file() and path.name != "dequantize_fast.cpp":
            target = overlay / path.relative_to(candidate)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
    return overlay


def evaluate(candidate, output, diagnostic, timeout, accuracy_only=False):
    candidate, manifest, before = validate_candidate(candidate)
    if output.exists():
        raise EvaluationError(f"Result directory must be new: {output}")
    output.mkdir(parents=True)
    evaluator = hashes(REPO, TRUSTED_FILES)
    result = {
        "status": "running",
        "candidate": str(candidate),
        "base_revision": manifest["base_revision"],
        "candidate_revision": git("rev-parse", "HEAD", cwd=candidate),
        "kernel_hashes": before,
        "evaluator_hashes": evaluator,
        "started_unix": time.time(),
    }
    (output / "candidate.patch").write_text(git("diff", manifest["base_revision"], cwd=candidate) + "\n")
    write_json(output / "result.json", result)
    try:
        stage = run_stage(
            candidate,
            output,
            "accuracy",
            ["--dev"],
            [OPERATIONS / "test_nvfp4_accuracy.py", OPERATIONS / "test_dequantize_nvfp4.py"],
            {"NVFP4_ACCURACY_REPORT": str(output / "accuracy.json")},
            timeout,
        )
        accuracy = read_json(output / "accuracy.json")
        if not accuracy["accuracy_gate"]["passed"]:
            raise EvaluationError("Numerical gate did not pass")
        header = str(KERNELS / "compute/nvfp4_sfpu.hpp")
        if accuracy["kernel_sha256"] != before[header]:
            raise EvaluationError("Accuracy test reported a different SFPU source than the candidate")
        result["accuracy_gate"] = accuracy["accuracy_gate"]
        if not accuracy_only:
            stage = run_stage(
                candidate,
                output,
                "pipeline",
                ["--profile"],
                [OPERATIONS / "test_nvfp4_perf.py"],
                {"NVFP4_BENCH_REPORT": str(output / "pipeline/workload.json")},
                timeout,
            )
            result["performance"] = profile_metrics(stage)
            save_disassembly(candidate, output, stage, result["performance"])
            if diagnostic:
                overlay = math_overlay(candidate, output, 128)
                stage = run_stage(
                    overlay,
                    output,
                    "math_isolate",
                    ["--profile"],
                    [OPERATIONS / "test_nvfp4_perf.py"],
                    {"NVFP4_BENCH_REPORT": str(output / "math_isolate/workload.json")},
                    timeout,
                )
                result["math_isolate"] = profile_metrics(stage, repetitions=128)
                save_disassembly(overlay, output, stage, result["math_isolate"])
                result["math_isolate"][
                    "scope"
                ] = "Repeated DEST face decoder; amortized pipeline, not standalone LLK MATH_ISOLATE"
        if before != validate_candidate(candidate)[2] or evaluator != hashes(REPO, TRUSTED_FILES):
            raise EvaluationError("Candidate or trusted evaluator changed during evaluation")
        result["status"] = "passed"
    except (EvaluationError, OSError, ValueError, KeyError, ET.ParseError, subprocess.CalledProcessError) as exc:
        result.update(status="failed", error=str(exc))
    result["finished_unix"] = time.time()
    write_json(output / "result.json", result)
    return result


def compare(baseline_path, candidate_path):
    baseline, candidate = read_json(baseline_path), read_json(candidate_path)
    if any(result["status"] != "passed" or "performance" not in result for result in [baseline, candidate]):
        raise EvaluationError("Comparison requires two passing accuracy+performance evaluations")
    if baseline["evaluator_hashes"] != candidate["evaluator_hashes"]:
        raise EvaluationError("Evaluations used different trusted evaluators")
    a, b = baseline["performance"], candidate["performance"]
    if a["workload"] != b["workload"] or a["clock_mhz"] != b["clock_mhz"]:
        raise EvaluationError("Workload or device clock differs")
    saved = a["pipeline_cycles_per_output_tile"] - b["pipeline_cycles_per_output_tile"]
    # A screening decision, followed by independent paired remeasurement.
    noise = max(3 * a["pipeline_mad"], 3 * b["pipeline_mad"], 0.01 * a["pipeline_cycles_per_output_tile"])
    return {
        "baseline_cycles": a["pipeline_cycles_per_output_tile"],
        "candidate_cycles": b["pipeline_cycles_per_output_tile"],
        "saved_cycles": saved,
        "improvement_percent": 100 * saved / a["pipeline_cycles_per_output_tile"],
        "screening_noise_floor_cycles": noise,
        "screening_improvement": saved > noise,
        "promotion_requires": "Independent paired hardware remeasurement; this comparison does not modify any branch",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--path", type=Path, required=True)
    create.add_argument("--ref", default="HEAD")
    run = commands.add_parser("evaluate")
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--math-isolate", action="store_true")
    run.add_argument("--accuracy-only", action="store_true")
    run.add_argument("--timeout-seconds", type=int, default=600)
    diff = commands.add_parser("compare")
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.path, args.ref)
        elif args.command == "compare":
            result = compare(args.baseline, args.candidate)
        else:
            result = evaluate(
                args.candidate, args.output.resolve(), args.math_isolate, args.timeout_seconds, args.accuracy_only
            )
    except (EvaluationError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"NVFP4 harness: {exc}\n")
    print(json.dumps(result, indent=2))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
