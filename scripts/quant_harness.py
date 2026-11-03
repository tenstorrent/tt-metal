#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Kernel-only worktrees and an accuracy-first hardware evaluator for quantized decoders.

One evaluator serves every format (NVFP4, MXFP8, ...). A FormatSpec captures what
differs: kernel directory, trusted evaluator files, test lists, profiler op code,
compute-kernel name, and how to find the decode sweep for the math diagnostic.

The agent calls prepare/evaluate/compare; this program does not invoke an LLM.
All hardware access goes through run_safe_pytest.sh. Runtime, Python environment,
tests, and compiler stay in the trusted checkout; sparse workers contain kernels.
"""

import argparse
import csv
from dataclasses import dataclass
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
from typing import Callable, Optional
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
OPERATIONS = Path("tests/ttnn/unit_tests/operations")
INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md", "AGENTS.override.md")
SHARED_TRUSTED_FILES = (
    Path("scripts/quant_harness.py"),
    Path("scripts/quant_harness_plugin.py"),
    Path("scripts/run_safe_pytest.sh"),
    Path("conftest.py"),
)


class EvaluationError(RuntimeError):
    pass


def _anchored_sweep(init_anchor, start_anchor, end_anchor):
    """Locate a decode sweep after `init_anchor` (or at `start_anchor`) and before `end_anchor`."""

    def locate(source):
        if (source.count(init_anchor) != 1 and source.count(start_anchor) != 1) or source.count(end_anchor) != 1:
            raise EvaluationError("Math diagnostic cannot instrument this candidate's complete decode sweep")
        start = (
            source.index(init_anchor) + len(init_anchor)
            if source.count(init_anchor) == 1
            else source.index(start_anchor)
        )
        return start, source.index(end_anchor)

    return locate


def _marked_sweep(begin_marker, end_marker):
    """Locate the lines strictly between two marker comment lines."""

    def locate(source):
        if source.count(begin_marker) != 1 or source.count(end_marker) != 1:
            raise EvaluationError(f"Math diagnostic requires exactly one {begin_marker} and one {end_marker} marker")
        begin = source.index(begin_marker)
        start = source.index("\n", begin) + 1
        end = source.rindex("\n", 0, source.index(end_marker)) + 1
        return start, end

    return locate


@dataclass(frozen=True)
class FormatSpec:
    name: str
    env_prefix: str
    kernels: Path
    manifest: str
    trusted_files: tuple
    op_code: str
    compute_kernel: str  # Compute kernel file stem under kernels/compute; also its JIT cache directory.
    accuracy_tests: tuple
    screen_tests: tuple
    perf_tests: tuple
    full_accuracy_expected_skips: int
    plugin: str
    locate_math_sweep: Callable[[str], tuple]
    # Path of the one kernel file the accuracy report hashes ("kernel_sha256"); None means
    # the report carries "kernel_hashes" for every kernel file, compared as a whole.
    accuracy_hashed_file: Optional[Path] = None
    supports_bit_permutation: bool = False

    def env(self, suffix):
        return f"{self.env_prefix}_{suffix}"


_NVFP4_KERNELS = Path("ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4/device/kernels")
_MXFP8_KERNELS = Path("ttnn/cpp/ttnn/operations/experimental/dequantize_mxfp8/device/kernels")

FORMATS = {
    "nvfp4": FormatSpec(
        name="nvfp4",
        env_prefix="NVFP4",
        kernels=_NVFP4_KERNELS,
        manifest=".nvfp4-worktree.json",
        trusted_files=SHARED_TRUSTED_FILES
        + (
            Path("scripts/nvfp4_harness.py"),
            Path("scripts/nvfp4_harness_plugin.py"),
            OPERATIONS / "test_nvfp4_accuracy.py",
            OPERATIONS / "test_dequantize_nvfp4.py",
            OPERATIONS / "test_nvfp4_perf.py",
            OPERATIONS / "test_nvfp4_candidate_screen.py",
            OPERATIONS / "test_nvfp4_transpose_output.py",
            Path("ttnn/ttnn/_experimental/nvfp4.py"),
        ),
        op_code="Nvfp4DeviceOperation",
        compute_kernel="dequantize_fast",
        accuracy_tests=(
            OPERATIONS / "test_nvfp4_accuracy.py",
            OPERATIONS / "test_dequantize_nvfp4.py",
            OPERATIONS / "test_nvfp4_transpose_output.py",
        ),
        screen_tests=(OPERATIONS / "test_nvfp4_candidate_screen.py",),
        perf_tests=(OPERATIONS / "test_nvfp4_perf.py",),
        # The six intentional UINT8/BF16-DEST exclusions are the only allowed skips.
        full_accuracy_expected_skips=6,
        plugin="scripts.nvfp4_harness_plugin",
        locate_math_sweep=_anchored_sweep(
            "        MATH((sfpu::nvfp4_init_raw()));\n",
            "#pragma GCC unroll 4\n        for (uint32_t plane = 0; plane < planes; ++plane) {",
            "        MATH((sfpu::nvfp4_convert_end()));",
        ),
        accuracy_hashed_file=_NVFP4_KERNELS / "compute/nvfp4_sfpu.hpp",
        supports_bit_permutation=True,
    ),
    "mxfp8": FormatSpec(
        name="mxfp8",
        env_prefix="MXFP8",
        kernels=_MXFP8_KERNELS,
        manifest=".mxfp8-worktree.json",
        trusted_files=SHARED_TRUSTED_FILES
        + (
            OPERATIONS / "mxfp8_reference.py",
            OPERATIONS / "test_mxfp8_accuracy.py",
            OPERATIONS / "test_dequantize_mxfp8.py",
            OPERATIONS / "test_mxfp8_perf.py",
            OPERATIONS / "test_mxfp8_candidate_screen.py",
            Path("ttnn/ttnn/_experimental/mxfp8.py"),
        ),
        op_code="Mxfp8DeviceOperation",
        compute_kernel="dequantize",
        accuracy_tests=(
            OPERATIONS / "test_mxfp8_accuracy.py",
            OPERATIONS / "test_dequantize_mxfp8.py",
        ),
        screen_tests=(OPERATIONS / "test_mxfp8_candidate_screen.py",),
        perf_tests=(OPERATIONS / "test_mxfp8_perf.py",),
        full_accuracy_expected_skips=0,
        plugin="scripts.quant_harness_plugin",
        locate_math_sweep=_marked_sweep("// MXFP8_MATH_SWEEP_BEGIN", "// MXFP8_MATH_SWEEP_END"),
    ),
}


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


def prepare(spec, path, ref):
    path = path.resolve()
    if path.exists():
        raise EvaluationError(f"Worktree path already exists: {path}")
    base = git("rev-parse", ref)
    git("worktree", "add", "--detach", "--no-checkout", path, base)
    git("sparse-checkout", "set", "--no-cone", f"/{spec.kernels}/", cwd=path)
    git("checkout", base, cwd=path)
    # Share existing instruction files; never copy instructions, venv or binaries.
    for name in INSTRUCTION_FILES:
        if (REPO / name).exists():
            (path / name).symlink_to(REPO / name)
    write_json(path / spec.manifest, {"runtime_root": str(REPO), "base_revision": base, "format": spec.name})
    return {"worktree": str(path), "base_revision": base, "kernel_directory": str(path / spec.kernels)}


def validate_candidate(spec, candidate):
    candidate = candidate.resolve(strict=True)
    manifest_path = candidate / spec.manifest
    if not manifest_path.exists():
        raise EvaluationError(f"Candidate has no {spec.manifest}; was it prepared for --format {spec.name}?")
    manifest = read_json(manifest_path)
    if manifest.get("format", spec.name) != spec.name:
        raise EvaluationError(f"Candidate was prepared for {manifest['format']}, not {spec.name}")
    if Path(manifest["runtime_root"]).resolve() != REPO:
        raise EvaluationError("Candidate belongs to another runtime checkout")
    if (
        Path(git("rev-parse", "--git-common-dir", cwd=candidate)).resolve()
        != Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    ):
        raise EvaluationError("Candidate is not a worktree of the trusted repository")
    permutation = manifest.get("input_bit_permutation")
    if permutation is not None:
        if not spec.supports_bit_permutation:
            raise EvaluationError(f"input_bit_permutation is not supported for {spec.name}")
        if (
            not isinstance(permutation, list)
            or len(permutation) != 16
            or any(type(bit) is not int for bit in permutation)
            or sorted(permutation) != list(range(16))
        ):
            raise EvaluationError("input_bit_permutation must be a bijection of bits 0..15")
    base = manifest["base_revision"]
    changed = git("diff", "--name-only", base, cwd=candidate).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", cwd=candidate).splitlines()
    for name in changed + untracked:
        path = Path(name)
        if name == spec.manifest or name in INSTRUCTION_FILES:
            continue
        if not path.is_relative_to(spec.kernels) or path.suffix not in (".cpp", ".h", ".hpp"):
            raise EvaluationError(f"Candidate changed an out-of-scope file: {name}")
        if (candidate / path).is_symlink():
            raise EvaluationError(f"Candidate kernel cannot be a symlink: {name}")
    for path in (candidate / spec.kernels).rglob("*"):
        if path.is_symlink():
            raise EvaluationError(f"Candidate kernel cannot be a symlink: {path}")
    paths = sorted(path.relative_to(candidate) for path in (candidate / spec.kernels).rglob("*") if path.is_file())
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


def run_stage(spec, candidate, output, name, flags, tests, extra_env, timeout, pytest_args=()):
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
        QUANT_KERNEL_ROOT=str(candidate),
        TT_METAL_CACHE=str(output.parent / "jit-cache" / hashlib.sha256(str(candidate).encode()).hexdigest()[:16]),
        TT_METAL_PROFILER_DIR=str(stage / "profiler"),
        TT_METAL_LOGGER_LEVEL="critical",
        PYTHONPATH=str(REPO) + os.pathsep + str(REPO / "ttnn"),
        PATH=str(REPO / "python_env/bin") + os.pathsep + env.get("PATH", ""),
    )
    # Format tests locate the candidate's sources for their reports through this.
    env[spec.env("KERNEL_ROOT")] = str(candidate)
    env.update(extra_env)
    if spec.supports_bit_permutation:
        manifest_path = candidate / spec.manifest
        source = manifest_path if manifest_path.exists() else output / "result.json"
        env[spec.env("INPUT_BIT_PERMUTATION")] = json.dumps(read_json(source).get("input_bit_permutation"))
    env["QUANT_PYTEST_METRICS"] = str(stage / "pytest_metrics.json")
    junit = stage / "pytest.xml"
    command = [
        str(REPO / "scripts/run_safe_pytest.sh"),
        *flags,
        *[str(REPO / test) for test in tests],
        "-q",
        "-p",
        spec.plugin,
        f"--junitxml={junit}",
        *pytest_args,
    ]
    write_json(stage / "command.json", {"argv": command, "candidate": str(candidate), "timeout_seconds": timeout})
    print(f"{spec.name.upper()} {name}: {candidate.name}", flush=True)
    started = time.time()
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
    write_json(stage / "timing.json", stage_timing(stage, started, time.time()))
    if code:
        raise EvaluationError(f"Stage {name} failed with exit code {code}; inspect {stage / 'run.log'}")
    screen = read_json(output / "result.json").get("validation_level") == "screen"
    expected_skips = spec.full_accuracy_expected_skips if name == "accuracy" and not screen else 0
    validate_junit(junit, expected_skips=expected_skips)
    return stage


def stage_timing(stage, started, finished):
    events = dict(
        (name, float(value))
        for name, value in re.findall(r"SAFE_PYTEST_TIMING: (\w+) ([\d.]+)", (stage / "run.log").read_text())
    )

    def interval(first, last):
        return events[last] - events[first] if first in events and last in events else None

    result = dict(
        started_unix=started,
        finished_unix=finished,
        wall_seconds=finished - started,
        lock_wait_seconds=interval("lock_wait_start", "lock_acquired"),
        preflight_seconds=interval("lock_acquired", "pytest_start"),
        test_command_seconds=interval("pytest_start", "pytest_end"),
    )
    metrics = stage / "pytest_metrics.json"
    if metrics.exists():
        payload = read_json(metrics)
        result["collected"] = payload["collected"]
        result["collection_seconds"] = payload["collection_seconds"]
        reports = payload["reports"]
        result["test_counts"] = {
            "passed": sum(r["phase"] == "call" and r["outcome"] == "passed" for r in reports),
            "failed": sum(r["outcome"] == "failed" for r in reports),
            "skipped": sum(r["outcome"] == "skipped" for r in reports),
        }
        result["pytest_phase_seconds"] = {
            phase: sum(r["seconds"] for r in reports if r["phase"] == phase) for phase in ("setup", "call", "teardown")
        }
        if "pytest_start" in events:
            result["python_startup_seconds"] = payload["session_start_unix"] - events["pytest_start"]
        if "pytest_end" in events:
            result["post_session_seconds"] = events["pytest_end"] - payload["session_end_unix"]
    return result


def profile_metrics(spec, stage, repetitions=1, pytest_repeats=1):
    files = list((stage / "profiler").glob("reports/*/ops_perf_results*.csv"))
    if len(files) != 1:
        raise EvaluationError(f"Expected one fresh profiler report, found {len(files)} in {stage}")
    with files[0].open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["OP CODE"] == spec.op_code]
    metadata = read_json(stage / "workload.json")
    expected_calls = metadata["calls"] * pytest_repeats
    if len(rows) != expected_calls:
        raise EvaluationError(f"Expected {expected_calls} profiled operations, got {len(rows)}")
    if pytest_repeats != 1:
        metadata["pytest_repeats"] = pytest_repeats
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
    for index, row in enumerate(rows):
        if index % metadata["calls"] < metadata["warmup_calls"]:
            continue
        if int(row["CORE COUNT"]) != metadata["cores"]:
            raise EvaluationError("Profiled core count differs from workload metadata")
        divisor = metadata["maximum_output_tiles_per_core"]
        samples.append(float(row["DEVICE KERNEL DURATION [ns]"]) * mhz / 1000 / divisor)
        math_samples.append(float(row["DEVICE TRISC1 KERNEL DURATION [ns]"]) * mhz / 1000 / divisor / repetitions)
    if not all(math.isfinite(value) and value > 0 for value in samples + math_samples):
        raise EvaluationError("Invalid timing samples")
    return {
        "op_code": spec.op_code,
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


def save_disassembly(spec, candidate, output, stage, metrics):
    with Path(metrics["profile_csv"]).open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["OP CODE"] == spec.op_code]
    kernel_hashes = set()
    for row in rows:
        kernel_hashes.update(
            re.findall(rf"{re.escape(spec.compute_kernel)}/(\d+)/", row.get("COMPUTE KERNEL HASH", ""))
        )
    if len(kernel_hashes) != 1:
        raise EvaluationError("Expected one compute kernel hash in the performance report")
    cache = output.parent / "jit-cache" / hashlib.sha256(str(candidate).encode()).hexdigest()[:16]
    elfs = list(cache.glob(f"*/kernels/{spec.compute_kernel}/{kernel_hashes.pop()}/trisc1/trisc1.elf"))
    if not elfs:
        raise EvaluationError("Could not locate the measured math ELF")
    elf = max(elfs, key=lambda path: path.stat().st_mtime_ns)
    source = candidate / spec.kernels / f"compute/{spec.compute_kernel}.cpp"
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


def math_overlay(spec, candidate, output, repeats):
    """Repeat a complete packed-tile decode sweep, preserving cross-face register state."""
    overlay = output / "math-overlay"
    compute = overlay / spec.kernels / "compute"
    compute.mkdir(parents=True)
    name = f"{spec.compute_kernel}.cpp"
    source = (candidate / spec.kernels / "compute" / name).read_text()
    start, end = spec.locate_math_sweep(source)
    if end <= start:
        raise EvaluationError("Math diagnostic found an invalid decode sweep boundary")
    sweep = source[start:end]
    replacement = (
        f"#pragma GCC unroll 0\n        for (uint32_t repeat = 0; repeat < {repeats}; ++repeat) {{\n"
        f"{sweep}        }}\n"
    )
    (compute / name).write_text(source[:start] + replacement + source[end:])
    for path in (candidate / spec.kernels).rglob("*"):
        if path.is_file() and path.name != name:
            target = overlay / path.relative_to(candidate)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
    return overlay


def check_accuracy_report(spec, accuracy, kernel_hashes):
    if not accuracy["accuracy_gate"]["passed"]:
        raise EvaluationError("Numerical gate did not pass")
    if spec.accuracy_hashed_file is not None:
        if accuracy.get("kernel_sha256") != kernel_hashes[str(spec.accuracy_hashed_file)]:
            raise EvaluationError("Accuracy test reported a different kernel source than the candidate")
    elif accuracy.get("kernel_hashes") != kernel_hashes:
        raise EvaluationError("Accuracy test reported different kernel sources than the candidate")


def evaluate(spec, candidate, output, diagnostic, timeout, accuracy_only=False, screen=False, perf_repeats=1):
    candidate, manifest, before = validate_candidate(spec, candidate)
    if output.exists():
        raise EvaluationError(f"Result directory must be new: {output}")
    output.mkdir(parents=True)
    evaluator = hashes(REPO, spec.trusted_files)
    result = {
        "status": "running",
        "format": spec.name,
        "op_code": spec.op_code,
        "validation_level": "screen" if screen else "full",
        "input_bit_permutation": manifest.get("input_bit_permutation"),
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
        run_stage(
            spec,
            candidate,
            output,
            "accuracy",
            ["--dev"],
            spec.screen_tests if screen else spec.accuracy_tests,
            {spec.env("ACCURACY_REPORT"): str(output / "accuracy.json")},
            timeout,
        )
        accuracy = read_json(output / "accuracy.json")
        check_accuracy_report(spec, accuracy, before)
        result["accuracy_gate"] = accuracy["accuracy_gate"]
        if not accuracy_only:
            stage = run_stage(
                spec,
                candidate,
                output,
                "pipeline",
                ["--profile"],
                spec.perf_tests,
                {spec.env("BENCH_REPORT"): str(output / "pipeline/workload.json")},
                timeout,
                pytest_args=[f"--count={perf_repeats}"],
            )
            result["performance"] = profile_metrics(spec, stage, pytest_repeats=perf_repeats)
            save_disassembly(spec, candidate, output, stage, result["performance"])
            if diagnostic:
                overlay = math_overlay(spec, candidate, output, 128)
                stage = run_stage(
                    spec,
                    overlay,
                    output,
                    "math_isolate",
                    ["--profile"],
                    spec.perf_tests,
                    {spec.env("BENCH_REPORT"): str(output / "math_isolate/workload.json")},
                    timeout,
                    pytest_args=[f"--count={perf_repeats}"],
                )
                result["math_isolate"] = profile_metrics(spec, stage, repetitions=128, pytest_repeats=perf_repeats)
                save_disassembly(spec, overlay, output, stage, result["math_isolate"])
                result["math_isolate"][
                    "scope"
                ] = "Repeated complete DEST decode sweep; amortized pipeline, not standalone LLK MATH_ISOLATE"
        after_manifest = read_json(candidate / spec.manifest)
        if (
            before != validate_candidate(spec, candidate)[2]
            or evaluator != hashes(REPO, spec.trusted_files)
            or manifest.get("input_bit_permutation") != after_manifest.get("input_bit_permutation")
        ):
            raise EvaluationError("Candidate or trusted evaluator changed during evaluation")
        result["status"] = "screened" if screen else "passed"
    except (EvaluationError, OSError, ValueError, KeyError, ET.ParseError, subprocess.CalledProcessError) as exc:
        result.update(status="failed", error=str(exc))
    result["stages"] = {p.parent.name: read_json(p) for p in output.glob("*/timing.json")}
    result["finished_unix"] = time.time()
    result["elapsed_seconds"] = result["finished_unix"] - result["started_unix"]
    write_json(output / "result.json", result)
    return result


def compare(baseline_path, candidate_path):
    baseline, candidate = read_json(baseline_path), read_json(candidate_path)
    if any(
        result["status"] not in ("passed", "screened") or "performance" not in result
        for result in [baseline, candidate]
    ):
        raise EvaluationError("Comparison requires two passing accuracy+performance evaluations")
    # Results recorded before the multi-format harness carry no format and are NVFP4.
    if baseline.get("format", "nvfp4") != candidate.get("format", "nvfp4"):
        raise EvaluationError("Evaluations are for different formats")
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
        "promotion_requires": "Full regression validation and independent paired hardware remeasurement; comparison does not modify branches",
    }


def main(default_format=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if default_format is None:
        parser.add_argument("--format", choices=sorted(FORMATS), required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--path", type=Path, required=True)
    create.add_argument("--ref", default="HEAD")
    run = commands.add_parser("evaluate")
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--math-isolate", action="store_true")
    run.add_argument("--accuracy-only", action="store_true")
    run.add_argument(
        "--screen",
        action="store_true",
        help="Primary-configuration reconstruction corpus only; broad regression required before promotion",
    )
    run.add_argument("--timeout-seconds", type=int, default=600)
    run.add_argument(
        "--perf-repeats",
        type=int,
        default=1,
        choices=range(1, 33),
        help="Repeat each eight-call benchmark, discarding two warmups per group",
    )
    diff = commands.add_parser("compare")
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    spec = FORMATS[default_format or args.format]
    try:
        if args.command == "prepare":
            result = prepare(spec, args.path, args.ref)
        elif args.command == "compare":
            result = compare(args.baseline, args.candidate)
        else:
            result = evaluate(
                spec,
                args.candidate,
                args.output.resolve(),
                args.math_isolate,
                args.timeout_seconds,
                args.accuracy_only,
                args.screen,
                args.perf_repeats,
            )
    except (EvaluationError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{spec.name.upper()} harness: {exc}\n")
    print(json.dumps(result, indent=2))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
