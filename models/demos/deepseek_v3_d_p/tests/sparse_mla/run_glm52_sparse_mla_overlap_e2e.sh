#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_glm52_sparse_mla_overlap_e2e.sh <firmware-bundle>

Runs the exact Galaxy GLM-5.2 Long production proxy in serial-overlap-serial
order. Iteration zero of each process is compile warmup; the remaining nine
synchronized whole-transformer samples determine each end-to-end median.

Required environment:
  GLM52_HF_MODEL                 GLM-5.2-FP8 model directory
  TT_GLM52_PREFILL_TTNN_CACHE    Complete TTNN weight-cache root
  PREFILL_TRACE_DIR              GLM-5.2 golden trace root

Optional environment:
  QUALIFICATION_ROOT             Evidence root (default: generated/profiler/qualification)
  DS_E2E_MIN_IMPROVEMENT         Noise-floor minimum (default: 0.001 = 0.1%)
  DS_E2E_MAX_SERIAL_DRIFT        Maximum bracketing-serial drift (default: 0.01 = 1%)
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

[[ $# -eq 1 ]] || {
    usage >&2
    exit 2
}
FIRMWARE_BUNDLE=$1
[[ -n "$FIRMWARE_BUNDLE" ]] || die "firmware-bundle must be non-empty"

command -v git >/dev/null || die "git is required"
command -v readelf >/dev/null || die "readelf is required"
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
if [[ -x "$REPO_ROOT/python_env/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/python_env/bin/python"
else
    PYTHON_BIN=$(command -v python) || die "cannot find Python"
fi
SAFE_PYTEST="$REPO_ROOT/scripts/run_safe_pytest.sh"
[[ -x "$SAFE_PYTEST" ]] || die "missing executable safe pytest runner: $SAFE_PYTEST"
[[ -z $(git status --porcelain) ]] ||
    die "worktree must be clean so the evidence identifies the exact tested code"
[[ -z "${PYTHONOPTIMIZE:-}" || "${PYTHONOPTIMIZE}" == 0 ]] ||
    die "PYTHONOPTIMIZE must be unset or zero; optimized Python would disable test assertions"
unset PYTHONOPTIMIZE

MIN_IMPROVEMENT=${DS_E2E_MIN_IMPROVEMENT:-0.001}
MAX_SERIAL_DRIFT=${DS_E2E_MAX_SERIAL_DRIFT:-0.01}
[[ "$MIN_IMPROVEMENT" =~ ^(0|[0-9]+\.[0-9]+)$ ]] ||
    die "DS_E2E_MIN_IMPROVEMENT must be a non-negative decimal"
[[ "$MAX_SERIAL_DRIFT" =~ ^(0|[0-9]+\.[0-9]+)$ ]] ||
    die "DS_E2E_MAX_SERIAL_DRIFT must be a non-negative decimal"

for required_env in GLM52_HF_MODEL TT_GLM52_PREFILL_TTNN_CACHE PREFILL_TRACE_DIR; do
    [[ -n "${!required_env:-}" ]] || die "$required_env must be set"
done
[[ -f "$GLM52_HF_MODEL/config.json" ]] || die "missing $GLM52_HF_MODEL/config.json"
[[ -f "$GLM52_HF_MODEL/model.safetensors.index.json" ]] ||
    die "missing $GLM52_HF_MODEL/model.safetensors.index.json"
[[ -d "$TT_GLM52_PREFILL_TTNN_CACHE" ]] ||
    die "TTNN cache root is not a directory: $TT_GLM52_PREFILL_TTNN_CACHE"
[[ -d "$PREFILL_TRACE_DIR" ]] || die "trace root is not a directory: $PREFILL_TRACE_DIR"

HEAD_COMMIT=$(git rev-parse HEAD)
HEAD_SHORT=$(git rev-parse --short=12 HEAD)
HEAD_TIME=$(git show -s --format=%ct HEAD)
SO_PATH=$(realpath ttnn/ttnn/_ttnn.so)
[[ -f "$SO_PATH" ]] || die "missing built extension $SO_PATH"
SO_MTIME=$(stat -c %Y "$SO_PATH")
[[ "$SO_MTIME" -ge "$HEAD_TIME" ]] ||
    die "$SO_PATH predates HEAD; rebuild the exact commit before qualification"
LATEST_BUILD_INPUT_MTIME=$(
    "$PYTHON_BIN" - <<'PY'
import pathlib
import subprocess

extensions = {".S", ".c", ".cc", ".cmake", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".in", ".inl", ".ipp"}
paths = subprocess.check_output(["git", "ls-files", "-z", "--recurse-submodules"]).decode().split("\0")
inputs = [
    pathlib.Path(path)
    for path in paths
    if path
    and (
        pathlib.Path(path).suffix in extensions
        or pathlib.Path(path).name in {"CMakeLists.txt", "CMakePresets.json"}
    )
]
if not inputs:
    raise SystemExit("no tracked C/C++/CMake build inputs found")
latest_ns = max(path.stat().st_mtime_ns for path in inputs)
print((latest_ns + 999_999_999) // 1_000_000_000)
PY
)
[[ "$SO_MTIME" -ge "$LATEST_BUILD_INPUT_MTIME" ]] ||
    die "$SO_PATH is older than a tracked C/C++/CMake input; rebuild the clean checkout"
SO_BUILD_ID=$(readelf -n "$SO_PATH" 2>/dev/null | sed -n 's/.*Build ID: //p')
[[ -n "$SO_BUILD_ID" && "$SO_BUILD_ID" != *$'\n'* ]] || die "could not read one ELF build ID from $SO_PATH"
PYTHON_BUILD_RECORD=$(
    "$PYTHON_BIN" - <<'PY'
import os
import re
import subprocess
import sys

import ttnn

loaded = os.path.realpath(ttnn._ttnn.__file__)
output = subprocess.check_output(["readelf", "-n", loaded], text=True, stderr=subprocess.STDOUT)
build_ids = re.findall(r"Build ID: ([0-9a-fA-F]+)", output)
if len(build_ids) != 1:
    raise SystemExit(f"expected one ELF build ID in loaded extension {loaded}, found {build_ids}")
print(f"{loaded}\t{build_ids[0]}\t{sys.flags.optimize}\t{os.path.realpath(sys.executable)}")
PY
)
IFS=$'\t' read -r PYTHON_LOADED_SO_PATH PYTHON_OBSERVED_BUILD_ID PYTHON_OPTIMIZE_FLAG PYTHON_EXECUTABLE \
    <<<"$PYTHON_BUILD_RECORD"
[[ "$PYTHON_LOADED_SO_PATH" == "$SO_PATH" ]] ||
    die "Python loaded $PYTHON_LOADED_SO_PATH, expected $SO_PATH"
[[ "$PYTHON_OBSERVED_BUILD_ID" == "$SO_BUILD_ID" ]] ||
    die "Python loaded build $PYTHON_OBSERVED_BUILD_ID, expected $SO_BUILD_ID"
[[ "$PYTHON_OPTIMIZE_FLAG" == 0 ]] || die "Python optimize flag is $PYTHON_OPTIMIZE_FLAG, expected zero"
[[ -n "$PYTHON_EXECUTABLE" ]] || die "could not observe the Python executable"

ACTUAL_DEVICES=$(
    "$PYTHON_BIN" - <<'PY'
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices

print(detect_num_devices())
PY
)
[[ "$ACTUAL_DEVICES" -eq 32 ]] || die "Galaxy end-to-end run requires 32 devices, detected $ACTUAL_DEVICES"

FW_FILES=(/sys/class/tenstorrent/tenstorrent\!*/tt_fw_bundle_ver)
[[ -e "${FW_FILES[0]}" ]] || die "cannot read installed firmware bundle from sysfs"
[[ "${#FW_FILES[@]}" -eq 32 ]] || die "expected 32 firmware records, found ${#FW_FILES[@]}"
INSTALLED_FW_VALUES=()
for fw_file in "${FW_FILES[@]}"; do
    fw_value=$(tr -d '[:space:]' <"$fw_file") || die "cannot read firmware bundle: $fw_file"
    [[ -n "$fw_value" ]] || die "empty firmware bundle: $fw_file"
    INSTALLED_FW_VALUES+=("$fw_value")
done
[[ "${#INSTALLED_FW_VALUES[@]}" -eq 32 ]] ||
    die "read ${#INSTALLED_FW_VALUES[@]} firmware bundles, expected 32"
INSTALLED_FIRMWARE=${INSTALLED_FW_VALUES[0]}
for fw_value in "${INSTALLED_FW_VALUES[@]}"; do
    [[ "$fw_value" == "$INSTALLED_FIRMWARE" ]] ||
        die "devices do not report one firmware bundle: ${INSTALLED_FW_VALUES[*]}"
done
FIRMWARE_MATCH=$(
    "$PYTHON_BIN" - "$FIRMWARE_BUNDLE" "$INSTALLED_FIRMWARE" <<'PY'
import sys


def canonical(version):
    fields = version.split(".")
    if len(fields) == 4 and fields[-1] == "0":
        fields.pop()
    return ".".join(fields)


print(int(canonical(sys.argv[1]) == canonical(sys.argv[2])))
PY
)
[[ "$FIRMWARE_MATCH" -eq 1 ]] ||
    die "requested firmware '$FIRMWARE_BUNDLE' does not match installed '$INSTALLED_FIRMWARE'"

RESOLVED_TRACE=$(
    "$PYTHON_BIN" - "$PREFILL_TRACE_DIR" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
if (root / "metadata.json").is_file():
    print(root)
else:
    matches = sorted(path for path in root.iterdir() if path.is_dir() and (path / "metadata.json").is_file())
    if len(matches) != 1:
        raise SystemExit(f"expected metadata.json at {root} or in one unique child, found {len(matches)}")
    print(matches[0])
PY
) || die "could not resolve a unique GLM trace from $PREFILL_TRACE_DIR"

EXPECTED_CACHE_DIR="$TT_GLM52_PREFILL_TTNN_CACHE/glm_5_2_bh_32dev/8x4"
[[ -d "$EXPECTED_CACHE_DIR" ]] ||
    die "missing expected 32-device 8x4 GLM-5.2 cache directory: $EXPECTED_CACHE_DIR"
compgen -G "$EXPECTED_CACHE_DIR/*.tensorbin" >/dev/null ||
    die "32-device 8x4 GLM-5.2 cache contains no tensorbin files: $EXPECTED_CACHE_DIR"

UTC_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SAFE_FIRMWARE=$(printf '%s' "$FIRMWARE_BUNDLE" | tr -c 'A-Za-z0-9._-' '_')
RUN_ID="galaxy_glm52_long_${HEAD_SHORT}_${UTC_STAMP}_$$"
QUALIFICATION_ROOT=${QUALIFICATION_ROOT:-generated/profiler/qualification}
EVIDENCE_DIR="$QUALIFICATION_ROOT/galaxy_80_40_e2e/${HEAD_SHORT}_${SAFE_FIRMWARE}_${UTC_STAMP}_$$"
mkdir -p "$(dirname "$EVIDENCE_DIR")"
mkdir "$EVIDENCE_DIR"
STATUS_FILE="$EVIDENCE_DIR/status"
echo running >"$STATUS_FILE"

RUN_PASSED=0
on_exit() {
    local exit_code=$?
    if [[ "$RUN_PASSED" -eq 1 && "$exit_code" -eq 0 ]]; then
        echo passed >"$STATUS_FILE"
    else
        echo failed >"$STATUS_FILE"
    fi
}
trap on_exit EXIT

assert_junit_exact() {
    local xml=$1
    "$PYTHON_BIN" - "$xml" <<'PY'
import pathlib
import sys
import xml.etree.ElementTree as ET

path = pathlib.Path(sys.argv[1])
root = ET.parse(path).getroot()
cases = root.findall(".//testcase")
failures = root.findall(".//failure")
errors = root.findall(".//error")
skipped = root.findall(".//skipped")
if len(cases) != 1:
    raise SystemExit(f"{path}: expected exactly one test, found {len(cases)}")
if failures or errors or skipped:
    raise SystemExit(f"{path}: failures={len(failures)} errors={len(errors)} skipped={len(skipped)}")
expected_suffix = (
    "test_glm_prefill_transformer_chunked_no_pcc"
    "[blackhole-glm52-mesh-8x4-L78-preload95k-chunks1-ten_iters]"
)
if cases[0].attrib.get("name") != expected_suffix:
    raise SystemExit(f"{path}: unexpected testcase {cases[0].attrib}")
PY
}

TEST_NODE="models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc[blackhole-glm52-mesh-8x4-L78-preload95k-chunks1-ten_iters]"

# Ambient profiling switches add synchronization or extra work and invalidate the production comparison.
unset TT_PREFILL_PROFILE_WARMUP TT_PREFILL_BLOCK_TIMING
export LOGURU_LEVEL=INFO

echo "Galaxy GLM-5.2 Long end-to-end overlap qualification"
echo "commit:       $HEAD_COMMIT"
echo "firmware:     $INSTALLED_FIRMWARE"
echo "ELF build id: $SO_BUILD_ID"
echo "run id:       $RUN_ID"
echo "trace:        $RESOLVED_TRACE"
echo "weight cache: $EXPECTED_CACHE_DIR"
echo "evidence:     $EVIDENCE_DIR"

SERIAL_BEFORE_LOG="$EVIDENCE_DIR/serial_before.log"
SERIAL_BEFORE_XML="$EVIDENCE_DIR/serial_before.xml"
env -u TT_SPARSE_MLA_OVERLAP_PROFILE \
    TT_PREFILL_EXPECT_SPARSE_MLA_OVERLAP_PROFILE=off \
    "$SAFE_PYTEST" "$TEST_NODE" -xvs --junitxml="$SERIAL_BEFORE_XML" 2>&1 | tee "$SERIAL_BEFORE_LOG"
assert_junit_exact "$SERIAL_BEFORE_XML"

OVERLAP_LOG="$EVIDENCE_DIR/overlap.log"
OVERLAP_XML="$EVIDENCE_DIR/overlap.xml"
TT_SPARSE_MLA_OVERLAP_PROFILE=galaxy_80_40 \
    TT_PREFILL_EXPECT_SPARSE_MLA_OVERLAP_PROFILE=galaxy_80_40 \
    "$SAFE_PYTEST" "$TEST_NODE" -xvs --junitxml="$OVERLAP_XML" 2>&1 | tee "$OVERLAP_LOG"
assert_junit_exact "$OVERLAP_XML"

SERIAL_AFTER_LOG="$EVIDENCE_DIR/serial_after.log"
SERIAL_AFTER_XML="$EVIDENCE_DIR/serial_after.xml"
env -u TT_SPARSE_MLA_OVERLAP_PROFILE \
    TT_PREFILL_EXPECT_SPARSE_MLA_OVERLAP_PROFILE=off \
    "$SAFE_PYTEST" "$TEST_NODE" -xvs --junitxml="$SERIAL_AFTER_XML" 2>&1 | tee "$SERIAL_AFTER_LOG"
assert_junit_exact "$SERIAL_AFTER_XML"

"$PYTHON_BIN" - \
    "$SERIAL_BEFORE_LOG" "$OVERLAP_LOG" "$SERIAL_AFTER_LOG" "$EVIDENCE_DIR/summary.json" \
    "$HEAD_COMMIT" "$FIRMWARE_BUNDLE" "$INSTALLED_FIRMWARE" "$RUN_ID" "$SO_MTIME" "$SO_BUILD_ID" \
    "$PYTHON_LOADED_SO_PATH" "$PYTHON_OBSERVED_BUILD_ID" "$PYTHON_OPTIMIZE_FLAG" "$PYTHON_EXECUTABLE" \
    "$LATEST_BUILD_INPUT_MTIME" "$ACTUAL_DEVICES" "$MIN_IMPROVEMENT" "$MAX_SERIAL_DRIFT" \
    "$GLM52_HF_MODEL" "$EXPECTED_CACHE_DIR" "$RESOLVED_TRACE" <<'PY'
import json
import pathlib
import re
import statistics
import sys

(
    serial_before_log,
    overlap_log,
    serial_after_log,
    output_path,
    commit,
    requested_firmware,
    installed_firmware,
    run_id,
    so_mtime,
    so_build_id,
    loaded_so_path,
    observed_build_id,
    python_optimize,
    python_executable,
    latest_build_input_mtime,
    device_count,
    minimum_improvement_floor,
    maximum_serial_drift,
    model_path,
    weight_cache_path,
    trace_path,
) = sys.argv[1:]

device_count = int(device_count)
python_optimize = int(python_optimize)
minimum_improvement_floor = float(minimum_improvement_floor)
maximum_serial_drift = float(maximum_serial_drift)

pattern = re.compile(r"iter ([0-9]+) done \(1 chunks\) in ([0-9]+(?:\.[0-9]+)?) seconds")
profile_pattern = re.compile(
    r"SPARSE_MLA_PROFILE_ASSERT variant=([^ ]+) expected=([^ ]+) active_layer_count=([0-9]+) "
    r"eligible_layer_count=([0-9]+) worker_grid=([0-9]+)x([0-9]+) mesh=([0-9]+)x([0-9]+) "
    r"topk_cores=([0-9]+) gather_cores=([0-9]+)"
)
execution_pattern = re.compile(
    r"SPARSE_MLA_EXECUTION_ASSERT expected=([^ ]+) eligible_layer_count=([0-9]+) "
    r"calls_per_layer=([0-9]+) serial_calls=([0-9]+) overlap_calls=([0-9]+)"
)
workload_pattern = re.compile(
    r"chunked transformer \(no-PCC\): num_layers=([0-9]+) mesh=\[([0-9]+), ([0-9]+)\] "
    r"n_chunks=([0-9]+) preload_isl=([0-9]+) total_len=([0-9]+) cache=([0-9]+) "
    r"chunk=([0-9]+) num_iters=([0-9]+)"
)


def parse_samples(path: str) -> list[float]:
    matches = pattern.findall(pathlib.Path(path).read_text())
    iterations = [int(iteration) for iteration, _ in matches]
    if iterations != list(range(10)):
        raise SystemExit(f"{path}: expected exactly iterations 0..9, got {iterations}")
    values = [float(seconds) for _, seconds in matches]
    if not all(value > 0 for value in values):
        raise SystemExit(f"{path}: non-positive sample in {values}")
    return values


def parse_profile(path: str) -> dict:
    matches = profile_pattern.findall(pathlib.Path(path).read_text())
    if len(matches) != 1:
        raise SystemExit(f"{path}: expected exactly one profile assertion, got {matches}")
    fields = matches[0]
    return {
        "variant": fields[0],
        "profile": fields[1],
        "active_layers": int(fields[2]),
        "eligible_layers": int(fields[3]),
        "worker_grid": [int(fields[4]), int(fields[5])],
        "mesh_shape": [int(fields[6]), int(fields[7])],
        "topk_owned_cores": int(fields[8]),
        "gather_owned_cores": int(fields[9]),
    }


def parse_execution(path: str) -> dict:
    matches = execution_pattern.findall(pathlib.Path(path).read_text())
    if len(matches) != 1:
        raise SystemExit(f"{path}: expected exactly one execution assertion, got {matches}")
    fields = matches[0]
    return {
        "profile": fields[0],
        "eligible_layers": int(fields[1]),
        "calls_per_layer": int(fields[2]),
        "serial_calls": int(fields[3]),
        "overlap_calls": int(fields[4]),
    }


def parse_workload(path: str) -> dict:
    matches = workload_pattern.findall(pathlib.Path(path).read_text())
    if len(matches) != 1:
        raise SystemExit(f"{path}: expected exactly one workload declaration, got {matches}")
    values = [int(value) for value in matches[0]]
    return {
        "layers": values[0],
        "mesh_shape": values[1:3],
        "chunks_per_iteration": values[3],
        "preloaded_tokens": values[4],
        "total_tokens": values[5],
        "cache_tokens": values[6],
        "chunk_tokens": values[7],
        "iterations": values[8],
    }


expected_serial_profile = {
    "variant": "glm_5_2",
    "profile": "off",
    "active_layers": 0,
    "eligible_layers": 21,
    "worker_grid": [12, 10],
    "mesh_shape": [8, 4],
    "topk_owned_cores": 120,
    "gather_owned_cores": 0,
}
expected_overlap_profile = {
    **expected_serial_profile,
    "profile": "galaxy_80_40",
    "active_layers": 21,
    "topk_owned_cores": 80,
    "gather_owned_cores": 40,
}
expected_serial_execution = {
    "profile": "off",
    "eligible_layers": 21,
    "calls_per_layer": 10,
    "serial_calls": 210,
    "overlap_calls": 0,
}
expected_overlap_execution = {
    **expected_serial_execution,
    "profile": "galaxy_80_40",
    "serial_calls": 0,
    "overlap_calls": 210,
}

serial_before_profile = parse_profile(serial_before_log)
overlap_profile = parse_profile(overlap_log)
serial_after_profile = parse_profile(serial_after_log)
if serial_before_profile != expected_serial_profile or serial_after_profile != expected_serial_profile:
    raise SystemExit(
        f"serial runtime profile mismatch: before={serial_before_profile}, after={serial_after_profile}"
    )
if overlap_profile != expected_overlap_profile:
    raise SystemExit(f"overlap runtime profile mismatch: {overlap_profile}")
serial_before_execution = parse_execution(serial_before_log)
overlap_execution = parse_execution(overlap_log)
serial_after_execution = parse_execution(serial_after_log)
if serial_before_execution != expected_serial_execution:
    raise SystemExit(f"serial-before execution mismatch: {serial_before_execution}")
if overlap_execution != expected_overlap_execution:
    raise SystemExit(f"overlap execution mismatch: {overlap_execution}")
if serial_after_execution != expected_serial_execution:
    raise SystemExit(f"serial-after execution mismatch: {serial_after_execution}")

workloads = [parse_workload(path) for path in (serial_before_log, overlap_log, serial_after_log)]
if not workloads[0] == workloads[1] == workloads[2]:
    raise SystemExit(f"serial/overlap workload mismatch: {workloads}")
workload = workloads[0]
expected_workload = {
    "layers": 78,
    "mesh_shape": [8, 4],
    "chunks_per_iteration": 1,
    "preloaded_tokens": 97280,
    "total_tokens": 102400,
    "cache_tokens": 102400,
    "chunk_tokens": 5120,
    "iterations": 10,
}
if workload != expected_workload:
    raise SystemExit(f"unexpected Long workload: {workload}")
if device_count != 32:
    raise SystemExit(f"expected 32 detected devices, got {device_count}")
if observed_build_id != so_build_id:
    raise SystemExit(f"loaded ELF build ID {observed_build_id} does not match shell-observed {so_build_id}")
if python_optimize != 0:
    raise SystemExit(f"Python optimize flag is {python_optimize}, expected zero")

serial_before_all = parse_samples(serial_before_log)
overlap_all = parse_samples(overlap_log)
serial_after_all = parse_samples(serial_after_log)
serial_before_samples = serial_before_all[1:]
overlap_samples = overlap_all[1:]
serial_after_samples = serial_after_all[1:]
serial_before_median = statistics.median(serial_before_samples)
serial_after_median = statistics.median(serial_after_samples)
serial_baseline_median = statistics.mean([serial_before_median, serial_after_median])
overlap_median = statistics.median(overlap_samples)
serial_drift = abs(serial_after_median - serial_before_median) / serial_baseline_median
serial_before_stdev = statistics.stdev(serial_before_samples)
overlap_stdev = statistics.stdev(overlap_samples)
serial_after_stdev = statistics.stdev(serial_after_samples)
pooled_stdev = statistics.mean(
    [serial_before_stdev**2, overlap_stdev**2, serial_after_stdev**2]
) ** 0.5
pooled_relative_stdev = pooled_stdev / serial_baseline_median
noise_aware_minimum = max(minimum_improvement_floor, 2.0 * pooled_relative_stdev)
improvement = 1.0 - overlap_median / serial_baseline_median
serial_drift_passed = serial_drift <= maximum_serial_drift
improvement_passed = improvement > noise_aware_minimum


def sample_summary(all_samples, post_warmup_samples):
    return {
        "all_iteration_seconds": all_samples,
        "post_warmup_seconds": post_warmup_samples,
        "median_seconds": statistics.median(post_warmup_samples),
        "stdev_seconds": statistics.stdev(post_warmup_samples),
        "minimum_seconds": min(post_warmup_samples),
        "maximum_seconds": max(post_warmup_samples),
    }

summary = {
    "schema_version": 2,
    "measurement_scope": "host_wall_clock_sync_bracketed_whole_transformer_chunk",
    "device_time_evidence": False,
    "candidate_profile": overlap_profile["profile"],
    "commit": commit,
    "firmware_bundle_requested": requested_firmware,
    "firmware_bundle_observed": installed_firmware,
    "qualification_run_id": run_id,
    "build": {
        "loaded_so_path": loaded_so_path,
        "so_mtime_epoch_seconds": int(so_mtime),
        "latest_tracked_build_input_mtime_epoch_seconds": int(latest_build_input_mtime),
        "observed_elf_build_id": observed_build_id,
        "shell_expected_elf_build_id": so_build_id,
    },
    "runtime": {"python_executable": python_executable, "python_optimize": python_optimize},
    "device_count": device_count,
    "mesh_shape": overlap_profile["mesh_shape"],
    "worker_grid": overlap_profile["worker_grid"],
    "serial_runtime_profile": serial_before_profile,
    "overlap_runtime_profile": overlap_profile,
    "serial_before_execution": serial_before_execution,
    "overlap_execution": overlap_execution,
    "serial_after_execution": serial_after_execution,
    "workload": {**workload, "variant": overlap_profile["variant"], "warmup_iterations_omitted": [0]},
    "assets": {
        "model_path": str(pathlib.Path(model_path).resolve()),
        "weight_cache_path": str(pathlib.Path(weight_cache_path).resolve()),
        "trace_path": str(pathlib.Path(trace_path).resolve()),
    },
    "test_node": (
        "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::"
        "test_glm_prefill_transformer_chunked_no_pcc"
        "[blackhole-glm52-mesh-8x4-L78-preload95k-chunks1-ten_iters]"
    ),
    "serial_before": sample_summary(serial_before_all, serial_before_samples),
    "overlap": sample_summary(overlap_all, overlap_samples),
    "serial_after": sample_summary(serial_after_all, serial_after_samples),
    "serial_bracket": {
        "baseline_median_seconds": serial_baseline_median,
        "relative_drift": serial_drift,
        "maximum_allowed_relative_drift": maximum_serial_drift,
        "passed": serial_drift_passed,
    },
    "whole_transformer_improvement": improvement,
    "pooled_relative_stdev": pooled_relative_stdev,
    "minimum_improvement_floor": minimum_improvement_floor,
    "noise_aware_minimum_required_improvement": noise_aware_minimum,
    "improvement_gate_passed": improvement_passed,
    "passed": serial_drift_passed and improvement_passed,
}
pathlib.Path(output_path).write_text(json.dumps(summary, indent=2) + "\n")
print(
    f"Galaxy GLM-5.2 Long end-to-end: serial-before={serial_before_median:.3f}s "
    f"overlap={overlap_median:.3f}s serial-after={serial_after_median:.3f}s "
    f"win={improvement:.2%} required>{noise_aware_minimum:.2%} drift={serial_drift:.2%}"
)
if not serial_drift_passed:
    raise SystemExit(
        f"bracketing serial medians drifted {serial_drift:.3%}, above {maximum_serial_drift:.3%}"
    )
if not improvement_passed:
    raise SystemExit(
        f"end-to-end improvement {improvement:.3%} did not exceed noise-aware minimum "
        f"{noise_aware_minimum:.3%}"
    )
PY

RUN_PASSED=1
echo "Galaxy GLM-5.2 Long end-to-end qualification passed: $EVIDENCE_DIR/summary.json"
