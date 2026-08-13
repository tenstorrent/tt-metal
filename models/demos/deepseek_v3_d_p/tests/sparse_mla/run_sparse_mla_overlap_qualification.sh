#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_sparse_mla_overlap_qualification.sh <loudbox_80_40|galaxy_80_40> <firmware-bundle>

Runs hardware correctness, Tensix-Watcher independent-progress coverage,
and the complete {warm,cold,long} x {BF16,scaled-FP8} serial/overlap matrix.
Every perf result must be a fresh schema-v8 manifest tied to the clean HEAD.

Environment:
  DS_PERF_OVERLAP_SAMPLES  Device samples for warm/long (default: 7)
  QUALIFICATION_ROOT       Evidence archive root (default: generated/profiler/qualification)
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

[[ $# -eq 2 ]] || {
    usage >&2
    exit 2
}

PROFILE=$1
REQUESTED_FIRMWARE_BUNDLE=$2
case "$PROFILE" in
    loudbox_80_40)
        EXPECTED_DEVICES=8
        EXPECTED_SP=2
        ;;
    galaxy_80_40)
        EXPECTED_DEVICES=32
        EXPECTED_SP=8
        ;;
    *)
        die "unsupported profile '$PROFILE'"
        ;;
esac
[[ -n "$REQUESTED_FIRMWARE_BUNDLE" ]] || die "firmware-bundle must be non-empty"

SAMPLES=${DS_PERF_OVERLAP_SAMPLES:-7}
[[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "DS_PERF_OVERLAP_SAMPLES must be a positive integer"
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
    die "worktree must be clean so manifests identify the exact tested code"
[[ -z "${PYTHONOPTIMIZE:-}" || "${PYTHONOPTIMIZE}" == 0 ]] ||
    die "PYTHONOPTIMIZE must be unset or zero; optimized Python would disable qualification assertions"
unset PYTHONOPTIMIZE

# Pin the production workload and prevent ambient report/test selectors from changing the matrix.
export DS_PERF_CACHE=51200
export DS_PERF_CHUNK=5120
export DS_PERF_LONG_CACHE=512000
unset DS_PERF_CSV DS_PERF_SCENARIO DS_PERF_ATTN_MODE

HEAD_COMMIT=$(git rev-parse HEAD)
HEAD_SHORT=$(git rev-parse --short=12 HEAD)
HEAD_TIME=$(git show -s --format=%ct HEAD)
UTC_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="${PROFILE}_${HEAD_SHORT}_${UTC_STAMP}_$$"
SAFE_FIRMWARE=$(printf '%s' "$REQUESTED_FIRMWARE_BUNDLE" | tr -c 'A-Za-z0-9._-' '_')
QUALIFICATION_ROOT=${QUALIFICATION_ROOT:-generated/profiler/qualification}
EVIDENCE_DIR="$QUALIFICATION_ROOT/$PROFILE/${HEAD_SHORT}_${SAFE_FIRMWARE}_${UTC_STAMP}_$$"
mkdir -p "$(dirname "$EVIDENCE_DIR")"
mkdir "$EVIDENCE_DIR"
mkdir "$EVIDENCE_DIR/cases"
STATUS_FILE="$EVIDENCE_DIR/status"
echo running >"$STATUS_FILE"

TMP_DIR=$(mktemp -d)
RUN_PASSED=0
on_exit() {
    local exit_code=$?
    if [[ "$RUN_PASSED" -eq 1 && "$exit_code" -eq 0 ]]; then
        echo passed >"$STATUS_FILE"
    else
        echo failed >"$STATUS_FILE"
    fi
    rm -rf "$TMP_DIR"
}
trap on_exit EXIT

assert_junit_exact() {
    local xml=$1
    local expected_tests=$2
    "$PYTHON_BIN" - "$xml" "$expected_tests" <<'PY'
import pathlib
import sys
import xml.etree.ElementTree as ET

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
root = ET.parse(path).getroot()
cases = root.findall(".//testcase")
failures = root.findall(".//failure")
errors = root.findall(".//error")
skipped = root.findall(".//skipped")
assert len(cases) == expected, f"{path}: expected exactly {expected} tests, found {len(cases)}"
assert not failures and not errors and not skipped, (
    f"{path}: failures={len(failures)} errors={len(errors)} skipped={len(skipped)}"
)
PY
}

validate_manifest() {
    local manifest=$1
    local enabled=$2
    local cache_format=$3
    local scenario=$4
    local serial_baseline_ns=$5
    local serial_branch_baseline_ns=$6
    "$PYTHON_BIN" - "$manifest" "$enabled" "$cache_format" "$scenario" "$PROFILE" \
        "$INSTALLED_FIRMWARE_BUNDLE" "$HEAD_COMMIT" "$EXPECTED_DEVICES" "$EXPECTED_SP" "$RUN_ID" \
        "$HEAD_TIME" "$SO_MTIME" "$SO_PATH" "$SO_BUILD_ID" "$LATEST_BUILD_INPUT_MTIME" \
        "$SAMPLES" "$serial_baseline_ns" "$serial_branch_baseline_ns" <<'PY'
import datetime
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_enabled = bool(int(sys.argv[2]))
expected_format = sys.argv[3]
expected_scenario = sys.argv[4]
expected_profile = sys.argv[5]
expected_firmware = sys.argv[6]
expected_commit = sys.argv[7]
expected_devices = int(sys.argv[8])
expected_sp = int(sys.argv[9])
expected_run_id = sys.argv[10]
head_time = int(sys.argv[11])
expected_so_mtime = int(sys.argv[12])
expected_loaded_so_path = sys.argv[13]
expected_build_id = sys.argv[14]
expected_build_input_mtime = int(sys.argv[15])
requested_samples = int(sys.argv[16])
expected_serial_baseline = float(sys.argv[17])
expected_serial_branch_baseline = float(sys.argv[18])

data = json.loads(path.read_text())
overlap = data["sparse_mla_overlap"]
assert data["schema_version"] == 8
assert data["commit"] == expected_commit
assert data["qualification_run_id"] == expected_run_id
assert data["device"]["num_devices"] == expected_devices
assert data["device"]["firmware_bundle"] == expected_firmware
so_mtime = datetime.datetime.fromisoformat(data["build"]["so_mtime"]).timestamp()
assert abs(so_mtime - expected_so_mtime) < 1.0
assert so_mtime >= head_time, "_ttnn.so predates the qualified commit"
assert data["build"]["loaded_so_path"] == expected_loaded_so_path
assert data["build"]["observed_elf_build_id"] == expected_build_id
assert data["build"]["qualification_expected_elf_build_id"] == expected_build_id
assert int(data["build"]["latest_tracked_build_input_mtime_epoch_seconds"]) == expected_build_input_mtime
assert int(data["build"]["qualification_expected_build_input_mtime_epoch_seconds"]) == expected_build_input_mtime
assert expected_so_mtime >= expected_build_input_mtime
assert data["runtime"]["python_optimize"] == 0
assert data["runtime"]["python_executable"]
assert overlap["profile"] == expected_profile
assert overlap["enabled"] is expected_enabled
assert overlap["scenario"] == expected_scenario
assert overlap["kv_cache_format"] == expected_format
expected_chunk = 5120 * expected_sp // 8
expected_cache = (512000 if expected_scenario == "long" else 51200) * expected_sp // 8
assert overlap["chunk_tokens"] == expected_chunk
assert overlap["cache_tokens"] == expected_cache
assert overlap["total_tokens"] == expected_cache + expected_chunk
assert overlap["topk_owned_cores"] == 80
assert overlap["gather_owned_cores"] == 40
assert overlap["worker_grid"] == [12, 10]
assert overlap["timing_source"] == "realtime_raw_device_ticks_per_chip_span"
assert overlap["measured_scenario_ns"] > 0
assert overlap["measured_branch_serialized_ns"] > 0
expected_samples = (
    overlap["cache_tokens"] // overlap["chunk_tokens"] + 1 if expected_scenario == "cold" else requested_samples
)
assert len(overlap["timing_samples_ns"]) == expected_samples

if expected_enabled:
    gather_budget = overlap["gather_scheduler_core_budget"]
    selected_workers = overlap["selected_workers_per_direction"]
    expected_budget = overlap["num_links"] * 2 * (selected_workers + (1 if selected_workers > 1 else 0))
    assert gather_budget == expected_budget
    assert gather_budget <= overlap["gather_owned_cores"]
    assert overlap["whole_forward_gate_applied"] is True
    assert overlap["branch_gate_applied"] is True
    assert overlap["whole_forward_improvement"] > 0
    assert overlap["branch_union_improvement"] >= 0.10
    assert overlap["measured_manager_boundary_gap_ns"] is not None
    assert math.isclose(overlap["serial_baseline_ns"], expected_serial_baseline, rel_tol=0, abs_tol=0.5)
    assert math.isclose(
        overlap["serial_branch_baseline_ns"], expected_serial_branch_baseline, rel_tol=0, abs_tol=0.5
    )
PY
}

ACTUAL_DEVICES=$(
    "$PYTHON_BIN" - <<'PY'
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices

print(detect_num_devices())
PY
)
[[ "$ACTUAL_DEVICES" -eq "$EXPECTED_DEVICES" ]] ||
    die "profile $PROFILE requires $EXPECTED_DEVICES devices, detected $ACTUAL_DEVICES"

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

FW_FILES=(/sys/class/tenstorrent/tenstorrent\!*/tt_fw_bundle_ver)
[[ -e "${FW_FILES[0]}" ]] || die "cannot read installed firmware bundle from sysfs"
[[ "${#FW_FILES[@]}" -eq "$EXPECTED_DEVICES" ]] ||
    die "expected $EXPECTED_DEVICES firmware records, found ${#FW_FILES[@]}"
INSTALLED_FW_VALUES=()
for fw_file in "${FW_FILES[@]}"; do
    fw_value=$(tr -d '[:space:]' <"$fw_file") || die "cannot read firmware bundle: $fw_file"
    [[ -n "$fw_value" ]] || die "empty firmware bundle: $fw_file"
    INSTALLED_FW_VALUES+=("$fw_value")
done
[[ "${#INSTALLED_FW_VALUES[@]}" -eq "$EXPECTED_DEVICES" ]] ||
    die "read ${#INSTALLED_FW_VALUES[@]} firmware bundles, expected $EXPECTED_DEVICES"
INSTALLED_FIRMWARE_BUNDLE=${INSTALLED_FW_VALUES[0]}
for fw_value in "${INSTALLED_FW_VALUES[@]}"; do
    [[ "$fw_value" == "$INSTALLED_FIRMWARE_BUNDLE" ]] ||
        die "devices do not report one firmware bundle: ${INSTALLED_FW_VALUES[*]}"
done
FIRMWARE_MATCH=$(
    "$PYTHON_BIN" - "$REQUESTED_FIRMWARE_BUNDLE" "$INSTALLED_FIRMWARE_BUNDLE" <<'PY'
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
    die "requested firmware '$REQUESTED_FIRMWARE_BUNDLE' does not match installed '$INSTALLED_FIRMWARE_BUNDLE'"
FIRMWARE_BUNDLE=$INSTALLED_FIRMWARE_BUNDLE

echo "qualification profile: $PROFILE"
echo "commit:               $HEAD_COMMIT"
echo "firmware:             $INSTALLED_FIRMWARE_BUNDLE"
echo "ELF build id:         $SO_BUILD_ID"
echo "run id:               $RUN_ID"
echo "evidence:             $EVIDENCE_DIR"

CORRECTNESS_XML="$EVIDENCE_DIR/correctness.xml"
DS_PERF_QUALIFICATION_RUN_ID="$RUN_ID" DS_PERF_FIRMWARE_VERSION="$FIRMWARE_BUNDLE" \
    "$SAFE_PYTEST" --run-all \
    models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_cache.py::test_glm52_sparse_mla_overlap_matches_serial \
    models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_cache.py::test_glm52_sparse_mla_overlap_growing_prefix_cache_and_lifetime \
    -q -k "$PROFILE" -s --junitxml="$CORRECTNESS_XML"
assert_junit_exact "$CORRECTNESS_XML" 3

WATCHER_XML="$EVIDENCE_DIR/watcher.xml"
WATCHER_MARKER="$TMP_DIR/watcher.marker"
touch "$WATCHER_MARKER"
TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
    DS_PERF_QUALIFICATION_RUN_ID="$RUN_ID" DS_PERF_FIRMWARE_VERSION="$FIRMWARE_BUNDLE" \
    "$SAFE_PYTEST" --run-all \
    tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py::test_high_bw_all_gather_external_semaphore_independent_progress_with_topk \
    tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py::test_high_bw_all_gather_external_semaphore_validation \
    -q -s --junitxml="$WATCHER_XML"
assert_junit_exact "$WATCHER_XML" 2
WATCHER_LOG=generated/watcher/watcher.log
[[ -f "$WATCHER_LOG" && "$WATCHER_LOG" -nt "$WATCHER_MARKER" ]] ||
    die "Watcher did not write a fresh $WATCHER_LOG"
cp "$WATCHER_LOG" "$EVIDENCE_DIR/watcher.log"

PERF_TEST=models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py::test_mla_chunked_perf
for CACHE_FORMAT in kv_bf16 kv_scaled_fp8; do
    for SCENARIO in warm cold long; do
        SERIAL_DIR="generated/profiler/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_serial_mla_perf"
        OVERLAP_DIR="generated/profiler/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_overlap_mla_perf"
        SERIAL_MANIFEST="$SERIAL_DIR/run_manifest_${SCENARIO}.json"
        OVERLAP_MANIFEST="$OVERLAP_DIR/run_manifest_${SCENARIO}.json"
        CASE_DIR="$EVIDENCE_DIR/cases/${CACHE_FORMAT}_${SCENARIO}"
        mkdir -p "$CASE_DIR"

        SERIAL_MARKER="$TMP_DIR/${CACHE_FORMAT}_${SCENARIO}_serial.marker"
        touch "$SERIAL_MARKER"
        DS_PERF_VARIANT=glm_5_2 \
        DS_PERF_OVERLAP_PROFILE="$PROFILE" \
        DS_PERF_OVERLAP_ENABLED=0 \
        DS_PERF_OVERLAP_SAMPLES="$SAMPLES" \
        DS_PERF_RT_OPS_DUMP=1 \
        DS_PERF_FIRMWARE_VERSION="$FIRMWARE_BUNDLE" \
        DS_PERF_QUALIFICATION_RUN_ID="$RUN_ID" \
        DS_PERF_QUALIFICATION_ELF_BUILD_ID="$SO_BUILD_ID" \
        DS_PERF_QUALIFICATION_BUILD_INPUT_MTIME="$LATEST_BUILD_INPUT_MTIME" \
            "$SAFE_PYTEST" "$PERF_TEST" -m perf \
            -k "glm_5_2 and $SCENARIO and sparse and $CACHE_FORMAT" -s \
            --junitxml="$CASE_DIR/serial.xml"
        assert_junit_exact "$CASE_DIR/serial.xml" 1
        [[ -f "$SERIAL_MANIFEST" && "$SERIAL_MANIFEST" -nt "$SERIAL_MARKER" ]] ||
            die "serial test did not write a fresh manifest: $SERIAL_MANIFEST"
        validate_manifest "$SERIAL_MANIFEST" 0 "$CACHE_FORMAT" "$SCENARIO" 0 0

        BASELINE_FILE="$TMP_DIR/${CACHE_FORMAT}_${SCENARIO}_baselines"
        "$PYTHON_BIN" - "$SERIAL_MANIFEST" >"$BASELINE_FILE" <<'PY'
import json
import sys

overlap = json.load(open(sys.argv[1]))["sparse_mla_overlap"]
print(overlap["measured_scenario_ns"], overlap["measured_branch_serialized_ns"])
PY
        read -r BASELINE_NS BRANCH_BASELINE_NS <"$BASELINE_FILE"

        OVERLAP_MARKER="$TMP_DIR/${CACHE_FORMAT}_${SCENARIO}_overlap.marker"
        touch "$OVERLAP_MARKER"
        DS_PERF_VARIANT=glm_5_2 \
        DS_PERF_OVERLAP_PROFILE="$PROFILE" \
        DS_PERF_OVERLAP_ENABLED=1 \
        DS_PERF_OVERLAP_SAMPLES="$SAMPLES" \
        DS_PERF_RT_OPS_DUMP=1 \
        DS_PERF_OVERLAP_BASELINE_NS="$BASELINE_NS" \
        DS_PERF_OVERLAP_BRANCH_BASELINE_NS="$BRANCH_BASELINE_NS" \
        DS_PERF_OVERLAP_MIN_IMPROVEMENT=0.10 \
        DS_PERF_OVERLAP_WHOLE_FORWARD_MIN_IMPROVEMENT=0.0 \
        DS_PERF_FIRMWARE_VERSION="$FIRMWARE_BUNDLE" \
        DS_PERF_QUALIFICATION_RUN_ID="$RUN_ID" \
        DS_PERF_QUALIFICATION_ELF_BUILD_ID="$SO_BUILD_ID" \
        DS_PERF_QUALIFICATION_BUILD_INPUT_MTIME="$LATEST_BUILD_INPUT_MTIME" \
            "$SAFE_PYTEST" "$PERF_TEST" -m perf \
            -k "glm_5_2 and $SCENARIO and sparse and $CACHE_FORMAT" -s \
            --junitxml="$CASE_DIR/overlap.xml"
        assert_junit_exact "$CASE_DIR/overlap.xml" 1
        [[ -f "$OVERLAP_MANIFEST" && "$OVERLAP_MANIFEST" -nt "$OVERLAP_MARKER" ]] ||
            die "overlap test did not write a fresh manifest: $OVERLAP_MANIFEST"
        validate_manifest \
            "$OVERLAP_MANIFEST" 1 "$CACHE_FORMAT" "$SCENARIO" "$BASELINE_NS" "$BRANCH_BASELINE_NS"

        cp "$SERIAL_MANIFEST" "$CASE_DIR/serial_manifest.json"
        cp "$OVERLAP_MANIFEST" "$CASE_DIR/overlap_manifest.json"
        cp "$SERIAL_DIR/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_serial_mla_perf_${SCENARIO}.csv" \
            "$CASE_DIR/serial_ops.csv"
        cp "$OVERLAP_DIR/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_overlap_mla_perf_${SCENARIO}.csv" \
            "$CASE_DIR/overlap_ops.csv"
        cp "$SERIAL_DIR/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_serial_mla_perf_${SCENARIO}_ops.csv" \
            "$CASE_DIR/serial_calls.csv"
        cp "$OVERLAP_DIR/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_overlap_mla_perf_${SCENARIO}_ops.csv" \
            "$CASE_DIR/overlap_calls.csv"
        if [[ "$SCENARIO" == cold ]]; then
            cp "$SERIAL_DIR/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_serial_mla_perf_cold_by_iter.csv" \
                "$CASE_DIR/serial_cold_by_iter.csv"
            cp "$OVERLAP_DIR/glm_5_2_sparse_${CACHE_FORMAT}_${PROFILE}_overlap_mla_perf_cold_by_iter.csv" \
                "$CASE_DIR/overlap_cold_by_iter.csv"
        fi

        "$PYTHON_BIN" - "$SERIAL_MANIFEST" "$OVERLAP_MANIFEST" "$CACHE_FORMAT" "$SCENARIO" \
            >"$CASE_DIR/result.json" <<'PY'
import json
import sys

serial = json.load(open(sys.argv[1]))["sparse_mla_overlap"]
overlap = json.load(open(sys.argv[2]))["sparse_mla_overlap"]
json.dump(
    {
        "cache_format": sys.argv[3],
        "scenario": sys.argv[4],
        "serial_forward_ns": serial["measured_scenario_ns"],
        "overlap_forward_ns": overlap["measured_scenario_ns"],
        "whole_forward_improvement": overlap["whole_forward_improvement"],
        "serial_branch_ns": serial["measured_branch_serialized_ns"],
        "overlap_branch_union_ns": overlap["measured_branch_union_ns"],
        "branch_union_improvement": overlap["branch_union_improvement"],
        "manager_boundary_gap_ns": overlap["measured_manager_boundary_gap_ns"],
    },
    sys.stdout,
    indent=2,
)
PY
    done
done

"$PYTHON_BIN" - "$EVIDENCE_DIR" "$PROFILE" "$REQUESTED_FIRMWARE_BUNDLE" "$INSTALLED_FIRMWARE_BUNDLE" \
    "$HEAD_COMMIT" "$RUN_ID" "$SAMPLES" "$SO_PATH" "$SO_MTIME" "$SO_BUILD_ID" \
    "$LATEST_BUILD_INPUT_MTIME" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cases = [json.loads(path.read_text()) for path in sorted(root.glob("cases/*/result.json"))]
assert len(cases) == 6, f"expected six qualification cases, found {len(cases)}"
summary = {
    "schema_version": 2,
    "profile": sys.argv[2],
    "firmware_bundle_requested": sys.argv[3],
    "firmware_bundle_observed": sys.argv[4],
    "commit": sys.argv[5],
    "qualification_run_id": sys.argv[6],
    "device_samples": int(sys.argv[7]),
    "build": {
        "loaded_so_path": sys.argv[8],
        "so_mtime_epoch_seconds": int(sys.argv[9]),
        "validated_loaded_elf_build_id": sys.argv[10],
        "latest_tracked_build_input_mtime_epoch_seconds": int(sys.argv[11]),
    },
    "branch_minimum_required_improvement": 0.10,
    "whole_forward_minimum_required_improvement": 0.0,
    "production_workload": {"cache_tokens": 51200, "chunk_tokens": 5120, "long_cache_tokens": 512000},
    "cases": cases,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

RUN_PASSED=1
echo "production sparse MLA overlap qualification passed: $EVIDENCE_DIR/summary.json"
