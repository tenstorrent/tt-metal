#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Purpose: Run focused multihost ULFM annotation smoke tests.
# These verify that:
#   - A rank failure on the remote host reports the remote hostname
#   - A FAULT_TOLERANT path emits a fault_tolerant annotation
#   - A FAST_FAIL path emits a fast_fail annotation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPIRUN="${MPIRUN:-$SCRIPT_DIR/mpirun_wrapper.sh}"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)}"
TEST_BIN="${TEST_BIN:-$TT_METAL_HOME/build/test/tt_metal/fault_tolerance_tests}"
HOSTFILE="${HOSTFILE:-/etc/mpirun/hostfile}"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: fault_tolerance_tests binary not found at $TEST_BIN" >&2
    echo "Build with: cmake --build build --target fault_tolerance_tests" >&2
    exit 1
fi

if [ ! -f "$HOSTFILE" ]; then
    echo "INFO: hostfile '$HOSTFILE' not present; skipping multihost ULFM annotation tests" >&2
    exit 0
fi

mapfile -t unique_hosts < <(awk '!/^[[:space:]]*#/ && NF {print $1}' "$HOSTFILE" | awk '!seen[$0]++')
if [ "${#unique_hosts[@]}" -lt 2 ]; then
    echo "INFO: hostfile '$HOSTFILE' has fewer than 2 unique hosts; skipping multihost ULFM annotation tests" >&2
    exit 0
fi

primary_host="${unique_hosts[0]}"
remote_host="${unique_hosts[1]}"

fail=0
export TT_METAL_HOME
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$TT_METAL_HOME/build/lib}"
# Mirror the known-good dual-T3K Open MPI networking setup so the smoke tests
# don't attempt to form BTL TCP connections over docker0 / loopback.
common_mpi_args=(
    --with-ft ulfm
    --hostfile "$HOSTFILE"
    --map-by ppr:1:node
    --bind-to none
    --mca btl_tcp_if_exclude docker0,lo
    -np 2
)

_extract_annotation_line() {
    local tmpout="$1"
    local expected_policy="$2"
    awk -v policy="$expected_policy" -v failed_host="$remote_host" -v detecting_host="$primary_host" '
        {
            pos = index($0, "::error ");
            if (!pos) {
                next;
            }
            line = substr($0, pos);
            if (index(line, "policy=" policy) &&
                index(line, "failed_hostname=" failed_host) &&
                index(line, "detecting_hostname=" detecting_host)) {
                print line;
                exit;
            }
        }
    ' "$tmpout"
}

_run_expected_annotation_test() {
    local expected_test="$1"
    local expected_policy="$2"
    local expect_success_markers="$3"
    shift 3

    local tmpout
    local cmd_status=0
    local annotation_line=""
    tmpout=$(mktemp)

    if "$@" 2>&1 | tee "$tmpout"; then
        cmd_status=0
    else
        cmd_status=${PIPESTATUS[0]}
    fi

    if [[ ! -s "$tmpout" ]]; then
        echo "ERROR: no output captured for ${expected_test}; launcher exited ${cmd_status}" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    if ! grep -Fq "[ RUN      ] ${expected_test}" "$tmpout"; then
        echo "ERROR: missing GTest start marker for ${expected_test}; launcher exited ${cmd_status}" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    if [[ "$expect_success_markers" == "yes" ]]; then
        if grep -q '\[  FAILED  \]' "$tmpout"; then
            echo "ERROR: GTest failures detected in ${expected_test}; launcher exited ${cmd_status}" >&2
            fail=$((fail + 1))
            rm -f "$tmpout"
            return
        fi
        if ! grep -Fq "[       OK ] ${expected_test}" "$tmpout" && ! grep -Fq "[  PASSED  ]" "$tmpout"; then
            echo "ERROR: missing successful GTest completion markers for ${expected_test}; launcher exited ${cmd_status}" >&2
            fail=$((fail + 1))
            rm -f "$tmpout"
            return
        fi
    else
        if [[ $cmd_status -eq 0 ]]; then
            echo "ERROR: ${expected_test} exited 0; ${expected_policy} should terminate mpirun non-zero" >&2
            fail=$((fail + 1))
            rm -f "$tmpout"
            return
        fi
    fi

    annotation_line="$(_extract_annotation_line "$tmpout" "$expected_policy")"
    if [[ -z "$annotation_line" ]]; then
        echo "ERROR: ${expected_test} did not emit ${expected_policy} annotation with failed_hostname=${remote_host} and detecting_hostname=${primary_host}" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    # Re-emit a clean workflow command after stripping any launcher prefixes.
    echo "$annotation_line"

    if [[ $cmd_status -ne 0 ]]; then
        echo "INFO: ${expected_test} emitted expected ${expected_policy} annotation despite mpirun exit ${cmd_status}" >&2
    fi

    rm -f "$tmpout"
}

echo "=== Multihost ULFM annotation smoke tests (${primary_host} -> ${remote_host}) ==="

_run_expected_annotation_test \
    "FaultTolerance.MPIRankFailureExceptionCarriesContext" \
    "fault_tolerant" \
    "yes" \
    env TT_METAL_GITHUB_ACTIONS_ANNOTATIONS=1 \
    "$MPIRUN" -x TT_METAL_GITHUB_ACTIONS_ANNOTATIONS -x TT_METAL_HOME -x LD_LIBRARY_PATH "${common_mpi_args[@]}" "$TEST_BIN" \
    --gtest_filter=FaultTolerance.MPIRankFailureExceptionCarriesContext

_run_expected_annotation_test \
    "FaultTolerance.FastFailEmitsGithubAnnotation" \
    "fast_fail" \
    "no" \
    env TT_METAL_GITHUB_ACTIONS_ANNOTATIONS=1 \
    "$MPIRUN" -x TT_METAL_GITHUB_ACTIONS_ANNOTATIONS -x TT_METAL_HOME -x LD_LIBRARY_PATH "${common_mpi_args[@]}" "$TEST_BIN" \
    --gtest_filter=FaultTolerance.FastFailEmitsGithubAnnotation

if [[ $fail -ne 0 ]]; then
    exit 1
fi
