#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Purpose: Run focused multihost ULFM rank-failure diagnostic smoke tests.
# FAST_FAIL runs first so CI (e.g. tooling-and-mpi-t3k) emits multihost ::error
# annotations before the FAULT_TOLERANT case.
# These verify that:
#   - A rank failure on the remote host reports the remote hostname (when ULFM
#     can resolve it; MPIX_ERR_REVOKED often yields failed_hostname=unknown-hostname)
#   - FAULT_TOLERANT / FAST_FAIL paths log policy=... with structured fields
#
# When GITHUB_ACTIONS is set, ulfm_github_workflow_wrappers.sh promotes structured
# ULFM diagnostic lines to ::error / ::warning as documented in that script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/multihost/ulfm_github_workflow_wrappers.sh
source "$SCRIPT_DIR/../../scripts/multihost/ulfm_github_workflow_wrappers.sh"

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
    echo "INFO: hostfile '$HOSTFILE' not present; skipping multihost ULFM diagnostic tests" >&2
    exit 0
fi

mapfile -t unique_hosts < <(awk '!/^[[:space:]]*#/ && NF {print $1}' "$HOSTFILE" | awk '!seen[$0]++')
if [ "${#unique_hosts[@]}" -lt 2 ]; then
    echo "INFO: hostfile '$HOSTFILE' has fewer than 2 unique hosts; skipping multihost ULFM diagnostic tests" >&2
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

# Prefer hostfile-aligned hostnames (strong multihost signal). Fall back to any
# structured line with the expected policy — required when failed ranks are not
# acked (e.g. MPIX_ERR_REVOKED / error 77) and C++ logs unknown-hostname.
_extract_ulfm_diagnostic_line() {
    local tmpout="$1"
    local expected_policy="$2"
    local line
    line="$(awk -v policy="$expected_policy" -v failed_host="$remote_host" -v detecting_host="$primary_host" '
        index($0, "ULFM detected a rank failure") &&
            index($0, "policy=" policy) &&
            index($0, "failed_hostname=" failed_host) &&
            index($0, "detecting_hostname=" detecting_host) {
            print $0;
            exit;
        }
    ' "$tmpout")"
    if [[ -n "$line" ]]; then
        echo "$line"
        return
    fi
    awk -v policy="$expected_policy" '
        index($0, "ULFM detected a rank failure") &&
            index($0, "policy=" policy) &&
            index($0, "failed_hostname=") &&
            index($0, "detecting_hostname=") {
            print $0;
            exit;
        }
    ' "$tmpout"
}

_run_expected_diagnostic_test() {
    local expected_test="$1"
    local expected_policy="$2"
    local expect_success_markers="$3"
    shift 3

    local tmpout
    local cmd_status=0
    local diag_line=""
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

    _emit_ulfm_github_annotations_from_file "$tmpout"

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

    diag_line="$(_extract_ulfm_diagnostic_line "$tmpout" "$expected_policy")"
    if [[ -z "$diag_line" ]]; then
        echo "ERROR: ${expected_test} did not emit a ${expected_policy} ULFM diagnostic line (expected ULFM detected... policy=${expected_policy} with failed_hostname= and detecting_hostname=)" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    # Log human-readable; GHA annotations already emitted from full log above.
    echo "$diag_line"

    if echo "$diag_line" | grep -q "failed_hostname=${remote_host}" && echo "$diag_line" | grep -q "detecting_hostname=${primary_host}"; then
        :
    else
        echo "INFO: ${expected_test}: diagnostic matched relaxed rules (hostfile hosts ${primary_host}/${remote_host} not both present — typical for REVOKED/unknown failed rank)" >&2
    fi

    if [[ $cmd_status -ne 0 ]]; then
        echo "INFO: ${expected_test} emitted expected ${expected_policy} diagnostic despite mpirun exit ${cmd_status}" >&2
    fi

    rm -f "$tmpout"
}

echo "=== Multihost ULFM rank-failure diagnostic smoke tests (${primary_host} -> ${remote_host}) ==="

_run_expected_diagnostic_test \
    "FaultTolerance.FastFailEmitsRankFailureDiagnostics" \
    "fast_fail" \
    "no" \
    "$MPIRUN" -x TT_METAL_HOME -x LD_LIBRARY_PATH "${common_mpi_args[@]}" "$TEST_BIN" \
    --gtest_filter=FaultTolerance.FastFailEmitsRankFailureDiagnostics

_run_expected_diagnostic_test \
    "FaultTolerance.MPIRankFailureExceptionCarriesContext" \
    "fault_tolerant" \
    "yes" \
    "$MPIRUN" -x TT_METAL_HOME -x LD_LIBRARY_PATH "${common_mpi_args[@]}" "$TEST_BIN" \
    --gtest_filter=FaultTolerance.MPIRankFailureExceptionCarriesContext

if [[ $fail -ne 0 ]]; then
    exit 1
fi
