#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Section 7.8 — Single-node ULFM gap tests
#
# These tests exercise ULFM control-plane behavior that does NOT require
# actual Tenstorrent hardware or a multi-host cluster. They verify:
#   - FAST_FAIL exit code 70 propagation
#   - GitHub Actions FAST_FAIL annotation emission
#   - MPI_Finalize watchdog (SIGALRM) path
#   - std::set_terminate handler
#   - MPIX_Comm_agree consensus
#   - FailurePolicy switching
#
# Run with: mpirun-ulfm --with-ft ulfm -np 2 (single host, no hostfile)

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the wrapper script to find mpirun-ulfm
MPIRUN="$SCRIPT_DIR/mpirun_wrapper.sh"

TT_METAL_HOME="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
TEST_BIN="${TT_METAL_HOME}/build/test/tt_metal/fault_tolerance_tests"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: fault_tolerance_tests binary not found at $TEST_BIN" >&2
    echo "Build with: cmake --build build --target fault_tolerance_tests" >&2
    exit 1
fi

fail=0

_run_expected_fast_fail_annotation_test() {
    local expected_test="$1"
    shift

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

    if [[ $cmd_status -eq 0 ]]; then
        echo "ERROR: ${expected_test} exited 0; FAST_FAIL should terminate mpirun non-zero" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    annotation_line="$(
        awk '
            {
                pos = index($0, "::error ");
                if (pos && index($0, "policy=fast_fail") && index($0, "failed_hostname=") && index($0, "detecting_hostname=")) {
                    print substr($0, pos);
                    exit;
                }
            }
        ' "$tmpout"
    )"

    if [[ -z "$annotation_line" ]]; then
        echo "ERROR: ${expected_test} did not emit the expected FAST_FAIL rank-failure annotation" >&2
        fail=$((fail + 1))
        rm -f "$tmpout"
        return
    fi

    # Re-emit a clean workflow command when mpirun prefixes the rank output.
    # GitHub only parses commands that begin at column 0.
    echo "$annotation_line"

    echo "INFO: ${expected_test} emitted FAST_FAIL annotation with hostname fields (mpirun exit ${cmd_status})" >&2
    rm -f "$tmpout"
}

echo "=== Single-node ULFM gap tests (Section 7.8) ==="

# ── 1. Agree consensus (no rank death, pure control-plane) ─────────────
echo "--- AgreeConsensus (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.AgreeConsensus || fail=$((fail + 1))

# ── 2. Failure policy switching (FAULT_TOLERANT mode, rank kill + recover)
echo "--- FailurePolicySwitching (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.FailurePolicySwitching || fail=$((fail + 1))

# ── 3. Fast-fail exit code 70 detection path ──────────────────────────
echo "--- FastFailExitCode70 (-np 2) ---"
"$MPIRUN" --with-ft ulfm -np 2 "$TEST_BIN" --gtest_filter=FaultTolerance.FastFailExitCode70 || fail=$((fail + 1))

# ── 3b. FAST_FAIL GitHub annotation coverage ─────────────────────────
echo "--- FastFailEmitsGithubAnnotation (-np 2) ---"
_run_expected_fast_fail_annotation_test \
  "FaultTolerance.FastFailEmitsGithubAnnotation" \
  env TT_METAL_GITHUB_ACTIONS_ANNOTATIONS=1 \
  "$MPIRUN" -x TT_METAL_GITHUB_ACTIONS_ANNOTATIONS --with-ft ulfm -np 2 "$TEST_BIN" --gtest_filter=FaultTolerance.FastFailEmitsGithubAnnotation

# ── 4. MPI_Finalize watchdog path verification ────────────────────────
echo "--- FinalizeWatchdogPath (-np 2) ---"
"$MPIRUN" --with-ft ulfm -np 2 "$TEST_BIN" --gtest_filter=FaultTolerance.FinalizeWatchdogPath || fail=$((fail + 1))

# ── 5. Terminate handler installed ────────────────────────────────────
echo "--- TerminateHandlerInstalled (-np 2) ---"
"$MPIRUN" --with-ft ulfm -np 2 "$TEST_BIN" --gtest_filter=FaultTolerance.TerminateHandlerInstalled || fail=$((fail + 1))

# ── 6. failed_ranks() before any failure (no rank death) ─────────────
echo "--- FailedRanksBeforeAnyFailure (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.FailedRanksBeforeAnyFailure || fail=$((fail + 1))

# ── 7. is_revoked() on healthy communicator ──────────────────────────
echo "--- IsRevokedFalseBeforeFailure (-np 2) ---"
"$MPIRUN" --with-ft ulfm -np 2 "$TEST_BIN" --gtest_filter=FaultTolerance.IsRevokedFalseBeforeFailure || fail=$((fail + 1))

# ── 8. is_revoked() after failure detection (before shrink) ──────────
echo "--- IsRevokedTrueAfterDetectionBeforeShrink (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.IsRevokedTrueAfterDetectionBeforeShrink || fail=$((fail + 1))

# ── 9. supports_fault_tolerance() consistency ────────────────────────
echo "--- SupportsFaultToleranceReported (-np 2) ---"
"$MPIRUN" --with-ft ulfm -np 2 "$TEST_BIN" --gtest_filter=FaultTolerance.SupportsFaultToleranceReported || fail=$((fail + 1))

# ── 10. set_failure_policy idempotency ───────────────────────────────
echo "--- SetFailurePolicyIsIdempotent (-np 2) ---"
"$MPIRUN" --with-ft ulfm -np 2 "$TEST_BIN" --gtest_filter=FaultTolerance.SetFailurePolicyIsIdempotent || fail=$((fail + 1))

# ── 11. Success path no error output ─────────────────────────────────
echo "--- SuccessPathNoErrorOutput (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.SuccessPathNoErrorOutput || fail=$((fail + 1))

# ── 12. agree() on single-rank sub-communicator ──────────────────────
echo "--- AgreeMixedVotesSingleRank (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.AgreeMixedVotesSingleRank || fail=$((fail + 1))

# ── 13. Double revoke guard (rank kill + recovery) ───────────────────
echo "--- DoubleRevokeGuard (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.DoubleRevokeGuard || fail=$((fail + 1))

# ── 14. agree() after revoke_and_shrink (rank kill + recovery) ───────
echo "--- AgreeAfterRevokeAndShrink (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.AgreeAfterRevokeAndShrink || fail=$((fail + 1))

# ── 15. MPIRankFailureException diagnostic context ───────────────────
echo "--- MPIRankFailureExceptionCarriesContext (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.MPIRankFailureExceptionCarriesContext || fail=$((fail + 1))

# ── 16. failed_ranks() after detection (rank kill + recovery) ────────
echo "--- FailedRanksAfterDetection (-np 4) ---"
"$MPIRUN" --with-ft ulfm -np 4 "$TEST_BIN" --gtest_filter=FaultTolerance.FailedRanksAfterDetection || fail=$((fail + 1))

echo "=== Single-node ULFM gap tests complete ==="

if [ $fail -ne 0 ]; then
    echo "ERROR: $fail test group(s) failed" >&2
    exit 1
fi
