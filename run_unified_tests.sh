#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Every unified suite, with the library's own checks TURNED ON.
#
# They are off by default and that is not a choice this repo makes: ASSERT_ENABLED is 1
# only under WATCHER_ENABLED or LIGHTWEIGHT_KERNEL_ASSERTS, so in a plain run every Block
# consume obligation, RetainedBlock occupancy check, moved-from poison and circular buffer
# capacity assert compiles to nothing. Correct kernels behave identically either way, which
# is exactly why it went unnoticed.
#
# THE TWO MODES ARE NOT INTERCHANGEABLE, and this uses each where it fits.
#
#   lightweight (default here)  ebreak. Halts the RISC, which the host cannot distinguish
#                               from a hang -- so a tripped assert shows up as this
#                               script's timeout, and the log tells you which suite. Crude,
#                               but cheap and stable across a long run.
#
#   watcher (--watcher)         Reports core, RISC, line and kernel, and throws to the
#                               host. The good diagnostic, and what to re-run a failing
#                               suite under. Costs about a second on a thirty second
#                               suite. Not the default only because lightweight is enough
#                               to catch a fired assert and cheaper to leave on.
#
# DO NOT hard-kill a run. Each device open takes hugepages and a killed process does not
# give them back -- the driver keeps them, and `tt-smi -r` does NOT release them. Enough
# kills and HugePages_Free reaches 0, after which EVERY device open falls back and hangs:
#
#     grep HugePages_Free /proc/meminfo      # 0 means this has happened
#
# The symptom is indistinguishable from a code regression -- suites that passed minutes ago
# hang from the first launch, in every assert mode, and still hang with the library changes
# stashed. That last check is what identifies it. Recovering needs the driver reset, not
# tt-smi. Ctrl-C a run and let it exit; do not kill -9 it.
#
# test_unified_negative.py sets its own environment per case and ignores both.
#
#     ./run_unified_tests.sh              # all suites, lightweight asserts
#     ./run_unified_tests.sh --watcher    # all suites, full watcher diagnostics
#     ./run_unified_tests.sh flash rope   # just those
#
# The negative suite runs LAST on purpose: its cases deliberately trip device asserts, and
# a tripped assert STOPS the device. A suite that follows a stopped device fails for
# reasons of its own, which is a confusing way to find out. If a run ends badly, or a suite
# fails right after work on deliberate hangs, reset before believing it:
#
#     tt-smi -r
#
# For perf work, run the bench scripts directly and use NEITHER mode: every number in
# unified_llama_prefill.md was taken with asserts off, and they stay comparable that way.

set -uo pipefail
cd "$(dirname "$0")"
export TT_METAL_HOME="$PWD"

MODE="lightweight"
if [ "${1:-}" = "--watcher" ]; then
    MODE="watcher"
    shift
fi
if [ "$MODE" = "watcher" ]; then
    export TT_METAL_WATCHER="${TT_METAL_WATCHER:-5}"
else
    export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
fi

SUITES=(
    unary binary bcast reduction add_exp mixed_format
    matmul matmul_bias matmul_mcast matmul_transpose matmul_blocked
    rmsnorm rope attention attention_proj flash
    layer negative
)
[ $# -gt 0 ] && SUITES=("$@")

# Reset between batches, if tt-smi is here to do it. See UNRESOLVED above.
RESET_EVERY="${RESET_EVERY:-8}"
have_reset=0
command -v tt-smi > /dev/null 2>&1 && have_reset=1

echo "  asserts: ${MODE}"
[ $have_reset -eq 0 ] && echo "  note: tt-smi not found, so no reset between batches"
pass=0
fail=0
run=0
failed_names=()

for suite in "${SUITES[@]}"; do
    if [ $have_reset -eq 1 ] && [ $run -gt 0 ] && [ $((run % RESET_EVERY)) -eq 0 ]; then
        echo "  -- resetting the device after ${run} suites"
        tt-smi -r > /dev/null 2>&1
    fi
    run=$((run + 1))
    script="test_unified_${suite}.py"
    if [ ! -f "$script" ]; then
        echo "  ${suite}: no such suite (${script})"
        fail=$((fail + 1))
        failed_names+=("$suite")
        continue
    fi
    start=$(date +%s)
    timeout 900 python3 "$script" > "/tmp/unified_${suite}.log" 2>&1
    rc=$?
    took=$(($(date +%s) - start))
    if [ $rc -eq 0 ]; then
        echo "  ${suite}: ok (${took}s)"
        pass=$((pass + 1))
    else
        if [ $rc -eq 124 ]; then
            echo "  ${suite}: TIMED OUT (${took}s) -- an assert may have halted a RISC;"
            echo "      re-run it under ./run_unified_tests.sh --watcher ${suite} for the line"
        else
            echo "  ${suite}: FAIL rc=${rc} (${took}s) -- /tmp/unified_${suite}.log"
        fi
        grep -iE "tripped an assert|static assertion failed|FAIL:" "/tmp/unified_${suite}.log" | head -3 | sed 's/^/      /'
        fail=$((fail + 1))
        failed_names+=("$suite")
    fi
done

echo "  ---- ${pass} passed, ${fail} failed"
if [ $fail -gt 0 ]; then
    echo "  failed: ${failed_names[*]}"
    exit 1
fi
