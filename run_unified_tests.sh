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
#                               host. The good diagnostic, and the first thing to re-run a
#                               failing suite under. Costs about a second on a thirty second
#                               suite. Not the default only because lightweight is enough
#                               to catch a fired assert and cheaper to leave on.
#
#                               A PASS UNDER THE WATCHER DOES NOT MEAN NO ASSERT FIRED.
#                               The watcher's own overhead perturbs timing, so a check on
#                               whether something has landed YET can stop failing under it.
#                               That is what hid hazard 30 (a kernel returning with a write
#                               still in flight) for as long as it hid. When lightweight
#                               hangs and the watcher is clean, do not conclude "not an
#                               assert" -- neuter lightweight_assert_trap in
#                               hw/inc/api/debug/assert.h instead, keeping ASSERT_ENABLED
#                               and every condition compiled, and see whether the suite
#                               passes. If it does, an assert IS firing, and
#                               `objdump -d` plus `addr2line -i` over the built ELFs in
#                               ~/.cache/tt-metal-cache enumerates the live ebreak sites
#                               with their inline stacks, which beats guessing.
#
# DO NOT hard-kill a run. A stopped device STAYS stopped until reset, so every run launched
# afterwards fails for a reason that has nothing to do with it -- and `kill -9` on the shell
# can leave a python child still holding the device, which fails everything until it is
# found with `pgrep -af test_unified`. The symptom is indistinguishable from a code
# regression: suites that passed minutes ago hang from the first launch, in every assert
# mode. Two cheap checks settle it -- `tt-smi -r` and retry, and stash the library changes
# to see whether the hang survives them. Ctrl-C a run and let it exit.
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
#
# A run begins with unified_selftest.cpp, on the host and before any device is touched:
# three projections compiled -Wall -Wextra -Werror and traced. SKIP_SELFTEST=1 skips it,
# SELFTEST_CXX names the compiler.

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

# matmul_blocked was briefly removed from this list as "the suite that hangs". It was not
# hanging: with asserts on it named a circular buffer the host had not allocated, the
# capacity assert fired, the device stopped, and every suite after it failed for that
# reason. Which read as a sequence-dependent stall and was not one. Fixed at the source --
# the harnesses now allocate the accumulator buffer unconditionally -- and it is back here
# where it belongs. A suite that "hangs" only in a full run deserves the same suspicion:
# check what ran before it, with asserts on, from a reset device.
SUITES=(
    unary binary bcast reduction add_exp mixed_format
    matmul matmul_bias matmul_mcast matmul_transpose matmul_blocked
    custom_compute mcast_share rmsnorm rope attention attention_proj flash
    layer negative
)
[ $# -gt 0 ] && SUITES=("$@")

# Per-suite timeout. NOT 900s: a suite that stalls should cost seconds of attention, not a
# quarter of an hour, and the whole point of a short bound is that a stall is a RESULT
# rather than something to sit through. Not 30s either, though, because real suites are
# slower than that -- binary is 29s and flash around 40 -- so the default is 120 with
# overrides for the two that legitimately run long: layer sweeps eight configs in two
# formats, and negative launches subprocesses that each open a device.
TIMEOUT_DEFAULT="${TIMEOUT_DEFAULT:-120}"
timeout_for() {
    case "$1" in
        layer) echo 600 ;;
        negative) echo 400 ;;
        *) echo "$TIMEOUT_DEFAULT" ;;
    esac
}

# Reset between batches, if tt-smi is here to do it.
RESET_EVERY="${RESET_EVERY:-8}"
have_reset=0
command -v tt-smi > /dev/null 2>&1 && have_reset=1

echo "  asserts: ${MODE}"
[ $have_reset -eq 0 ] && echo "  note: tt-smi not found, so no reset between batches"
pass=0
fail=0
run=0
failed_names=()

# The host selftest, FIRST and before any device is touched. It compiles the headers with
# -Wall -Wextra -Werror -- the only build that does, since the device build turns those off
# -- and runs the example kernels once per thread projection, checking that every circular
# buffer balances and that the free-function and method spellings emit identical traces.
#
# It is here because it was NOT here, and went unbuilt from stage 1b until the port was
# finished: nothing referenced it, so four separate changes broke it a little more and no
# run said so. Three seconds against nine minutes of device time also puts it in the right
# order -- a header that does not compile should not cost a suite run to discover.
#
# SKIP_SELFTEST=1 skips it. It needs no device, so there is rarely a reason to.
SELFTEST_CXX="${SELFTEST_CXX:-clang++-20}"
if [ "${SKIP_SELFTEST:-0}" != "1" ]; then
    if ! command -v "$SELFTEST_CXX" > /dev/null 2>&1; then
        echo "  selftest: skipped, no ${SELFTEST_CXX} (set SELFTEST_CXX)"
    else
        selftest_rc=0
        for spec in "DM0 -DIS_DM_THREAD=1 -DTT_DM_THREAD_ID=0" \
                    "DM1 -DIS_DM_THREAD=1 -DTT_DM_THREAD_ID=1" \
                    "COMPUTE -DIS_COMPUTE_THREAD=1"; do
            # shellcheck disable=SC2086
            set -- $spec
            label=$1
            shift
            log="/tmp/unified_selftest_${label}.log"
            if ! "$SELFTEST_CXX" -std=c++17 -Wall -Wextra -Werror -I. "$@" \
                -DTT_LABEL="\"${label}\"" unified_selftest.cpp -o "/tmp/u_${label}" \
                > "$log" 2>&1; then
                echo "  selftest ${label}: BUILD FAILED -- ${log}"
                selftest_rc=1
                continue
            fi
            if ! "/tmp/u_${label}" >> "$log" 2>&1; then
                echo "  selftest ${label}: FAILED -- ${log}"
                selftest_rc=1
            fi
        done
        if [ $selftest_rc -eq 0 ]; then
            echo "  selftest: ok (DM0, DM1, COMPUTE balanced; spellings agree)"
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
            failed_names+=("selftest")
        fi
    fi
fi

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
    limit=$(timeout_for "$suite")
    start=$(date +%s)
    timeout "$limit" python3 "$script" > "/tmp/unified_${suite}.log" 2>&1
    rc=$?

    # A timeout leaves the device stopped, and a stopped device fails everything after it
    # for reasons of its own. So reset and give the suite one more go: that also separates
    # the two causes worth telling apart -- a suite that passes on the retry was stalled by
    # what ran before it, while one that times out twice is stalled by itself.
    if [ $rc -eq 124 ] && [ $have_reset -eq 1 ]; then
        echo "  ${suite}: timed out at ${limit}s -- resetting and retrying once"
        tt-smi -r > /dev/null 2>&1
        start=$(date +%s)
        timeout "$limit" python3 "$script" > "/tmp/unified_${suite}.log" 2>&1
        rc=$?
        [ $rc -eq 0 ] && echo "      passed on the retry, so the stall came from what ran before it"
    fi

    took=$(($(date +%s) - start))
    if [ $rc -eq 0 ]; then
        echo "  ${suite}: ok (${took}s)"
        pass=$((pass + 1))
    else
        if [ $rc -eq 124 ]; then
            echo "  ${suite}: TIMED OUT twice (${limit}s each) -- an assert may have halted a RISC;"
            echo "      re-run it under ./run_unified_tests.sh --watcher ${suite} for the line."
            echo "      If the watcher is CLEAN, that does not clear it: see the watcher note above."
            [ $have_reset -eq 1 ] && tt-smi -r > /dev/null 2>&1
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
