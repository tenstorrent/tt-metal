#!/usr/bin/env bash
# Run AFTER:  source setup.sh
# Guards ensure the emulator + ASAN actually ran; results are otherwise rejected.
#
# Covers tests/ttnn/nightly/unit_tests/operations/experimental/ — experimental
# ops: minimal matmul, bcast_to, MLA w/o, MoE, rotary embedding, topk router,
# DIT fused ops, isin, and the deepseek_prefill subdirectory (moe_post_combine,
# prefill extract/insert, moe_grouped_topk, single_routed_expert).
#
# Resumable: every test file is recorded in $STATE once attempted, so ending the
# run (Ctrl-C / kill) and re-running CONTINUES from where it stopped and APPENDS
# to $LOG instead of restarting the whole suite.
#   - Start over:        ./experimental_nightly_tests.sh --fresh
#   - Re-run one test:   remove its line from $STATE, then re-run.
# A per-test timeout ($TEST_TIMEOUT secs, default 900) keeps a single hung test
# from blocking the suite — it is killed, logged, and the loop moves on.
set -uo pipefail
LOG=experimental_nightly_output.log
STATE=experimental_nightly_output.state
TEST_TIMEOUT=${TEST_TIMEOUT:-900}

type emule_preflight >/dev/null 2>&1 || { echo "Run 'source setup.sh' first."; exit 1; }
emule_preflight || exit 1

# --- Fresh vs resume ---
if [[ "${1:-}" == "--fresh" || "${FRESH:-0}" == "1" ]]; then
    : > "$LOG"
    : > "$STATE"
    echo "Fresh run: cleared $LOG and $STATE." | tee -a "$LOG"
elif [[ ! -f "$STATE" ]]; then
    if [[ -s "$LOG" ]]; then
        grep -E '^Running tests/.*\.py\.\.\.$' "$LOG" \
            | sed -E 's/^Running (.*)\.\.\.$/\1/' | sort -u > "$STATE"
        echo "Resuming: seeded $(wc -l < "$STATE") attempted test(s) from existing $LOG; appending." | tee -a "$LOG"
    else
        : > "$STATE"
    fi
else
    echo "Resuming: $(wc -l < "$STATE") test(s) already attempted; appending to $LOG." | tee -a "$LOG"
fi

for file in \
    tests/ttnn/nightly/unit_tests/operations/experimental/test_*.py \
    tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_*.py; do
    if grep -qxF "$file" "$STATE"; then
        echo "Skipping $file (already attempted)." | tee -a "$LOG"
        continue
    fi
    echo "Running $file..." | tee -a "$LOG"
    echo "$file" >> "$STATE"
    timeout "$TEST_TIMEOUT" pytest "$file" -v 2>&1 | tee -a "$LOG"
    if [[ "${PIPESTATUS[0]}" == "124" ]]; then
        echo "TIMEOUT after ${TEST_TIMEOUT}s: $file (killed, continuing)." | tee -a "$LOG"
    fi
done

emule_postflight "$LOG"
