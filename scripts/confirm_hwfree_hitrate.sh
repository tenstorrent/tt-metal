#!/bin/bash
# CONFIRM: does the hardware-free (slow-dispatch + mock) warmup cause the low JIT-cache hit rate?
# Same local N300 box that got 99.4% with a real-device/fast-dispatch warmup. Only variable changed:
# warm the cache via scripts/precompile_warm.sh (slow-dispatch + mock — the CI method), then run the
# suite normally (fast dispatch) and read the hit rate. If it drops ~99% -> ~42%, slow-dispatch/mock
# kernel divergence is proven as the cause (no CI-host confound).
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
T="tests/ttnn/nightly/unit_tests/operations/conv"
MARK="not disable_fast_runtime_mode"
RES=/tmp/conv_hwfree_results.txt
CACHE=/tmp/conv_hwfree_cache
export SAFE_PYTEST_DISPATCH_TIMEOUT=120
cd "$WT"
rm -rf "$CACHE" && mkdir -p "$CACHE"
rm -f /tmp/tt_precompile_cluster_desc.yaml /tmp/conf_hwfree_DONE   # fresh local descriptor

echo "HWFREE hit-rate confirm  $(date)  host=$(hostname)"            | tee "$RES"
echo "cache=$CACHE  (compare vs earlier real-device warmup = 99.4%)" | tee -a "$RES"

# --- [1] hardware-free warmup (slow-dispatch + mock) = the CI method ---
echo "" | tee -a "$RES"; echo "=== [1] HARDWARE-FREE WARMUP (slow-dispatch+mock)  $(date '+%T') ===" | tee -a "$RES"
W0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE="$CACHE" \
    bash scripts/precompile_warm.sh "$T" -m "$MARK" > /tmp/conf_hwfree_warmup.log 2>&1
echo "[1] warmup exit=$?  wall=$(($(date +%s)-W0))s" | tee -a "$RES"
grep -E "PRECOMPILE:|UP_FRONT_COLLECT:" /tmp/conf_hwfree_warmup.log | tee -a "$RES"

# --- [2] warm run: REAL device, FAST dispatch, reuse the hwfree cache -> read hit rate ---
echo "" | tee -a "$RES"; echo "=== [2] WARM RUN (fast dispatch, reuse hwfree cache)  $(date '+%T') ===" | tee -a "$RES"
R0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE="$CACHE" \
    scripts/run_safe_pytest.sh --run-all "$T" -m "$MARK" --timeout=600 > /tmp/conf_hwfree_run.log 2>&1
echo "[2] run exit=$?  wall=$(($(date +%s)-R0))s" | tee -a "$RES"
echo ">>> HIT RATE (hwfree warmup):" | tee -a "$RES"
grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)[^(]*" /tmp/conf_hwfree_run.log | tail -1 | tee -a "$RES"
grep -aE "= .*(passed|failed|skipped|xfailed).* (in|=)" /tmp/conf_hwfree_run.log | tail -1 | tee -a "$RES"
echo "(reference: real-device warmup hit 99.4%; CI hwfree warmup hit 42.5%)" | tee -a "$RES"
echo "DONE $(date)" | tee -a "$RES"
touch /tmp/conf_hwfree_DONE
