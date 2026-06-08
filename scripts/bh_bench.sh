#!/bin/bash
# Local WH benchmark of the precompile speedup, in the order: PREWARM -> WARM RUN -> COLD.
# (warm phases first so their numbers land before the slow cold leg; isolated caches so warm is clean.)
#   [1] PREWARM  : hardware-free warm (fingerprint + compute-grid replay) into a fresh cache  -> shippable cost
#   [2] WARM RUN : pytest reusing that cache (zero inline compile)                            -> the payoff
#   [3] COLD     : pytest with a fresh cache (== today's CI)                                  -> baseline
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
T="tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py"
MARK="not disable_fast_runtime_mode"
RES=/tmp/bh_bench_results.txt
export SAFE_PYTEST_DISPATCH_TIMEOUT=120
cd "$WT"
CW=/tmp/bench_warmcache; CC=/tmp/bench_coldcache
rm -rf "$CW" "$CC" && mkdir -p "$CW" "$CC"
rm -f /tmp/bh_bench_DONE /tmp/bh_bench_WARMDONE

echo "BENCH conv2d (PREWARM->WARM->COLD)  $(date)  host=$(hostname)  branch=$(git branch --show-current)" | tee "$RES"

# ---------- [1] PREWARM (hardware-free, fingerprint + grid replay) ----------
echo "" | tee -a "$RES"; echo "=== [1] PREWARM (hardware-free) $(date '+%T') ===" | tee -a "$RES"
t0=$(date +%s)
TT_METAL_CACHE="$CW" PYTHONPATH="$WT" bash scripts/ci_precompile_warm.sh "$T" -m "$MARK" > /tmp/bench_prewarm.log 2>&1
PW=$(( $(date +%s)-t0 ))
echo "[1] prewarm wall=${PW}s" | tee -a "$RES"
grep -aE "PRECOMPILE:|fingerprint=|compiled [0-9]+ programs" /tmp/bench_prewarm.log | sed -E 's/\x1b\[[0-9;]*m//g' | tail -5 | tee -a "$RES"

# ---------- [2] WARM RUN (reuse the prewarmed cache) ----------
echo "" | tee -a "$RES"; echo "=== [2] WARM RUN (reuse) $(date '+%T') ===" | tee -a "$RES"
t0=$(date +%s)
TT_METAL_CACHE="$CW" PYTHONPATH="$WT" scripts/run_safe_pytest.sh --run-all "$T" -m "$MARK" --timeout=600 > /tmp/bench_warm.log 2>&1
WR=$(( $(date +%s)-t0 ))
echo "[2] warm-run wall=${WR}s" | tee -a "$RES"
grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)" /tmp/bench_warm.log | tail -1 | tee -a "$RES"
grep -aE "= .*(passed|failed|skipped|xfailed).* (in|=)" /tmp/bench_warm.log | tail -1 | tee -a "$RES"
touch /tmp/bh_bench_WARMDONE   # warm numbers are in -> cold leg starts now

# ---------- [3] COLD (fresh cache == today's CI) ----------
echo "" | tee -a "$RES"; echo "=== [3] COLD (fresh) $(date '+%T') ===" | tee -a "$RES"
t0=$(date +%s)
TT_METAL_CACHE="$CC" PYTHONPATH="$WT" scripts/run_safe_pytest.sh --run-all "$T" -m "$MARK" --timeout=600 > /tmp/bench_cold.log 2>&1
CO=$(( $(date +%s)-t0 ))
echo "[3] cold wall=${CO}s" | tee -a "$RES"
grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)" /tmp/bench_cold.log | tail -1 | tee -a "$RES"
grep -aE "= .*(passed|failed|skipped|xfailed).* (in|=)" /tmp/bench_cold.log | tail -1 | tee -a "$RES"

# ---------- summary ----------
echo "" | tee -a "$RES"; echo "================= SUMMARY =================" | tee -a "$RES"
printf "PREWARM (hw-free): %5ds (%dm%02ds)\nWARM RUN (reuse):  %5ds (%dm%02ds)\nCOLD (inline):     %5ds (%dm%02ds)\n" \
  $PW $((PW/60)) $((PW%60)) $WR $((WR/60)) $((WR%60)) $CO $((CO/60)) $((CO%60)) | tee -a "$RES"
[ "$WR" -gt 0 ] && printf "warm-reuse speedup (cold/warm): %.2fx\nin-job (cold/(prewarm+warm)):   %.2fx\n" \
  "$(echo "scale=3;$CO/$WR"|bc)" "$(echo "scale=3;$CO/($PW+$WR)"|bc)" | tee -a "$RES"
echo "DONE $(date)" | tee -a "$RES"
touch /tmp/bh_bench_DONE
