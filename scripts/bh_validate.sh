#!/bin/bash
# Validate the precompile-blackhole-fix end-to-end on the LOCAL Wormhole N300.
# The fix (JitBuildFingerprint capture/replay) should make the hardware-free warmup compile kernels
# that MATCH the real fast-dispatch run -> JIT cache hit rate should jump from the broken ~42.5%
# (measured pre-fix on this same box) to ~high (~99%). Uses a fresh cache so the warm pass is the
# only thing that populates it. Conv subset = where the 42.5% was measured (kernel-diverse).
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
T="tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py"
KSEL="resnet50 or vae_sdxl"          # ~160 kernel-diverse cases, fast enough to iterate
CACHE=/tmp/bh_validate_cache
RES=/tmp/bh_validate_results.txt
export SAFE_PYTEST_DISPATCH_TIMEOUT=120
cd "$WT"
rm -rf "$CACHE" && mkdir -p "$CACHE"
rm -f /tmp/bh_validate_DONE

echo "BH-FIX validate  $(date)  host=$(hostname)"                       | tee "$RES"
echo "branch=$(git branch --show-current)  cache=$CACHE (fresh)"         | tee -a "$RES"
echo "binding present in _ttnn.so: $(strings build/lib/_ttnn.so 2>/dev/null | grep -c capture_jit_build_fingerprint)" | tee -a "$RES"
echo "selection: $T -k '$KSEL'"                                          | tee -a "$RES"

echo "" | tee -a "$RES"; echo "=== run_safe_pytest --precompile (fresh cache) $(date '+%T') ===" | tee -a "$RES"
t0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE="$CACHE" \
    scripts/run_safe_pytest.sh --precompile --run-all "$T" -k "$KSEL" > /tmp/bh_validate_run.log 2>&1
echo "exit=$?  wall=$(($(date +%s)-t0))s" | tee -a "$RES"

echo "" | tee -a "$RES"; echo "--- PRECOMPILE diagnostics (build_key match? warm ok?) ---" | tee -a "$RES"
grep -aE "PRECOMPILE:|UP_FRONT_COLLECT:" /tmp/bh_validate_run.log | sed -E 's/\x1b\[[0-9;]*m//g' | tee -a "$RES"
echo "--- JIT cache HIT RATE on the real run (the payoff) ---" | tee -a "$RES"
grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)[^(]*" /tmp/bh_validate_run.log | tail -1 | tee -a "$RES"
grep -aE "= .*(passed|failed|skipped|xfailed).* (in|=)" /tmp/bh_validate_run.log | tail -1 | tee -a "$RES"
echo "(pre-fix this same hardware-free warmup gave 42.5%; fix target ~99%)" | tee -a "$RES"
echo "DONE $(date)" | tee -a "$RES"
touch /tmp/bh_validate_DONE
