#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# worker_sweep.sh — how does the precompile WARMUP compile scale with worker count?
# This isolates precompile's core mechanism (parallel up-front compile). For each worker
# count it does a hardware-free warmup over the 75-test suite with a FRESH ccache + JIT cache
# (so every kernel really compiles — cold, the CI-relevant case) and reports the compile
# sub-phase wall (from the plugin's "compiled N in Xs") and the whole-warmup getrusage CPU.
# Hardware-free: no device lock needed. Reuses the captured fingerprint/descriptor.
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
cd "$WT"; source python_env/bin/activate 2>/dev/null
BENCH_DIR="$WT/scripts/precompile_bench"
mapfile -t IDS < "$BENCH_DIR/layernorm_75.txt"
TIMER="$BENCH_DIR/run_and_time.py"
FP=/tmp/tt_precompile_build_fingerprint.txt; DESC=/tmp/tt_precompile_cluster_desc.yaml
OUT="${OUT:-/tmp/ln_worker_sweep}"; rm -rf "$OUT"; mkdir -p "$OUT"
JITC="$OUT/jit"; CCD="$OUT/cc"
PB=(-o "addopts=--import-mode=importlib" -p no:cacheprovider -q --no-header)
WORKERS_LIST="${WORKERS_LIST:-1 2 4 8 16}"
RES="$OUT/results.txt"

[[ -f "$FP" && -f "$DESC" ]] || { echo "need $FP and $DESC (run a --precompile/probe once first)"; exit 1; }
echo "WORKER SWEEP  $(date)  nproc=$(nproc)  tests=${#IDS[@]}" | tee "$RES"
printf "%-8s %-12s %-14s %-10s %-10s %s\n" "workers" "warmup_wall" "compile_wall" "%CPU" "cores" "speedup_vs_w1" | tee -a "$RES"

base=""
for w in $WORKERS_LIST; do
  rm -rf "$JITC" "$CCD"; mkdir -p "$JITC" "$CCD"   # cold ccache + cold JIT -> full real compile
  log="$OUT/w${w}.log"; tf="$OUT/w${w}.timing"
  TT_METAL_FORCE_2_ERISC_MODE=1 TT_METAL_JIT_BUILD_FINGERPRINT="$FP" TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC" UP_FRONT_COLLECT=1 UP_FRONT_META_COLLECT=1 UP_FRONT_COLLECT_WORKERS=$w \
  TT_METAL_CACHE="$JITC" CCACHE_DIR="$CCD" LOGURU_LEVEL=ERROR PYTHONPATH="$WT" \
    python "$TIMER" "$tf" "$log" pytest "${IDS[@]}" -p up_front_collect_plugin "${PB[@]}"
  wall=$(sed -n 's/.*wall=\([0-9.]*\).*/\1/p' "$tf")
  user=$(sed -n 's/.*user=\([0-9.]*\).*/\1/p' "$tf"); sys=$(sed -n 's/.*sys=\([0-9.]*\).*/\1/p' "$tf")
  cwall=$(grep -aoE "compiled [0-9]+ programs in [0-9.]+s" "$log" | tail -1 | grep -oE "[0-9.]+s" | tr -d s)
  cpu=$(awk -v u="$user" -v s="$sys" -v w="$wall" 'BEGIN{if(w>0)printf "%.0f",100*(u+s)/w; else print 0}')
  cores=$(awk -v c="$cpu" 'BEGIN{printf "%.1f", c/100}')
  [[ -z "$base" ]] && base="$cwall"
  spd=$(awk -v b="$base" -v c="$cwall" 'BEGIN{if(c>0)printf "%.2fx", b/c; else print "-"}')
  printf "%-8s %-12s %-14s %-10s %-10s %s\n" "$w" "${wall}s" "${cwall}s" "${cpu}%" "$cores" "$spd" | tee -a "$RES"
done
echo "DONE_SWEEP" | tee -a "$RES"
