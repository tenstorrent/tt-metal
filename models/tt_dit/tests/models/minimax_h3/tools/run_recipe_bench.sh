#!/usr/bin/env bash
# Bench only (not for merge): MiniMax-H3 on one Blackhole Galaxy (4x8), SDPA recipes FAST / STANDARD /
# LOW_PRECISION (BFP8 K/V), 16:9 at 5/10/15 s, TDP capped at 190 W.
#   tools/run_recipe_bench.sh OUT_DIR [block|pipeline|all]
# Needs MINIMAX_H3_MODEL_PATH (pipeline) and ideally TT_DIT_CACHE_DIR. Run from the tt-metal root.
set -u
OUT=${1:?usage: run_recipe_bench.sh OUT_DIR [block|pipeline|all]}; WHAT=${2:-all}
mkdir -p "$OUT"; SUMMARY="$OUT/summary.txt"
export TT_METAL_SHM_TRACKING_DISABLED=1 TT_METAL_INSPECTOR=0 TT_METAL_LOGS_PATH=/tmp/tt-logs
export TT_METAL_TDP_LIMIT_WATTS=190 PYTHONUNBUFFERED=1
T=models/tt_dit/tests/models/minimax_h3
RECIPES=("FAST:bf16" "STANDARD:bf16" "LOW_PRECISION:bfp8")
PROMPT_DEFAULT="A woman in her thirties with shoulder-length dark hair sits at a sunlit kitchen table, looking straight into the camera, and speaks warmly and clearly: \"Good morning everyone, today I want to tell you about my favorite recipe, a simple lemon cake my grandmother taught me.\" She smiles and gestures with her hands as she talks."
export H3_PROMPT=${H3_PROMPT:-$PROMPT_DEFAULT}

# Every chip must read back 190 W; otherwise the numbers are not at the requested TDP.
check_tdp() {
  local log=$1 ok all warn
  ok=$(grep -c "firmware TDP limit on chip .* is now 190 W" "$log")
  all=$(grep -c "firmware TDP limit on chip" "$log")
  warn=$(grep -c "TT_METAL_TDP_LIMIT_WATTS: leaving" "$log")
  echo "  TDP check: $ok of $all chips read back 190 W, $warn warnings" | tee -a "$SUMMARY"
  [ "$ok" -gt 0 ] && [ "$ok" -eq "$all" ] && [ "$warn" -eq 0 ]
}

run() {  # name, env..., -- pytest args
  local name=$1; shift; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local log="$OUT/$name.log"; echo "== $name" | tee -a "$SUMMARY"
  local t0=$(date +%s)
  env "${envs[@]}" pytest "$@" -s > "$log" 2>&1; local rc=$?
  echo "  rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a "$SUMMARY"
  check_tdp "$log" || { echo "  ABORT: TDP not at 190 W on every chip ($log)" | tee -a "$SUMMARY"; exit 2; }
  grep -aoE "BLOCKPERF.*|denoise breakdown:.*|Denoising \(total\).*" "$log" | sed 's/^/  /' | tee -a "$SUMMARY"
}

if [ "$WHAT" = block ] || [ "$WHAT" = all ]; then
  for rk in "${RECIPES[@]}"; do r=${rk%%:*}; kv=${rk##*:}
    run "block_${r}_${kv}" H3_SDPA_RECIPE=$r H3_SDPA_KV=$kv H3_BLOCK_ITERS=5 -- \
      "$T/test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf" -k "4x8sp1tp0nl2 and 512_text_tokens and sp_sim1"
  done
fi
if [ "$WHAT" = pipeline ] || [ "$WHAT" = all ]; then
  for d in 5 10 15; do for rk in "${RECIPES[@]}"; do r=${rk%%:*}; kv=${rk##*:}
    run "t2va_16x9_${d}s_${r}_${kv}" H3_SDPA_RECIPE=$r H3_SDPA_KV=$kv H3_DUMP_DIR="$OUT/latents/${d}s_${r}_${kv}" \
      H3_DUMP_LATENT_STEPS=1,10,25,40,50 -- \
      "$T/test_performance_minimax_h3.py::test_t2va_performance" -k "4x8 and 16x9_${d}s"
  done; done
  python3 "$T/tools/compare_recipe_latents.py" "$OUT/latents" | tee -a "$SUMMARY"
  cp ~/h3_t2va_artifacts/t2va_16x9_* "$OUT/" 2>/dev/null
fi
echo "summary: $SUMMARY"
