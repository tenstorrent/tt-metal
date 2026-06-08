#!/bin/bash
# SD 1.4 whole-UNet (512x512, batch 2), REAL device, 3 legs: COLD vs PRECOMPILE-warmup vs WARM-reuse.
# Mirrors scripts/unet_precompile_bench.sh (SDXL) for CompVis/stable-diffusion-v1-4. ccache OFF on ALL
# legs (must match between warmup+warm or warm silently misses; off everywhere => visible compile + no
# A->B contamination). Shallow fast-collect on by default. Weights cached locally (HF_HUB_OFFLINE).
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
cd "$WT"; export PYTHONPATH="$WT" HF_HUB_OFFLINE=1
TEST="models/demos/vision/generative/stable_diffusion/wormhole/tests/test_unet_2d_condition_model.py"
SEL="2-4-64-64"
RES=/tmp/sd14_precompile_results.txt
JIT_COLD=/tmp/sd14_cold; JIT_WARM=/tmp/sd14_warm
rm -rf "$JIT_COLD" "$JIT_WARM" /tmp/sd14_precompile_DONE
echo "SD1.4 precompile bench  $(date)  host=$(hostname)" | tee "$RES"
echo "test=$TEST -k $SEL  (whole SD1.4 UNet, real device, ccache OFF all legs, shallow on)" | tee -a "$RES"

echo "" | tee -a "$RES"; echo "=== [A] COLD (fresh JIT, ccache off, inline compile) $(date '+%T') ===" | tee -a "$RES"
t0=$(date +%s)
env -u TT_METAL_CCACHE_KERNEL_SUPPORT TT_METAL_CACHE="$JIT_COLD" \
  timeout 2400 python3 -m pytest "$TEST" -k "$SEL" -p no:cacheprovider >/tmp/sd14_A.log 2>&1
A=$(( $(date +%s)-t0 )); echo "[A] cold wall=${A}s" | tee -a "$RES"
sed -E 's/\x1b\[[0-9;]*m//g' /tmp/sd14_A.log | grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)|[0-9]+ (passed|failed)" | tail -2 | tee -a "$RES"

echo "" | tee -a "$RES"; echo "=== [B] PRECOMPILE warmup (fast+shallow, real-alloc, parallel compile w8, ccache off) $(date '+%T') ===" | tee -a "$RES"
t0=$(date +%s)
env -u TT_METAL_CCACHE_KERNEL_SUPPORT UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_WORKERS=8 LOGURU_LEVEL=ERROR \
  TT_METAL_CACHE="$JIT_WARM" \
  timeout 2400 python3 -m pytest "$TEST" -k "$SEL" -p up_front_collect_plugin -p no:cacheprovider >/tmp/sd14_B.log 2>&1
B=$(( $(date +%s)-t0 )); echo "[B] precompile-warmup wall=${B}s" | tee -a "$RES"
sed -E 's/\x1b\[[0-9;]*m//g' /tmp/sd14_B.log | grep -aoE "UP_FRONT_COLLECT: [0-9]+ ops stashed.*|fast-shallow.*|compiled [0-9]+ programs in [0-9.]+s.*" | tail -3 | tee -a "$RES"

echo "" | tee -a "$RES"; echo "=== [C] WARM (reuse, ccache off, no compile) $(date '+%T') ===" | tee -a "$RES"
t0=$(date +%s)
env -u TT_METAL_CCACHE_KERNEL_SUPPORT TT_METAL_CACHE="$JIT_WARM" \
  timeout 1200 python3 -m pytest "$TEST" -k "$SEL" -p no:cacheprovider >/tmp/sd14_C.log 2>&1
C=$(( $(date +%s)-t0 )); echo "[C] warm wall=${C}s" | tee -a "$RES"
sed -E 's/\x1b\[[0-9;]*m//g' /tmp/sd14_C.log | grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)|[0-9]+ (passed|failed)" | tail -2 | tee -a "$RES"

pcompile=$(sed -E 's/\x1b\[[0-9;]*m//g' /tmp/sd14_B.log | grep -aoE "compiled [0-9]+ programs in [0-9.]+s" | tail -1)
echo "" | tee -a "$RES"; echo "================= SUMMARY =================" | tee -a "$RES"
printf "[A] COLD (inline)      : %5ds\n[B] PRECOMPILE warmup  : %5ds   (%s)\n[C] WARM (reuse)       : %5ds\n" \
  "$A" "$B" "$pcompile" "$C" | tee -a "$RES"
echo "inline serial compile  ~= A-C = $((A-C))s" | tee -a "$RES"
[ "$C" -gt 0 ] && echo "ship-the-cache (A/C)   = $(echo "scale=2;$A/$C"|bc)x   |  in-job two-pass (A/(B+C)) = $(echo "scale=2;$A/($B+$C)"|bc)x" | tee -a "$RES"
echo "DONE $(date)" | tee -a "$RES"; touch /tmp/sd14_precompile_DONE
