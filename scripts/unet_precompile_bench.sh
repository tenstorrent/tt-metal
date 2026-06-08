#!/bin/bash
# Whole SDXL UNet (512x512, 1x4x64x64), REAL device, 3 legs: COLD vs PRECOMPILE-warmup vs WARM-reuse.
# Goal: see the difference precompile (parallel up-front compile) makes vs cold (serial inline) vs warm (reuse).
# ccache is DISABLED on the compile legs (A,B) so the serial-vs-parallel compile difference is visible
# (a warm ccache would serve objects and hide it). Real device used (no mock, no xdist).
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
cd "$WT"; export PYTHONPATH="$WT" HF_HUB_OFFLINE=1
TEST="models/demos/stable_diffusion_xl_base/tests/pcc/test_module_tt_unet.py"
SEL="image_resolution2-"          # the 512x512 (1,4,64,64) whole-UNet case
RES=/tmp/unet_precompile_results.txt
JIT_COLD=/tmp/unet_cold; JIT_WARM=/tmp/unet_warm
rm -rf "$JIT_COLD" "$JIT_WARM" /tmp/unet_*_DONE
echo "UNet precompile bench  $(date)  host=$(hostname)" | tee "$RES"
echo "test=$TEST -k $SEL  (whole SDXL UNet, real device, ccache OFF on ALL legs)" | tee -a "$RES"
# NOTE: ccache on/off MUST match between the warmup (B) and the warm run (C), else the warm run
# silently misses 100% (cache validity encodes the ccache state). So C is ccache-off too. ccache
# being off everywhere also guarantees no A->B contamination (ccache is never read or written).

# ---------- A: COLD (inline serial compile, ccache off) ----------
echo "" | tee -a "$RES"; echo "=== [A] COLD (fresh JIT, ccache off, inline compile) $(date '+%T') ===" | tee -a "$RES"
echo 'CMD: env -u TT_METAL_CCACHE_KERNEL_SUPPORT TT_METAL_CACHE=/tmp/unet_cold \' | tee -a "$RES"
echo "       pytest $TEST -k $SEL" | tee -a "$RES"
t0=$(date +%s)
env -u TT_METAL_CCACHE_KERNEL_SUPPORT TT_METAL_CACHE="$JIT_COLD" \
  timeout 2400 python3 -m pytest "$TEST" -k "$SEL" -p no:cacheprovider >/tmp/unet_A.log 2>&1
A=$(( $(date +%s)-t0 )); echo "[A] cold wall=${A}s" | tee -a "$RES"
sed -E 's/\x1b\[[0-9;]*m//g' /tmp/unet_A.log | grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)|PCC is [0-9.]+|[0-9]+ (passed|failed)" | tail -3 | tee -a "$RES"

# ---------- B: PRECOMPILE warmup (real-device collect + PARALLEL compile, ccache off) ----------
echo "" | tee -a "$RES"; echo "=== [B] PRECOMPILE warmup (fast torch, real-alloc, parallel compile w8, ccache off) $(date '+%T') ===" | tee -a "$RES"
echo 'CMD: env -u TT_METAL_CCACHE_KERNEL_SUPPORT UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 \' | tee -a "$RES"
echo "       UP_FRONT_COLLECT_WORKERS=8 TT_METAL_CACHE=/tmp/unet_warm \\" | tee -a "$RES"
echo "       pytest $TEST -k $SEL -p up_front_collect_plugin" | tee -a "$RES"
t0=$(date +%s)
env -u TT_METAL_CCACHE_KERNEL_SUPPORT UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_WORKERS=8 LOGURU_LEVEL=ERROR \
  TT_METAL_CACHE="$JIT_WARM" \
  timeout 2400 python3 -m pytest "$TEST" -k "$SEL" -p up_front_collect_plugin -p no:cacheprovider >/tmp/unet_B.log 2>&1
B=$(( $(date +%s)-t0 )); echo "[B] precompile-warmup wall=${B}s" | tee -a "$RES"
sed -E 's/\x1b\[[0-9;]*m//g' /tmp/unet_B.log | grep -aoE "UP_FRONT_COLLECT: [0-9]+ ops stashed.*|compiled [0-9]+ programs in [0-9.]+s.*" | tail -3 | tee -a "$RES"

# ---------- C: WARM (reuse the precompile cache, no compile) ----------
echo "" | tee -a "$RES"; echo "=== [C] WARM (reuse /tmp/unet_warm, no compile) $(date '+%T') ===" | tee -a "$RES"
echo "CMD: env -u TT_METAL_CCACHE_KERNEL_SUPPORT TT_METAL_CACHE=/tmp/unet_warm  pytest $TEST -k $SEL" | tee -a "$RES"
t0=$(date +%s)
env -u TT_METAL_CCACHE_KERNEL_SUPPORT TT_METAL_CACHE="$JIT_WARM" \
  timeout 1200 python3 -m pytest "$TEST" -k "$SEL" -p no:cacheprovider >/tmp/unet_C.log 2>&1
C=$(( $(date +%s)-t0 )); echo "[C] warm wall=${C}s" | tee -a "$RES"
sed -E 's/\x1b\[[0-9;]*m//g' /tmp/unet_C.log | grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)|PCC is [0-9.]+|[0-9]+ (passed|failed)" | tail -3 | tee -a "$RES"

# ---------- summary ----------
pcompile=$(sed -E 's/\x1b\[[0-9;]*m//g' /tmp/unet_B.log | grep -aoE "compiled [0-9]+ programs in [0-9.]+s" | tail -1)
echo "" | tee -a "$RES"; echo "================= SUMMARY =================" | tee -a "$RES"
printf "[A] COLD (inline)      : %5ds\n[B] PRECOMPILE warmup  : %5ds   (%s)\n[C] WARM (reuse)       : %5ds\n" \
  "$A" "$B" "$pcompile" "$C" | tee -a "$RES"
echo "inline serial compile  ~= A-C = $((A-C))s   (cold minus warm; same body, diff = inline compile)" | tee -a "$RES"
echo "parallel compile (B)   = $pcompile" | tee -a "$RES"
[ "$C" -gt 0 ] && echo "ship-the-cache (A/C)   = $(echo "scale=2;$A/$C"|bc)x   |  in-job two-pass (A/(B+C)) = $(echo "scale=2;$A/($B+$C)"|bc)x" | tee -a "$RES"
echo "DONE $(date)" | tee -a "$RES"
touch /tmp/unet_precompile_DONE
