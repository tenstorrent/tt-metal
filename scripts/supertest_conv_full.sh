#!/bin/bash
# SUPERTEST: FULL conv nightly dir — cold inline vs precompile (meta-collect) two-pass.
# Mirrors scripts/supertest.sh but targets the whole operations/conv dir that CI runs, and
# warms via UP_FRONT_META_COLLECT (the generic path --precompile / CI will use). The warm
# RUN (pass2) is what bounds the achievable CI speedup; pass1 is the (relocatable) warm cost.
#
#   A  cold inline           : fresh cache, full suite (== today's CI model)
#   B1 warm pass (meta)      : fresh cache, collect+parallel-compile, NO real exec (warm cost)
#   B2 warm run              : reuse B1 cache, full suite (== CI with a pre-warmed cache)
#
# --run-all (no -x) so failures never truncate; dispatch timeout raised so heavy conv3d/VAE
# ops aren't mistaken for hangs. Same selection + marker as CI for fidelity.
set -uo pipefail

WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
T="tests/ttnn/nightly/unit_tests/operations/conv"
MARK="not disable_fast_runtime_mode"
RES=/tmp/supertest_conv_full_results.txt
export SAFE_PYTEST_DISPATCH_TIMEOUT=120   # heavy conv ops are not hangs
cd "$WT"

rm -f /tmp/conv_full_DONE
echo "SUPERTEST conv-full start $(date)"            | tee "$RES"
echo "scope: full $T  (-m '$MARK'), --run-all, meta-collect, host=$(hostname) nproc=$(nproc)" | tee -a "$RES"

# ---------- Phase A: COLD INLINE ----------
rm -rf /tmp/conv_cold && mkdir -p /tmp/conv_cold
echo "" | tee -a "$RES"; echo "=== [A] COLD INLINE  $(date '+%T') ===" | tee -a "$RES"
A0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE=/tmp/conv_cold \
    scripts/run_safe_pytest.sh --run-all "$T" -m "$MARK" --timeout=600 > /tmp/conv_cold.log 2>&1
echo "[A] COLD exit=$?  wall=$(($(date +%s)-A0))s" | tee -a "$RES"
grep -E "= .*(passed|failed|skipped|xfailed|error).* (in|=)" /tmp/conv_cold.log | tail -1 | tee -a "$RES"

# ---------- Phase B1: WARM PASS (meta collect + parallel compile, hardware-free body) ----------
rm -rf /tmp/conv_warm && mkdir -p /tmp/conv_warm
echo "" | tee -a "$RES"; echo "=== [B1] WARM PASS (meta collect+compile)  $(date '+%T') ===" | tee -a "$RES"
B0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE=/tmp/conv_warm \
    UP_FRONT_COLLECT=1 UP_FRONT_META_COLLECT=1 \
    scripts/run_safe_pytest.sh --run-all "$T" -m "$MARK" -p up_front_collect_plugin -s > /tmp/conv_warm_pass1.log 2>&1
echo "[B1] PASS1 exit=$?  wall=$(($(date +%s)-B0))s" | tee -a "$RES"
grep "UP_FRONT_COLLECT:" /tmp/conv_warm_pass1.log | tee -a "$RES"

# ---------- Phase B2: WARM RUN (reuse B1 cache) ----------
echo "" | tee -a "$RES"; echo "=== [B2] WARM RUN (reuse cache)  $(date '+%T') ===" | tee -a "$RES"
C0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE=/tmp/conv_warm \
    scripts/run_safe_pytest.sh --run-all "$T" -m "$MARK" --timeout=600 > /tmp/conv_warm_pass2.log 2>&1
echo "[B2] PASS2 exit=$?  wall=$(($(date +%s)-C0))s" | tee -a "$RES"
grep -E "= .*(passed|failed|skipped|xfailed|error).* (in|=)" /tmp/conv_warm_pass2.log | tail -1 | tee -a "$RES"

# ---------- SUMMARY ----------
COLD=$(($(grep -oP '\[A\] COLD exit=\d+\s+wall=\K\d+' "$RES")))
P1=$(($(grep -oP '\[B1\] PASS1 exit=\d+\s+wall=\K\d+' "$RES")))
P2=$(($(grep -oP '\[B2\] PASS2 exit=\d+\s+wall=\K\d+' "$RES")))
echo "" | tee -a "$RES"; echo "================= SUMMARY =================" | tee -a "$RES"
printf "COLD inline:        %5ds (%dm%02ds)\n" $COLD $((COLD/60)) $((COLD%60)) | tee -a "$RES"
printf "WARM pass1 (warm):  %5ds (%dm%02ds)\n" $P1   $((P1/60))   $((P1%60))   | tee -a "$RES"
printf "WARM pass2 (reuse): %5ds (%dm%02ds)\n" $P2   $((P2/60))   $((P2%60))   | tee -a "$RES"
[ "$P2" -gt 0 ] && printf "WARM-REUSE SPEEDUP: %.2fx   (in-job two-pass %.2fx)\n" \
    "$(echo "scale=3;$COLD/$P2"|bc)" "$(echo "scale=3;$COLD/($P1+$P2)"|bc)" | tee -a "$RES"
echo "SUPERTEST conv-full done $(date)" | tee -a "$RES"
touch /tmp/conv_full_DONE
