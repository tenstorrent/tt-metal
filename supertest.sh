#!/bin/bash
# SUPERTEST: cold inline vs precompile two-pass (fast collect) for an op's nightly suite.
# Both paths use the committed fast PCC (torch.corrcoef). Fast-collect monkeyswaps apply to pass 1.
# --run-all (no -x) so failures don't truncate; each phase timed; continues if a phase errors/hangs.
#
# Usage: supertest.sh <name> <test_target> [more_targets...]
#   e.g. supertest.sh layernorm tests/.../fused/test_layernorm.py tests/.../fused/test_layernorm_sharded.py
set -uo pipefail

WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
NAME="${1:-conv2d}"; shift || true
if [ "$#" -eq 0 ]; then
    T=("tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py")
else
    T=("$@")
fi
RES="/tmp/supertest_${NAME}_results.txt"
cd "$WT"

rm -f "/tmp/supertest_${NAME}_DONE"
echo "SUPERTEST[$NAME] start $(date)" | tee "$RES"
echo "targets: ${T[*]}  (--run-all, PYTHONPATH=$WT, fast PCC live)" | tee -a "$RES"

# ---------- Phase A: COLD INLINE ----------
rm -rf "/tmp/super_${NAME}_cold" && mkdir -p "/tmp/super_${NAME}_cold"
echo "" | tee -a "$RES"; echo "=== [A] COLD INLINE  start $(date '+%T') ===" | tee -a "$RES"
A0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE="/tmp/super_${NAME}_cold" \
    scripts/run_safe_pytest.sh --run-all "${T[@]}" > "/tmp/super_${NAME}_cold.log" 2>&1
ARC=$?
A1=$(date +%s)
echo "[A] COLD exit=$ARC  wall=$((A1-A0))s" | tee -a "$RES"
grep -E "passed|failed|skipped|error|xfailed" "/tmp/super_${NAME}_cold.log" | tail -2 | tee -a "$RES"

# ---------- Phase B1: PRECOMPILE FAST COLLECT ----------
rm -rf "/tmp/super_${NAME}_pre" && mkdir -p "/tmp/super_${NAME}_pre"
echo "" | tee -a "$RES"; echo "=== [B1] PRECOMPILE FAST COLLECT  start $(date '+%T') ===" | tee -a "$RES"
B0=$(date +%s)
PYTHONPATH="$WT" UP_FRONT_COLLECT=1 UP_FRONT_FAST_COLLECT=1 TT_METAL_CACHE="/tmp/super_${NAME}_pre" \
    scripts/run_safe_pytest.sh --run-all "${T[@]}" -p up_front_collect_plugin -s > "/tmp/super_${NAME}_pass1.log" 2>&1
B1RC=$?
B1=$(date +%s)
echo "[B1] PASS1 exit=$B1RC  wall=$((B1-B0))s" | tee -a "$RES"
grep "UP_FRONT_COLLECT:" "/tmp/super_${NAME}_pass1.log" | tee -a "$RES"
grep -E "passed|failed|skipped|error" "/tmp/super_${NAME}_pass1.log" | tail -1 | tee -a "$RES"

# ---------- Phase B2: WARM REAL RUN ----------
echo "" | tee -a "$RES"; echo "=== [B2] WARM REAL RUN  start $(date '+%T') ===" | tee -a "$RES"
C0=$(date +%s)
PYTHONPATH="$WT" TT_METAL_CACHE="/tmp/super_${NAME}_pre" \
    scripts/run_safe_pytest.sh --run-all "${T[@]}" > "/tmp/super_${NAME}_pass2.log" 2>&1
C2RC=$?
C1=$(date +%s)
echo "[B2] PASS2 exit=$C2RC  wall=$((C1-C0))s" | tee -a "$RES"
grep -E "passed|failed|skipped|error|xfailed" "/tmp/super_${NAME}_pass2.log" | tail -2 | tee -a "$RES"

# ---------- SUMMARY ----------
COLD=$((A1-A0)); P1=$((B1-B0)); P2=$((C1-C0)); TWO=$((P1+P2))
echo "" | tee -a "$RES"
echo "================= SUMMARY [$NAME] =================" | tee -a "$RES"
printf "COLD inline:          %5ds  (%dm%02ds)\n" $COLD $((COLD/60)) $((COLD%60)) | tee -a "$RES"
printf "NEW pass1 (collect):  %5ds  (%dm%02ds)\n" $P1 $((P1/60)) $((P1%60)) | tee -a "$RES"
printf "NEW pass2 (warm):     %5ds  (%dm%02ds)\n" $P2 $((P2/60)) $((P2%60)) | tee -a "$RES"
printf "NEW two-pass total:   %5ds  (%dm%02ds)\n" $TWO $((TWO/60)) $((TWO%60)) | tee -a "$RES"
if [ "$TWO" -lt "$COLD" ]; then
    printf "RESULT: precompile WINS by %ds (%sx)\n" $((COLD-TWO)) "$(awk "BEGIN{printf \"%.2f\", $COLD/$TWO}")" | tee -a "$RES"
else
    printf "RESULT: precompile LOSES by %ds (%sx)\n" $((TWO-COLD)) "$(awk "BEGIN{printf \"%.2f\", $TWO/$COLD}")" | tee -a "$RES"
fi
echo "SUPERTEST[$NAME] done $(date)" | tee -a "$RES"
touch "/tmp/supertest_${NAME}_DONE"
