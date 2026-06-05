#!/bin/bash
# SUPERTEST: cold inline vs precompile two-pass (fast collect) for an op's nightly suite.
# Both paths use the committed fast PCC. Fast-collect monkeyswaps apply to pass 1.
# --run-all (no -x) so failures don't truncate; each phase timed; continues if a phase errors/hangs.
#
# Usage: supertest.sh <name> <pytest target/args...>
#   env SUPERTEST_ORDER=cold_first (default) | twopass_first
#   env SAFE_PYTEST_DISPATCH_TIMEOUT=<sec> (default 5; raise for slow ops to avoid false hangs)
set -uo pipefail

WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
NAME="${1:-conv2d}"; shift || true
if [ "$#" -eq 0 ]; then
    T=("tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py")
else
    T=("$@")
fi
ORDER="${SUPERTEST_ORDER:-cold_first}"
DT="${SAFE_PYTEST_DISPATCH_TIMEOUT:-5}"
RES="/tmp/supertest_${NAME}_results.txt"
cd "$WT"
rm -f "/tmp/supertest_${NAME}_DONE"
echo "SUPERTEST[$NAME] start $(date)  order=$ORDER dispatch_timeout=${DT}s loguru=${LOGURU_LEVEL:-default}" | tee "$RES"
echo "targets: ${T[*]}  (--run-all, PYTHONPATH=$WT, fast PCC live)" | tee -a "$RES"

COLD=0; P1=0; P2=0; TWO=0

phase_cold() {
    rm -rf "/tmp/super_${NAME}_cold" && mkdir -p "/tmp/super_${NAME}_cold"
    echo "" | tee -a "$RES"; echo "=== [A] COLD INLINE  start $(date '+%T') ===" | tee -a "$RES"
    local t0 t1; t0=$(date +%s)
    SAFE_PYTEST_DISPATCH_TIMEOUT="$DT" PYTHONPATH="$WT" TT_METAL_CACHE="/tmp/super_${NAME}_cold" \
        scripts/run_safe_pytest.sh --run-all "${T[@]}" > "/tmp/super_${NAME}_cold.log" 2>&1
    local rc=$?; t1=$(date +%s); COLD=$((t1-t0))
    echo "[A] COLD exit=$rc  wall=${COLD}s" | tee -a "$RES"
    grep -E "passed|failed|skipped|error|xfailed" "/tmp/super_${NAME}_cold.log" | tail -2 | tee -a "$RES"
}

phase_twopass() {
    rm -rf "/tmp/super_${NAME}_pre" && mkdir -p "/tmp/super_${NAME}_pre"
    echo "" | tee -a "$RES"; echo "=== [B1] PRECOMPILE FAST COLLECT  start $(date '+%T') ===" | tee -a "$RES"
    local t0 t1; t0=$(date +%s)
    SAFE_PYTEST_DISPATCH_TIMEOUT="$DT" PYTHONPATH="$WT" UP_FRONT_COLLECT=1 UP_FRONT_FAST_COLLECT=1 TT_METAL_CACHE="/tmp/super_${NAME}_pre" \
        scripts/run_safe_pytest.sh --run-all "${T[@]}" -p up_front_collect_plugin > "/tmp/super_${NAME}_pass1.log" 2>&1
    local rc1=$?; t1=$(date +%s); P1=$((t1-t0))
    echo "[B1] PASS1 exit=$rc1  wall=${P1}s" | tee -a "$RES"
    grep "UP_FRONT_COLLECT:" "/tmp/super_${NAME}_pass1.log" | tee -a "$RES"
    grep -E "passed|failed|skipped|error" "/tmp/super_${NAME}_pass1.log" | tail -1 | tee -a "$RES"

    echo "" | tee -a "$RES"; echo "=== [B2] WARM REAL RUN  start $(date '+%T') ===" | tee -a "$RES"
    local t2 t3; t2=$(date +%s)
    SAFE_PYTEST_DISPATCH_TIMEOUT="$DT" PYTHONPATH="$WT" TT_METAL_CACHE="/tmp/super_${NAME}_pre" \
        scripts/run_safe_pytest.sh --run-all "${T[@]}" > "/tmp/super_${NAME}_pass2.log" 2>&1
    local rc2=$?; t3=$(date +%s); P2=$((t3-t2)); TWO=$((P1+P2))
    echo "[B2] PASS2 exit=$rc2  wall=${P2}s" | tee -a "$RES"
    grep -E "passed|failed|skipped|error|xfailed" "/tmp/super_${NAME}_pass2.log" | tail -2 | tee -a "$RES"
}

if [ "$ORDER" = "twopass_first" ]; then phase_twopass; phase_cold; else phase_cold; phase_twopass; fi

echo "" | tee -a "$RES"
echo "================= SUMMARY [$NAME] =================" | tee -a "$RES"
printf "COLD inline:          %5ds  (%dm%02ds)\n" $COLD $((COLD/60)) $((COLD%60)) | tee -a "$RES"
printf "NEW pass1 (collect):  %5ds  (%dm%02ds)\n" $P1 $((P1/60)) $((P1%60)) | tee -a "$RES"
printf "NEW pass2 (warm):     %5ds  (%dm%02ds)\n" $P2 $((P2/60)) $((P2%60)) | tee -a "$RES"
printf "NEW two-pass total:   %5ds  (%dm%02ds)\n" $TWO $((TWO/60)) $((TWO%60)) | tee -a "$RES"
if [ "$TWO" -gt 0 ] && [ "$TWO" -lt "$COLD" ]; then
    printf "RESULT: precompile WINS by %ds (%sx)\n" $((COLD-TWO)) "$(awk "BEGIN{printf \"%.2f\", $COLD/$TWO}")" | tee -a "$RES"
elif [ "$TWO" -gt 0 ]; then
    printf "RESULT: precompile LOSES by %ds (%sx)\n" $((TWO-COLD)) "$(awk "BEGIN{printf \"%.2f\", $TWO/$COLD}")" | tee -a "$RES"
fi
echo "SUPERTEST[$NAME] done $(date)" | tee -a "$RES"
touch "/tmp/supertest_${NAME}_DONE"
