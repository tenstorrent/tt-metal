#!/bin/bash
# §43: lap timer over the DRISC main loop (scratchpad lap.py on top of the user's v4; v4 restored at the end).
cd /home/aagarwal/tt-metal
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
python3 $S/lap.py ${LAPV:-lap} || exit 0
tt-smi -r all > /dev/null 2>&1
for y in ${LAPY:-mix:fabric_exp/streams/p3_mixes.yaml 4352:fabric_exp/exp_2d_4352.yaml}; do
    l=${LAPL:-dqlap}_15226240_${y%%:*}
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 900 fabric_exp/run_exp.sh $l ${y#*:} TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=15226240 TT_FABRIC_DRISC_PHASE_PROFILE=2 \
        TT_FABRIC_PHASE_MASK=0 TT_FABRIC_PHASE_NO_ITER=1 > /dev/null 2>&1
    s=$?; r=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$l.log)
    echo "$l: ${r:-NO RESULT (exit $s)}"
    if [ $s -ne 0 ] || ! echo "$r" | grep -q ", 0 failed"; then pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; break; fi
done
python3 $S/lap.py v4
echo ALL_DONE
