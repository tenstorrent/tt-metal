#!/bin/bash
# §39: DRISC phase timers one phase at a time (TT_FABRIC_PHASE_MASK, no per-iteration timing), to check how much
# timing every phase at once inflates each phase. Offload 15226240 on the command mixes.
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
O=TT_FABRIC_EXP_FLAGS=15226240
tt-smi -r all > /dev/null 2>&1
P=${1:-dqm}
for m in visit:0x800 admit:0x10 local:0x40 forward:0x20 completion:0x80; do
    l=${P}_${m%%:*}_15226240_mix
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 900 fabric_exp/run_exp.sh $l fabric_exp/streams/p3_mixes.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 $KA $O TT_FABRIC_DRISC_PHASE_PROFILE=2 TT_FABRIC_PHASE_MASK=${m##*:} \
        TT_FABRIC_PHASE_NO_ITER=1 > /dev/null 2>&1
    s=$?; r=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$l.log)
    echo "$l: ${r:-NO RESULT (exit $s)}"
    if [ $s -ne 0 ] || ! echo "$r" | grep -q ", 0 failed"; then pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; exit 0; fi
done
echo ALL_DONE
