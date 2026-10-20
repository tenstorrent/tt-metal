#!/bin/bash
# P3: run p3_mixes.yaml (usage: p3_runs.sh [label prefix]) on stock (branch worktree) and the study's builds (experiment tree).
cd /home/aagarwal/tt-metal
P=${1:-p3}  # label prefix for result files
Y=fabric_exp/streams/p3_mixes.yaml
C="TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
check() {  # label log
    echo "$1: $(grep -oE 'Tests: .*executed, [0-9]+ failed' $2) $(grep -cE 'TT_FATAL|TT_THROW' $2) fatal"
}
tt-smi -r all > /dev/null 2>&1
timeout 1200 fabric_exp/streams/run_branch.sh ${P}_stock $Y > /dev/null || { pkill -x test_tt_fabric; tt-smi -r all > /dev/null 2>&1; }
check ${P}_stock fabric_exp/streams/results/${P}_stock.log
for cfg in "546176|TT_FABRIC_EXP_FLAGS=546176" "13129088|TT_FABRIC_EXP_FLAGS=13129088" \
           "13129088_w32|TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"; do
    l=${P}_${cfg%%|*}
    timeout 1200 fabric_exp/run_exp.sh $l $Y $C ${cfg#*|} > /dev/null || { pkill -x test_tt_fabric; tt-smi -r all > /dev/null 2>&1; }
    check $l fabric_exp/results/$l.log
done
echo P3_RUNS_DONE
