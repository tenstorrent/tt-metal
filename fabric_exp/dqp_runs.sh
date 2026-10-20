#!/bin/bash
# §39 DRISC timing: the phase profiler (TT_FABRIC_PHASE_PROFILE) on E0, E1 and the offload DRISC (block 2 of E1's
# profile buffer). Fast dispatch. Level 1 = counters + per-iteration busy/blocked/idle, level 2 adds phase timers.
# Default: the DRISC alone (TT_FABRIC_DRISC_PHASE_PROFILE) at levels 1 and 2 on the 4352 B suite and the command mixes.
# Arg `all`: first every core (TT_FABRIC_PHASE_PROFILE), offload and same-tree baseline (13129088). Any failure or hang stops the script. Summaries: fabric_exp/phase_report.py <label>_phase.csv.
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
run() {  # label yaml [env...]; returns 1 on failure or hang
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 $KA "${@:3}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    local why=$(grep -m1 -oE 'MMIO per-op timeout[^,]*' fabric_exp/results/$1.log 2>/dev/null)
    local last=$(grep -oE 'Running Test: [A-Za-z0-9_]+' fabric_exp/results/$1.log 2>/dev/null | tail -1)
    echo "$1: ${res:-NO RESULT (exit $status)} drisc rows=$(python3 -c "import csv,sys; print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r['erisc']=='2'))" fabric_exp/results/$1_phase.csv 2>/dev/null)${why:+ | $why | last: ${last#Running Test: }}"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1
        echo "reset done; $(lspci | grep -c -i 1e52) cards on PCIe"
        return 1
    fi
    return 0
}
O=TT_FABRIC_EXP_FLAGS=15226240
B=TT_FABRIC_EXP_FLAGS=13129088
tt-smi -r all > /dev/null 2>&1
if [ "${1:-}" = all ]; then  # every core profiled (ERISCs too: E0-bound at ~287, see RESULTS §39)
run dqp1_15226240_4352 fabric_exp/exp_2d_4352.yaml $O TT_FABRIC_PHASE_PROFILE=1 || exit 0
run dqp2_15226240_4352 fabric_exp/exp_2d_4352.yaml $O TT_FABRIC_PHASE_PROFILE=2 || exit 0
run dqpb2_13129088_4352 fabric_exp/exp_2d_4352.yaml $B TT_FABRIC_PHASE_PROFILE=2 || exit 0
run dqp2_15226240_mix fabric_exp/streams/p3_mixes.yaml $O TT_FABRIC_PHASE_PROFILE=2 || exit 0
run dqpb2_13129088_mix fabric_exp/streams/p3_mixes.yaml $B TT_FABRIC_PHASE_PROFILE=2 || exit 0
fi
# DRISC only (TT_FABRIC_DRISC_PHASE_PROFILE): ERISCs unprofiled, so the DRISC sees full traffic.
run dqd1_15226240_4352 fabric_exp/exp_2d_4352.yaml $O TT_FABRIC_DRISC_PHASE_PROFILE=1 || exit 0
run dqd2_15226240_4352 fabric_exp/exp_2d_4352.yaml $O TT_FABRIC_DRISC_PHASE_PROFILE=2 || exit 0
run dqd1_15226240_mix fabric_exp/streams/p3_mixes.yaml $O TT_FABRIC_DRISC_PHASE_PROFILE=1 || exit 0
run dqd2_15226240_mix fabric_exp/streams/p3_mixes.yaml $O TT_FABRIC_DRISC_PHASE_PROFILE=2 || exit 0
echo ALL_DONE
