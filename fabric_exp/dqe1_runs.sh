#!/bin/bash
# §54: E1's per-packet time on 1-hop writes, current best ERISC-only v2 build (13129088 + E1 bits 27/28 = 415782272),
# fast dispatch, 4352 B suite. (1) slack test: TT_FABRIC_EXP_RX_DELAY spins on E1 per packet; E1 busy ~= the E0-bound
# interval minus the largest delay that leaves 1-hop writes unchanged. (2) phase profile level 1 (busy/blocked/idle per
# iteration, no phase timers). Labels dqe1_rxd<D>_415782272_4352, dqe1_prof1_415782272_4352.
cd /home/aagarwal/tt-metal
F=TT_FABRIC_EXP_FLAGS=415782272
run() {
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" fabric_exp/exp_2d_4352.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 $F "${@:2}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)}"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"; return 1
    fi
}
tt-smi -r all > /dev/null 2>&1
for d in ${RXD:-20 40 50 60 80}; do run dqe1_rxd${d}_415782272_4352 TT_FABRIC_EXP_RX_DELAY=$d; done
run dqe1_prof1_415782272_4352 TT_FABRIC_PHASE_PROFILE=1
echo ALL_DONE
