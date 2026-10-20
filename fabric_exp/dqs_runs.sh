#!/bin/bash
# §39 DRISC delay sweep: TT_FABRIC_EXP_DRISC_DELAY=d spins d cycles per packet on the offload DRISC only (profiling off).
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
run() {  # label yaml [env...]; returns 1 on failure or hang
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 $KA "${@:3}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    local why=$(grep -m1 -oE 'MMIO per-op timeout[^,]*' fabric_exp/results/$1.log 2>/dev/null)
    local last=$(grep -oE 'Running Test: [A-Za-z0-9_]+' fabric_exp/results/$1.log 2>/dev/null | tail -1)
    echo "$1: ${res:-NO RESULT (exit $status)}${why:+ | $why | last: ${last#Running Test: }}"
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
# Once the DRISC binds, cycles/packet = DRISC loop + d (slope 1), so DRISC loop = 2 * rate(d) - rate(2d).
for d in 0 32 64 96; do
    run dqs${d}_15226240_4352 fabric_exp/exp_2d_4352.yaml $O TT_FABRIC_EXP_DRISC_DELAY=$d || exit 0
    run dqs${d}_15226240_mix fabric_exp/streams/p3_mixes.yaml $O TT_FABRIC_EXP_DRISC_DELAY=$d || exit 0
done
echo ALL_DONE
