#!/bin/bash
# §45: neighbor exchange (1D speedy path) on main's router: stock, no flags, BW telemetry off, keepalive at the
# firmware's 8000 (kafw) and with the keepalive fix (ka1000). Sweep = all-pairs exchange; oneway = one sender.
cd /home/aagarwal/tt-metal
run() {
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 900 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 "${@:3}" > /dev/null 2>&1
    local s=$? r=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log)
    echo "$1: ${r:-NO RESULT (exit $s)}"
}
tt-smi -r all > /dev/null 2>&1
for ka in fw 1000; do
    E=""; [ $ka != fw ] && E=TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=$ka
    run dqnbr_stock_ka${ka}_sweep fabric_exp/exp_nbr_sweep.yaml $E
    run dqnbr_stock_ka${ka}_oneway fabric_exp/exp_nbr_oneway.yaml $E
done
echo ALL_DONE
