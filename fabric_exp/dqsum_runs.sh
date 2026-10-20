#!/bin/bash
# §44 summary: same-tree stock router (no flags) at the firmware keepalive (register untouched, 8000, as main ships) and with the keepalive
# fix (1000), on the suites the offload versions ran. Fast dispatch, no BW telemetry.
cd /home/aagarwal/tt-metal
run() {
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 900 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 "${@:3}" > /dev/null 2>&1
    local s=$? r=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log)
    echo "$1: ${r:-NO RESULT (exit $s)}"
}
tt-smi -r all > /dev/null 2>&1
# kafw: timeout register left at the firmware's 8000 (router does not touch it, as on main); ka1000: keepalive fix.
for ka in fw 1000; do
    E=""; [ $ka != fw ] && E=TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=$ka
    run dqs_stock_ka${ka}_4352 fabric_exp/exp_2d_4352.yaml $E
    run dqs_stock_ka${ka}_mix fabric_exp/streams/p3_mixes.yaml $E
    run dqs_stock_ka${ka}_4352_trim fabric_exp/exp_2d_trim_4352.yaml $E
    run dqs_stock_ka${ka}_lat fabric_exp/exp_latency.yaml $E
done
echo ALL_DONE
