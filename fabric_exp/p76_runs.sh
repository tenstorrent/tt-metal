#!/bin/bash
# §76: 1D on the current tree (commit 5ef745cb2a0: v2 + the user's s74, flags 13129088, keepalive 1000) against the stock
# router ("main": v1 kernel, no flags, firmware keepalive), on fabric_exp/exp_1d_4352.yaml (Linear 1-hop E/W, N/S,
# bidirectional, 3-hop; linear and unidirectional multicast; half ring; neighbor exchange; write + scatter, 4352 B).
#   v: 1D validation on v2: test_fabric_sanity_at_least_2x2_mesh.yaml (Linear, Ring, Mesh)       label p76_val
#   a: current unprofiled, +20 spin on E0 / E1; stock at the firmware keepalive and at 1000     labels p76_{cur,txd20,rxd20,stock_kafw,stock_ka1000}
#   b: level-2 single-phase profiles of the current tree: E0 1 2 4 8 1024, E1 16 32 64 128 2048  labels p76_m<mask>
# Broker job: no resets, line-buffered output, heartbeat. Fast dispatch, telemetry off. Router sources are not touched.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
Y=fabric_exp/exp_1d_4352.yaml
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
check() {  # label
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
cur() {  # label yaml [ENV=VALUE ...]   v2, best flags, keepalive 1000
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 "${@:3}" \
        | grep --line-buffered -v 'Running Test'
    check $1
}
stock() {  # label yaml [ENV=VALUE ...]   v1 kernel, no experiment flags
    env -u TT_METAL_SLOW_DISPATCH_MODE -u TT_FABRIC_KERNEL_VERSION -u TT_FABRIC_EXP_FLAGS timeout 1200 \
        fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 "${@:3}" | grep --line-buffered -v 'Running Test'
    check $1
}
case "${1:-a}" in
v)  cur p76_val $SAN/test_fabric_sanity_at_least_2x2_mesh.yaml || { echo STOPPED; exit 1; } ;;
a)  cur p76_cur $Y || { echo STOPPED; exit 1; }
    stock p76_stock_kafw $Y || { echo STOPPED; exit 1; }
    stock p76_stock_ka1000 $Y TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 || { echo STOPPED; exit 1; }
    cur p76_txd20 $Y TT_FABRIC_EXP_TX_DELAY=20 || { echo STOPPED; exit 1; }
    cur p76_rxd20 $Y TT_FABRIC_EXP_RX_DELAY=20 || { echo STOPPED; exit 1; } ;;
b)  for m in 1 2 4 8 1024 16 32 64 128 2048; do
        cur p76_m$m $Y TT_FABRIC_PHASE_PROFILE=2 TT_FABRIC_PHASE_MASK=$m TT_FABRIC_PHASE_NO_ITER=1 || { echo STOPPED; exit 1; }
    done ;;
esac
echo ALL_DONE
