#!/bin/bash
# §72: where the current tree stands (the user's s71, unchanged) and how each packet's cycles split between E0 (TX) and
# E1 (RX). Two yamls: streams/p72_profile.yaml (1-hop E/W, N/S write and scatter; E/W fused, atomic, 16 B write;
# forwarded 4352 B and 16 B writes) and exp_2d_4352.yaml (adds bidirectional, 3-hop, multicast). Per yaml: no profiling;
# slack with a +20 spin on E0 (TT_FABRIC_EXP_TX_DELAY) and on E1 (TT_FABRIC_EXP_RX_DELAY); level-2 single-phase
# profiles (TT_FABRIC_PHASE_MASK, no per-iteration timing): E0 send 1, TX-queue spin 2, completion 4, connection 8,
# whole sender visit 1024; E1 decode/admit 16, forward 32, local 64, completion 128, whole receiver step 2048.
# Broker job: no resets, line-buffered output, heartbeat. Labels p72_<yaml>_<run>. Flags 13129088, fast dispatch,
# telemetry off. Router sources are not touched.
cd /home/aagarwal/tt-metal
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
run() {  # label yaml [ENV=VALUE ...]
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 600 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 "${@:3}" \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
for y in p72:fabric_exp/streams/p72_profile.yaml 4352:fabric_exp/exp_2d_4352.yaml; do
    n=${y%%:*}; Y=${y#*:}
    run p72_${n}_noprof $Y || { echo STOPPED; exit 1; }
    run p72_${n}_txd20 $Y TT_FABRIC_EXP_TX_DELAY=20 || { echo STOPPED; exit 1; }
    run p72_${n}_rxd20 $Y TT_FABRIC_EXP_RX_DELAY=20 || { echo STOPPED; exit 1; }
    for m in 1 2 4 8 1024 16 32 64 128 2048; do
        run p72_${n}_m$m $Y TT_FABRIC_PHASE_PROFILE=2 TT_FABRIC_PHASE_MASK=$m TT_FABRIC_PHASE_NO_ITER=1 || { echo STOPPED; exit 1; }
    done
done
echo ALL_DONE
