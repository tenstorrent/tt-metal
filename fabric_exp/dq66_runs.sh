#!/bin/bash
# §66, first runs on bh-lb-120-a08u24 (through the tt-device-mcp broker: no resets here, output streamed, heartbeat
# for the broker's 300 s silence reaper).
# 1. Host calibration: v1 at 13129088 on 4352 / mix, against a04u24's dqib / dqv1b (v1 is unchanged since).
# 2. The user's §66 deferral (variants/s66): validation (stream + mesh), 4352 / mix / latency, flushed-signal yaml.
# Flags 13129088, fast dispatch, telemetry off. Labels a8v1_13129088_<suite>, dq66_13129088_<suite>.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
run() {  # kernel_version label yaml
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=$1 timeout 1200 fabric_exp/run_exp.sh "$2" "$3" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$2.log 2>/dev/null)
    echo "$2: ${res:-NO RESULT (exit $status)}"
    [ $status -eq 0 ] && [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
run 1 a8v1_13129088_4352 fabric_exp/exp_2d_4352.yaml || { echo STOPPED; exit 1; }
run 1 a8v1_13129088_mix fabric_exp/streams/p3_mixes.yaml || { echo STOPPED; exit 1; }
run 2 dq66_13129088_stream $SAN/test_fabric_stream_sanity.yaml || { echo STOPPED; exit 1; }
run 2 dq66_13129088_mesh $SAN/test_fabric_mesh_sanity_common.yaml || { echo STOPPED; exit 1; }
run 2 dq66_13129088_4352 fabric_exp/exp_2d_4352.yaml || { echo STOPPED; exit 1; }
run 2 dq66_13129088_mix fabric_exp/streams/p3_mixes.yaml || { echo STOPPED; exit 1; }
run 2 dq66_13129088_lat fabric_exp/exp_latency.yaml || { echo STOPPED; exit 1; }
run 2 dq66_13129088_flushmix fabric_exp/streams/flush_mix.yaml || { echo STOPPED; exit 1; }
echo ALL_DONE
