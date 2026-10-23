#!/bin/bash
# §65: the user's slimmer deferral: every flushed atomic is staged and marked pending (no inline "already landed"
# check), the idle-iteration drain is a non-blocking check before the completion block (partial batches wait while an
# atomic is pending), the inner-loop-end drain moves to the main-loop exit, and deferral is limited to ERISCs with no
# sender channels. Flags 13129088, fast dispatch. Validation (stream + mesh), 4352 / mix / latency, and the
# flushed-signal yaml (streams/flush_mix.yaml). Labels dqps_13129088_<suite>.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
run() {  # label yaml
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"; return 1
    fi
}
tt-smi -r all > /dev/null 2>&1
run dqps_13129088_stream $SAN/test_fabric_stream_sanity.yaml || { echo ALL_DONE; exit 0; }
run dqps_13129088_mesh $SAN/test_fabric_mesh_sanity_common.yaml || { echo ALL_DONE; exit 0; }
run dqps_13129088_4352 fabric_exp/exp_2d_4352.yaml
run dqps_13129088_mix fabric_exp/streams/p3_mixes.yaml
run dqps_13129088_lat fabric_exp/exp_latency.yaml
run dqps_13129088_flushmix fabric_exp/streams/flush_mix.yaml
echo ALL_DONE
