#!/bin/bash
# §63: the user's v2 with the flushed atomic deferred: execute_local_semaphore_inc stages the atomic; if the payload writes
# have not landed it marks it pending and returns, and E1 drains it (flush + send) before the next packet's first NoC op,
# on an idle/blocked iteration, or at the end of the inner loop. Flags 13129088 (ERISC-only), fast dispatch.
# Validation (mesh + stream sanity), 4352 / mix / latency, then fused write+atomic with the harness flush flag off and
# forced on (temporary edit of tests/.../kernels/tt_fabric_test_kernels_utils.hpp, restored and checked).
# Labels dqpd_13129088_<suite>, dqpd_fl<off|on>_13129088. Env: ORIG_HDR, FLUSH_HDR.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
H=tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_kernels_utils.hpp
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
run dqpd_13129088_stream $SAN/test_fabric_stream_sanity.yaml || { echo ALL_DONE; exit 0; }
run dqpd_13129088_mesh $SAN/test_fabric_mesh_sanity_common.yaml || { echo ALL_DONE; exit 0; }
run dqpd_13129088_4352 fabric_exp/exp_2d_4352.yaml
run dqpd_13129088_mix fabric_exp/streams/p3_mixes.yaml
run dqpd_13129088_lat fabric_exp/exp_latency.yaml
run dqpd_floff_13129088 fabric_exp/streams/p3_fused.yaml
cp "$FLUSH_HDR" $H
run dqpd_flon_13129088 fabric_exp/streams/p3_fused.yaml
cp "$ORIG_HDR" $H
cmp $H "$ORIG_HDR" && echo "harness header restored"
echo ALL_DONE
