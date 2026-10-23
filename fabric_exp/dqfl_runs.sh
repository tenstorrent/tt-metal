#!/bin/bash
# §58: fused write+atomic with the harness's flush flag off (as every test sends it) and forced on (temporary edit of
# tests/.../kernels/tt_fabric_test_kernels_utils.hpp, restored and checked). Best ERISC-only v2 build (348673408),
# fabric_exp/streams/p3_fused.yaml. Labels dqfl_<off|on>_348673408. Env: ORIG_HDR, FLUSH_HDR.
cd /home/aagarwal/tt-metal
H=tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_kernels_utils.hpp
run() {
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" fabric_exp/streams/p3_fused.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=348673408 > /dev/null 2>&1
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"; fi
}
tt-smi -r all > /dev/null 2>&1
run dqfl_off_348673408
cp "$FLUSH_HDR" $H
run dqfl_on_348673408
cp "$ORIG_HDR" $H
cmp $H "$ORIG_HDR" && echo "harness header restored"
echo ALL_DONE
