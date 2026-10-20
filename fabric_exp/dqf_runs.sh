#!/bin/bash
# §38 option 1 (user): DRISC offload (15226240) under FAST dispatch, which runs no host L1 barrier over the DRAM tiles
# during traffic. Validation (stream + mesh sanity twice), same-tree FD baselines (${PB}_*), benchmarks, small, latency.
# Any offload failure or hang stops the script. Labels: <P>_* and <PB>_* (default dqf/dqfb; args 1 and 2).
cd /home/aagarwal/tt-metal
P=${1:-dqf}
PB=${2:-dqfb}
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
W="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
run() {  # label yaml [env...]; returns 1 on failure or hang
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 $KA "${@:3}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    local why=$(grep -m1 -oE 'MMIO per-op timeout[^,]*' fabric_exp/results/$1.log 2>/dev/null)
    local last=$(grep -oE 'Running Test: [A-Za-z0-9_]+' fabric_exp/results/$1.log 2>/dev/null | tail -1)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)${why:+ | $why | last: ${last#Running Test: }}"
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
echo "== phase 1: validation (fast dispatch)"
run ${P}_15226240_stream $SAN/test_fabric_stream_sanity.yaml $O || exit 0
run ${P}_15226240_mesh $SAN/test_fabric_mesh_sanity_common.yaml $O || exit 0
run ${P}_15226240_mesh2 $SAN/test_fabric_mesh_sanity_common.yaml $O || exit 0
echo "== phase 2a: same-tree baselines"
run ${PB}_13129088_4352 fabric_exp/exp_2d_4352.yaml $B
run ${PB}_13129088_mix fabric_exp/streams/p3_mixes.yaml $B
run ${PB}_13129088_4352_trim fabric_exp/exp_2d_trim_4352.yaml $B
run ${PB}_13129088_mix_trim fabric_exp/streams/p3_mixes_trim.yaml $B
echo "== phase 2: benchmarks"
run ${P}_15226240_4352 fabric_exp/exp_2d_4352.yaml $O || exit 0
run ${P}_15226240_mix fabric_exp/streams/p3_mixes.yaml $O || exit 0
run ${P}_15226240_4352_trim fabric_exp/exp_2d_trim_4352.yaml $O || exit 0
run ${P}_15226240_mix_trim fabric_exp/streams/p3_mixes_trim.yaml $O || exit 0
run ${P}_15226240w32_4352 fabric_exp/exp_2d_4352.yaml $O $W || exit 0
run ${P}_15226240w32_mix fabric_exp/streams/p3_mixes.yaml $O $W || exit 0
echo "== phase 3: small packets and latency"
run ${PB}_13129088_small fabric_exp/exp_2d_trim_small.yaml $B
run ${P}_15226240_small fabric_exp/exp_2d_trim_small.yaml $O || exit 0
run ${PB}_13129088_lat fabric_exp/exp_latency.yaml $B
run ${P}_15226240_lat fabric_exp/exp_latency.yaml $O || exit 0
echo ALL_DONE
