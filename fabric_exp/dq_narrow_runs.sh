#!/bin/bash
# §38 follow-up (user-requested): does the mesh-sanity hang reproduce with scatter + multicast + 2 links alone?
# Stops at the first hang. Labels: dqn_*.
cd /home/aagarwal/tt-metal
ENV="TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_SLOW_DISPATCH_MODE=1 TT_FABRIC_EXP_FLAGS=15226240"
run() {  # label yaml; returns 1 on failure or hang
    timeout 900 fabric_exp/run_exp.sh "$1" "$2" $ENV > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    local why=$(grep -m1 -oE 'MMIO per-op timeout[^,]*|TT_THROW[^\n]{0,80}|Timeout[^\n]{0,80}' fabric_exp/results/$1.log 2>/dev/null)
    local last=$(grep -oE 'Running Test: [A-Za-z0-9_]+' fabric_exp/results/$1.log 2>/dev/null | tail -1)
    echo "$1: ${res:-NO RESULT (exit $status)} ${why:+| $why} | last: ${last#Running Test: }"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1
        echo "reset done; $(lspci | grep -c -i 1e52) cards on PCIe"
        return 1
    fi
    return 0
}
tt-smi -r all > /dev/null 2>&1
run dqn_scatter_mcast_2l_1024 fabric_exp/dq/scatter_mcast_2l_1024.yaml || exit 0
run dqn_scatter_mcast_2l_sizes fabric_exp/dq/scatter_mcast_2l_sizes.yaml || exit 0
run dqn_mesh_1024_seq fabric_exp/dq/mesh_1024_seq.yaml || exit 0
echo ALL_PASSED
