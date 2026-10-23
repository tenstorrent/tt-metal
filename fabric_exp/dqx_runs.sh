#!/bin/bash
# §53: E1 experiments in router v2 (TT_FABRIC_KERNEL_VERSION=2), fast dispatch. Bits 26 (one header word-40 load),
# 27 (stateful NoC1 forwarding on two-ERISC E/W routers), 28 (compile-time admit stream reads, local-only early
# return). Per variant: mesh validation, then 4352 / mix / latency. A failing variant is reset and skipped.
# Labels: dqx_<flags>_<suite>.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
run() {  # label yaml flags; returns 1 on failure or hang
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=$3 > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1
        echo "reset done; $(lspci | grep -c -i 1e52) cards on PCIe"
        return 1
    fi
    return 0
}
variant() {  # flags suites...
    local f=$1; shift
    run dqx_${f}_mesh $SAN/test_fabric_mesh_sanity_common.yaml $f || return 0
    for s in "$@"; do
        case $s in
            4352) run dqx_${f}_4352 fabric_exp/exp_2d_4352.yaml $f || return 0 ;;
            mix) run dqx_${f}_mix fabric_exp/streams/p3_mixes.yaml $f || return 0 ;;
            lat) run dqx_${f}_lat fabric_exp/exp_latency.yaml $f || return 0 ;;
        esac
    done
}
tt-smi -r all > /dev/null 2>&1
echo "== ERISC-only"
for f in ${DQX_VARIANTS:-482891136 13129088 80237952 281564544 147346816}; do variant $f 4352 mix lat; done
echo "== offload"
for f in ${DQX_OFFLOAD_VARIANTS-15226240 82335104}; do variant $f 4352 mix; done
echo ALL_DONE
