#!/bin/bash
# forced_mocks.sh: each support-matrix TT_FATAL of validate_router_config triggered once, on a host built with the
# temporary force patch (force.patch, never committed): a bh_2d mock (P150x8 descriptor, two ERISCs, 2D mesh) per check,
# with TT_DEBUG_FORCE_ROUTER_ARGS / TT_DEBUG_FORCE_ROUTER_ARCH setting the values that check rejects. Each run must
# fail with that check's message and no earlier one. CPU only (mock cluster).
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
cd /home/aagarwal/tt-metal-mainline/.scratch
export TT_COV_TREE=/home/aagarwal/tt-metal-mainline TT_COV_DROP_CACHE=1
run() {  # name expected-message env-json
    export TT_COV_DIR=/home/aagarwal/.tt-metal-mainline-cov-force-$1 TT_COV_EXTRA_ENV="$3"
    rm -rf "${TT_COV_DIR:?}"
    timeout 900 python3 -B cov.py mock bh_2d > /dev/null 2>&1
    local log=$TT_COV_DIR/cases/bh_2d/mock.log
    local hits=$(grep -ac "TT_FATAL" "$log")
    if grep -aq -- "$2" "$log"; then echo "PASS $1: \"$2\" ($hits TT_FATAL lines)"; else
        echo "FAIL $1: expected \"$2\"; first TT_FATAL: $(grep -a -m1 'TT_FATAL' "$log" | cut -c1-200)"; fi
    rm -rf "${TT_COV_DIR:?}"
}
F() { echo "{\"TT_DEBUG_FORCE_ROUTER_ARGS\": \"$1\"}"; }
run arch "the router supports Blackhole only" '{"TT_DEBUG_FORCE_ROUTER_ARCH": "1"}'
run muxvc1 "VC1 with the Tensix mux extension is not supported" "$(F FABRIC_TENSIX_EXTENSION_MUX_MODE=1,ACTUAL_VC1_SENDER_CHANNELS=1)"
run vc2 "VC2 is supported as neighbour exchange on 2D routers" "$(F ACTUAL_VC2_SENDER_CHANNELS=2)"
run flavc1 "first-level acks on VC1 / VC2 are not supported" "$(F ENABLE_FIRST_LEVEL_ACK_VC1=1)"
run vc2x "VC2's crossover onto VC1 is not supported" "$(F VC2_CROSSOVER_TO_VC1=1)"
run udm "UDM is supported on one-ERISC 2D meshes with one relay" "$(F UDM_MODE=1)"
run speedy "the speedy path needs VC0 without bubble flow control" "$(F ENABLE_SPEEDY_VC0=1,DISABLE_RX_CH0_FORWARDING=0)"
run cache "the CPU data cache is not supported" "$(F ENABLE_RISC_CPU_DATA_CACHE=1)"
run spin "the TX queue's spin-wait send is not supported" "$(F ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA=1)"
run perf "the perf recorder supports low-resolution bandwidth only" "$(F PERF_TELEMETRY_MODE=2)"
run mux2e "Tensix mux mode is supported on one ERISC" "$(F FABRIC_TENSIX_EXTENSION_MUX_MODE=1)"
run tensix "local Tensix connections are supported in mux mode and UDM only" "$(F NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS=1)"
echo FORCED_MOCKS_DONE
