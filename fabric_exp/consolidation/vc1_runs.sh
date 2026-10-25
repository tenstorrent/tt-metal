#!/bin/bash
# VC1 on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat): the P150x8 as two 2x2 meshes
# (fabric_exp/vc1: MGD + rank bindings, two MPI ranks via ttrun.py), which turns VC1 on. fabric_exp/vc1/vc1_split.yaml:
# multi-mesh sanity (data checked), then intra-mesh 1-hop E/W and N/S, inter-mesh neighbor exchange, intra-mesh
# multicast benchmarks. The tree's router (s79mc, run as v2), three ways, each in its own fresh kernel cache with
# compile commands logged (for offline VC1 builds):
#   default (the builder picks -Os because VC1 is active)   label vc1auto_
#   -O3 (does VC1 fit and run at -O3?)                       label vc1o3_
#   -O2                                                      label vc1o2_
# A failed or hung level is reported and the job moves on (each run is bounded by `timeout`).
# Flags 13129088, fast dispatch, telemetry off, keepalive 1000. The tree is not modified; checked on exit.
cd /home/aagarwal/tt-metal
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
check_tree() { cmp -s fabric_exp/variants/s79mc/fabric_erisc_router_v2.cpp $R && echo "tree is s79mc" || echo "!!! TREE IS NOT s79mc"; }
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; check_tree' EXIT
out=fabric_exp/results
# Launch: the hand-written rank bindings (two ranks on one host). Job 210 failed in the control plane with them:
# "SAT placement: mesh 'M0' has no PGD grouping and no embeddable MGD fallback". ttrun's discovery mode
# (-m MGD) needs one host per rank (job 211: "--hosts is required", duplicates rejected), so it cannot split one host.
LAUNCH="--rank-binding fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml"
LOGLEVEL=${LOGLEVEL:-}
run_vc1() {  # label cache opt|auto
    local l=$1 c=$2 o=$3 optenv=""
    [ "$o" = auto ] || optenv="TT_METAL_FABRIC_OPT_LEVEL=$o"
    rm -rf "${c:?}"; mkdir -p $c
    rm -f generated/fabric/bandwidth_summary_results_blackhole.csv generated/fabric/bandwidth_results_blackhole.csv \
        $out/${l}_13129088_vc1.csv $out/${l}_13129088_vc1_tests.csv
    echo "=== ${l}_13129088_vc1: opt $o, cache $c ==="
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_METAL_CLEAR_L1=1 \
        TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$c \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 $optenv ${LOGLEVEL:+TT_LOGGER_LEVEL=$LOGLEVEL} \
        timeout 600 python_env/bin/python ttnn/ttnn/distributed/ttrun.py $LAUNCH \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/fabric_exp/vc1/vc1_split.yaml \
        > $out/${l}_13129088_vc1.log 2>&1
    echo "exit $? ; $(grep -aoE 'Tests: .*executed, [0-9]+ failed' $out/${l}_13129088_vc1.log | sort | uniq -c | tr '\n' ' ')"
    grep -aE 'Running Test:|TT_FATAL|error:|HUNG|Timeout|timed out|Physical groupings' $out/${l}_13129088_vc1.log | grep -v 'g++' | head -40
    [ -f generated/fabric/bandwidth_summary_results_blackhole.csv ] && cp generated/fabric/bandwidth_summary_results_blackhole.csv $out/${l}_13129088_vc1.csv
    [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_13129088_vc1_tests.csv
    local n=$(grep -a 'g++ compile cmd' $out/${l}_13129088_vc1.log | grep -c fabric_erisc_router_v2)
    echo "router kernels compiled: $n"
    grep -aq 'Running Test' $out/${l}_13129088_vc1.log || { echo "!!! no test started (control plane / launch failure)"; exit 1; }
}
check_tree
run_vc1 vc1auto /home/aagarwal/.tt-metal-cache-vc1-auto auto
run_vc1 vc1o3 /home/aagarwal/.tt-metal-cache-vc1-O3 O3
run_vc1 vc1o2 /home/aagarwal/.tt-metal-cache-vc1-O2 O2
echo ALL_DONE
