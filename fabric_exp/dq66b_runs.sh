#!/bin/bash
# §66 A/B on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat).
# 1. v1 latency calibration (a8v1_13129088_lat, against a04u24's dqib).
# 2. §63 rebuilt from the transcript (variants/s63r) swapped in: validation, 4352 / mix / latency, flushed-signal yaml
#    (labels a8s63_13129088_<suite>).
# 3. §66 (variants/s66) restored and re-run on 4352 / mix / latency / flushed-signal (labels dq66r2_13129088_<suite>),
#    which also gives this host's run-to-run noise against dq66_*.
# Flags 13129088, fast dispatch, telemetry off. The tree is restored to §66 on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
T=tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
V=fabric_exp/variants
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_edm_packet_transmission.hpp $T && echo "== router: $1"; }
restore() { swap s66 > /dev/null; cmp -s $V/s66/fabric_erisc_router_v2.cpp $R && cmp -s $V/s66/fabric_edm_packet_transmission.hpp $T \
    && echo "tree restored to s66" || echo "!!! RESTORE FAILED"; }
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; restore' EXIT
run() {  # kernel_version label yaml
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=$1 timeout 1200 fabric_exp/run_exp.sh "$2" "$3" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$2.log 2>/dev/null)
    echo "$2: ${res:-NO RESULT (exit $status)}"
    [ $status -eq 0 ] && [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
run 1 a8v1_13129088_lat fabric_exp/exp_latency.yaml || { echo STOPPED; exit 1; }
swap s63r
run 2 a8s63_13129088_stream $SAN/test_fabric_stream_sanity.yaml || { echo STOPPED; exit 1; }
run 2 a8s63_13129088_mesh $SAN/test_fabric_mesh_sanity_common.yaml || { echo STOPPED; exit 1; }
run 2 a8s63_13129088_4352 fabric_exp/exp_2d_4352.yaml || { echo STOPPED; exit 1; }
run 2 a8s63_13129088_mix fabric_exp/streams/p3_mixes.yaml || { echo STOPPED; exit 1; }
run 2 a8s63_13129088_lat fabric_exp/exp_latency.yaml || { echo STOPPED; exit 1; }
run 2 a8s63_13129088_flushmix fabric_exp/streams/flush_mix.yaml || { echo STOPPED; exit 1; }
swap s66
run 2 dq66r2_13129088_4352 fabric_exp/exp_2d_4352.yaml || { echo STOPPED; exit 1; }
run 2 dq66r2_13129088_mix fabric_exp/streams/p3_mixes.yaml || { echo STOPPED; exit 1; }
run 2 dq66r2_13129088_lat fabric_exp/exp_latency.yaml || { echo STOPPED; exit 1; }
run 2 dq66r2_13129088_flushmix fabric_exp/streams/flush_mix.yaml || { echo STOPPED; exit 1; }
echo ALL_DONE
