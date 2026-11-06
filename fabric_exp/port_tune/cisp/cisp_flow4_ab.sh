#!/bin/bash
# Slow-state check X4 (coordinator 2026-11-05, observation only, yaml only): two outbound flows with placed sender
#   cores (pcs_nx_x4_x8.yaml, mk_flow_yamls.py: SOdef / SOswap / SOfar on chips 1 and 2, SOdef on chips 0 and 3)
#   against the NX write rows (A, pcs_nx_wonly_x8.yaml, same-job control); profiler on and off for each. Measure:
#   the router telemetry per router and iteration (cisp_router_tally.py) and the sender CSV.
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/cisp/cisp_flow4_ab.sh
set -uo pipefail
cd /home/aagarwal/tt-metal
D=fabric_exp/port_tune/cisp
TREE=/home/aagarwal/tt-metal-ci-sp
B=$TREE/build_Release
G=$TREE/generated/fabric
RUN_TIMEOUT=300
out=/home/aagarwal/tt-metal/fabric_exp/results/cisp_flow4ab_$(date +%Y%m%d-%H%M%S)
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 57c55f98118 ] || { echo "!!! $TREE is not at 57c55f98118"; exit 1; }
[ "$(git -C $TREE diff | md5sum | cut -d' ' -f1)" = "$(md5sum < $D/host_link_solve_warn.patch | cut -d' ' -f1)" ] \
    || { echo "!!! $TREE does not carry exactly cisp/host_link_solve_warn.patch"; exit 1; }
grep -q -- "-DTRACY_ENABLE" $B/build.ninja || { echo "!!! $B is not a Tracy build"; exit 1; }
declare -A YAML=([A]=pcs_nx_wonly_x8.yaml [X4]=pcs_nx_x4_x8.yaml)
for y in "${YAML[@]}"; do [ -f $D/$y ] || { echo "!!! missing $D/$y"; exit 1; }; done
mkdir -p $out
echo "cisp_flow4_ab: $(hostname), results $out"

run() {  # arm (on | off) variant (S A B C)
    local arm=$1 v=$2 label=flow_$2_$1 rc t0=$SECONDS
    local log=$out/$label.log cache=/home/aagarwal/.tt-metal-cache-cisp-$arm-ci
    local env_extra=()
    [ $arm = on ] && env_extra+=(TT_METAL_STREAMING_PROFILER=1)
    echo "=== $label: cache $cache, env ${env_extra[*]:-(none)}"
    rm -f $G/bandwidth_results_blackhole.csv $G/bandwidth_summary_results_blackhole.csv \
        $G/bandwidth_summary_results_blackhole_upload.csv $G/*_raw_telemetry.csv
    (cd $TREE && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_FABRIC_NO_BW_TELEMETRY \
        TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE TT_METAL_CACHE=$cache \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib "${env_extra[@]}" \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$D/${YAML[$v]} > $log 2>&1 < /dev/null)
    rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    cp $G/bandwidth_results_blackhole.csv $out/${label}_bandwidth_results.csv 2>/dev/null
    cp $G/bandwidth_summary_results_blackhole.csv $out/${label}_bandwidth_summary.csv 2>/dev/null
    mkdir -p $out/${label}_raw_telemetry && cp $G/*_raw_telemetry.csv $out/${label}_raw_telemetry/ 2>/dev/null
    echo "$label: exit $rc after $((SECONDS - t0)) s; $(grep -ac 'Running Test:' $log) iterations;" \
        "profiler capture starts $(grep -ac '\[streaming profiler\] active on' $log);" \
        "sender rows $(($(wc -l < $out/${label}_bandwidth_results.csv 2>/dev/null || echo 1) - 1))"
    grep -aE "TT_FATAL|TT_THROW|terminate called|profiler stalls|dropped" $log | head -4 | sed 's/^/   | /'
    [ $rc = 0 ] || { echo "!!! $label: test rc $rc: stopping"; return 1; }
}
for spec in on:A off:A on:X4 off:X4; do
    run ${spec%%:*} ${spec##*:} || exit 1
done
echo "cisp_flow4_ab done (${SECONDS} s)"
