#!/bin/bash
# CI streaming-profiler stall hunt, ship-threshold A/B (approved 2026-11-05; local profiler-kernel edits, never for the
#   PR): arms base (stock, profiler on), deferC (drisc_relay kShipMinPct 50 + kMaxDeferSweeps 4096), deferD
#   (kShipMinPct 50 + kMaxDeferSweeps 1024), then off (the sender baseline); each on arm as cisp_zones.sh's.
# From cisp_zones.sh (step 1): the profiler's own timeline over
#   the 819 rows (cisp/pcs_nx_ci_x8.yaml: ubench's NeighborExchangeUnicast group as CI runs it, 100k packets x 3
#   iterations, num_links 1 and 2, 8 repeats in one process) on the port tree 57c55f98118 (tt-metal-ci-sp, Tracy,
#   TELEM=ci: router telemetry on, as in CI). Arms: on (TT_METAL_STREAMING_PROFILER=1 with TT_METAL_STREAMING_PROFILER_
#   ZONE_CSV and _OPS_CSV into the results dir), then off (the sender-side baseline). Each arm keeps its log, the test's
#   bandwidth_results / bandwidth_summary CSVs (cleared before the run) and the raw telemetry CSVs.
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/cisp/cisp_defer_ab.sh
set -uo pipefail
cd /home/aagarwal/tt-metal
D=fabric_exp/port_tune/cisp
TREE=/home/aagarwal/tt-metal-ci-sp
B=$TREE/build_Release
G=$TREE/generated/fabric
RUN_TIMEOUT=600
out=/home/aagarwal/tt-metal/fabric_exp/results/cisp_deferab_$(date +%Y%m%d-%H%M%S)
K=tt_metal/impl/streaming_profiler/kernels/drisc_relay.cpp
GR=/home/aagarwal/tt-metal-port-gap-roots
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 57c55f98118 ] || { echo "!!! $TREE is not at 57c55f98118"; exit 1; }
[ "$(git -C $TREE diff | md5sum | cut -d' ' -f1)" = "$(md5sum < $D/host_link_solve_warn.patch | cut -d' ' -f1)" ] \
    || { echo "!!! $TREE does not carry exactly cisp/host_link_solve_warn.patch"; exit 1; }
grep -q -- "-DTRACY_ENABLE" $B/build.ninja || { echo "!!! $B is not a Tracy build"; exit 1; }
cmp -s $GR/cisp_deferC/$K $D/shipab/drisc_relay_deferC.cpp && cmp -s $GR/cisp_deferD/$K $D/shipab/drisc_relay_deferD.cpp \
    || { echo "!!! the A/B roots' drisc_relay.cpp are not cisp/shipab's"; exit 1; }
mkdir -p $out
echo "cisp_defer_ab: $(hostname), results $out"

run() {  # arm
    local arm=$1 label=ab_$1 rc t0=$SECONDS root=$TREE
    case $arm in deferC) root=$GR/cisp_deferC ;; deferD) root=$GR/cisp_deferD ;; esac
    local log=$out/$label.log cache=/home/aagarwal/.tt-metal-cache-cisp-ab-$arm-ci
    local env_extra=()
    [ $arm != off ] && env_extra+=(TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_ZONE_CSV=$out/${label}_zones.csv
                                 TT_METAL_STREAMING_PROFILER_OPS_CSV=$out/${label}_ops.csv)
    echo "=== $label: cache $cache, env ${env_extra[*]:-(none)}"
    rm -f $G/bandwidth_results_blackhole.csv $G/bandwidth_summary_results_blackhole.csv \
        $G/bandwidth_summary_results_blackhole_upload.csv $G/*_raw_telemetry.csv
    (cd $root && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_FABRIC_NO_BW_TELEMETRY \
        TT_METAL_HOME=$root TT_METAL_RUNTIME_ROOT=$root TT_METAL_CACHE=$cache \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib "${env_extra[@]}" \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$D/pcs_nx_ci_x8.yaml > $log 2>&1 < /dev/null)
    rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    cp $G/bandwidth_results_blackhole.csv $out/${label}_bandwidth_results.csv 2>/dev/null
    cp $G/bandwidth_summary_results_blackhole.csv $out/${label}_bandwidth_summary.csv 2>/dev/null
    mkdir -p $out/${label}_raw_telemetry && cp $G/*_raw_telemetry.csv $out/${label}_raw_telemetry/ 2>/dev/null
    echo "$label: exit $rc after $((SECONDS - t0)) s; $(grep -ac 'Running Test:' $log) iterations;" \
        "profiler capture starts $(grep -ac '\[streaming profiler\] active on' $log);" \
        "zone csv $(wc -l < $out/${label}_zones.csv 2>/dev/null || echo -) lines, ops csv $(wc -l < $out/${label}_ops.csv 2>/dev/null || echo -) lines;" \
        "sender rows $(($(wc -l < $out/${label}_bandwidth_results.csv 2>/dev/null || echo 1) - 1))"
    grep -aE "TT_FATAL|TT_THROW|terminate called|profiler stalls|dropped" $log | head -4 | sed 's/^/   | /'
    [ $rc = 0 ] || { echo "!!! $label: test rc $rc: stopping"; return 1; }
}
for arm in base deferC deferD off; do
    run $arm || exit 1
done
echo "cisp_defer_ab done (${SECONDS} s)"
