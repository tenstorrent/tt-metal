#!/bin/bash
# Slow-state check X7 (coordinator 2026-11-05, approved, read-only): router PC samples, profiler on, contrasting
#   (i)   the stock L1/L2 alternation of the NX write rows (pcs_nx_wonly_x8.yaml), where the slow state builds over the
#         L1 rows that follow an L2 block;
#   (ii)  L1-only blocks (pcs_nx_l1only_x8.yaml), where link 1 never carries traffic and nothing slows;
#   (gap) the alternation with a near-idle pause after each L2 block (pcs_nx_gap_x8.yaml, mk_order_yamls.py).
#   Per sampled arm (profiler on), while the test runs: consolidation/pc_sampler.py --only-loaded (every loaded router's
#   ERISC0 and ERISC1, debug-bus PC reads, tt-exalens safe mode) from the first "Running Test:" to the test's end,
#   timestamped (t_start_epoch). Then the three yamls profiler off, unsampled (the router tally's baselines).
#   Tree / build / environment as cisp_ab.sh (TELEM=ci). Any sampler or test error stops the job.
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/cisp/cisp_pcs3.sh
set -uo pipefail
cd /home/aagarwal/tt-metal
D=fabric_exp/port_tune/cisp
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
TREE=/home/aagarwal/tt-metal-ci-sp
B=$TREE/build_Release
RUN_TIMEOUT=600
out=/home/aagarwal/tt-metal/fabric_exp/results/cisp_pcs3_$(date +%Y%m%d-%H%M%S)
G=$TREE/generated/fabric
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 57c55f98118 ] || { echo "!!! $TREE is not at 57c55f98118"; exit 1; }
[ "$(git -C $TREE diff | md5sum | cut -d' ' -f1)" = "$(md5sum < $D/host_link_solve_warn.patch | cut -d' ' -f1)" ] \
    || { echo "!!! $TREE does not carry exactly cisp/host_link_solve_warn.patch"; exit 1; }
grep -q -- "-DTRACY_ENABLE" $B/build.ninja || { echo "!!! $B is not a Tracy build"; exit 1; }
mkdir -p $out
echo "cisp_pcs3: $(hostname), results $out"

run() {  # arm variant sample(0|1)
    local arm=$1 v=$2 smp=$3 label=x7_$2_$1 rc src=0
    local log=$out/$label.log cache=/home/aagarwal/.tt-metal-cache-cisp-$arm-ci
    local env_extra=()
    [ $arm = on ] && env_extra+=(TT_METAL_STREAMING_PROFILER=1)
    echo "=== $label: cache $cache, env ${env_extra[*]:-(none)}"
    rm -f $G/bandwidth_results_blackhole.csv $G/bandwidth_summary_results_blackhole.csv $G/*_raw_telemetry.csv
    (cd $TREE && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_FABRIC_NO_BW_TELEMETRY \
        TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE TT_METAL_CACHE=$cache \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib "${env_extra[@]}" \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$D/${YAML[$v]} > $log 2>&1 < /dev/null) & local tp=$!
    if [ $smp = 1 ]; then
        timeout $((RUN_TIMEOUT + 30)) /home/aagarwal/tt-metal/python_env/bin/python $CON/pc_sampler.py $out/${label}_samples \
            --start-log $log --start-pattern "Running Test:" --stop-pattern "Tests: " --wait-s $RUN_TIMEOUT \
            --max-s $RUN_TIMEOUT --only-loaded > $out/${label}_sampler.log 2>&1
        src=$?
        grep -aE '^sampler: |Error|error|Traceback' $out/${label}_sampler.log | head -6 | sed 's/^/   | /'
    fi
    wait $tp; rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    cp $G/bandwidth_results_blackhole.csv $out/${label}_bandwidth_results.csv 2>/dev/null
    echo "$label: test exit $rc, sampler exit $src; $(grep -ac 'BW (GB/s)=' $log) BW lines;" \
        "mean $(grep -aoE 'BW \(GB/s\)=[0-9.]+' $log | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.3f GB/s", s/n}');" \
        "profiler capture starts $(grep -ac '\[streaming profiler\] active on' $log)"
    grep -aE "TT_FATAL|TT_THROW|terminate called" $log | head -3 | sed 's/^/   | /'
    [ $rc = 0 ] && [ $src = 0 ] || { echo "!!! $label: test rc $rc, sampler rc $src: stopping"; return 1; }
}
declare -A YAML=([i]=pcs_nx_wonly_x8.yaml [ii]=pcs_nx_l1only_x8.yaml [gap]=pcs_nx_gap_x8.yaml)
for spec in on:i:1 on:ii:1 on:gap:1 off:i:0 off:ii:0 off:gap:0; do
    IFS=: read -r a v smp <<< "$spec"
    run $a $v $smp || exit 1
done
echo "cisp_pcs3 done (${SECONDS} s)"
