#!/bin/bash
# CI streaming-profiler stall hunt, step 3 (observation only, approved 2026-11-05): relays + senders on one timeline;
# (from step 2) PC samples of the test's sender
#   and sync workers (Tensix BRISC + NCRISC over the debug bus; the 48 cores of cisp/tensix_senders.txt, from the step-1
#   zone CSV) over whole CI-length runs, with each core's mailbox head every 50 ms (which kernel image ran), to see where
#   the senders spend the ~0.43 ms of the slow iterations (cisp_followups_20261105.md (g)).
#   Tree / build / environment as cisp_ab.sh (TELEM=ci: the router's bandwidth telemetry on, as in CI; the user's
#   exception for this investigation). Rows: cisp/pcs_nx_ci_x8.yaml (ubench.yaml's NeighborExchangeUnicast group as CI
#   runs it, 100k packets x 3 iterations, num_links 1 and 2, repeated 8 times in one process). Per arm, while the test
#   runs: consolidation/pc_sampler.py --tensix cisp/tensix_senders.txt (debug-bus PC reads, tt-exalens safe mode) from
#   the first "Running Test:" to the test's end, timestamped (t_start_epoch) to line up with the log; the test's
#   bandwidth_results CSV per arm (cleared first) marks the slow iterations.
#   Order: on, off; each arm its own JIT cache (cisp_ab.sh's). Any sampler or test error stops the job (no workaround).
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/cisp/cisp_relays.sh
set -uo pipefail
cd /home/aagarwal/tt-metal
D=fabric_exp/port_tune/cisp
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
TREE=/home/aagarwal/tt-metal-ci-sp
B=$TREE/build_Release
RUN_TIMEOUT=600
out=/home/aagarwal/tt-metal/fabric_exp/results/cisp_relays_$(date +%Y%m%d-%H%M%S)
G=$TREE/generated/fabric
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 57c55f98118 ] || { echo "!!! $TREE is not at 57c55f98118"; exit 1; }
[ "$(git -C $TREE diff | md5sum | cut -d' ' -f1)" = "$(md5sum < $D/host_link_solve_warn.patch | cut -d' ' -f1)" ] \
    || { echo "!!! $TREE does not carry exactly cisp/host_link_solve_warn.patch"; exit 1; }
grep -q -- "-DTRACY_ENABLE" $B/build.ninja || { echo "!!! $B is not a Tracy build"; exit 1; }
mkdir -p $out
echo "cisp_relays: $(hostname), results $out"

run() {  # arm
    local arm=$1 label=relays_$1 rc src
    local log=$out/$label.log cache=/home/aagarwal/.tt-metal-cache-cisp-$arm-ci
    local env_extra=()
    [ $arm = on ] && env_extra+=(TT_METAL_STREAMING_PROFILER=1)
    echo "=== $label: cache $cache, env ${env_extra[*]:-(none)}"
    rm -f $G/bandwidth_results_blackhole.csv $G/bandwidth_summary_results_blackhole.csv $G/*_raw_telemetry.csv
    (cd $TREE && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_FABRIC_NO_BW_TELEMETRY \
        TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE TT_METAL_CACHE=$cache \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib "${env_extra[@]}" \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$D/pcs_nx_ci_x8.yaml > $log 2>&1 < /dev/null) & local tp=$!
    timeout $((RUN_TIMEOUT + 30)) /home/aagarwal/tt-metal/python_env/bin/python $CON/pc_sampler.py $out/${label}_samples \
        --start-log $log --start-pattern "Running Test:" --stop-pattern "Tests: " --wait-s $RUN_TIMEOUT \
        --max-s $RUN_TIMEOUT --cores /home/aagarwal/tt-metal/$D/relay_cores.txt > $out/${label}_sampler.log 2>&1
    src=$?
    grep -aE '^sampler: |Error|error|Traceback' $out/${label}_sampler.log | head -6 | sed 's/^/   | /'
    wait $tp; rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    cp $G/bandwidth_results_blackhole.csv $out/${label}_bandwidth_results.csv 2>/dev/null
    echo "$label: test exit $rc, sampler exit $src; $(grep -ac 'BW (GB/s)=' $log) BW lines;" \
        "mean $(grep -aoE 'BW \(GB/s\)=[0-9.]+' $log | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.3f GB/s", s/n}');" \
        "profiler capture starts $(grep -ac '\[streaming profiler\] active on' $log)"
    grep -aE "TT_FATAL|TT_THROW|terminate called" $log | head -3 | sed 's/^/   | /'
    [ $rc = 0 ] && [ $src = 0 ] || { echo "!!! $label: test rc $rc, sampler rc $src: stopping"; return 1; }
}
for arm in on off; do
    run $arm || exit 1
done
echo "cisp_relays done (${SECONDS} s)"
