#!/bin/bash
# CI streaming-profiler follow-up (a): PC samples of the port's routers on the CI's worst row, profiler on and off.
#   Tree / build / environment as cisp_ab.sh (TELEM=ci: the router's bandwidth telemetry on, as in CI; the user's
#   exception for these runs). Rows: cisp/pcs_nx4096w_l{2,1}.yaml (ubench.yaml's NeighborExchangeUnicast 4096 B
#   unicast_write at 2 and 1 links, one 80M-packet iteration, several seconds of steady traffic). Per run, while the
#   test runs: consolidation/pc_sampler.py on every chip's ERISC 0 and 1 (debug-bus PC reads; tt-exalens safe mode),
#   from 1 s after "Launching programs" for 4 s. Order: on l2, off l2, off l1, on l1; each arm its own JIT cache
#   (cisp_ab.sh's). Any sampler or test error stops the job (no workaround).
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/cisp/cisp_pcs.sh
set -uo pipefail
cd /home/aagarwal/tt-metal
D=fabric_exp/port_tune/cisp
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
TREE=/home/aagarwal/tt-metal-ci-sp
B=$TREE/build_Release
RUN_TIMEOUT=300
out=/home/aagarwal/tt-metal/fabric_exp/results/cisp_pcs_$(date +%Y%m%d-%H%M%S)
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 57c55f98118 ] || { echo "!!! $TREE is not at 57c55f98118"; exit 1; }
[ "$(git -C $TREE diff | md5sum | cut -d' ' -f1)" = "$(md5sum < $D/host_link_solve_warn.patch | cut -d' ' -f1)" ] \
    || { echo "!!! $TREE does not carry exactly cisp/host_link_solve_warn.patch"; exit 1; }
grep -q -- "-DTRACY_ENABLE" $B/build.ninja || { echo "!!! $B is not a Tracy build"; exit 1; }
mkdir -p $out
echo "cisp_pcs: $(hostname), results $out"

run() {  # arm links
    local arm=$1 L=$2 label=pcs_$1_l$2 rc src
    local log=$out/$label.log cache=/home/aagarwal/.tt-metal-cache-cisp-$arm-ci
    local env_extra=()
    [ $arm = on ] && env_extra+=(TT_METAL_STREAMING_PROFILER=1)
    echo "=== $label: cache $cache, env ${env_extra[*]:-(none)}"
    (cd $TREE && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_FABRIC_NO_BW_TELEMETRY \
        TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE TT_METAL_CACHE=$cache \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib "${env_extra[@]}" \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$D/pcs_nx4096w_l$L.yaml > $log 2>&1 < /dev/null) & local tp=$!
    timeout 120 /home/aagarwal/tt-metal/python_env/bin/python $CON/pc_sampler.py $out/${label}_samples --start-log $log \
        --start-pattern "Launching programs" --start-delay-s 1 --max-s 4 --eriscs 0,1 > $out/${label}_sampler.log 2>&1
    src=$?
    grep -aE '^sampler: |Error|error|Traceback' $out/${label}_sampler.log | head -6 | sed 's/^/   | /'
    wait $tp; rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    echo "$label: test exit $rc, sampler exit $src; $(grep -ac 'BW (GB/s)=' $log) BW lines;" \
        "mean $(grep -aoE 'BW \(GB/s\)=[0-9.]+' $log | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.3f GB/s", s/n}');" \
        "profiler capture starts $(grep -ac '\[streaming profiler\] active on' $log)"
    grep -aE "TT_FATAL|TT_THROW|terminate called" $log | head -3 | sed 's/^/   | /'
    [ $rc = 0 ] && [ $src = 0 ] || { echo "!!! $label: test rc $rc, sampler rc $src: stopping"; return 1; }
}
for spec in on:2 off:2 off:1 on:1; do
    run ${spec%%:*} ${spec##*:} || exit 1
done
echo "cisp_pcs done (${SECONDS} s)"
