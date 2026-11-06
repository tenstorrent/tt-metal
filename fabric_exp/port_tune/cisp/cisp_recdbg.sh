#!/bin/bash
# CI streaming-profiler stall study (B), user-approved 2026-11-05: the perf recorder's counted rounds per iteration, from
#   the runtime root /home/aagarwal/tt-metal-port-gap-roots/cisp_recdbg (57c55f98118's sources with cisp/recdbg/'s
#   router_telemetry.hpp and router_round.hpp: a per-iteration ring in router L1 at 0x60000, recdbg.diff; local only,
#   never for the PR). After each arm's test: consolidation/hang_capture.py (read-only L1 dump of every Ethernet core),
#   from which cisp/recdbg/recdbg_report.py reads the ring.
#   Tree / build / environment as cisp_ab.sh (TELEM=ci: the router's bandwidth telemetry on, as in CI; the user's
#   exception for this investigation). Rows: cisp/pcs_nx_ci_x8.yaml (ubench.yaml's NeighborExchangeUnicast group as CI
#   runs it, 100k packets x 3 iterations, num_links 1 and 2, repeated 8 times in one process). Order: on, off; each arm
#   its own JIT cache. Any test or capture error stops the job (no workaround).
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/cisp/cisp_recdbg.sh
set -uo pipefail
cd /home/aagarwal/tt-metal
D=fabric_exp/port_tune/cisp
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
TREE=/home/aagarwal/tt-metal-ci-sp
B=$TREE/build_Release
RUN_TIMEOUT=600
out=/home/aagarwal/tt-metal/fabric_exp/results/cisp_recdbg_$(date +%Y%m%d-%H%M%S)
ROOT=/home/aagarwal/tt-metal-port-gap-roots/cisp_recdbg
RR=$ROOT/tt_metal/fabric/hw/inc/edm_fabric/router
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 57c55f98118 ] || { echo "!!! $TREE is not at 57c55f98118"; exit 1; }
[ "$(git -C $TREE diff | md5sum | cut -d' ' -f1)" = "$(md5sum < $D/host_link_solve_warn.patch | cut -d' ' -f1)" ] \
    || { echo "!!! $TREE does not carry exactly cisp/host_link_solve_warn.patch"; exit 1; }
grep -q -- "-DTRACY_ENABLE" $B/build.ninja || { echo "!!! $B is not a Tracy build"; exit 1; }
for f in router_telemetry.hpp router_round.hpp; do
    cmp -s $RR/$f $D/recdbg/$f || { echo "!!! $RR/$f is not cisp/recdbg/$f"; exit 1; }
done
mkdir -p $out
echo "cisp_recdbg: $(hostname), root $ROOT, results $out"

run() {  # arm
    local arm=$1 label=recdbg_$1 rc src
    local log=$out/$label.log cache=/home/aagarwal/.tt-metal-cache-cisp-recdbg-$arm-ci
    local env_extra=()
    [ $arm = on ] && env_extra+=(TT_METAL_STREAMING_PROFILER=1)
    echo "=== $label: cache $cache, env ${env_extra[*]:-(none)}"
    (cd $ROOT && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_FABRIC_NO_BW_TELEMETRY \
        TT_METAL_HOME=$ROOT TT_METAL_RUNTIME_ROOT=$ROOT TT_METAL_CACHE=$cache \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib "${env_extra[@]}" \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$D/pcs_nx_ci_x8.yaml > $log 2>&1 < /dev/null)
    rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    echo "$label: test exit $rc; $(grep -ac 'BW (GB/s)=' $log) BW lines;" \
        "mean $(grep -aoE 'BW \(GB/s\)=[0-9.]+' $log | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.3f GB/s", s/n}');" \
        "profiler capture starts $(grep -ac '\[streaming profiler\] active on' $log)"
    grep -aE "TT_FATAL|TT_THROW|terminate called" $log | head -3 | sed 's/^/   | /'
    [ $rc = 0 ] || { echo "!!! $label: test rc $rc: stopping"; return 1; }
    timeout 300 /home/aagarwal/tt-metal/python_env/bin/python $CON/hang_capture.py $out/${label}_cap --no-workers \
        --time-budget 240 --meta label=$label > $out/${label}_cap.log 2>&1
    src=$?
    tail -2 $out/${label}_cap.log | sed 's/^/   | /'
    [ $src = 0 ] || { echo "!!! $label: capture rc $src: stopping"; return 1; }
}
for arm in on off; do
    run $arm || exit 1
done
echo "cisp_recdbg done (${SECONDS} s)"
