#!/bin/bash
# CI failure 41963172504 ("BH LoudBox streaming profiler fabric overhead") reproduced on a09u32: the bandwidth ratio
# with the streaming profiler on / off of test_streaming_profiler_fabric_overhead, on the pushed router (57c55f98118,
# image-identical to the failing cc63d9b0284 apart from comments).
#   Tree: /home/aagarwal/tt-metal-ci-sp at 57c55f98118, its own Tracy build (the streaming profiler needs TRACY_ENABLE).
#   Config: the test's own (cisp/ubench.yaml: test_fabric_ubench_at_least_2x2_mesh.yaml with each latency test repeated
#           40 times at 1024 samples, as the test's ubench_config fixture writes it), --filter name.*Unicast*; bandwidth
#           parsed as the test does: per "Running Test:" the mean of its "BW (GB/s)=" lines, latency tests left out.
#   Environment as the test's run(): the job's environment without TT_METAL_STREAMING_PROFILER*, plus the arm's.
#   TELEM: ci (default: TT_FABRIC_NO_BW_TELEMETRY unset, as in CI, so benchmark tests run with the router's bandwidth
#          telemetry on) or off (TT_FABRIC_NO_BW_TELEMETRY=1).
#   Arms (ARMS, default "off on"): off; on (TT_METAL_STREAMING_PROFILER=1); nohook (on, from the runtime root
#          /home/aagarwal/tt-metal-port-gap-roots/cisp_nohook, whose router_round.hpp skips link_sync::RouterHook::step).
#   Pass a: the arms in order; pass b: reversed. One JIT cache per arm and telemetry mode.
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/cisp/cisp_ab.sh
# DRY=1: the checks and the runs it would do.
set -uo pipefail
cd /home/aagarwal/tt-metal
D=fabric_exp/port_tune/cisp
# CISP_TREE / CISP_SHA / CISP_TAG run another tree the same way (main: /home/aagarwal/tt-metal-main at d2bd12fcf1f, its
# own Tracy build; only the off / on arms; its sources are not edited, and the one local change allowed there is the
# test-side TT_FABRIC_NO_BW_TELEMETRY switch, inert with TELEM=ci).
TREE=${CISP_TREE:-/home/aagarwal/tt-metal-ci-sp}
SHA=${CISP_SHA:-57c55f98118}
TAG=${CISP_TAG:+_$CISP_TAG}
B=$TREE/build_Release
ARMS=${ARMS:-off on}
TELEM=${TELEM:-ci}
RUN_TIMEOUT=420
out=/home/aagarwal/tt-metal/fabric_exp/results/cisp_${TELEM}${TAG}_$(date +%Y%m%d-%H%M%S)
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
echo "cisp_ab: $(hostname), arms '$ARMS', telemetry $TELEM, results $out"

[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = $SHA ] || { echo "!!! $TREE is not at $SHA"; exit 1; }
if [ -z "${CISP_TREE:-}" ]; then
# The worktree carries exactly the host-only patch cisp/host_link_solve_warn.patch (the no-hook arm's link-solve
# warnings; no router or test change), so every arm runs one host binary.
[ -z "$(git -C $TREE status --porcelain -- tests)" ] || { echo "!!! $TREE has test changes"; exit 1; }
[ "$(git -C $TREE diff | md5sum | cut -d' ' -f1)" = "$(md5sum < $D/host_link_solve_warn.patch | cut -d' ' -f1)" ] \
    || { echo "!!! $TREE does not carry exactly cisp/host_link_solve_warn.patch"; exit 1; }
[ -z "$(git -C $TREE diff --name-only | grep -v '^tt_metal/impl/streaming_profiler/sync/')" ] \
    || { echo "!!! $TREE has changes outside the profiler's host sync code"; exit 1; }
else
    [ "$ARMS" = "off on" ] || [ "$ARMS" = "on off" ] || { echo "!!! another tree runs the off / on arms only"; exit 1; }
    [ "$(git -C $TREE diff --name-only)" = tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp ] \
        || { echo "!!! $TREE differs from HEAD beyond test_tt_fabric.cpp's TT_FABRIC_NO_BW_TELEMETRY switch"; exit 1; }
    [ $TELEM = ci ] || { echo "!!! another tree runs TELEM=ci only"; exit 1; }
    echo "tree $TREE at $(git -C $TREE rev-parse --short=11 HEAD); local change: $(git -C $TREE diff --stat | tail -1)"
fi
NH=/home/aagarwal/tt-metal-port-gap-roots/cisp_nohook/tt_metal/fabric/hw/inc/edm_fabric/router/router_round.hpp
if [[ " $ARMS " == *" nohook "* ]]; then
    [ "$(git -C $TREE show HEAD:tt_metal/fabric/hw/inc/edm_fabric/router/router_round.hpp | sed 's#^    link_sync::RouterHook::step(counters.heartbeat);$#    // LOCAL (fabric_exp/port_tune/cisp, the profiler split'"'"'s no-hook arm; never committed): the hook is not called.#' | md5sum | cut -d' ' -f1)" = \
      "$(md5sum < $NH | cut -d' ' -f1)" ] || { echo "!!! the no-hook root's router_round.hpp is not HEAD's minus the hook call"; exit 1; }
fi
grep -q -- "-DTRACY_ENABLE" $B/build.ninja || { echo "!!! $B is not a Tracy build"; exit 1; }
[ "$(md5sum < $D/ubench.yaml | cut -d' ' -f1)" = ae65dcfa93fc798bc763b39dc228c9ec ] || { echo "!!! ubench.yaml changed"; exit 1; }
root_of() { case $1 in nohook) echo /home/aagarwal/tt-metal-port-gap-roots/cisp_nohook ;; *) echo $TREE ;; esac; }
for arm in $ARMS; do
    r=$(root_of $arm)
    [ -d $r ] || { echo "!!! no root $r for arm $arm"; exit 1; }
done
[ -n "${DRY:-}" ] || mkdir -p $out

run_arm() {  # pass arm
    local pass=$1 arm=$2 r label log t0=$SECONDS rc
    r=$(root_of $arm); label=sp${pass}_$arm; log=$out/$label.log
    local env_extra=()
    case $arm in on|nohook) env_extra+=(TT_METAL_STREAMING_PROFILER=1) ;; esac
    [ $TELEM = off ] && env_extra+=(TT_FABRIC_NO_BW_TELEMETRY=1)
    local cache=/home/aagarwal/.tt-metal-cache-cisp-${arm}-$TELEM$TAG
    echo "=== $label: root $r, cache $cache, env ${env_extra[*]:-(none)}"
    if [ -n "${DRY:-}" ]; then return 0; fi
    # the test's own sender-side results and the raw telemetry, this arm's only (each run writes or appends them)
    rm -f $r/generated/fabric/bandwidth_summary_results_blackhole.csv $r/generated/fabric/bandwidth_results_blackhole.csv \
        $r/generated/fabric/bandwidth_summary_results_blackhole_upload.csv $r/generated/fabric/*_raw_telemetry.csv
    (cd $r && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_FABRIC_NO_BW_TELEMETRY \
        TT_METAL_HOME=$r TT_METAL_RUNTIME_ROOT=$r TT_METAL_CACHE=$cache \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib "${env_extra[@]}" \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$D/ubench.yaml --filter 'name.*Unicast*' > $log 2>&1 < /dev/null)
    rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    echo "$label: exit $rc after $((SECONDS - t0)) s; $(grep -ac 'Running Test:' $log) tests run;" \
        "capture starts $(grep -ac '\[streaming profiler\] active on' $log);" \
        "declined: $(grep -aoE '\[streaming profiler\] not capturing: .*' $log | head -1)"
    grep -aE "TT_FATAL|TT_THROW|terminate called" $log | head -3 | sed 's/^/   | /'
    cp $r/generated/fabric/latency_results_blackhole.csv $out/${label}_latency.csv 2>/dev/null
    cp $r/generated/fabric/bandwidth_results_blackhole.csv $out/${label}_bandwidth_results.csv 2>/dev/null
    cp $r/generated/fabric/bandwidth_summary_results_blackhole.csv $out/${label}_bandwidth_summary.csv 2>/dev/null
    mkdir -p $out/${label}_raw_telemetry && cp $r/generated/fabric/*_raw_telemetry.csv $out/${label}_raw_telemetry/ 2>/dev/null
    return $rc
}

rc=0
for pass in a b; do
    order=$ARMS
    [ $pass = b ] && order=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
    for arm in $order; do
        if [ $((SECONDS + RUN_TIMEOUT + 60)) -gt 1440 ]; then echo "=== stopping: job at $SECONDS s"; rc=1; break 2; fi
        run_arm $pass $arm || { echo "!!! sp${pass}_$arm failed: stopping"; rc=1; break 2; }
    done
done
[ -n "${DRY:-}" ] && { echo "DRY done"; exit 0; }
python3 $D/cisp_report.py $out | tee $out/report.md
# What the profiler build changes in the router images, from this job's caches (pairs by named args, the profiler's
# LINK_SYNC_* args left out of the key; LINK_SYNC_ROLE counts show which routers run a link port).
P2=fabric_exp/port_tune/dio/pair_images2.py
for pair in "off on" "on nohook"; do
    set -- $pair
    [[ " $ARMS " == *" $1 "* && " $ARMS " == *" $2 "* ]] || continue
    echo "--- router images: $1 vs $2" | tee -a $out/report.md
    python3 $P2 /home/aagarwal/.tt-metal-cache-cisp-$1-$TELEM$TAG /home/aagarwal/.tt-metal-cache-cisp-$2-$TELEM$TAG \
        --ignore-args 'LINK_SYNC_' --json $out/pairs_$1_$2.json | tee -a $out/report.md
    for proc in active_erisc subordinate_active_erisc; do
        echo "--- functions, $1 vs $2, $proc (code with addresses and immediates masked)" | tee -a $out/report.md
        python3 $D/func_diff.py $out/pairs_$1_$2.json $proc | tee -a $out/report.md
    done
done
echo "cisp_ab done (rc $rc, ${SECONDS} s)"
exit $rc
