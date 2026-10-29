#!/bin/bash
# v4 rings and tori on one Blackhole Galaxy node (RESULTS §98, family 2): data-checked validation of main's router and
# v4 on the Galaxy's 2D torus and rings (the yamls vetted on the mock Galaxy, fabric_exp/v4/glxprobe), then ring rows.
#
#   CONFIGS="main v4r" STEPS="main:ds v4r:ds ... v4r:nt" sbatch fabric_exp/galaxy/frs_glx_torus.sbatch
#
# STEPS are config:suite pairs run in order. Suites (yamls vetted on the mock Galaxy): ds = test_fabric_2d_torus_
# deadlock_stability, st = the torus stability suite's _short tests, nt = test_fabric_2d_torus_nightly (long: v4 only by
# default; about 27 min on v4, so its run gets NT_TIMEOUT, 3000 s), sa = the sanity suite's Ring tests, ring = glx_ring.yaml
# (benchmark rows), mcs = glx_ring_mcast_sweep.yaml (full-ring multicast over sizes), mesh / linear = frs_glx.sbatch's
# 2D mesh and 1D line benchmark rows (fabric_exp/galaxy/glx_{mesh,linear}.yaml). Every other run gets 1500 s.
# Express / Z (the sanity yaml test_fabric_express_routing_sanity_common's groups, fabric_exp/express): xr =
# glx_express_ring.yaml on the 8x4 RING x RING express descriptor, xl = glx_express_line.yaml on
# express_links_8x4_line_ring_mesh_graph_descriptor.textproto (in yamls/; one mesh of CI's quad 8x4); a suite ending in
# 1 (xr1, xl1) runs the same on one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1).
# CONFIGS lists the trees to copy:
#   main    tree "base", router v1 at flags 0 (main's router)
#   v4*     tree of that name (fabric_exp/variants/<name> in v2's place), flags 13129088, host keepalive
#   v3q     tree "v3q"
#SBATCH --partition=bh_sc36_6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --job-name=frs-glx-torus
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
CONFIGS=${CONFIGS:-main v4r}
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) configs=$CONFIGS"
# Nodes are also used outside Slurm (jobs 155971, 157726, 157804: another user's tests held the device on an --exclusive
# node, and the first step waited on the KMD lock after the tree copies). Look for other users' device work first and
# stop before the slow copies (IGNORE_BUSY=1 skips the check).
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $(hostname): other users' processes that may hold the device; stopping before the tree copies:"
    echo "$busy"
    exit 1
fi
trees="base"
for c in $CONFIGS; do case $c in main) ;; *) trees="$trees $c" ;; esac; done
for t in $trees; do cp -al $G/trees/$t $J/$t || { echo "!!! copy of $t failed"; exit 1; }; done
echo "trees copied ($trees) $(date -u +%T)"

run() {  # label cache tree kernel_version flags yaml [ENV=VAL...]
    local label=$1 cache=$J/cache_$2 tree=$J/$3 kv=$4 flags=$5 yaml=$6 to=1500 n=2; shift 6
    case $label in *_val_nt) to=${NT_TIMEOUT:-3000} ;; esac
    local base=$label
    while [ -e $J/results/$label.log ]; do label=${base}_r$n; n=$((n + 1)); done  # a repeated step: _r2, _r3, ...
    local lib=$tree/build_Release
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    echo "=== $label ($(date -u +%T)): tree ${tree##*/}, kernel v$kv, flags $flags, $(basename $yaml), timeout $to s $*"
    local log=$J/results/$label.log t0=$SECONDS busy=0
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$cache TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_KERNEL_VERSION=$kv TT_FABRIC_EXP_FLAGS=$flags "$@" \
        timeout $to ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml \
        > $log 2>&1) &
    local pid=$!
    # A device another process holds (job 155971: a KMD lock held on an --exclusive node for the whole timeout): no test
    # starts while UMD waits for the lock. Stop after 180 s of that instead of at the timeout.
    while kill -0 $pid 2>/dev/null; do
        sleep 10
        if [ $((SECONDS - t0)) -ge ${BUSY_S:-180} ] && ! grep -aq "Running Test" $log && grep -aq "Waiting for KMD lock" $log; then
            busy=1
            pkill -TERM -P $pid
            sleep 15
            pkill -KILL -f "test_tt_fabric --test_config $yaml"
            break
        fi
    done
    wait $pid
    local rc=$?
    if [ $busy = 1 ]; then
        echo "!!! $label: device held by another process (KMD lock wait > ${BUSY_S:-180} s, no test ran): stopping"
        grep -a "Waiting for KMD lock" $log | tail -1
        ps -eo user:16,pid,etime,args | awk -v me="$USER" 'NR == 1 || ($1 != me && /tt[-_]|tenstorrent|python|pytest|test_/)' |
            cut -c1-200 | head -20
        exit 1
    fi
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && cp $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $J/results/$label.log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    echo "$label: exit $rc; ${res:-NO RESULT}"
    sed 's/\x1b\[[0-9;]*m//g' $J/results/$label.log | grep -aE 'TT_FATAL|TT_THROW|FAILED' | head -3
    [ $rc = 124 ] && { echo "!!! $label timed out (hang?): stopping"; exit 1; }
    return $rc
}

config() {  # config label-suffix yaml [ENV=VAL...]
    local c=$1 s=$2 yaml=$3; shift 3
    case $c in
        main) run main_$s main base 1 0 $yaml "$@" ;;
        v3q) run v3q_$s v3q v3q 2 13129088 $yaml "$@" ;;
        v4*) run ${c}_$s $c $c 2 13129088 $yaml "$@" ;;
        *) echo "!!! unknown config $c" ;;
    esac
}

S=$J/base/tests/tt_metal/tt_fabric/test_infra/test_yamls
Y=$G/yamls
suite_yaml() {
    case $1 in
        ds) echo $S/test_fabric_2d_torus_deadlock_stability.yaml ;;
        st) echo $Y/glx_torus_stability_short.yaml ;;
        nt) echo $S/test_fabric_2d_torus_nightly.yaml ;;
        sa) echo $Y/glx_ring_sanity.yaml ;;
        ring) echo $Y/glx_ring.yaml ;;
        mcs) echo $Y/glx_ring_mcast_sweep.yaml ;;
        mesh) echo $J/base/fabric_exp/galaxy/glx_mesh.yaml ;;
        linear) echo $J/base/fabric_exp/galaxy/glx_linear.yaml ;;
        xr|xr1) echo $Y/glx_express_ring.yaml ;;
        xl|xl1) echo $Y/glx_express_line.yaml ;;
        *) echo "" ;;
    esac
}
suite_env() {  # the suite's extra environment: its mesh graph descriptor, the ERISC mode
    case $1 in
        xr|xr1) echo TT_MESH_GRAPH_DESC_PATH=$J/base/tests/tt_metal/tt_fabric/custom_mesh_descriptors/express_links_8x4_mesh_graph_descriptor.textproto ;;
        xl|xl1) echo TT_MESH_GRAPH_DESC_PATH=$Y/express_links_8x4_line_ring_mesh_graph_descriptor.textproto ;;
    esac
    case $1 in x?1) echo TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 ;; esac
}
# STEPS: config:suite pairs, run in order (validation suites are labelled val_<suite>).
for step in ${STEPS:-main:ds v4r:ds main:st v4r:st main:sa v4r:sa main:ring v4r:ring v4r:nt}; do
    c=${step%%:*}; s=${step##*:}; yaml=$(suite_yaml $s)
    [ -z "$yaml" ] && { echo "!!! unknown suite $s"; continue; }
    case $s in ring|mcs|mesh|linear) label=$s ;; *) label=val_$s ;; esac
    config $c $label $yaml $(suite_env $s)
done
echo "done $(date -u +%T)"
