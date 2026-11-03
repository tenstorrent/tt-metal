#!/bin/bash
# REFERENCE ONLY (plan change 2026-11-02): a separate Galaxy agent runs the Galaxy points 2, 4 and 5 with its own
# job; this breakdown task submits no Galaxy job. Kept as a worked example of what the mapping needs (two trees,
# source snapshots, kept caches, compile logs): fabric_exp/report/BREAKDOWN_PLAN.md, "Galaxy samples: what to save".
# Cycle breakdown, phase 2 on one Blackhole Galaxy (fabric_exp/report/BREAKDOWN_PLAN.md): PC samples (pc_sampler.py,
# tt-exalens from the tree's python_env, read-only over the debug bus) of both ERISCs of every router while one long row
# runs, for points 2 (2D 1-hop write 1024 B), 4 (7-hop N/S write 1024 B) and 5 (full-ring multicast write 4352 B), on
# main's router as shipped (tree bd_main_TAG: kernel v1, flags 0, firmware keepalive) and the new router (tree
# bd_new_TAG: kernel v2 slot, flags 13129088, keepalive 1000), two passes in reversed router order (a: main then new;
# b: new then main), then one unsampled run of the new router's point-2 row (the sampler's perturbation check).
# Trees: fabric_exp/consolidation/bd_mk_glx_trees.sh NEW_VARIANT TAG.
#
#   TAG=<tag> sbatch fabric_exp/consolidation/bd_glx.sbatch
#   dry run first (stubs, no Slurm, no device):
#     G=<scratch> SLURM_JOB_ID=dry TAG=<tag> TSMI=<stub> NODE=bh-glx-110-a00u00 IGNORE_BUSY=1 bash bd_glx.sbatch
#
# Rules (memory, TODO.md): ONLY bh-glx-110-a* nodes (partition bh_sc36_6; the job stops at once anywhere else); one study
# Galaxy allocation at a time (check squeue -u $USER first: another agent's job may hold the slot); dry-run first;
# BW telemetry off. Results in $G/jobs/<id>/results; the kernel caches $G/jobs/<id>/cache_{main,new} and the source
# snapshots results/src_{main,new}.tgz are what the mapping reads: keep them.
#SBATCH --partition=bh_sc36_6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=bd-glx
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
TAG=${TAG:?TAG=<tag>: trees bd_main_TAG and bd_new_TAG}
J=$G/jobs/${SLURM_JOB_ID:-manual}
TSMI=${TSMI:-tt-smi}
NODE=${NODE:-${SLURMD_NODENAME:-$(hostname)}}
SAMPLE_S=${SAMPLE_S:-90}
echo "node=$NODE job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) tag=$TAG"
case $NODE in
    bh-glx-110-a*) ;;
    *) echo "!!! $NODE is not a bh-glx-110-a* node: stopping (study rule)"; exit 1 ;;
esac
mkdir -p $J/results

busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $NODE: other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi
for r in main new; do
    [ -d $G/trees/bd_${r}_$TAG ] || { echo "!!! no tree $G/trees/bd_${r}_$TAG"; exit 1; }
    cp -al $G/trees/bd_${r}_$TAG $J/tree_$r || { echo "!!! copy of bd_${r}_$TAG failed"; exit 1; }
    tar czf $J/results/src_$r.tgz -C $J/tree_$r tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric \
        tt_metal/hw/inc tt_metal/hostdevcommon
done
echo "trees copied $(date -u +%T)"
if ! $J/tree_new/python_env/bin/python -c "import ttexalens, yaml" 2>$J/results/preflight.log; then
    echo "!!! the tree's python_env (tt-exalens) does not run here:"; cat $J/results/preflight.log; exit 1
fi

glx_reset() {  # reason
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$? chips
    chips=$($TSMI -ls 2>/dev/null | grep -c "Blackhole")
    echo "reset: exit $rc, Blackhole chips listed: $chips"
    [ $rc = 0 ] && [ "$chips" -ge 32 ]
}
glx_reset "before the steps" || { echo "!!! reset failed: stopping"; exit 1; }
kill_mine() { pkill -KILL -u "$USER" -f "$J/tree_" 2>/dev/null; sleep 5; }

run() {  # label router yaml timeout sample(0|1)
    local label=$1 r=$2 yaml=$3 to=$4 smp=$5 tree=$J/tree_$2 t0=$SECONDS
    local lib=$tree/build_Release log=$J/results/$label.log env
    if [ $r = main ]; then env="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
    else env="TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"; fi
    echo "=== $label ($(date -u +%T)): $r, $(basename $yaml)$([ $smp = 1 ] && echo ', sampled')"
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE -u TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$J/cache_$r TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_FABRIC_OPT_LEVEL=O3 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 $env \
        timeout $to ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $tree/$yaml > $log 2>&1) &
    local pid=$!
    if [ $smp = 1 ]; then
        timeout $((SAMPLE_S + 300)) $tree/python_env/bin/python $tree/fabric_exp/consolidation/pc_sampler.py \
            $J/results/${label}_samples --start-log $log --start-delay-s 5 --max-s $SAMPLE_S --wait-s 600 \
            > $J/results/${label}_sampler.log 2>&1
        echo "   $(grep -E '^sampler: [0-9]+ samples' $J/results/${label}_sampler.log || tail -n 2 $J/results/${label}_sampler.log | tr '\n' ' ')"
    fi
    wait $pid
    local rc=$?
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && mv $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}; $(awk -F, 'NR>1 {printf "%s %s B %.1f cycles/packet; ", $1, $7, 1.35e9/$10}' $J/results/$label.csv 2>/dev/null)"
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing this job's processes, resetting, stopping"
        kill_mine; glx_reset "after a timeout in $label"; exit 1
    fi
    if [ $rc != 0 ] || ! echo "$res" | grep -q ", 0 failed"; then
        echo "!!! $label failed: stopping"; exit 1
    fi
    if [ $smp = 1 ] && ! grep -qE '^sampler: [0-9]+ samples in .* 0 read errors' $J/results/${label}_sampler.log; then
        echo "!!! $label: the sampler did not finish cleanly (see ${label}_sampler.log): stopping"; exit 1
    fi
}

Y=fabric_exp/consolidation/bd_rows
points() {  # pass router
    run $1_$2_ew1024 $2 $Y/glx_ew1024.yaml 1200 1
    run $1_$2_ns7 $2 $Y/glx_ns7_1024.yaml 1200 1
    run $1_$2_fullring $2 $Y/glx_fullring4352.yaml 1500 1
}
run c_main_mesh main $Y/glx_compile_mesh.yaml 1200 0
run c_main_ring main $Y/glx_compile_ring.yaml 1200 0
points a main
run c_new_mesh new $Y/glx_compile_mesh.yaml 1200 0
run c_new_ring new $Y/glx_compile_ring.yaml 1200 0
points a new
points b new
points b main
run u_new_ew1024 new $Y/glx_ew1024.yaml 1200 0   # unsampled: the sampler's perturbation check against a/b_new_ew1024
echo "done $(date -u +%T)"
