#!/bin/bash
# The bubble-flow-control injection rule (RING_JAM_FIX.md §10) on one BH Galaxy at 200G, base against branch, in one
# job (frs_glx_frgap.sbatch's run loop for two mainline /data trees, run through per-job symlink farms; no study
# variant, no study switches):
#   base   $BASE_TREE (default /data/aagarwal/regress/trees/bfcbase: 4db4f45790b, the pushed port, built like the branch
#          in /home/aagarwal/tt-metal-mainline-bfcbase; `bfc_stage_glx.sh base`)
#   bfc    $BFC_TREE  (default /data/aagarwal/regress/trees/bfc: branch aagarwal/fabric-router-mainline-bfc;
#          `bfc_stage_glx.sh bfc`)
# Steps (default): the 2D check on each tree; the data-checked full-ring flow-control validation (glx_frg_val: Ring and
# XY-torus full-ring multicast and all-to-all, unicast write + fused atomic inc, 4096 B) on the branch, then on the base;
# pass a suite by suite (base, bfc), pass b reversed (bfc, base), suites frg_nsrx (8-ring 3072 / 3584 / 4352 B),
# frg_ew (4-ring 4352 B), frg_short4k (ring 1-hop and half ring, 4352 B). STEPS=PASS:CONFIG:SUITE ... replaces them.
# Yamls and the capture tool come from $BFC_DATA (bfc_stage_glx.sh copies them: fabric_exp/galaxy/frgap,
# glx_check_2d.yaml, bfc/hang_capture.py); the capture runs with $CAPTURE_PY (default: the main /data tree's python_env,
# which has tt-exalens; the two trees have none).
# Every run: TT_FABRIC_NO_BW_TELEMETRY=1, the host's default opt level and slots (every TT_* variable of the submitting
# shell unset first), TT_METAL_CLEAR_L1=1, compile commands logged. A run past its timeout is a hang: this job's processes
# are killed, the routers captured read-only, the Galaxy reset, and the job stops. A failed run stops the job.
#   sbatch --exclude=<list> fabric_exp/galaxy/frs_glx_bfc.sbatch
#   dry run (no Slurm, no device): G=<scratch> SLURM_JOB_ID=dry TSMI=<stub> DRY_NODE_OK=1 IGNORE_BUSY=1 bash frs_glx_bfc.sbatch
#   preflight only (the real /data trees, nothing written): PREFLIGHT_ONLY=1 DRY_NODE_OK=1 IGNORE_BUSY=1 bash frs_glx_bfc.sbatch
# Rules: only bh-glx-110-a* nodes (the job stops at once anywhere else); other users' device work on the node stops the
# job before anything (IGNORE_BUSY=1 skips); tt-smi -glx_reset before the steps.
#SBATCH --partition=bh_sc36_6,bh_110a_sc16
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=frs-glx-bfc
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
BASE_TREE=${BASE_TREE:-/data/aagarwal/regress/trees/bfcbase}
CAPTURE_PY=${CAPTURE_PY:-/data/aagarwal/regress/trees/main_d2bd12f/python_env/bin/python}
BFC_TREE=${BFC_TREE:-/data/aagarwal/regress/trees/bfc}
BFC_DATA=${BFC_DATA:-/data/aagarwal/frs-glx/bfc}
TSMI=${TSMI:-tt-smi}
SUITES="frg_nsrx frg_ew frg_short4k"
DEF="a:base:check a:bfc:check a:bfc:frg_val a:base:frg_val"
for s in $SUITES; do DEF="$DEF a:base:$s a:bfc:$s"; done
for s in $SUITES; do DEF="$DEF b:bfc:$s b:base:$s"; done
STEPS=${STEPS:-$DEF}
NODE=${SLURMD_NODENAME:-$(hostname)}
echo "node=$NODE job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) base=$BASE_TREE bfc=$BFC_TREE"
echo "steps=$STEPS"
# bh-glx-120-c* (partition bh_sc36_2) only with GLX_SC36_2_EXCEPTION=1 at submit (user's one-time exception,
# 2026-11-04); the default guard is 110-a only.
case "$NODE" in
    bh-glx-110-a*) ;;
    bh-glx-120-c*) [ "${GLX_SC36_2_EXCEPTION:-0}" = 1 ] || [ -n "${DRY_NODE_OK:-}" ] ||
        { echo "!!! node $NODE is a 120-c machine and GLX_SC36_2_EXCEPTION is not set: stopping"; exit 1; } ;;
    *) [ -z "${DRY_NODE_OK:-}" ] && { echo "!!! node $NODE is not a bh-glx-110-a* machine: stopping"; exit 1; } ;;
esac
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $NODE: other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi
suite_yaml() {  # suite
    case $1 in
        check) echo $BFC_DATA/yamls/glx_check_2d.yaml ;;
        frg_*) echo $BFC_DATA/yamls/glx_$1.yaml ;;
        *) echo "" ;;
    esac
}
tree_of() { case $1 in base) echo $BASE_TREE ;; bfc) echo $BFC_TREE ;; esac; }
# Preflight on the source trees (before the copies and the reset).
miss=0
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    t=$(tree_of $c); y=$(suite_yaml $s)
    if [ -z "$p" ] || [ -z "$t" ] || [ -z "$y" ] || [ ! -f "$y" ]; then echo "!!! step $st: no tree/yaml (${y:-?})"; miss=1; fi
    for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric build/lib/libtt_metal.so; do
        [ -n "$t" ] && [ ! -e $t/$f ] && { echo "!!! step $st: no $t/$f"; miss=1; }
    done
done
for f in $CAPTURE_PY $BFC_DATA/hang_capture.py $BASE_TREE/.bfc_tree.json $BFC_TREE/.bfc_tree.json; do
    [ -e $f ] || { echo "!!! no $f"; miss=1; }
done
[ $miss = 0 ] || { echo "!!! preflight failed: stopping before the copies"; exit 1; }
echo "preflight: $(echo $STEPS | wc -w) steps, every yaml and tool present"
echo "base tree: $(tr -d '\n ' < $BASE_TREE/.bfc_tree.json)"
echo "bfc tree: $(tr -d '\n ' < $BFC_TREE/.bfc_tree.json)"
[ -n "${PREFLIGHT_ONLY:-}" ] && exit 0
mkdir -p $J/results
env | grep -E '^TT_' | sort > $J/results/submit_env.txt
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
unset PYTHONPATH VIRTUAL_ENV

# Trees: a farm per tree (symlinks to the tree's top-level entries, this job's own generated/): /data is short of
# inodes, so no copies. The binaries load the tree's own libraries (LD_LIBRARY_PATH, in run).
copy_tree() {  # source dir
    mkdir -p $J/$2/generated || { echo "!!! farm $2 failed"; exit 1; }
    local e
    for e in $(ls -A $1); do [ $e = generated ] || ln -s $1/$e $J/$2/$e; done
    tar czf $J/results/src_$2.tgz -C $1 tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric \
        tt_metal/hw/inc tt_metal/hostdevcommon tt_metal/fabric/erisc_datamover_builder.cpp \
        tt_metal/fabric/builder/fabric_builder_config.hpp
}
need_base=0; need_bfc=0
for st in $STEPS; do case $st in *:base:*) need_base=1 ;; *:bfc:*) need_bfc=1 ;; esac; done
[ $need_base = 1 ] && copy_tree $BASE_TREE base
[ $need_bfc = 1 ] && copy_tree $BFC_TREE bfc
echo "tree farms made $(date -u +%T) (base: $need_base, bfc: $need_bfc)"
PY=$CAPTURE_PY
# Kernel caches on the node's /tmp (/data is short of inodes; dyn_glx.sbatch's LOCAL_CACHE=1). At exit the router
# builds (named args and ELFs, no XIP copies) are copied to $J/router_builds_CONFIG/<hash>/ for analysis, then the
# node-local caches are removed. LOCAL_CACHE=0 keeps the caches in $J instead.
CR=$J
if [ "${LOCAL_CACHE:-1}" = 1 ]; then
    CR=/tmp/$USER-bfcglx-${SLURM_JOB_ID:-manual}; mkdir -p $CR || { echo "!!! no node-local cache dir $CR"; exit 1; }
    echo "kernel caches node-local: $CR ($(df -h /tmp | tail -1 | awk '{print $4}') free on /tmp)"
fi
save_router_builds() {
    local c d h k n=0
    for c in base bfc; do
        for d in $CR/cache_$c/*/kernels/fabric_erisc_router/*/; do
            [ -f $d/named_ct_arg_map_generated.h ] || continue
            h=$(basename $d); mkdir -p $J/router_builds_$c/$h; cp $d/named_ct_arg_map_generated.h $J/router_builds_$c/$h/
            for k in active_erisc subordinate_active_erisc; do
                [ -f $d/$k/$k.elf ] && { mkdir -p $J/router_builds_$c/$h/$k; cp $d/$k/$k.elf $J/router_builds_$c/$h/$k/; }
            done
            n=$((n + 1))
        done
    done
    echo "router builds saved: $n"
}
finish() { [ "$CR" != "$J" ] && { save_router_builds; rm -rf "${CR:?}"; }; }
trap finish EXIT

glx_reset() {  # reason
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T): $($TSMI --version 2>&1 | head -1)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$? chips
    chips=$($TSMI -ls 2>/dev/null | grep -c "Blackhole")
    echo "reset: exit $rc, Blackhole chips listed: $chips"
    [ $rc = 0 ] && [ "$chips" -ge 32 ]
}
glx_reset "before the steps" || { echo "!!! reset failed: stopping"; exit 1; }
kill_mine() { pkill -KILL -u "$USER" -f "$J/(base|bfc)/" 2>/dev/null; sleep 5; }
capture() {  # label cache
    [ "${CAPTURE:-1}" = 1 ] || return 0
    echo "=== hang capture ${1}_hangcap ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-600} $PY $BFC_DATA/hang_capture.py $J/results/${1}_hangcap \
        --time-budget ${CAPTURE_BUDGET:-540} --meta cache=$2 --meta label=$1 --meta node=$NODE \
        > $J/results/${1}_hangcap.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/${1}_hangcap 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/${1}_hangcap.log)"
}

run() {  # label config suite
    local label=$1 c=$2 s=$3 tree=$J/$2 to=${BW_TIMEOUT:-1200} t0=$SECONDS busy=0
    local yaml; yaml=$(suite_yaml $s)
    case $s in check | frg_val*) to=${ROW_TIMEOUT:-900} ;; esac
    local log=$J/results/$label.log cache=$CR/cache_$c
    [ -e $log ] && { echo "!!! $label ran already: stopping"; exit 1; }
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    echo "=== $label ($(date -u +%T)): tree $c, $(basename $yaml), timeout $to s"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree LD_LIBRARY_PATH=$(readlink -f $tree/build)/lib \
        TT_METAL_CACHE=$cache TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        timeout $to $tree/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml > $log 2>&1) &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        sleep 5
        if [ $((SECONDS - t0)) -ge ${BUSY_S:-180} ] && ! grep -aq "Running Test" $log && grep -aq "Waiting for KMD lock" $log; then
            busy=1
            kill_mine
            break
        fi
    done
    wait $pid
    local rc=$?
    if [ $busy = 1 ]; then
        echo "!!! $label: device held by another process (KMD lock wait > ${BUSY_S:-180} s, no test ran): stopping"
        exit 1
    fi
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && cp $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    if [ $rc != 0 ] && [ -z "$res" ] && grep -aq "could not fit in the discovered physical topology" $log; then
        echo "!!! $label: this node's links do not map the ring/torus graph (nothing ran): stopping (exclude the node)"
        sed 's/\x1b\[[0-9;]*m//g' $log | grep -a "TT_FATAL" | head -1 | cut -c1-300
        exit 3
    fi
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}"
    case $s in check | frg_*) awk -F, 'NR>1 {printf "   %s %s B %.1f cycles/packet (1.35 GHz)\n", $1, $7, 1.35e9/$10}' $J/results/$label.csv 2>/dev/null ;; esac
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED|mismatch' | head -3
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing this job's processes, capturing, resetting, stopping"
        kill_mine; capture $label $cache; glx_reset "after a timeout in $label"; exit 1
    fi
    if [ $rc != 0 ] || ! echo "$res" | grep -q ", 0 failed"; then
        echo "!!! $label failed: stopping"; glx_reset "after a failure in $label"; exit 1
    fi
}

for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    [ -n "$p" ] && [ -n "$c" ] && [ -n "$s" ] || { echo "!!! bad step $st: stopping"; exit 1; }
    run ${p}_${c}_$s $c $s
done
echo "done $(date -u +%T)"
