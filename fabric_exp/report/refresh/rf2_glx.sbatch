#!/bin/bash
# Report refresh 2 at 200G on one Blackhole Galaxy (coordinator 2026-11-06): fused write + atomic on the 1D line, ring,
# XY torus (across the wrap) and neighbor exchange (1 link) rows (refresh/yamls/rf2g_*.yaml, benchmark mode), main
# against the port in one job, two passes in reversed router order, stock test infra, telemetry off.
#   main  /data/aagarwal/lb2/trees/rf2_main_b078.tar.zst: b078540a262 (the PR's base) + the TT_FABRIC_NO_BW_TELEMETRY switch
#   port  /data/aagarwal/lb2/trees/rf2_port_8047.tar.zst: 8047cfa60ca (the PR tip) + the same switch
#   (pr_perf/stage_pr_tree.sh: test binary, host libraries, runtime and JIT sources, MD5SUMS and COMMIT)
# Both trees are unpacked on the node's local disk (/tmp; nothing on /data but the results: its inodes are short),
# with every run's kernel cache there too, deleted once the run's results are copied; the local copy is removed at exit.
# Steps: pass a: check (glx_check_2d.yaml, data-checked) main, port; per suite main, port. Pass b: suites reversed, port,
# main. A run past its timeout is a hang: killed, captured (hang_capture.py), reset, the job stops. A failed run resets
# and the job goes on. A node whose links don't map ring / torus skips those steps (exit 3).
#   sbatch --exclude=... rf2_glx.sbatch     PREFLIGHT_ONLY=1 bash rf2_glx.sbatch: inputs only
#SBATCH --partition=bh_sc36_6,bh_110a_sc16
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=frs-glx-rf2
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
Y=${Y:-$G/yamls/rf2}
T=/data/aagarwal/lb2/trees
declare -A TAR=([main]=${TAR_MAIN:-$T/rf2_main_b078.tar.zst} [port]=${TAR_PORT:-$T/rf2_port_8047.tar.zst})  # TAR_*: the dry run's stubs only
declare -A WANT=([main]=b078540a262 [port]=8047cfa60ca)
PY=${PY:-/data/aagarwal/regress/trees/main_d2bd12f/python_env/bin/python}
TSMI=${TSMI:-tt-smi}
SUITES=(linear ring torus nx)
STEPS="a:main:check a:port:check"
for s in "${SUITES[@]}"; do STEPS="$STEPS a:main:$s a:port:$s"; done
for ((i = ${#SUITES[@]} - 1; i >= 0; i--)); do STEPS="$STEPS b:port:${SUITES[$i]} b:main:${SUITES[$i]}"; done
NODE=${SLURMD_NODENAME:-$(hostname)}
L=${LOCAL_ROOT:-/tmp/aagarwal-rf2-${SLURM_JOB_ID:-manual}}
echo "node=$NODE job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) local=$L"
echo "steps: $STEPS"
suite_yaml() { case $1 in check) echo $Y/glx_check_2d.yaml ;; *) echo $Y/rf2g_$1.yaml ;; esac; }
miss=0
for c in main port; do [ -f ${TAR[$c]} ] && [ -f ${TAR[$c]}.sha256 ] || { echo "!!! no ${TAR[$c]}"; miss=1; }; done
for s in check "${SUITES[@]}"; do [ -f $(suite_yaml $s) ] || { echo "!!! no $(suite_yaml $s)"; miss=1; }; done
[ -f $Y/hang_capture.py ] && [ -x $PY ] || { echo "!!! hang capture tools missing"; miss=1; }
[ $miss = 0 ] || { echo "!!! preflight failed: stopping"; exit 1; }
echo "preflight: $(echo $STEPS | wc -w) steps; tarballs, yamls and tools present"
[ -n "${PREFLIGHT_ONLY:-}" ] && exit 0
case "$NODE" in bh-glx-110-a*) ;; *) [ -n "${DRY_NODE_OK:-}" ] || { echo "!!! node $NODE is not a bh-glx-110-a* machine: stopping"; exit 1; } ;; esac
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then echo "!!! $NODE: other users' processes that may hold the device; stopping:"; echo "$busy"; exit 1; fi
free_gb=$(df -BG --output=avail $(dirname $L) | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 12 ] || { echo "!!! $(dirname $L) has ${free_gb} GB free (< 12): stopping"; exit 1; }
mkdir -p $J/results
trap 'rm -rf "${L:?}"' EXIT
env | grep -E '^TT_' | sort > $J/results/submit_env.txt
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
unset PYTHONPATH VIRTUAL_ENV
for c in main port; do
    t0=$SECONDS; mkdir -p $L/$c
    (cd $(dirname ${TAR[$c]}) && sha256sum -c --quiet $(basename ${TAR[$c]}).sha256) || { echo "!!! ${TAR[$c]} fails its sha256"; exit 1; }
    tar -I zstd -xf ${TAR[$c]} -C $L/$c || { echo "!!! unpack of ${TAR[$c]} failed"; exit 1; }
    (cd $L/$c && md5sum -c --quiet MD5SUMS) || { echo "!!! $c: binaries differ from MD5SUMS"; exit 1; }
    case $(cat $L/$c/COMMIT) in ${WANT[$c]}*) ;; *) echo "!!! $c tree at $(cat $L/$c/COMMIT), want ${WANT[$c]}"; exit 1 ;; esac
    ln -sfn build_Release $L/$c/build; mkdir -p $L/$c/generated
    cp $L/$c/COMMIT $J/results/tree_${c}_commit.txt; cp $L/$c/PATCH $J/results/tree_${c}_patch.txt
    echo "$c: $(cut -c1-11 $L/$c/COMMIT) unpacked at $L/$c ($(du -sh $L/$c | cut -f1), $((SECONDS - t0)) s)"
done
glx_reset() {
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$? chips; chips=$($TSMI -ls 2>/dev/null | grep -c "Blackhole")
    echo "reset: exit $rc, Blackhole chips listed: $chips"
    [ $rc = 0 ] && [ "$chips" -ge 32 ]
}
glx_reset "before the steps" || { echo "!!! reset failed: stopping"; exit 1; }
kill_mine() { pkill -KILL -u "$USER" -f "$L/(main|port)/" 2>/dev/null; sleep 5; }
FAILED=""; NO_RING=0; SKIPPED=""
run() {  # label tree suite pass
    local label=$1 c=$2 s=$3 p=$4 tree=$L/$2 t0=$SECONDS busy=0 to=1500 yaml cache log lib
    yaml=$(suite_yaml $s); lib=$tree/build_Release; log=$J/results/$label.log; cache=$L/cache_$label
    [ $s = check ] && to=900
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    echo "=== $label ($(date -u +%T)): $c, $(basename $yaml), timeout $to s"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree TT_METAL_RUNTIME_ROOT=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$cache TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        timeout $to ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml > $log 2>&1) &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        sleep 5
        if [ $((SECONDS - t0)) -ge 180 ] && ! grep -aq "Running Test" $log && grep -aq "Waiting for KMD lock" $log; then busy=1; kill_mine; break; fi
    done
    wait $pid; local rc=$?
    [ $busy = 1 ] && { echo "!!! $label: device held by another process: stopping"; exit 1; }
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && cp $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    local clean; clean=$(sed 's/\x1b\[[0-9;]*m//g' $log)
    local res; res=$(echo "$clean" | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    if [ $rc != 0 ] && [ -z "$res" ] && echo "$clean" | grep -aq "could not fit in the discovered physical topology"; then
        NO_RING=1; SKIPPED="$SKIPPED $label"; rm -rf "${cache:?}"
        echo "!!! $label: this node's links do not map the $s graph (nothing ran): skipping the ring/torus steps"
        return 0
    fi
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}"
    awk -F, 'NR>1 {printf "   %s %s B %.1f cycles/packet (1.35 GHz)\n", $1, $7, 1.35e9/$10}' $J/results/$label.csv 2>/dev/null
    echo "$clean" | grep -aE 'TT_FATAL|TT_THROW|FAILED|mismatch' | head -3
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing, capturing, resetting, stopping"; kill_mine
        timeout 600 $PY $Y/hang_capture.py $J/results/${label}_hangcap --time-budget 540 --meta cache=$cache --meta label=$label \
            --meta node=$NODE > $J/results/${label}_hangcap.log 2>&1
        echo "hang capture: exit $? ($(tail -n 1 $J/results/${label}_hangcap.log))"
        glx_reset "after a timeout in $label"; exit 1
    fi
    rm -rf "${cache:?}"  # results copied: the run's kernel cache goes (node-local disk)
    if [ $rc != 0 ] || ! echo "$res" | grep -q ", 0 failed"; then
        echo "!!! $label failed: resetting, going on"; FAILED="$FAILED $label"
        glx_reset "after a failure in $label" || { echo "!!! reset failed: stopping"; exit 1; }
    fi
}
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    if [ $NO_RING = 1 ] && case $s in ring | torus | nx) true ;; *) false ;; esac; then
        echo "skipped ${p}_${c}_$s (no ring/torus mapping on $NODE)"; SKIPPED="$SKIPPED ${p}_${c}_$s"; continue
    fi
    run ${p}_${c}_$s $c $s $p
done
echo "done $(date -u +%T)"
[ -z "$FAILED" ] || echo "FAILED:$FAILED"
[ -z "$SKIPPED" ] || { echo "SKIPPED on $NODE:$SKIPPED"; exit 3; }
