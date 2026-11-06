#!/bin/bash
# PR perf refresh at 200G on one Blackhole Galaxy (user, 2026-11-05; the ring jam is fixed): main as shipped against the
# port tip, same job, two passes in reversed order, telemetry off for both.
#   main  /data/aagarwal/regress/trees/main_d2bd12f: d2bd12fcf1f + the TT_FABRIC_NO_BW_TELEMETRY switch
#         (fabric_exp/testplan/patches/main_no_bw_telemetry.patch, test harness only), built
#   port  /data/aagarwal/frs-glx/trees/port_e7fe189: e7fe1899049, a farm of regress/trees/mainline_c3c1813
#         (pr_perf/mkfarm_pr.py: the three router headers at e7fe1899049, the bring-up tools it deletes left out; host
#         binaries and test_tt_fabric.cpp from c3c1813c42a, which equal e7fe1899049 + the switch patch)
# Suites (inputs staged in /data/aagarwal/frs-glx/yamls/pr_e7fe): check = glx_check_2d (data-checked, first per router);
#   mesh / linear / ring / torus = glx_sizes_*.yaml (bandwidth_glx_sweep.json's suites, 256-4352 B, incl. the multicast
#   and full-ring multicast rows); sweep = pr_glx_size_sweep.yaml (2D and 1D 1-hop E/W write at 256, 512, 1024, 2048,
#   3072, 4096, 4352 B).
#   PR_SUITES="..." overrides the suite list (default: mesh linear ring torus sweep). The gap runs (coordinator,
#   2026-11-05; mk_pr_gap_yamls.py, 512 and 4096 B): k4mesh / k4linear / k4ring / k4torus = pr_glx_k4_<suite>.yaml;
#   nx = pr_glx_nx_sweep.yaml (NeighborExchange write, 1 and 2 links, 256-4352 B; released by the user 2026-11-05).
#   lat = pr_glx_latency.yaml (one-packet round trips; per-sample results <label>_latency.csv).
# Pass a: check main, port; then per suite main, port. Pass b: check port, main; then the suites reversed, port, main.
# Every run: TT_FABRIC_NO_BW_TELEMETRY=1, TT_METAL_CLEAR_L1=1, profiler off, compile commands logged, every inherited
# TT_* variable unset first, one kernel cache per router and pass. Labels <pass>_<main|port>_<suite> (.log, .csv summary,
# _tests.csv per sender) in $J/results. Each tree runs from a job farm ($J/main, $J/port: top-level symlinks, generated/ a
# real directory), so outputs stay in the job.
# A run past its timeout is a hang: this job's processes killed, the routers captured read-only (hang_capture.py), the
# Galaxy reset, the job stops. A failed run resets and stops. A node whose links don't map the ring / torus graphs
# skips those steps (exit 3). Rules: bh-glx-110-a* only; other users' device work stops the job; tt-smi -glx_reset first.
#   sbatch [--exclude ...] fabric_exp/pr_perf/frs_glx_pr.sbatch       PREFLIGHT_ONLY=1 bash ...: inputs only, no device
#SBATCH --partition=bh_sc36_6,bh_110a_sc16
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=frs-glx-pr
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
Y=${Y:-$G/yamls/pr_e7fe}
declare -A SRC=([main]=/data/aagarwal/regress/trees/main_d2bd12f [port]=$G/trees/port_e7fe189)
TSMI=${TSMI:-tt-smi}
SUITES=(${PR_SUITES:-mesh linear ring torus sweep})
STEPS="a:main:check a:port:check"
for s in "${SUITES[@]}"; do STEPS="$STEPS a:main:$s a:port:$s"; done
STEPS="$STEPS b:port:check b:main:check"
for ((i = ${#SUITES[@]} - 1; i >= 0; i--)); do STEPS="$STEPS b:port:${SUITES[$i]} b:main:${SUITES[$i]}"; done
STEPS=${STEPS_OVERRIDE:-$STEPS}
NODE=${SLURMD_NODENAME:-$(hostname)}
echo "node=$NODE job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) main=${SRC[main]} port=${SRC[port]}"
echo "steps=$STEPS"
suite_yaml() {
    case $1 in
        check) echo $Y/glx_check_2d.yaml ;; mesh | linear | ring | torus) echo $Y/glx_sizes_$1.yaml ;;
        sweep) echo $Y/pr_glx_size_sweep.yaml ;;
        k4mesh | k4linear | k4ring | k4torus) echo $Y/pr_glx_k4_${1#k4}.yaml ;; nx) echo $Y/pr_glx_nx_sweep.yaml ;;
        lat) echo $Y/pr_glx_latency.yaml ;;
        *) echo "" ;;
    esac
}
miss=0
grep -q TT_FABRIC_NO_BW_TELEMETRY ${SRC[main]}/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp || { echo "!!! main tree lacks the telemetry switch"; miss=1; }
grep -q TT_FABRIC_NO_BW_TELEMETRY ${SRC[port]}/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp || { echo "!!! port tree lacks the telemetry switch"; miss=1; }
grep -q '^d2bd12fcf1f' <(python3 -c "import json; print(json.load(open('${SRC[main]}/.regress_tree.json'))['sha'])") || { echo "!!! main tree is not d2bd12fcf1f"; miss=1; }
grep -q '^e7fe1899049' ${SRC[port]}/.tree_sha || { echo "!!! port tree is not e7fe1899049"; miss=1; }
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    y=$(suite_yaml $s)
    [ -n "$p" ] && [ -n "${SRC[$c]:-}" ] && [ -n "$y" ] && [ -f "$y" ] || { echo "!!! step $st: no tree/yaml (${y:-?})"; miss=1; }
done
for c in main port; do
    for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric build_Release/tt_metal/libtt_metal.so python_env/bin/python; do
        [ -e ${SRC[$c]}/$f ] || { echo "!!! $c: no ${SRC[$c]}/$f"; miss=1; }
    done
done
[ -f $Y/hang_capture.py ] || { echo "!!! no $Y/hang_capture.py"; miss=1; }
[ $miss = 0 ] || { echo "!!! preflight failed: stopping"; exit 1; }
echo "preflight: $(echo $STEPS | wc -w) steps; trees, yamls and tools present; both trees read TT_FABRIC_NO_BW_TELEMETRY"
[ -n "${PREFLIGHT_ONLY:-}" ] && exit 0
case "$NODE" in bh-glx-110-a*) ;; *) echo "!!! node $NODE is not a bh-glx-110-a* machine: stopping"; exit 1 ;; esac
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then echo "!!! $NODE: other users' processes that may hold the device; stopping:"; echo "$busy"; exit 1; fi
mkdir -p $J/results
env | grep -E '^TT_' | sort > $J/results/submit_env.txt
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
unset PYTHONPATH VIRTUAL_ENV
for c in main port; do  # job farm: top-level symlinks to the tree, generated/ real
    mkdir -p $J/$c/generated
    for e in ${SRC[$c]}/* ${SRC[$c]}/.[!.]*; do
        b=$(basename $e); [ "$b" = generated ] || ln -s $e $J/$c/$b
    done
done
PY=${SRC[main]}/python_env/bin/python
cp ${SRC[port]}/.tree_sha $J/results/port_tree_sha.txt; cp ${SRC[main]}/.regress_tree.json $J/results/main_tree.json

glx_reset() {
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$? chips; chips=$($TSMI -ls 2>/dev/null | grep -c "Blackhole")
    echo "reset: exit $rc, Blackhole chips listed: $chips"
    [ $rc = 0 ] && [ "$chips" -ge 32 ]
}
glx_reset "before the steps" || { echo "!!! reset failed: stopping"; exit 1; }
kill_mine() { pkill -KILL -u "$USER" -f "$J/(main|port)/" 2>/dev/null; sleep 5; }
capture() {
    echo "=== hang capture ${1}_hangcap ($(date -u +%T))"
    timeout 600 $PY $Y/hang_capture.py $J/results/${1}_hangcap --time-budget 540 --meta cache=$2 --meta label=$1 --meta node=$NODE \
        > $J/results/${1}_hangcap.log 2>&1
    echo "hang capture: exit $? ($(tail -n 1 $J/results/${1}_hangcap.log))"
}
run() {  # label config suite pass
    local label=$1 c=$2 s=$3 p=$4 tree=$J/$2 t0=$SECONDS busy=0 to=1500
    local yaml; yaml=$(suite_yaml $s)
    [ $s = check ] && to=900
    local lib=$tree/build_Release log=$J/results/$label.log cache=$J/cache_${c}_$p
    [ -e $log ] && { echo "!!! $label ran already: stopping"; exit 1; }
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv \
        $tree/generated/fabric/latency_results_blackhole.csv
    echo "=== $label ($(date -u +%T)): $c, $(basename $yaml), timeout $to s"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
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
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests latency_results_blackhole:_latency; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && cp $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    if [ $rc != 0 ] && [ -z "$res" ] && grep -aq "could not fit in the discovered physical topology" $log; then
        NO_RING=1; SKIPPED="$SKIPPED $label"
        echo "!!! $label: this node's links do not map the $s suite's graph (nothing ran): skipping the ring/torus steps"
        return 0
    fi
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}"
    [ $s = check ] && awk -F, 'NR>1 {printf "   %s %s B %.1f cycles/packet (1.35 GHz)\n", $1, $7, 1.35e9/$10}' $J/results/$label.csv 2>/dev/null
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED|mismatch' | head -3
    if [ $rc = 124 ]; then echo "!!! $label timed out (hang?): killing, capturing, resetting, stopping"; kill_mine; capture $label $cache; glx_reset "after a timeout in $label"; exit 1; fi
    if [ $rc != 0 ] || ! echo "$res" | grep -q ", 0 failed"; then echo "!!! $label failed: stopping"; glx_reset "after a failure in $label"; exit 1; fi
}
NO_RING=0; SKIPPED=""
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    if [ $NO_RING = 1 ] && case $s in *ring | *torus) true ;; *) false ;; esac; then echo "skipped ${p}_${c}_$s (no ring/torus mapping on $NODE)"; SKIPPED="$SKIPPED ${p}_${c}_$s"; continue; fi
    run ${p}_${c}_$s $c $s $p
done
echo "done $(date -u +%T)"
[ -z "$SKIPPED" ] || { echo "SKIPPED on $NODE:$SKIPPED"; exit 3; }
