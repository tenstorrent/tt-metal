#!/bin/bash
# 200G NeighborExchange A/B on one Blackhole Galaxy, port e7fe1899049 only (coordinator 2026-11-05; diagnostic arms, local
# patches, never committed): why the port trails main on 200G NX at wire-bound sizes (179756: 3072 / 4096 / 4352 B).
# Rows: pr_glx_nxab.yaml (NX write, 1 link, 3072 / 4096 / 4352 B, 200,000 packets). Every arm runs the A/B's host
# build (trees/nxab_host: e7fe1899049 + the telemetry switch + nxab/rx_slots_host.patch, which does nothing unless
# TT_FABRIC_AB_VC0_RX_SLOTS is set); device-side arms are farms of it with patched router headers (nxab/mk_nxab_trees.py).
#   a     default (receiver 32, worker 51)
#   b10 / b30 / b60   E0 busy-wait of 10 / 30 / 60 core cycles per packet sent (nxab/spin_sites.patch + spin_mcycle.hpp.in)
#   c16 / c64         VC0 receiver depth 16 / 64, the worker takes the difference (67 / 19): TT_FABRIC_AB_VC0_RX_SLOTS
#   d64   IDLE_PASSES_BEFORE_PARTIAL_CREDIT 64 (stock 4; nxab/idle_passes64.patch)
#   e60   E1 busy-wait of 60 core cycles per packet received and handled
# Pass a: check (arm a), then the arms in order; pass b: check, then the arms reversed. Every run: TT_FABRIC_NO_BW_TELEMETRY=1,
# TT_METAL_CLEAR_L1=1, profiler off, compile commands logged, inherited TT_* variables unset, one kernel cache per arm and
# pass. A run past its timeout is a hang: killed, captured read-only, Galaxy reset, job stops; a failure resets and stops.
# Rules: bh-glx-110-a* only; other users' device work stops the job; tt-smi -glx_reset first.
#   sbatch [--exclude ...] fabric_exp/pr_perf/nxab/frs_glx_nxab.sbatch       PREFLIGHT_ONLY=1 bash ...: inputs only
#SBATCH --partition=bh_sc36_6,bh_110a_sc16
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=frs-glx-nxab
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
Y=${Y:-$G/yamls/pr_e7fe}
declare -A SRC=([a]=$G/trees/nxab_host [b10]=$G/trees/nxab_b10 [b30]=$G/trees/nxab_b30 [b60]=$G/trees/nxab_b60
    [c16]=$G/trees/nxab_host [c64]=$G/trees/nxab_host [d64]=$G/trees/nxab_d64 [e60]=$G/trees/nxab_e60)
declare -A ARM_ENV=([c16]="TT_FABRIC_AB_VC0_RX_SLOTS=16" [c64]="TT_FABRIC_AB_VC0_RX_SLOTS=64")
PY=/data/aagarwal/regress/trees/main_d2bd12f/python_env/bin/python
TSMI=${TSMI:-tt-smi}
ARMS=(${NXAB_ARMS:-a b10 b30 b60 c16 c64 d64 e60})
STEPS="a:a:check"
for c in "${ARMS[@]}"; do STEPS="$STEPS a:$c:nx"; done
STEPS="$STEPS b:a:check"
for ((i = ${#ARMS[@]} - 1; i >= 0; i--)); do STEPS="$STEPS b:${ARMS[$i]}:nx"; done
STEPS=${STEPS_OVERRIDE:-$STEPS}
NODE=${SLURMD_NODENAME:-$(hostname)}
echo "node=$NODE job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) arms=${ARMS[*]}"
echo "steps=$STEPS"
suite_yaml() {
    case $1 in check) echo $Y/glx_check_2d.yaml ;; nx) echo $Y/pr_glx_nxab.yaml ;; *) echo "" ;; esac
}
miss=0
for c in "${ARMS[@]}"; do
    T=${SRC[$c]:-}
    [ -n "$T" ] && grep -q '^e7fe1899049\|^nxab arm' $T/.tree_sha 2>/dev/null || { echo "!!! arm $c: no tree (${T:-?})"; miss=1; continue; }
    grep -q TT_FABRIC_NO_BW_TELEMETRY $T/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp || { echo "!!! arm $c: no telemetry switch"; miss=1; }
    for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric build_Release/tt_metal/libtt_metal.so; do
        [ -e $T/$f ] || { echo "!!! arm $c: no $T/$f"; miss=1; }
    done
done
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    y=$(suite_yaml $s)
    [ -n "$p" ] && [ -n "${SRC[$c]:-}" ] && [ -n "$y" ] && [ -f "$y" ] || { echo "!!! step $st: no tree/yaml (${y:-?})"; miss=1; }
done
[ -x $PY ] || { echo "!!! no $PY"; miss=1; }
[ -f $Y/hang_capture.py ] || { echo "!!! no $Y/hang_capture.py"; miss=1; }
[ $miss = 0 ] || { echo "!!! preflight failed: stopping"; exit 1; }
echo "preflight: $(echo $STEPS | wc -w) steps; arm trees, yamls and tools present"
[ -n "${PREFLIGHT_ONLY:-}" ] && exit 0
case "$NODE" in bh-glx-110-a*) ;; *) echo "!!! node $NODE is not a bh-glx-110-a* machine: stopping"; exit 1 ;; esac
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then echo "!!! $NODE: other users' processes that may hold the device; stopping:"; echo "$busy"; exit 1; fi
mkdir -p $J/results
env | grep -E '^TT_' | sort > $J/results/submit_env.txt
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
unset PYTHONPATH VIRTUAL_ENV
for c in "${ARMS[@]}"; do  # job farm: top-level symlinks to the tree, generated/ real
    mkdir -p $J/$c/generated
    for e in ${SRC[$c]}/* ${SRC[$c]}/.[!.]*; do
        b=$(basename $e); [ "$b" = generated ] || ln -s $e $J/$c/$b
    done
done
for c in "${ARMS[@]}"; do cp ${SRC[$c]}/.tree_sha $J/results/tree_${c}.txt; done

glx_reset() {
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$? chips; chips=$($TSMI -ls 2>/dev/null | grep -c "Blackhole")
    echo "reset: exit $rc, Blackhole chips listed: $chips"
    [ $rc = 0 ] && [ "$chips" -ge 32 ]
}
glx_reset "before the steps" || { echo "!!! reset failed: stopping"; exit 1; }
kill_mine() { pkill -KILL -u "$USER" -f "$J/[a-z0-9]+/" 2>/dev/null; sleep 5; }
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
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 ${ARM_ENV[$c]:-} \
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
    run ${p}_${c}_$s $c $s $p
done
echo "done $(date -u +%T)"
[ -z "$SKIPPED" ] || { echo "SKIPPED on $NODE:$SKIPPED"; exit 3; }
