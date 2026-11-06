#!/bin/bash
# Report data refresh at 200G on one Blackhole Galaxy (user 2026-11-05): main as shipped against the port (PR tip),
# same job, two passes in reversed router order, telemetry off for both. One allocation.
#   main   /data/aagarwal/regress/trees/main_d2bd12f: d2bd12fcf1f + the TT_FABRIC_NO_BW_TELEMETRY switch, built
#   port   frs-glx/trees/rf_port_d51e: d51e42cd78a (origin/aagarwal/fabric-router-mainline tip) + the same switch; a farm
#          of regress/trees/mainline_c3c1813 (refresh/mk_rf_farm.py: the files that differ at d51e42cd78a, host libraries
#          and test_tt_fabric from /home/aagarwal/tt-metal-rf-port's build)
#   mainh / porth  frs-glx/trees/rf_mainh_d2bd, rf_porth_d51e: the same two routers with the A/B harness only (study test
#          infra test_infra_6dad, adapter_vc2_credits_6dad, the diag launch delay; router images untouched), for the
#          multi-VC rows; farms of main_d2bd12f / mainline_c3c1813 with the rf-mainh / rf-porth worktrees' builds
# Steps PASS:TREE:SUITE (inputs in /data/aagarwal/frs-glx/yamls/rf, refresh/mk_rf_yamls.py and copies):
#   check                 glx_check_2d.yaml (data-checked, flow control), every tree, pass a first
#   linear ring torus nx  rfg_*.yaml: the page's missing 200G cells, 256-4352 B
#   c_mesh c_ring         compile rows of the breakdown cache (galaxy/pcs/glx_compile_*.yaml)
#   s_ew1024 s_ns7 s_fullring  breakdown points 2, 4, 5 with pc_sampler.py beside the test (debug-bus PC reads, both
#                         ERISCs, every chip; galaxy/pcs/glx_{ew1024,ns7_1024,fullring4352}.yaml)
#   ew4352vc2 vc02ew<S> vc02bew<S> vc2ew<S>  VC0 + VC2 on one 2D E/W link (pcs3 rows, chip 0 -> 1, VC2 on)
#   vc02m<S>              the same with counts matched from this job's pass a (rf_matched.py, run after pass a's rows)
#   B_main_<g> B_new_<g>  VC0 + VC1 on the Galaxy as two 2x2 meshes (glx_2x2_split, slot study P2's layout; two ranks
#                         through tt-run), vc1/mvc3 rows, groups base, lag1, lag0 (200 ms launch delay), match
# Every run: TT_FABRIC_NO_BW_TELEMETRY=1, TT_METAL_CLEAR_L1=1, profiler off, compile commands logged, every inherited
# TT_* variable unset first; one kernel cache per tree and pass (the breakdown its own, kept). Labels PASS_TREE_SUITE in
# $J/results. A run past its timeout is a hang: killed, captured (hang_capture.py), reset, the job stops. A failed run
# resets the Galaxy and the job goes on. A node whose links don't map ring / torus skips those steps (exit 3).
#   sbatch --exclude=... fabric_exp/report/refresh/rf_glx.sbatch      PREFLIGHT_ONLY=1 bash ...: inputs only
#SBATCH --partition=bh_sc36_6,bh_110a_sc16
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --job-name=frs-glx-rf
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
Y=${Y:-$G/yamls/rf}
declare -A SRC=([main]=${SRC_MAIN:-/data/aagarwal/regress/trees/main_d2bd12f} [port]=$G/trees/rf_port_d51e
                [mainh]=$G/trees/rf_mainh_d2bd [porth]=$G/trees/rf_porth_d51e)  # SRC_MAIN: the dry run's stub only
TSMI=${TSMI:-tt-smi}
MPI_ARGS=${MPI_ARGS:---oversubscribe --mca btl self,sm}
MPI_BIN=${MPI_BIN:-/opt/openmpi-v5.0.7-ulfm/bin}  # as slotstudy/sls_glx.sbatch (the Galaxy split, job 167464)
export PATH=$MPI_BIN:$PATH
SAMPLE_S=${SAMPLE_S:-25}
VC2S="ew4352vc2 vc02ew256 vc02bew256 vc2ew256 vc02ew1024 vc02bew1024 vc2ew1024 vc02ew4352 vc02bew4352 vc2ew4352"
VC2SR=$(echo $VC2S | tr ' ' '\n' | tac | tr '\n' ' ')
cells() { local p=$1 a=$2 b=$3 s out=""; for s in linear ring torus nx; do out="$out $p:$a:$s $p:$b:$s"; done; echo $out; }
cellsr() { local p=$1 a=$2 b=$3 s out=""; for s in nx torus ring linear; do out="$out $p:$a:$s $p:$b:$s"; done; echo $out; }
bd() { local p=$1 t=$2; echo "$p:$t:c_mesh $p:$t:s_ew1024 $p:$t:s_ns7 $p:$t:c_ring $p:$t:s_fullring"; }
vc2() { local p=$1 t=$2 l=$3 s out=""; for s in $l; do out="$out $p:$t:$s"; done; echo $out; }
vc2m() { local p=$1 t=$2 s out=""; for s in 256 1024 4352; do out="$out $p:$t:vc02m$s"; done; echo $out; }
mvc() { local p=$1 t=$2 set=$3 g out=""; shift 3; for g in "$@"; do out="$out $p:$t:${set}_$g"; done; echo $out; }
STEPS="a:main:check a:port:check a:mainh:check a:porth:check $(cells a main port) $(bd a main) $(bd a port)
 $(vc2 a mainh "$VC2S") $(vc2 a porth "$VC2S") $(mvc a mainh B_main base lag1 lag0) $(mvc a porth B_new base lag1 lag0)
 a:-:matched $(vc2m a mainh) $(vc2m a porth) $(mvc a mainh B_main match) $(mvc a porth B_new match)
 b:porth:check b:mainh:check b:port:check b:main:check
 $(mvc b porth B_new match) $(mvc b mainh B_main match) $(vc2m b porth) $(vc2m b mainh)
 $(mvc b porth B_new lag0 lag1 base) $(mvc b mainh B_main lag0 lag1 base) $(vc2 b porth "$VC2SR") $(vc2 b mainh "$VC2SR")
 $(bd b port) $(bd b main) $(cellsr b port main)"
STEPS=$(echo ${STEPS_OVERRIDE:-$STEPS})
NODE=${SLURMD_NODENAME:-$(hostname)}
echo "node=$NODE job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ)"
for c in main port mainh porth; do echo "  $c = ${SRC[$c]}"; done
echo "steps ($(echo $STEPS | wc -w)): $STEPS"
suite_yaml() {  # suite pass tree
    case $1 in
        check) echo $Y/glx_check_2d.yaml ;; linear | ring | torus | nx) echo $Y/rfg_$1.yaml ;;
        c_mesh) echo $Y/glx_compile_mesh.yaml ;; c_ring) echo $Y/glx_compile_ring.yaml ;;
        s_ew1024) echo $Y/glx_ew1024.yaml ;; s_ns7) echo $Y/glx_ns7_1024.yaml ;; s_fullring) echo $Y/glx_fullring4352.yaml ;;
        ew4352vc2) echo $Y/ew4352.yaml ;; vc02ew* | vc02bew* | vc2ew*) echo $Y/$1.yaml ;;
        vc02m*) echo $J/results/a_$3_$1.yaml ;;
        B_main_match | B_new_match) echo $J/results/a_$3_$1.yaml ;;
        B_main_* | B_new_*) echo $Y/mvc3_$1.yaml ;;
        *) echo "" ;;
    esac
}
miss=0
for c in main port mainh porth; do
    grep -q TT_FABRIC_NO_BW_TELEMETRY ${SRC[$c]}/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp || { echo "!!! $c lacks the telemetry switch"; miss=1; }
    for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric build_Release/tt_metal/libtt_metal.so python_env/bin/python; do
        [ -e ${SRC[$c]}/$f ] || { echo "!!! $c: no ${SRC[$c]}/$f"; miss=1; }
    done
done
grep -q '^d2bd12fcf1f' <(python3 -c "import json; print(json.load(open('${SRC[main]}/.regress_tree.json'))['sha'])") || { echo "!!! main is not d2bd12fcf1f"; miss=1; }
grep -q '^d51e42cd78a' ${SRC[port]}/.tree_sha || { echo "!!! port is not d51e42cd78a"; miss=1; }
grep -q '^d51e42cd78a' ${SRC[porth]}/.tree_sha || { echo "!!! porth is not d51e42cd78a"; miss=1; }
grep -q '^d2bd12fcf1f' ${SRC[mainh]}/.tree_sha || { echo "!!! mainh is not d2bd12fcf1f"; miss=1; }
for c in mainh porth; do grep -q "Shared link" ${SRC[$c]}/tests/tt_metal/tt_fabric/test_infra/tt_fabric_test_bandwidth_profiler.cpp || { echo "!!! $c lacks the harness"; miss=1; }; done
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    [ $s = matched ] && continue
    y=$(suite_yaml $s $p $c)
    case $y in $J/*) continue ;; esac
    [ -n "$p" ] && [ -n "${SRC[$c]:-}" ] && [ -n "$y" ] && [ -f "$y" ] || { echo "!!! step $st: no tree/yaml (${y:-?})"; miss=1; }
done
for f in hang_capture.py pc_sampler.py rf_matched.py build_multivc3.py mvc3_yamls.py rb_glx_std.yaml rb_glx_lag0.yaml glx_2x2_split_mgd.textproto; do
    [ -f $Y/$f ] || { echo "!!! no $Y/$f"; miss=1; }
done
[ $miss = 0 ] || { echo "!!! preflight failed: stopping"; exit 1; }
echo "preflight: $(echo $STEPS | wc -w) steps; trees, yamls and tools present"
[ -n "${PREFLIGHT_ONLY:-}" ] && exit 0
case "$NODE" in bh-glx-110-a*) ;; *) echo "!!! node $NODE is not a bh-glx-110-a* machine: stopping"; exit 1 ;; esac
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then echo "!!! $NODE: other users' processes that may hold the device; stopping:"; echo "$busy"; exit 1; fi
mkdir -p $J/results
env | grep -E '^TT_' | sort > $J/results/submit_env.txt
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
unset PYTHONPATH VIRTUAL_ENV
for c in main port mainh porth; do  # job farm: top-level symlinks to the tree, generated/ real
    mkdir -p $J/$c/generated
    for e in ${SRC[$c]}/* ${SRC[$c]}/.[!.]*; do b=$(basename $e); [ "$b" = generated ] || ln -s $e $J/$c/$b; done
    cp ${SRC[$c]}/.tree_sha $J/results/tree_$c.txt 2>/dev/null || cp ${SRC[$c]}/.regress_tree.json $J/results/tree_$c.txt
    tar czf $J/results/src_$c.tgz -C ${SRC[$c]}/ --dereference tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric \
        tt_metal/hw/inc tt_metal/hostdevcommon
done
PY=${SRC[main]}/python_env/bin/python
$PY -c "import ttexalens, yaml" > $J/results/preflight_py.log 2>&1 || { echo "!!! the python_env (tt-exalens) does not run here"; cat $J/results/preflight_py.log; exit 1; }

glx_reset() {
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$? chips; chips=$($TSMI -ls 2>/dev/null | grep -c "Blackhole")
    echo "reset: exit $rc, Blackhole chips listed: $chips"
    [ $rc = 0 ] && [ "$chips" -ge 32 ]
}
glx_reset "before the steps" || { echo "!!! reset failed: stopping"; exit 1; }
kill_mine() { pkill -KILL -u "$USER" -f "$J/(main|port|mainh|porth)/" 2>/dev/null; pkill -KILL -u "$USER" -f "pc_sampler.py $J/" 2>/dev/null; sleep 5; }
capture() {
    echo "=== hang capture ${1}_hangcap ($(date -u +%T))"
    timeout 600 $PY $Y/hang_capture.py $J/results/${1}_hangcap --time-budget 540 --meta cache=$2 --meta label=$1 --meta node=$NODE \
        > $J/results/${1}_hangcap.log 2>&1
    echo "hang capture: exit $? ($(tail -n 1 $J/results/${1}_hangcap.log))"
}
FAILED=""
run() {  # label tree suite pass
    local label=$1 c=$2 s=$3 p=$4 tree=$J/$2 t0=$SECONDS busy=0 to=1500 smp=0 spid="" ranks=1 env=() cache rb
    local yaml; yaml=$(suite_yaml $s $p $c)
    local lib=$tree/build_Release log=$J/results/$label.log
    cache=$J/cache_${c}_$p
    case $s in
        check | c_*) to=900 ;;
        s_*) to=900; smp=1 ;;
        ew4352vc2 | vc02* | vc2ew*) to=600; env=(TT_METAL_ENABLE_FABRIC_VC2=1) ;;
        B_main_* | B_new_*) to=900; ranks=2; rb=$Y/rb_glx_std.yaml; cache=$J/cache_${c}_${p}_mvc ;;
    esac
    case $s in c_* | s_*) cache=$J/bdcache_${c}_$p ;; *_lag1) env=(TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS=200) ;;
        *_lag0) env=(TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS=200); rb=$Y/rb_glx_lag0.yaml ;; esac
    [ -f "$yaml" ] || { echo "!!! $label: no yaml $yaml: skipped"; FAILED="$FAILED $label"; return 0; }
    [ -e $log ] && { echo "!!! $label ran already: stopping"; exit 1; }
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    local cmd=(./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml)
    [ $ranks = 2 ] && cmd=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py --rank-binding $rb --mpi-args "$MPI_ARGS" "${cmd[@]}")
    echo "=== $label ($(date -u +%T)): $c, $(basename $yaml), timeout $to s ${env[*]}$([ $smp = 1 ] && echo ', sampled')$([ $ranks = 2 ] && echo ", 2 ranks ($(basename $rb))")"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree TT_METAL_RUNTIME_ROOT=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$cache TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_LOG_JIT_TELEMETRY=1 "${env[@]}" \
        timeout $to "${cmd[@]}" > $log 2>&1) &
    local pid=$!
    if [ $smp = 1 ]; then
        timeout $((SAMPLE_S + 600)) $PY $Y/pc_sampler.py $J/results/${label}_samples --start-log $log --start-delay-s 3 \
            --max-s $SAMPLE_S --wait-s 600 --eriscs 0,1 > $J/results/${label}_sampler.log 2>&1 &
        spid=$!
    fi
    while kill -0 $pid 2>/dev/null; do
        sleep 5
        if [ $((SECONDS - t0)) -ge 180 ] && ! grep -aq "Running Test" $log && grep -aq "Waiting for KMD lock" $log; then busy=1; kill_mine; break; fi
    done
    wait $pid; local rc=$?
    if [ -n "$spid" ]; then
        local t1=$SECONDS
        while kill -0 $spid 2>/dev/null && [ $((SECONDS - t1)) -lt $((SAMPLE_S + 120)) ]; do sleep 2; done
        kill -0 $spid 2>/dev/null && { kill -TERM $spid; sleep 5; kill -KILL $spid 2>/dev/null; }
        wait $spid
        echo "   sampler: exit $?; $(grep -E '^sampler: ([0-9]+ targets|[0-9]+ samples)' $J/results/${label}_sampler.log | tr '\n' ' ')"
    fi
    [ $busy = 1 ] && { echo "!!! $label: device held by another process: stopping"; exit 1; }
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && cp $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    local clean; clean=$(sed 's/\x1b\[[0-9;]*m//g' $log)
    local res; res=$(echo "$clean" | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | sort | uniq -c | tr -s ' \n' ' ')
    if [ $rc != 0 ] && [ -z "$res" ] && echo "$clean" | grep -aq "could not fit in the discovered physical topology"; then
        NO_RING=1; SKIPPED="$SKIPPED $label"
        echo "!!! $label: this node's links do not map the $s graph (nothing ran): skipping the ring/torus steps"
        return 0
    fi
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}"
    case $s in check | s_* | c_*) awk -F, 'NR>1 {printf "   %s %s B %.1f cycles/packet (1.35 GHz)\n", $1, $7, 1.35e9/$10}' $J/results/$label.csv 2>/dev/null ;; esac
    [ $ranks = 2 ] && echo "   router compile levels: $(echo "$clean" | grep -a 'g++ compile cmd' | grep -a 'fabric_erisc_router' | grep -oE ' -O[0-9s]+ ' | sort | uniq -c | tr -s ' \n' ' ')"
    echo "$clean" | grep -aE 'TT_FATAL|TT_THROW|FAILED|mismatch' | head -3
    if [ $rc = 124 ]; then echo "!!! $label timed out (hang?): killing, capturing, resetting, stopping"; kill_mine; capture $label $cache; glx_reset "after a timeout in $label"; exit 1; fi
    local want=$ranks ok; ok=$(echo "$clean" | grep -acE 'Tests: [0-9]+ executed, 0 failed')
    if [ $rc != 0 ] || [ "$ok" -lt $want ]; then
        echo "!!! $label failed: resetting, going on"; FAILED="$FAILED $label"
        glx_reset "after a failure in $label" || { echo "!!! reset failed: stopping"; exit 1; }
    fi
}
NO_RING=0; SKIPPED=""
needs_ring() { case $1 in ring | torus | nx | c_ring | s_fullring) return 0 ;; esac; return 1; }
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    if [ $s = matched ]; then
        echo "=== matched counts from pass a ($(date -u +%T))"
        $PY $Y/rf_matched.py vc02 $J/results a mainh porth 2>&1 | sed 's/^/   /'
        $PY $Y/rf_matched.py mvc3 $J/results a_mainh B_main 2>&1 | sed 's/^/   /'
        $PY $Y/rf_matched.py mvc3 $J/results a_porth B_new 2>&1 | sed 's/^/   /'
        continue
    fi
    if [ $NO_RING = 1 ] && needs_ring $s; then echo "skipped ${p}_${c}_$s (no ring/torus mapping on $NODE)"; SKIPPED="$SKIPPED ${p}_${c}_$s"; continue; fi
    run ${p}_${c}_$s $c $s $p
done
echo "done $(date -u +%T)"
# keep the breakdown caches' router builds (ELFs with the JIT's -g, generated headers) and the multi-VC routers' headers;
# drop the other caches (inodes on /data)
for d in $J/bdcache_* $J/cache_*; do
    [ -d $d ] || continue
    (cd $d && find . -path '*/kernels/fabric_erisc_router*/*' \( -name '*.elf' -o -name '*.h' -o -name '*.hpp' \) -print | tar czf $J/results/$(basename $d)_routers.tgz -T -)
done
for d in $J/cache_*; do [ -d "$d" ] && rm -rf "${d:?}"; done
[ -z "$FAILED" ] || echo "FAILED:$FAILED"
[ -z "$SKIPPED" ] || { echo "SKIPPED on $NODE:$SKIPPED"; exit 3; }
