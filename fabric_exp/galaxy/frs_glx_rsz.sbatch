#!/bin/bash
# Report runs at 200G on one Blackhole Galaxy node (2026-11-02): main's router against the new router in the same job,
# (1) bandwidth over packet sizes and (2) PC samples of the cycle breakdown's Galaxy points (report/BREAKDOWN_PLAN.md
# §5.2, "Galaxy samples: what to save").
#
#   PASS=a sbatch fabric_exp/galaxy/frs_glx_rsz.sbatch       bandwidth pass a (main, mainka, $NEWCFG per suite + validation)
#   PASS=b sbatch fabric_exp/galaxy/frs_glx_rsz.sbatch       bandwidth pass b ($NEWCFG, mainka, main per suite)
#   STEPS="..." sbatch fabric_exp/galaxy/frs_glx_rsz.sbatch  any steps (the breakdown samples: fabric_exp/galaxy/pcs/STEPS)
#   dry run (stubs, no Slurm, no device):
#     G=<scratch> SLURM_JOB_ID=dry TSMI=<stub> DRY_NODE_OK=1 IGNORE_BUSY=1 STEPS=... bash frs_glx_rsz.sbatch
#
# Trees (fabric_exp/galaxy/mk_glx_base.sh, then BASE=... mk_glx_tree.sh): $BASE (default base_1102, the live study
# tree of 2026-11-02 with no variant in it) and $NEW (default v4rxg3_1102 = $BASE + fabric_exp/variants/v4rxg3).
# NEWCFG (default v4rxg3) names the new router's config: its steps, labels, cache and source snapshot, with the settings
# below (2026-11-03, the refresh on the final router: NEW=v4txa11_1103 NEWCFG=v4txa11).
# Configs (every run: TT_FABRIC_NO_BW_TELEMETRY=1, -O3, default slots (every TT_FABRIC_* / TT_METAL_* variable of the
# submitting shell is unset first), TT_METAL_CLEAR_L1=1, compile commands logged):
#   main    tree $BASE, main's router as shipped: TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0, firmware keepalive
#   mainka  the same with the study's keepalive (TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000): the keepalive's share alone
#   $NEWCFG tree $NEW, kernel v2 slot, TT_FABRIC_EXP_FLAGS=13129088, keepalive 1000 (the P150 run_cap settings)
# STEPS: PASS:CONFIG:SUITE, run in order; label PASS_CONFIG_SUITE (.log, .csv = summary, _tests.csv = per sender).
#   check = fabric_exp/galaxy/glx_check_2d.yaml: one short data-checked 2D row, first for each tree a job uses (fail fast)
#   bandwidth suites: mesh / linear / ring = fabric_exp/galaxy/glx_sizes_{mesh,linear,ring}.yaml (glx_{suite}.yaml at
#     256 / 1024 / 2048 / 3072 / 4352 B), torus = glx_sizes_torus.yaml (XY torus, mock-vetted), val = the data-checked
#     mesh sanity suite (96 tests, ~3 min)
#   breakdown rows (fabric_exp/galaxy/pcs/): c_mesh, c_ring = compile rows (short); s_ew1024, s_ns7, s_fullring =
#     points 2, 4, 5 with pc_sampler.py beside the test (both ERISCs, every chip); u_ew1024 etc. = the same row unsampled
#     (the sampler's perturbation check). Samples: results/LABEL_samples.{bin,json}, sampler log LABEL_sampler.log.
# Each config has its own kernel cache $J/cache_CONFIG, kept (the PC mapping reads its ELFs). Source snapshots of each
# tree: results/src_{main,$NEWCFG}.tgz. A run past its timeout is a hang: this job's processes are killed, the routers
# captured read-only (hang_capture.py, CAPTURE=0 skips), the Galaxy reset, and the job stops. A failed run stops it too.
# Rules: only bh-glx-110-a* nodes (partition bh_sc36_6; the job stops at once anywhere else); other users' device work
# on the node stops the job before anything (IGNORE_BUSY=1 skips); tt-smi -glx_reset before the steps.
#SBATCH --partition=bh_sc36_6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=frs-glx-rsz
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
BASE=${BASE:-base_1102}
NEW=${NEW:-v4rxg3_1102}
NEWCFG=${NEWCFG:-v4rxg3}
[[ $NEWCFG =~ ^v4[A-Za-z0-9]+$ ]] || { echo "!!! NEWCFG '$NEWCFG': want v4<name> (letters and digits)"; exit 1; }
TSMI=${TSMI:-tt-smi}
SAMPLE_S=${SAMPLE_S:-25}
SAMPLE_DELAY_S=${SAMPLE_DELAY_S:-3}
case ${PASS:-a} in
    a) BW="a:main:check a:$NEWCFG:check a:main:val a:$NEWCFG:val"
       for s in mesh linear ring torus; do BW="$BW a:main:$s a:mainka:$s a:$NEWCFG:$s"; done ;;
    b) BW="b:$NEWCFG:check b:main:check"
       for s in mesh linear ring torus; do BW="$BW b:$NEWCFG:$s b:mainka:$s b:main:$s"; done ;;
    *) BW="" ;;
esac
STEPS=${STEPS:-$BW}
[ -n "$STEPS" ] || { echo "!!! no steps (PASS=a|b or STEPS=...)"; exit 1; }
NODE=${SLURMD_NODENAME:-$(hostname)}
echo "node=$NODE job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) base=$BASE new=$NEW newcfg=$NEWCFG"
echo "steps=$STEPS"
# User rule (2026-11-02): Galaxy jobs only on the 110-a machines (partition bh_sc36_6, nodes bh-glx-110-a*).
case "$NODE" in
    bh-glx-110-a*) ;;
    *) [ -z "${DRY_NODE_OK:-}" ] && { echo "!!! node $NODE is not a bh-glx-110-a* machine: stopping"; exit 1; } ;;
esac
# Other users' device work (the Galaxies also run outside Slurm: jobs 155971, 157726, 157804): stop before anything.
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $NODE: other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi
suite_yaml() {  # suite tree-dir
    local Y=$2/fabric_exp/galaxy
    case $1 in
        mesh | linear | ring | torus) echo $Y/glx_sizes_$1.yaml ;;
        check) echo $Y/glx_check_2d.yaml ;;
        val) echo $2/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_mesh_sanity_common.yaml ;;
        c_mesh) echo $Y/pcs/glx_compile_mesh.yaml ;;
        c_ring) echo $Y/pcs/glx_compile_ring.yaml ;;
        [su]_ew1024) echo $Y/pcs/glx_ew1024.yaml ;;
        [su]_ns7) echo $Y/pcs/glx_ns7_1024.yaml ;;
        [su]_fullring) echo $Y/pcs/glx_fullring4352.yaml ;;
        *) echo "" ;;
    esac
}
tree_of() { case $1 in main | mainka) echo $G/trees/$BASE ;; "$NEWCFG") echo $G/trees/$NEW ;; esac; }
# Preflight on the source trees (before the copies and the reset): every step's yaml and the tools it needs.
# PREFLIGHT_ONLY=1 stops here (a dry run against the real /data trees: no copy, no device, nothing written).
miss=0
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    t=$(tree_of $c); y=$([ -n "$t" ] && suite_yaml $s $t)
    if [ -z "$p" ] || [ -z "$t" ] || [ -z "$y" ] || [ ! -f "$y" ]; then echo "!!! step $st: no tree/yaml (${y:-?})"; miss=1; fi
    for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric python_env/bin/python \
             fabric_exp/consolidation/pc_sampler.py fabric_exp/consolidation/hang_capture.py; do
        [ -n "$t" ] && [ ! -e $t/$f ] && { echo "!!! step $st: no $t/$f"; miss=1; }
    done
done
[ $miss = 0 ] || { echo "!!! preflight failed: stopping before the copies"; exit 1; }
echo "preflight: $(echo $STEPS | wc -w) steps, every yaml and tool present"
[ -n "${PREFLIGHT_ONLY:-}" ] && exit 0
mkdir -p $J/results
# The submitting shell's environment reaches the job (sbatch --export=ALL): log, then drop every TT_* setting, so slots,
# keepalive, flags and paths are exactly the ones set per run below.
env | grep -E '^TT_' | sort > $J/results/submit_env.txt
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
unset PYTHONPATH VIRTUAL_ENV

# Trees: copied (hard links) so generated/ is this job's; only the trees the steps use.
need_main=0; need_new=0; sampled=0
for st in $STEPS; do
    case $st in *:main:* | *:mainka:*) need_main=1 ;; *:"$NEWCFG":*) need_new=1 ;; esac
    case $st in *:s_*) sampled=1 ;; esac
done
copy_tree() {  # tree-name dir router-name
    [ -d $G/trees/$1 ] || { echo "!!! no tree $G/trees/$1"; exit 1; }
    cp -al $G/trees/$1 $J/$2 || { echo "!!! copy of $1 failed"; exit 1; }
    tar czf $J/results/src_$3.tgz -C $J/$2 tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric \
        tt_metal/hw/inc tt_metal/hostdevcommon
}
[ $need_main = 1 ] && copy_tree $BASE base main
[ $need_new = 1 ] && copy_tree $NEW new $NEWCFG
echo "trees copied $(date -u +%T) (base: $need_main, new: $need_new)"
T=$J/base; [ -d $T ] || T=$J/new  # the tree the tools (python_env, pc_sampler.py, hang_capture.py) come from
PY=$T/python_env/bin/python
if [ $sampled = 1 ] && ! $PY -c "import ttexalens, yaml" > $J/results/preflight.log 2>&1; then
    echo "!!! the tree's python_env (tt-exalens) does not run here:"; cat $J/results/preflight.log; exit 1
fi

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
kill_mine() { pkill -KILL -u "$USER" -f "$J/(base|new)/" 2>/dev/null; pkill -KILL -u "$USER" -f "pc_sampler.py $J/" 2>/dev/null; sleep 5; }
capture() {  # label cache
    [ "${CAPTURE:-1}" = 1 ] || return 0
    echo "=== hang capture ${1}_hangcap ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-600} $PY $T/fabric_exp/consolidation/hang_capture.py $J/results/${1}_hangcap \
        --time-budget ${CAPTURE_BUDGET:-540} --meta cache=$2 --meta label=$1 --meta node=$NODE \
        > $J/results/${1}_hangcap.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/${1}_hangcap 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/${1}_hangcap.log)"
}


run() {  # label config suite
    local label=$1 c=$2 s=$3 kv flags tree to=${BW_TIMEOUT:-1500} t0=$SECONDS busy=0 smp=0 spid=""
    local ka=(-u TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT)
    case $c in
        main) tree=$J/base; kv=1; flags=0 ;;
        mainka) tree=$J/base; kv=1; flags=0; ka=(TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000) ;;
        "$NEWCFG") tree=$J/new; kv=2; flags=13129088; ka=(TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000) ;;
        *) echo "!!! unknown config $c: stopping"; exit 1 ;;
    esac
    local yaml; yaml=$(suite_yaml $s $tree)
    [ -n "$yaml" ] && [ -f "$yaml" ] || { echo "!!! no yaml for suite $s: stopping"; exit 1; }
    case $s in s_*) smp=1; to=${ROW_TIMEOUT:-900} ;; c_* | u_* | check) to=${ROW_TIMEOUT:-900} ;; esac
    local lib=$tree/build_Release log=$J/results/$label.log cache=$J/cache_$c
    [ -e $log ] && { echo "!!! $label ran already: stopping"; exit 1; }
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    echo "=== $label ($(date -u +%T)): tree ${tree##*/}, kernel v$kv, flags $flags, ${ka[*]}, $(basename $yaml), timeout $to s$([ $smp = 1 ] && echo ', sampled')"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE "${ka[@]}" TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$cache TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_METAL_FABRIC_OPT_LEVEL=O3 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        TT_FABRIC_KERNEL_VERSION=$kv TT_FABRIC_EXP_FLAGS=$flags \
        timeout $to ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml > $log 2>&1) &
    local pid=$!
    if [ $smp = 1 ]; then  # the sampler opens the device after the test logs "Running Test:" (read-only debug-bus reads)
        timeout $((SAMPLE_S + 600)) $PY $T/fabric_exp/consolidation/pc_sampler.py $J/results/${label}_samples \
            --start-log $log --start-delay-s $SAMPLE_DELAY_S --max-s $SAMPLE_S --wait-s 600 --eriscs 0,1 \
            > $J/results/${label}_sampler.log 2>&1 &
        spid=$!
    fi
    # A device another process holds: UMD waits for the KMD lock and no test starts. Stop after BUSY_S of that.
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
    if [ -n "$spid" ]; then  # the sampler stops at the test's "Tests: " line or after SAMPLE_S; never wait long for it
        local t1=$SECONDS
        while kill -0 $spid 2>/dev/null && [ $((SECONDS - t1)) -lt $((SAMPLE_S + 120)) ]; do sleep 2; done
        kill -0 $spid 2>/dev/null && { echo "   sampler still running $((SAMPLE_S + 120)) s after the test: killing it"; kill -TERM $spid; sleep 5; kill -KILL $spid 2>/dev/null; }
        wait $spid
        echo "   sampler: exit $?; $(grep -E '^sampler: (device open|[0-9]+ targets|[0-9]+ samples|start)' $J/results/${label}_sampler.log | tr '\n' ' ')"
    fi
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
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    if [ $rc != 0 ] && [ -z "$res" ] && grep -aq "could not fit in the discovered physical topology" $log; then
        # The host could not map the ring / torus mesh graph onto this Galaxy's links (job 169343 on a02u14: the
        # 4x8 TORUSXY grouping): nothing ran on the device. Skip the remaining ring / torus steps, keep going.
        NO_RING=1
        echo "!!! $label: this node's links do not map the $s suite's ring/torus graph (nothing ran): skipping the ring/torus steps"
        sed 's/\x1b\[[0-9;]*m//g' $log | grep -a "TT_FATAL" | head -1 | cut -c1-300
        SKIPPED="$SKIPPED $label"
        return 0
    fi
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}"
    case $s in check | c_* | [su]_*) awk -F, 'NR>1 {printf "   %s %s B %.1f cycles/packet (1.35 GHz)\n", $1, $7, 1.35e9/$10}' $J/results/$label.csv 2>/dev/null ;; esac
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED|mismatch' | head -3
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing this job's processes, capturing, resetting, stopping"
        kill_mine; capture $label $cache; glx_reset "after a timeout in $label"; exit 1
    fi
    if [ $rc != 0 ] || ! echo "$res" | grep -q ", 0 failed"; then
        echo "!!! $label failed: stopping"; glx_reset "after a failure in $label"; exit 1
    fi
    if [ $smp = 1 ] && ! grep -qE '^sampler: [1-9][0-9]* samples in .* 0 read errors' $J/results/${label}_sampler.log; then
        echo "!!! $label: the sampler did not finish cleanly (see ${label}_sampler.log): stopping"
        tail -n 5 $J/results/${label}_sampler.log
        exit 1
    fi
}

NO_RING=0; SKIPPED=""
needs_ring() { case $1 in ring | torus | c_ring | [su]_fullring) return 0 ;; esac; return 1; }
for st in $STEPS; do
    IFS=: read -r p c s <<< "$st"
    [ -n "$p" ] && [ -n "$c" ] && [ -n "$s" ] || { echo "!!! bad step $st: stopping"; exit 1; }
    if [ $NO_RING = 1 ] && needs_ring $s; then echo "skipped ${p}_${c}_$s (no ring/torus mapping on $NODE)"; SKIPPED="$SKIPPED ${p}_${c}_$s"; continue; fi
    run ${p}_${c}_$s $c $s
done
echo "done $(date -u +%T)"
[ -z "$SKIPPED" ] || { echo "SKIPPED on $NODE (no ring/torus mapping):$SKIPPED"; exit 3; }
