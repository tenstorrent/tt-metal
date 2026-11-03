#!/bin/bash
# N/S fix 1 on the Galaxy (V4_NS_PRESETS.md; 2026-11-03): V4_NS_FLOOR §1's 2D rows (the whole Galaxy as one 8x4 mesh,
# VC0 only) per router, at 2 KiB slots (sls_2d_glx_2k.yaml, 22 tests) and at 4352 B slots (sls_2d_glx_4k.yaml, 33 tests),
# the builder's default slots (logged). A trimmed copy of fabric_exp/slotstudy/sls_glx2k.sbatch (same run / reset /
# hang handling): kinds 2d2k and 2dk only, yamls from $G/yamls/nsfix, the bh-glx-110-a* node guard, and a main_fw arm.
#
#   STEPS="2d2k@nsf_nsa:a:base=d 2dk:a:base=d 2d2k@base_1102:a:main=d ..." TREE=nsf_g3 XTREES="nsf_nsa base_1102" \
#       sbatch --partition=bh_sc36_6 --exclude=bh-glx-110-a02u14 fabric_exp/consolidation/ns_glx.sbatch
#   dry run: G=<scratch> SLURM_JOB_ID=dry TREE=<stub> XTREES=<stubs> TSMI=<stub> NODE_OK=1 IGNORE_BUSY=1 bash ns_glx.sbatch
#
# Steps run in order; label <kind>_<pass>_<tag> (results/<label>.log, <label>.csv, <label>_tests.csv). KIND@NAME runs the
# step in tree trees/NAME (listed in XTREES, copied to $J/tree_NAME) instead of $TREE. A tag starting "main" runs main's
# router as shipped (TT_FABRIC_KERNEL_VERSION=1, flags 0, firmware keepalive, the builder's opt level: the P150 census's
# main_fw); give it the base tree, whose fabric headers are the study tree's own (no variant swapped in). Every other tag
# runs the study router (kernel 2, flags 13129088, keepalive 1000, -O3), BW telemetry off for all.
# After each run: exit 0 with every test passing, else stop. A timeout is a hang: kill this job's processes (command lines
# naming $J/tree only), read-only capture (hang_capture.py), tt-smi -glx_reset, stop.
#SBATCH --partition=bh_sc36_6
#SBATCH --exclude=bh-glx-110-a02u14
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=nsf-glx
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
TREE=${TREE:-nsf_g3}
J=$G/jobs/${SLURM_JOB_ID:-manual}
STEPS=${STEPS:-}
TSMI=${TSMI:-tt-smi}
Y=${Y:-$G/yamls/nsfix}
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) tree=$TREE xtrees=${XTREES:-}"
echo "steps=$STEPS"
[ -n "$STEPS" ] || { echo "!!! no STEPS"; exit 1; }
case $(hostname) in
    bh-glx-110-a*) ;;
    *) [ -n "${NODE_OK:-}" ] || { echo "!!! $(hostname) is not a bh-glx-110-a* node: stopping"; exit 1; } ;;
esac

busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $(hostname): other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi

cp -al $G/trees/$TREE $J/tree || { echo "!!! copy of $TREE failed"; exit 1; }
for t in ${XTREES:-}; do cp -al $G/trees/$t $J/tree_$t || { echo "!!! copy of $t failed"; exit 1; }; done
echo "tree copied ($TREE ${XTREES:-}) $(date -u +%T)"
if ! $J/tree/python_env/bin/python -c "import click, yaml, loguru, pydantic" 2>$J/results/preflight.log; then
    echo "!!! the tree's python_env does not run here:"; cat $J/results/preflight.log; exit 1
fi

glx_reset() {  # reason
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T): $($TSMI --version 2>&1 | head -1)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$?
    if [ $rc != 0 ]; then
        echo "glx_reset: exit $rc ($(tail -n 2 $rlog | tr '\n' ' ')); trying tt-smi -r"
        timeout 900 $TSMI -r >> $rlog 2>&1
        rc=$?
    fi
    local chips; chips=$($TSMI -ls 2>/dev/null | grep -c "Blackhole")
    echo "reset: exit $rc, Blackhole chips listed: $chips"
    [ $rc = 0 ] && [ "$chips" -ge 32 ]
}
glx_reset "before the steps" || { echo "!!! reset failed: stopping"; exit 1; }

kill_mine() {  # this job's test processes only: every process whose command line names one of this job's trees
    pkill -KILL -u "$USER" -f "$J/tree" 2>/dev/null
    sleep 5
}

capture() {  # out-name label chips: read-only capture of the routers and test workers
    echo "=== hang capture $1 ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-420} $J/tree/python_env/bin/python $J/tree/fabric_exp/consolidation/hang_capture.py \
        $J/results/$1 --chips $3 --time-budget ${CAPTURE_BUDGET:-360} --meta cache=$J/cache_$2 \
        --meta label=$2 --meta node=$(hostname) > $J/results/$1.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/$1 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/$1.log)"
}

run() {  # label yaml timeout expected-tests router-env...
    local label=$1 yaml=$2 to=$3 want=$4; shift 4
    local tree=$J/${TD:-tree} lib=$J/${TD:-tree}/build_Release log=$J/results/$label.log t0=$SECONDS busy=0
    local launch=(./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml)
    local chips; chips=$(seq -s, 0 31)
    echo "=== $label ($(date -u +%T)): $(basename $yaml), tree $(basename $tree), timeout $to s, $*"
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$J/cache_$label TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_EXP_LOG_SLOTS=1 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 "$@" \
        timeout $to "${launch[@]}" > $log 2>&1) &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        sleep 10
        if [ $((SECONDS - t0)) -ge ${BUSY_S:-180} ] && ! grep -aq "Running Test" $log && grep -aq "Waiting for KMD lock" $log; then
            busy=1
            pkill -TERM -P $pid
            sleep 15
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
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && mv $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | sort | uniq -c | tr '\n' ';')
    local slots; slots=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'VC0 sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c | tr -s ' \n' ' ')
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}"
    echo "   slots: $slots"
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED' | head -5
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing this job's processes, capturing, resetting the Galaxy, stopping"
        kill_mine
        local w; for w in $(seq 1 60); do pgrep -u "$USER" -f "$J/tree" > /dev/null || break; sleep 1; done
        [ "${CAPTURE:-1}" = 1 ] && capture ${label}_hangcap $label $chips
        glx_reset "after a timeout in $label"
        exit 1
    fi
    if [ $rc != 0 ] || ! sed 's/\x1b\[[0-9;]*m//g' $log | grep -aqE "Tests: $want executed, 0 failed"; then
        echo "!!! $label: exit $rc, not '$want executed, 0 failed': stopping"
        exit 1
    fi
}

STUDY_ENV="TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_FABRIC_OPT_LEVEL=O3"
MAIN_ENV="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
for step in $STEPS; do
    kind=${step%%:*}; rest=${step#*:}; pass=${rest%%:*}; rest=${rest#*:}; tag=${rest%%=*}
    TD=tree
    if [[ $kind == *@* ]]; then TD=tree_${kind#*@}; kind=${kind%@*}; [ -d $J/$TD ] || { echo "!!! $step: tree $TD not copied (XTREES)"; exit 1; }; fi
    label=${kind}_${pass}_${tag}
    [ -e $J/results/$label.log ] && { echo "!!! $step: label $label already used (give each tree its own tag)"; exit 1; }
    renv=$STUDY_ENV; [[ $tag == main* ]] && renv=$MAIN_ENV
    case $kind in
        2d2k) run $label $Y/sls_2d_glx_2k.yaml ${D2_TO:-1200} 22 $renv ;;
        2dk) run $label $Y/sls_2d_glx_4k.yaml ${D2_TO:-1200} 33 $renv ;;
        *) echo "!!! unknown step $step"; exit 1 ;;
    esac
    rm -rf $J/cache_$label
done
echo "done $(date -u +%T)"
