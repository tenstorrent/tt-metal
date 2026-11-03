#!/bin/bash
# Slot study (SLOT_STUDY.md, 2026-11-01): slot rows on one Blackhole Galaxy node, the lead v4rxv2i3 at -O3 (tree
# trees/$TREE, default sls_lead: mk_sls_tree.sh), BW telemetry off, flags 13129088, keepalive 1000, slots logged.
#
#   STEPS="vc1:a:today=19,9,16,5,4 2d:a:s17=17,16,16 ..." sbatch fabric_exp/slotstudy/sls_glx.sbatch
#   dry run: G=<scratch> SLURM_JOB_ID=dry TREE=<stub tree> TSMI=<stub> MPI_BIN=<stub dir> IGNORE_BUSY=1 bash sls_glx.sbatch
#
# Steps, in order (label <kind>_<pass>_<tag>; results/<label>.log, <label>.csv (summary), <label>_tests.csv (per test)):
#   vc1:PASS:TAG=W,F0,R0,F1,R1   P2: the Galaxy split into two 2x2 meshes (fabric_exp/slotstudy/glx_2x2_split_*, one MPI
#                                rank per mesh via ttrun.py, copied from the VC2 line agent's harness), sls_glx_vc1.yaml
#                                (98 finishes), TT_FABRIC_EXP_SLOTS_VC1
#   2d:PASS:TAG=W,F,R            P3: the whole Galaxy as one 8x4 mesh (VC0 only), sls_2d_glx.yaml, TT_FABRIC_EXP_SLOTS_2D
#   1d:PASS:TAG=W,F,R            the same on 1D lines (sls_1d_glx.yaml), TT_FABRIC_EXP_SLOTS_1D
#   vc1x44:PASS:TAG=W,F0,R0,F1,R1, vc1x42:...   P4 (prepared, refused without P4_OK=1 until the command-buffer-2 fix):
#                                two 4x4 meshes (sls_glx_vc1_4x4.yaml, 2 ranks) or four 4x2 meshes (sls_glx_vc1_4x2.yaml,
#                                4 ranks), glx_4x4_split_* / glx_4x2_split_*
# After each run: the logged slots must match the request, the run must exit 0 with every test passing; otherwise stop.
# A timeout is a hang: kill this job's processes (command lines naming $J/tree only), read-only capture of the chips the
# step uses (hang_capture.py, tt-exalens from the tree's python_env), tt-smi -glx_reset, stop.
# Before the steps: the busy check (other users' device work: stop; IGNORE_BUSY=1 skips), the tree copy (cp -al), the MPI
# and Python preflight, tt-smi -glx_reset. MPI: --oversubscribe (one-task allocation) and btl self,sm (job 167436: TCP
# start-up hung on a node with overlay networks), as the VC2 line agent's vc2l_glx_mm.sbatch.
#SBATCH --partition=bh_sc36_6,bh_sc36_2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:15:00
#SBATCH --job-name=sls-glx
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
TREE=${TREE:-sls_lead}
J=$G/jobs/${SLURM_JOB_ID:-manual}
STEPS=${STEPS:-}
TSMI=${TSMI:-tt-smi}
MPI_BIN=${MPI_BIN:-/opt/openmpi-v5.0.7-ulfm/bin}
MPI_ARGS=${MPI_ARGS:---oversubscribe --mca btl self,sm}
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) tree=$TREE"
echo "steps=$STEPS"
[ -n "$STEPS" ] || { echo "!!! no STEPS"; exit 1; }

busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $(hostname): other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi

cp -al $G/trees/$TREE $J/tree || { echo "!!! copy of $TREE failed"; exit 1; }
echo "tree copied $(date -u +%T)"
export PATH=$MPI_BIN:$PATH
echo "mpirun-ulfm: $(command -v mpirun-ulfm || echo none); mpirun: $(command -v mpirun || echo none)"
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

kill_mine() {  # this job's test processes only: every process whose command line names this job's tree
    pkill -KILL -u "$USER" -f "$J/tree/" 2>/dev/null
    sleep 5
}

capture() {  # out-name label chips: read-only capture of the routers and test workers
    echo "=== hang capture $1 ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-420} $J/tree/python_env/bin/python $J/tree/fabric_exp/consolidation/hang_capture.py \
        $J/results/$1 --chips $3 --time-budget ${CAPTURE_BUDGET:-360} --meta cache=$J/cache_$2 \
        --meta label=$2 --meta node=$(hostname) > $J/results/$1.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/$1 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/$1.log)"
}

run() {  # label kind yaml timeout expected-tests [ENV=VAL...]
    local label=$1 kind=$2 yaml=$3 to=$4 want=$5; shift 5
    local tree=$J/tree lib=$J/tree/build_Release log=$J/results/$label.log t0=$SECONDS busy=0 chips
    local launch=(./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml)
    if [ $kind = vc1 ]; then
        launch=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py
            --rank-binding fabric_exp/slotstudy/glx_2x2_split_rank_bindings.yaml --mpi-args "$MPI_ARGS" "${launch[@]}")
        chips=0,1,2,3,4,5,6,7
    elif [ $kind = vc1x44 ] || [ $kind = vc1x42 ]; then
        local rb=fabric_exp/slotstudy/glx_${kind:4:1}x${kind:5:1}_split_rank_bindings.yaml
        launch=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py --rank-binding $rb --mpi-args "$MPI_ARGS" "${launch[@]}")
        chips=$(seq -s, 0 31)
    else
        chips=$(seq -s, 0 31)
    fi
    echo "=== $label ($(date -u +%T)): $(basename $yaml), timeout $to s $*"
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$J/cache_$label TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_LOG_SLOTS=1 TT_METAL_FABRIC_OPT_LEVEL=O3 \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 "$@" \
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
    local slots; slots=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'VC[01] sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c | tr -s ' \n' ' ')
    echo "$label: exit $rc in $((SECONDS - t0)) s; per rank: ${res:-NO RESULT}"
    echo "   slots: $slots"
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED' | head -5
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing this job's processes, capturing, resetting the Galaxy, stopping"
        kill_mine
        local w; for w in $(seq 1 60); do pgrep -u "$USER" -f "$J/tree/" > /dev/null || break; sleep 1; done
        [ "${CAPTURE:-1}" = 1 ] && capture ${label}_hangcap $label $chips
        glx_reset "after a timeout in $label"
        exit 1
    fi
    local ok; ok=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acE "Tests: $want executed, 0 failed")
    local nrank=1; [ $kind = vc1 ] || [ $kind = vc1x44 ] && nrank=2; [ $kind = vc1x42 ] && nrank=4
    if [ $rc != 0 ] || [ "$ok" -lt $nrank ]; then
        echo "!!! $label: exit $rc, $ok of $nrank ranks reported '$want executed, 0 failed': stopping"
        exit 1
    fi
}

check_slots() {  # label kind spec: the logged depths must be the requested ones
    local label=$1 kind=$2 spec=$3 log=$J/results/$1.log w f0 r0 f1 r1 n0 n1=1
    IFS=, read w f0 r0 f1 r1 <<< "$spec"
    n0=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acF "VC0 sender slots {$w, $f0, $f0, $f0} receiver slots {$r0}")
    if [[ $kind == vc1* ]]; then
        n1=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acF "VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}")
    elif [ $kind = 1d ]; then
        n0=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acE "VC0 sender slots \{$w, $f0\} receiver slots \{$r0\}")
    fi
    [ "$n0" -gt 0 ] && [ "$n1" -gt 0 ] || { echo "!!! $label: slots not as requested ($spec)"; exit 1; }
}

Y=$J/tree/fabric_exp/slotstudy
for step in $STEPS; do
    kind=${step%%:*}; rest=${step#*:}; pass=${rest%%:*}; rest=${rest#*:}; tag=${rest%%=*}; spec=${rest#*=}
    label=${kind}_${pass}_${tag}
    case $kind in
        vc1) run $label vc1 $Y/sls_glx_vc1.yaml ${VC1_TO:-900} 49 TT_FABRIC_EXP_SLOTS_VC1=$spec ;;
        2d) run $label 2d $Y/sls_2d_glx.yaml ${D2_TO:-1200} 22 TT_FABRIC_EXP_SLOTS_2D=$spec ;;
        1d) run $label 1d $Y/sls_1d_glx.yaml ${D1_TO:-1200} 10 TT_FABRIC_EXP_SLOTS_1D=$spec ;;
        vc1x44|vc1x42)  # P4: VC1 with meshes larger than 2x2; refused until main says the command-buffer-2 fix is in
            [ "${P4_OK:-0}" = 1 ] || { echo "!!! $step: P4 steps need P4_OK=1 (the command-buffer-2 fix; V4_BOUNDS.md §4)"; exit 1; }
            run $label $kind $Y/sls_glx_vc1_${kind:4:1}x${kind:5:1}.yaml ${VC1_TO:-900} $([ $kind = vc1x44 ] && echo 48 || echo 24) \
                TT_FABRIC_EXP_SLOTS_VC1=$spec ;;
        *) echo "!!! unknown step $step"; exit 1 ;;
    esac
    check_slots $label $kind $spec
    rm -rf $J/cache_$label
done
echo "done $(date -u +%T)"
