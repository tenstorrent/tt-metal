#!/bin/bash
# VC2 line mode bring-up, phases 3 and 4 (VC2_LINE_BRINGUP.md): one Blackhole Galaxy node split into two 2x2 meshes
# (fabric_exp/vc2line/glx_2x2_split_*: cluster-descriptor chips {0,1,4,5} | {2,3,6,7}, one MPI rank per mesh through
# ttrun.py), data-checked, the worktree's router and host (tree trees/$TREE), routers at -O3 (TT_METAL_FABRIC_OPT_LEVEL;
# the host's default for VC1 routers is -Os) unless a step says otherwise.
#
#   STEPS="mm_fanin ..." sbatch fabric_exp/vc2line/vc2l_glx_mm.sbatch          (from the worktree)
#   default STEPS: phase 4's regression (the all-VC data check in three modes, VC2 at 2 slots, repeats, perf, the window A/B,
#   the capture check), about 10 minutes; pin a node known free of other users' processes (--nodelist), e.g. bh-glx-120-c04u08
#   dry run: G=<scratch> SLURM_JOB_ID=dry TREE=<stub tree> TSMI=<stub> IGNORE_BUSY=1 bash fabric_exp/vc2line/vc2l_glx_mm.sbatch
#
# Steps (in order; each with a timeout; a timeout is treated as a hang: kill this job's processes, reset, stop):
#   mm_fanin         fabric_exp/vc2line/glx_vc2_fanin.yaml (VC2 line mode in the yaml): VC2 into VC1 across meshes
#   mm_fanin_os      the same at the host's default level (-Os)
#   mm_sanity_nbr    fabric_exp/vc1/vc1_sanity.yaml (VC0 multi-mesh all-to-all, inter-mesh neighbour exchange), VC2 on in
#                    neighbour mode (env)
#   mm_sanity_line   the same, VC2 on in line mode (env; VC2 idle)
#   mm_alloc_line    fabric_exp/vc1/vc1_alloc.yaml (VC0/VC1 traffic, data-checked), VC2 line mode (env; VC2 idle)
#   mm_vc2_line      test_fabric_vc2_at_least_2x2_mesh.yaml (VC2 within each mesh, with VC0 in some tests), line mode
#   allvc            phase 4: fabric_exp/vc2line/glx_allvc.yaml, VC0, VC1 and VC2 all carrying traffic at once
#   allvc_os         the same at the host's default level (-Os)
#   allvc_small      the same with VC2's windows at 2 slots (TT_FABRIC_EXP_SLOTS_VC1=$SMALL_VC2: VC2 receiver 2)
#   perf             fabric_exp/vc2line/glx_allvc_perf.yaml (benchmark mode; the CSVs are kept as results/<step>*.csv)
#   perfns           fabric_exp/vc2line/glx_allvc_perf_nosync.yaml (benchmark, sync off: per-flow rates of several flows)
#   perfab           fabric_exp/vc2line/glx_allvc_perf_ab.yaml (VC2 flows alone, benchmark) at the default row;
#   perfab_vc1x2     the same with VC1 at 8/8 (TT_FABRIC_EXP_SLOTS_VC1=$AB_VC1: slots from VC0's worker and forwarding)
#   perfab_vc2x2     the same with VC2 at 8/8/8 (TT_FABRIC_EXP_SLOTS_VC1=$AB_VC2: slots from VC0's worker)
#   split_idle_off   perf: fabric_exp/vc2line/glx_idle_split.yaml (VC0 rows, VC1 compiled in and idle), VC2 off, VC0 and VC1
#                    pinned at $IDLE_ROW (TT_FABRIC_EXP_SLOTS_VC1), slots logged
#   split_idle_nbr   the same with VC2 on in neighbour mode, idle ($IDLE_ROW,4,4)
#   split_idle_line  the same with VC2 on in line mode, idle ($IDLE_ROW,4,4,4)
#   full             perf: fabric_exp/vc2line/glx_full_split.yaml (each VC alone and together on one link, the intermesh
#                    link, one chip's links), the default row
#   full_vc1x2       the same with VC1 at 8/8 ($FULL_VC1: VC0's worker kept at 16; its forwarding 2, receiver 15)
#   full_vc2x2       the same with VC2 at 8/8/8 ($FULL_VC2: VC0's worker kept at 16; its forwarding 3)
#   capself          a hang capture of the idle device after the last step (checks the capture's mechanics here)
#   <step>_1e        the same on one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1); <step>_r<N>: a repeat
# Every step logs the slot rows (TT_FABRIC_EXP_LOG_SLOTS=1). Hang capture is armed: on a timeout, this job's processes
# are killed, then fabric_exp/consolidation/hang_capture.py (read-only, tt-exalens from the tree's python_env) writes
# results/<step>_hangcap/ for chips 0-7 before the reset (CAPTURE=0 disables it).
# Before the steps: the busy check (other users' device work: stop; IGNORE_BUSY=1 skips), the tree copy (cp -al), a
# preflight of the MPI launcher and ttrun.py's Python, then `tt-smi -glx_reset` on this node. Both ranks run on this
# node in a one-task allocation, so mpirun gets --oversubscribe (job 167319: without it no rank started), and the ranks
# talk through shared memory (btl self,sm; MPI_ARGS overrides): with ttrun's default (TCP, excluding lo and docker0) both
# ranks of job 167436 (bh-glx-110-a05u14, a node with a flannel/calico overlay and many bridges) waited in MPI_Comm_dup at
# start-up until cancelled, before any device work.
#SBATCH --partition=bh_sc36_6,bh_sc36_2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=vc2l-glx-mm
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
TREE=${TREE:-vc2l_p4c}
J=$G/jobs/${SLURM_JOB_ID:-manual}
STEPS=${STEPS:-allvc allvc_1e allvc_os allvc_small allvc_r2 allvc_r3 perf perfns perfab perfab_vc1x2 perfab_vc2x2 capself}
# The small VC2 window row (VC0+VC1+VC2 line): VC0 worker (the rest), VC0 forwarding, VC0 receiver, VC1 forwarding, VC1
# receiver, VC2 worker, VC2 receiver, VC2 forwarding.
SMALL_VC2=${SMALL_VC2:-0,7,16,4,4,4,2,4}
AB_VC1=${AB_VC1:-6,5,16,8,8,4,4,4}
AB_VC2=${AB_VC2:-4,7,16,4,4,8,8,8}
IDLE_ROW=${IDLE_ROW:-16,7,16,4,4}
FULL_VC1=${FULL_VC1:-16,2,15,8,8,4,4,4}
FULL_VC2=${FULL_VC2:-16,3,16,4,4,8,8,8}
TSMI=${TSMI:-tt-smi}
MPI_BIN=${MPI_BIN:-/opt/openmpi-v5.0.7-ulfm/bin}
MPI_ARGS=${MPI_ARGS:---oversubscribe --mca btl self,sm}
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) tree=$TREE steps=$STEPS"

busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $(hostname): other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi

cp -al $G/trees/$TREE $J/tree || { echo "!!! copy of $TREE failed"; exit 1; }
echo "tree copied $(date -u +%T)"

# Preflight: ttrun.py runs on the tree's Python environment and launches mpirun-ulfm (or mpirun) from PATH.
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

kill_mine() {  # this job's test processes only (a timeout can leave mpirun's ranks behind): every process whose command
    # line names this job's tree (ttrun.py, mpirun and the ranks all carry the test yaml's path under $J/tree)
    pkill -KILL -u "$USER" -f "$J/tree/" 2>/dev/null
    sleep 5
}

run() {  # label yaml timeout [ENV=VAL...]
    local label=$1 yaml=$2 to=$3; shift 3
    local tree=$J/tree lib=$J/tree/build_Release log=$J/results/$label.log t0=$SECONDS busy=0
    echo "=== $label ($(date -u +%T)): $(basename $yaml), timeout $to s $*"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$J/cache_$label TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_LOG_SLOTS=1 "$@" \
        timeout $to ./python_env/bin/python ttnn/ttnn/distributed/ttrun.py \
        --rank-binding fabric_exp/vc2line/glx_2x2_split_rank_bindings.yaml --mpi-args "$MPI_ARGS" \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml > $log 2>&1) &
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
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | sort | uniq -c | tr '\n' ';')
    echo "$label: exit $rc in $((SECONDS - t0)) s; per rank: ${res:-NO RESULT}"
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED|Watcher|watcher' | head -5
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && mv $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing this job's processes, capturing, resetting the Galaxy, stopping"
        kill_mine
        local w; for w in $(seq 1 60); do pgrep -u "$USER" -f "$J/tree/" > /dev/null || break; sleep 1; done
        [ "${CAPTURE:-1}" = 1 ] && capture ${label}_hangcap $label
        glx_reset "after a timeout in $label"
        exit 1
    fi
    return $rc
}

capture() {  # out-name label: read-only capture of the routers and test workers of chips 0-7 (the split's)
    echo "=== hang capture $1 ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-420} $J/tree/python_env/bin/python $J/tree/fabric_exp/consolidation/hang_capture.py \
        $J/results/$1 --chips 0,1,2,3,4,5,6,7 --time-budget ${CAPTURE_BUDGET:-360} --meta cache=$J/cache_$2 \
        --meta label=$2 --meta node=$(hostname) > $J/results/$1.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/$1 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/$1.log)"
}

S=$J/tree/tests/tt_metal/tt_fabric/test_infra/test_yamls
V=$J/tree/fabric_exp/vc2line
V1=$J/tree/fabric_exp/vc1
O3="TT_METAL_FABRIC_OPT_LEVEL=O3"
VC2LINE="TT_METAL_ENABLE_FABRIC_VC2=1 TT_METAL_FABRIC_VC2_LINE=1"
VC2NBR="TT_METAL_ENABLE_FABRIC_VC2=1"
ONE="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
last=""
for step in $STEPS; do
    base=${step%_r[0-9]}; one=${base%_1e}; extra=""
    [ "$one" != "$base" ] && extra=$ONE
    base=$one
    case $base in
        mm_fanin) run $step $V/glx_vc2_fanin.yaml 900 $O3 $extra ;;
        mm_fanin_os) run $step $V/glx_vc2_fanin.yaml 900 $extra ;;
        mm_sanity_nbr) run $step $V1/vc1_sanity.yaml 1200 $O3 $VC2NBR $extra ;;
        mm_sanity_line) run $step $V1/vc1_sanity.yaml 1200 $O3 $VC2LINE $extra ;;
        mm_alloc_line) run $step $V1/vc1_alloc.yaml 1200 $O3 $VC2LINE $extra ;;
        mm_vc2_line) run $step $S/test_fabric_vc2_at_least_2x2_mesh.yaml 900 $O3 $VC2LINE $extra ;;
        allvc) run $step $V/glx_allvc.yaml 600 $O3 $extra ;;
        allvc_os) run $step $V/glx_allvc.yaml 600 $extra ;;
        allvc_small) run $step $V/glx_allvc.yaml 600 $O3 TT_FABRIC_EXP_SLOTS_VC1=$SMALL_VC2 $extra ;;
        perf) run $step $V/glx_allvc_perf.yaml 1200 $O3 $extra ;;
        perfab) run $step $V/glx_allvc_perf_ab.yaml 600 $O3 $extra ;;
        perfns) run $step $V/glx_allvc_perf_nosync.yaml 1200 $O3 $extra ;;
        split_idle_off) run $step $V/glx_idle_split.yaml 900 $O3 TT_FABRIC_EXP_SLOTS_VC1=$IDLE_ROW $extra ;;
        split_idle_nbr) run $step $V/glx_idle_split.yaml 900 $O3 $VC2NBR TT_FABRIC_EXP_SLOTS_VC1=$IDLE_ROW,4,4 $extra ;;
        split_idle_line) run $step $V/glx_idle_split.yaml 900 $O3 $VC2LINE TT_FABRIC_EXP_SLOTS_VC1=$IDLE_ROW,4,4,4 $extra ;;
        full) run $step $V/glx_full_split.yaml 1200 $O3 $extra ;;
        full_vc1x2) run $step $V/glx_full_split.yaml 1200 $O3 TT_FABRIC_EXP_SLOTS_VC1=$FULL_VC1 $extra ;;
        full_vc2x2) run $step $V/glx_full_split.yaml 1200 $O3 TT_FABRIC_EXP_SLOTS_VC1=$FULL_VC2 $extra ;;
        perfab_vc1x2) run $step $V/glx_allvc_perf_ab.yaml 600 $O3 TT_FABRIC_EXP_SLOTS_VC1=$AB_VC1 $extra ;;
        perfab_vc2x2) run $step $V/glx_allvc_perf_ab.yaml 600 $O3 TT_FABRIC_EXP_SLOTS_VC1=$AB_VC2 $extra ;;
        capself) [ -n "$last" ] && capture capself_after_$last $last ;;
        *) echo "!!! unknown step $step" ;;
    esac
    last=$step
done
echo "done $(date -u +%T)"
