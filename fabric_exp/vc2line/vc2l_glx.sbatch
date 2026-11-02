#!/bin/bash
# VC2 line mode bring-up on one Blackhole Galaxy node (VC2_LINE_BRINGUP.md): data-checked runs of the worktree's router
# and host (tree trees/$TREE, a lean copy of /home/aagarwal/tt-metal-vc2line; the router in v2's place).
#
#   STEPS="nbr_vc2 nbr_sanity line ..." sbatch fabric_exp/vc2line/vc2l_glx.sbatch       (from the worktree)
#   dry run: G=<scratch> SLURM_JOB_ID=dry TREE=<stub tree> bash fabric_exp/vc2line/vc2l_glx.sbatch
#
# Steps (run in order; each with a timeout; a timeout is treated as a hang: reset the Galaxy and stop):
#   nbr_vc2        VC2 neighbour mode: test_fabric_vc2_at_least_2x2_mesh.yaml (TT_METAL_ENABLE_FABRIC_VC2 via the yaml)
#   nbr_sanity     VC0 only: test_fabric_mesh_sanity_common.yaml
#   line           VC2 line mode: fabric_exp/vc2line/glx_vc2_line.yaml (vc2_line in the yaml)
#   line_vc2       the VC2 neighbour yaml with every router in line mode (TT_METAL_FABRIC_VC2_LINE=1): 1-hop lines
#   line_sanity    the mesh sanity yaml (VC0) with VC2 on in line mode (env), VC2 idle
#   ring           VC2 rings on tori (phase 2): fabric_exp/vc2line/glx_vc2_ring.yaml (X-only, Y-only, XY tori)
#   ring_small     the same with a 2-slot VC2 receiver ring (TT_FABRIC_EXP_SLOTS_VC2=$SMALL_RING, slots logged)
#   tds_nbr        the torus deadlock-stability suite (test_fabric_2d_torus_deadlock_stability.yaml), VC2 neighbour mode
#   tds_line       the same, VC2 line mode (rings on torus axes; VC2 idle, VC0's traffic)
#   tst_nbr        the short torus stability suite (fabric_exp/vc2line/glx_torus_stability_short.yaml), VC2 neighbour
#   tst_line       the same, VC2 line mode
#   idle_off       perf: fabric_exp/vc2line/glx_idle_mesh.yaml (VC0 rows, benchmark, no sync), VC2 off, VC0 pinned at
#                  $IDLE_VC0 (TT_FABRIC_EXP_SLOTS), -O3, slots logged
#   idle_nbr       the same with VC2 on in neighbour mode, idle (TT_FABRIC_EXP_SLOTS_VC2=$IDLE_VC0,4,4)
#   idle_line      the same with VC2 on in line mode, idle (TT_FABRIC_EXP_SLOTS_VC2=$IDLE_VC0,4,4,4)
#   <step>_1e      the same on one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1); <step>_r<N>: a repeat
# A timeout is treated as a hang: this job's processes are killed (command lines naming its tree), then
# fabric_exp/consolidation/hang_capture.py (read-only, tt-exalens from the tree's python_env) captures every chip into
# results/<step>_hangcap/ before the reset (CAPTURE=0 disables it).
# Before the steps: a check for other users' device work (stop at once if any; IGNORE_BUSY=1 skips it), the tree copy
# (cp -al, so generated/ is this job's), then `tt-smi -glx_reset` on this node.
#SBATCH --partition=bh_sc36_6,bh_sc36_2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=vc2l-glx
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
TREE=${TREE:-vc2l_p2}
J=$G/jobs/${SLURM_JOB_ID:-manual}
STEPS=${STEPS:-ring ring_small ring_1e}
# The small-ring row (line mode, VC0+VC2): worker (the rest), VC0 forwarding, VC0 receiver, VC2 worker, VC2 receiver,
# VC2 forwarding.
SMALL_RING=${SMALL_RING:-0,12,16,4,2,4}
# VC0's depths for the idle-VC rows (worker, forwarding, receiver), the same in every configuration of those steps.
IDLE_VC0=${IDLE_VC0:-16,7,16}
TSMI=${TSMI:-tt-smi}
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) tree=$TREE steps=$STEPS"

# Other users' device work (the Galaxies also run outside Slurm: jobs 155971, 157726, 157804): stop before anything.
busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $(hostname): other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi

cp -al $G/trees/$TREE $J/tree || { echo "!!! copy of $TREE failed"; exit 1; }
echo "tree copied $(date -u +%T)"

glx_reset() {  # reason
    echo "=== tt-smi -glx_reset ($1) $(date -u +%T): $($TSMI --version 2>&1 | head -1)"
    local rlog=$J/results/glx_reset_$(date -u +%H%M%S).log
    timeout 900 $TSMI -glx_reset > $rlog 2>&1
    local rc=$?
    if [ $rc != 0 ]; then
        # This tt-smi may not take -glx_reset (the user's own Galaxy jobs reset with `tt-smi -r`): log its options and
        # reset with -r instead.
        echo "glx_reset: exit $rc ($(tail -n 2 $rlog | tr '\n' ' ')); trying tt-smi -r"
        $TSMI --help 2>&1 | grep -iE "reset|glx" | head -5
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
capture() {  # out-name label: read-only capture of the routers and test workers of every chip
    echo "=== hang capture $1 ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-600} $J/tree/python_env/bin/python $J/tree/fabric_exp/consolidation/hang_capture.py \
        $J/results/$1 --time-budget ${CAPTURE_BUDGET:-540} --meta cache=$J/cache_$2 --meta label=$2 \
        --meta node=$(hostname) > $J/results/$1.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/$1 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/$1.log)"
}

run() {  # label yaml timeout [ENV=VAL...]
    local label=$1 yaml=$2 to=$3; shift 3
    local tree=$J/tree lib=$J/tree/build_Release log=$J/results/$label.log t0=$SECONDS busy=0
    echo "=== $label ($(date -u +%T)): $(basename $yaml), timeout $to s $*"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$J/cache_$label TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 "$@" \
        timeout $to ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml \
        > $log 2>&1) &
    local pid=$!
    # A device another process holds: UMD waits for the KMD lock and no test starts. Stop after BUSY_S of that.
    while kill -0 $pid 2>/dev/null; do
        sleep 10
        if [ $((SECONDS - t0)) -ge ${BUSY_S:-180} ] && ! grep -aq "Running Test" $log && grep -aq "Waiting for KMD lock" $log; then
            busy=1
            pkill -TERM -P $pid
            sleep 15
            pkill -KILL -u "$USER" -f "$J/tree/"  # this job's processes only (their command lines name its tree)
            break
        fi
    done
    wait $pid
    local rc=$?
    if [ $busy = 1 ]; then
        echo "!!! $label: device held by another process (KMD lock wait > ${BUSY_S:-180} s, no test ran): stopping"
        exit 1
    fi
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    echo "$label: exit $rc in $((SECONDS - t0)) s; ${res:-NO RESULT}"
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED|Watcher|watcher' | head -5
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

S=$J/tree/tests/tt_metal/tt_fabric/test_infra/test_yamls
V=$J/tree/fabric_exp/vc2line
VC2LINE="TT_METAL_ENABLE_FABRIC_VC2=1 TT_METAL_FABRIC_VC2_LINE=1"
VC2NBR="TT_METAL_ENABLE_FABRIC_VC2=1"
ONE="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
O3="TT_METAL_FABRIC_OPT_LEVEL=O3"
LOGS="TT_FABRIC_EXP_LOG_SLOTS=1"
for step in $STEPS; do
    base=${step%_r[0-9]}; one=${base%_1e}; extra=""
    [ "$one" != "$base" ] && extra=$ONE
    base=$one
    case $base in
        nbr_vc2) run $step $S/test_fabric_vc2_at_least_2x2_mesh.yaml 900 $extra ;;
        nbr_sanity) run $step $S/test_fabric_mesh_sanity_common.yaml 1500 $extra ;;
        line) run $step $V/glx_vc2_line.yaml 900 $extra ;;
        line_vc2) run $step $S/test_fabric_vc2_at_least_2x2_mesh.yaml 900 $VC2LINE $extra ;;
        line_sanity) run $step $S/test_fabric_mesh_sanity_common.yaml 1500 $VC2LINE $extra ;;
        ring) run $step $V/glx_vc2_ring.yaml 1200 $extra ;;
        ring_small) run $step $V/glx_vc2_ring.yaml 1200 TT_FABRIC_EXP_SLOTS_VC2=$SMALL_RING TT_FABRIC_EXP_LOG_SLOTS=1 $extra ;;
        tds_nbr) run $step $S/test_fabric_2d_torus_deadlock_stability.yaml 1800 $VC2NBR $extra ;;
        tds_line) run $step $S/test_fabric_2d_torus_deadlock_stability.yaml 1800 $VC2LINE $extra ;;
        tst_nbr) run $step $V/glx_torus_stability_short.yaml 1800 $VC2NBR $extra ;;
        tst_line) run $step $V/glx_torus_stability_short.yaml 1800 $VC2LINE $extra ;;
        idle_off) run $step $V/glx_idle_mesh.yaml 900 $O3 $LOGS TT_FABRIC_EXP_SLOTS=$IDLE_VC0 $extra ;;
        idle_nbr) run $step $V/glx_idle_mesh.yaml 900 $O3 $LOGS $VC2NBR TT_FABRIC_EXP_SLOTS_VC2=$IDLE_VC0,4,4 $extra ;;
        idle_line) run $step $V/glx_idle_mesh.yaml 900 $O3 $LOGS $VC2LINE TT_FABRIC_EXP_SLOTS_VC2=$IDLE_VC0,4,4,4 $extra ;;
        *) echo "!!! unknown step $step" ;;
    esac
done
echo "done $(date -u +%T)"
