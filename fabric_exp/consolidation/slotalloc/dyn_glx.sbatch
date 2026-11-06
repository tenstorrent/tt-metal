#!/bin/bash
# Dynamic slot allocator A/B on one Blackhole Galaxy (SLOT_ALLOCATOR.md, 2026-11-03; a copy of
# slotstudy/sls_glx2k.sbatch with slot arms instead of explicit rows): tree trees/$TREE (default v4txa11_dyn: the
# router v4txa11 on a base tree whose host has TT_FABRIC_EXP_DYN_SLOTS), -O3, BW telemetry off, flags 13129088,
# keepalive 1000, slots logged and checked against each step's expected row.
#
#   STEPS="2d3k:a:dyn=27,24,16 2d3k:a:tier=35,16,32 ..." sbatch fabric_exp/consolidation/slotalloc/dyn_glx.sbatch
#   dry run: G=<scratch> SLURM_JOB_ID=dry TREE=<stub tree> TSMI=<stub> MPI_BIN=<stub dir> IGNORE_BUSY=1 bash dyn_glx.sbatch
#
# A step is KIND:PASS:ARM=EXPECT (label <kind>_<pass>_<arm>; results/<label>.log, .csv, _tests.csv):
#   ARM    tier  TT_FABRIC_EXP_TIER_SLOTS=1 (main's tier pick)   dyn  TT_FABRIC_EXP_DYN_SLOTS=1
#          def   no switch (the study router's default row)     dynw24  dyn with policy worker_cap_slots=24
#          row   the kind's explicit row switch set to EXPECT (TT_FABRIC_EXP_SLOTS_2D / _1D / _VC1 / _VC2)
#          dynfloor  dyn with the 2D VC0 receiver floor (default policy since 2026-11-04); skipped on a tree whose host
#                library lacks it. On such a library dyn and dynw24 add vc0_receiver_floor=off (the first A/B's rows).
#   EXPECT W,F,R (2D, 1D), W,F0,R0,F1,R1 (VC1 split), W,F0,R0+W2,R2 (VC2 on): the depths the log must show
#          (slot_alloc_table.sh prints the dyn and tier rows)
#   KIND   2d / 2d3k: the Galaxy as one 8x4 mesh (sls_2d_glx.yaml, 4352 B slots; sls_2d_glx_3k.yaml, 3072 B)
#          1d: lines (sls_1d_glx.yaml)
#          vc1 / vc13k: two 2x2 meshes, one MPI rank each (sls_glx_vc1.yaml / sls_glx_vc1_3k.yaml)
#          vc2k / vc23k: VC2-on rows (TT_METAL_ENABLE_FABRIC_VC2=1; sls_vc2_glx_4k.yaml / sls_vc2_glx_3k.yaml)
#          slot-need sweep (SLOT_ALLOCATOR.md section 12, consolidation/slotalloc/knee/yamls in the tree): kn2w / kn2f
#          (2D worker-receiver / forwarding rows), vc1kn (2x2 split), vc1knx44 (4x4 split), vc1x44san (the 4x4
#          split's data-checked sanity); a step's PASS field carries the arm name (e.g. kn2w:a_w16:row=16,2,32)
# KIND@NAME runs the step in tree trees/NAME (listed in XTREES, copied to $J/tree_NAME) instead of $TREE.
# After each run: the logged slots must match EXPECT, the run must exit 0 with every test passing; otherwise stop.
# A timeout is a hang: kill this job's processes (command lines naming $J/tree only), read-only capture of the chips the
# step uses (hang_capture.py, tt-exalens from the tree's python_env), tt-smi -glx_reset, stop.
# Before the steps: the node check (bh-glx-110-a* only, user 2026-11-02), the busy check (other users' device work: stop; IGNORE_BUSY=1 skips), the tree copy (cp -al), the MPI
# and Python preflight, tt-smi -glx_reset. MPI: --oversubscribe (one-task allocation) and btl self,sm.
#SBATCH --partition=bh_sc36_6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:20:00
#SBATCH --job-name=dyn-glx
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
TREE=${TREE:-v4txa11_dyn}
J=$G/jobs/${SLURM_JOB_ID:-manual}
STEPS=${STEPS:-}
TSMI=${TSMI:-tt-smi}
MPI_BIN=${MPI_BIN:-/opt/openmpi-v5.0.7-ulfm/bin}
MPI_ARGS=${MPI_ARGS:---oversubscribe --mca btl self,sm}
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) tree=$TREE"
case "${SLURMD_NODENAME:-}" in
    bh-glx-110-a*) ;;
    # One-time exception (user, 2026-11-04): single Galaxies of bh_sc36_2 only when the submitter sets it.
    bh-glx-120-c*) [ "${GLX_SC36_2_EXCEPTION:-0}" = 1 ] || { echo "!!! node ${SLURMD_NODENAME}: bh-glx-120-c* needs GLX_SC36_2_EXCEPTION=1 at submit: stopping"; exit 1; } ;;
    "") [ "${SLURM_JOB_ID:-}" = "" ] || [[ ${SLURM_JOB_ID} == dry* ]] || { echo "!!! no SLURMD_NODENAME: stopping"; exit 1; } ;;
    *) echo "!!! node ${SLURMD_NODENAME}: study Galaxy runs are on bh-glx-110-a* only (user, 2026-11-02): stopping"; exit 1 ;;
esac
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
for t in ${XTREES:-}; do cp -al $G/trees/$t $J/tree_$t || { echo "!!! copy of $t failed"; exit 1; }; done
echo "tree copied ($TREE ${XTREES:-}) $(date -u +%T)"
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

kill_mine() {  # this job's test processes only: every process whose command line names one of this job's trees
    pkill -KILL -u "$USER" -f "$J/tree" 2>/dev/null
    sleep 5
}

capture() {  # out-name label chips: read-only capture of the routers and test workers
    echo "=== hang capture $1 ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-420} $J/tree/python_env/bin/python $J/tree/fabric_exp/consolidation/hang_capture.py \
        $J/results/$1 --chips $3 --time-budget ${CAPTURE_BUDGET:-360} --meta cache=$CR/cache_$2 \
        --meta label=$2 --meta node=$(hostname) > $J/results/$1.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/$1 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/$1.log)"
}

run() {  # label kind yaml timeout expected-tests [ENV=VAL...]
    local label=$1 kind=$2 yaml=$3 to=$4 want=$5; shift 5
    local tree=$J/${TD:-tree} lib=$J/${TD:-tree}/build_Release log=$J/results/$label.log t0=$SECONDS busy=0 chips
    local launch=(./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml)
    if [ $kind = vc1 ] || [ $kind = vc12k ] || [ $kind = vc1k ] || [ $kind = vc13k ] || [ $kind = vc135k ] || [ $kind = vc1kn ] || [ $kind = vc1kn3 ]; then
        launch=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py
            --rank-binding fabric_exp/slotstudy/glx_2x2_split_rank_bindings.yaml --mpi-args "$MPI_ARGS" "${launch[@]}")
        chips=0,1,2,3,4,5,6,7
    elif [ $kind = vc1knx44 ] || [ $kind = vc1x44san ] || [ $kind = vc1knx443 ] || [ $kind = vc1x44san3 ]; then
        launch=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py
            --rank-binding fabric_exp/slotstudy/glx_4x4_split_rank_bindings.yaml --mpi-args "$MPI_ARGS" "${launch[@]}")
        chips=$(seq -s, 0 31)
    elif [ $kind = vc1x44 ] || [ $kind = vc1x42 ]; then
        local rb=fabric_exp/slotstudy/glx_${kind:4:1}x${kind:5:1}_split_rank_bindings.yaml
        launch=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py --rank-binding $rb --mpi-args "$MPI_ARGS" "${launch[@]}")
        chips=$(seq -s, 0 31)
    else
        chips=$(seq -s, 0 31)
    fi
    echo "=== $label ($(date -u +%T)): $(basename $yaml), timeout $to s $ARMENV $*"
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$CR/cache_$label TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_LOG_SLOTS=1 TT_METAL_FABRIC_OPT_LEVEL=O3 \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 $ARMENV "$@" \
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
        local w; for w in $(seq 1 60); do pgrep -u "$USER" -f "$J/tree" > /dev/null || break; sleep 1; done
        [ "${CAPTURE:-1}" = 1 ] && capture ${label}_hangcap $label $chips
        glx_reset "after a timeout in $label"
        exit 1
    fi
    local ok; ok=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acE "Tests: $want executed, 0 failed")
    local nrank=1; case $kind in vc1|vc1x44|vc12k|vc1k|vc13k|vc135k|vc1kn|vc1knx44|vc1x44san|vc1kn3|vc1knx443|vc1x44san3) nrank=2 ;; vc1x42) nrank=4 ;; esac
    if [ $rc != 0 ] || [ "$ok" -lt $nrank ]; then
        echo "!!! $label: exit $rc, $ok of $nrank ranks reported '$want executed, 0 failed': stopping"
        exit 1
    fi
}

check_slots() {  # label kind expect: the logged depths must be EXPECT
    local label=$1 kind=$2 e=$3 log=$J/results/$1.log w f0 r0 f1 r1 w2 r2 v0 v1="" n0 n1=1
    case $kind in
        vc1*) IFS=, read w f0 r0 f1 r1 <<< "$e"; v1="VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}" ;;
        vc2*) IFS=, read w f0 r0 <<< "${e%+*}"; IFS=, read w2 r2 <<< "${e#*+}"; v1="VC2 sender slots {$w2} receiver slots {$r2}" ;;
        *) IFS=, read w f0 r0 <<< "$e" ;;
    esac
    case $kind in 1d) v0="VC0 sender slots {$w, $f0} receiver slots {$r0}" ;; *) v0="VC0 sender slots {$w, $f0, $f0, $f0} receiver slots {$r0}" ;; esac
    n0=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acF "$v0")
    [ -n "$v1" ] && n1=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acF "$v1")
    echo "   slots check: '$v0' x$n0${v1:+, '$v1' x$n1}"
    [ "$n0" -gt 0 ] && [ "$n1" -gt 0 ] || { echo "!!! $label: slots not as expected ($e)"; exit 1; }
}

Y=$J/tree/fabric_exp/slotstudy
# Kernel caches: $J on /data by default; LOCAL_CACHE=1 puts them on the node's /tmp (removed at the end) so a job does
# not spend /data inodes (2026-11-04: /data ran out of inodes).
CR=$J
if [ "${LOCAL_CACHE:-0}" = 1 ]; then
    CR=/tmp/$USER-dynglx-${SLURM_JOB_ID:-manual}; mkdir -p $CR || { echo "!!! no node-local cache dir $CR"; exit 1; }
    trap 'rm -rf "$CR"' EXIT
    echo "kernel caches node-local: $CR ($(df -h /tmp | tail -1 | awk '{print $4}') free on /tmp)"
fi
KY=$J/tree/fabric_exp/consolidation/slotalloc/knee/yamls
VC2=TT_METAL_ENABLE_FABRIC_VC2=1
for step in $STEPS; do
    kind=${step%%:*}; rest=${step#*:}; pass=${rest%%:*}; rest=${rest#*:}; arm=${rest%%=*}; e=${rest#*=}
    [ "$e" != "$rest" ] || { echo "!!! $step: no expected row"; exit 1; }
    TD=tree
    if [[ $kind == *@* ]]; then TD=tree_${kind#*@}; kind=${kind%@*}; [ -d $J/$TD ] || { echo "!!! $step: tree $TD not copied (XTREES)"; exit 1; }; fi
    label=${kind}_${pass}_${arm}
    case $kind in
        2d|2d3k|kn2w|kn2f|knm|knmval) rowsw=TT_FABRIC_EXP_SLOTS_2D ;; 1d) rowsw=TT_FABRIC_EXP_SLOTS_1D ;;
        vc1|vc13k|vc1kn|vc1knx44|vc1x44san|vc1kn3|vc1knx443|vc1x44san3) rowsw=TT_FABRIC_EXP_SLOTS_VC1 ;; vc2k|vc23k) rowsw=TT_FABRIC_EXP_SLOTS_VC2 ;;
        *) echo "!!! unknown kind in $step"; exit 1 ;;
    esac
    FLOOR_OFF=""; grep -aq vc0_receiver_floor $J/$TD/build_Release/tt_metal/libtt_metal.so && FLOOR_OFF=vc0_receiver_floor=off
    case $arm in
        tier) ARMENV="TT_FABRIC_EXP_TIER_SLOTS=1" ;;
        dyn) ARMENV="TT_FABRIC_EXP_DYN_SLOTS=1${FLOOR_OFF:+ TT_FABRIC_EXP_DYN_SLOTS_POLICY=$FLOOR_OFF}" ;;
        dynw24) ARMENV="TT_FABRIC_EXP_DYN_SLOTS=1 TT_FABRIC_EXP_DYN_SLOTS_POLICY=worker_cap_slots=24${FLOOR_OFF:+,$FLOOR_OFF}" ;;
        dynfloor) [ -n "$FLOOR_OFF" ] || { echo "=== skip $step: $TD's host library has no VC0 receiver floor"; continue; }
                  ARMENV="TT_FABRIC_EXP_DYN_SLOTS=1" ;;
        def) ARMENV="" ;; row) ARMENV="$rowsw=${e/+/,}" ;;
        row_g[124]) ARMENV="$rowsw=${e/+/,} TT_FABRIC_EXP_COMPLETION_GROUP=${arm#row_g}" ;;  # section 12.10's switch
        *) echo "!!! unknown arm in $step"; exit 1 ;;
    esac
    case $kind in
        2d) run $label 2d $Y/sls_2d_glx.yaml ${D2_TO:-1200} 22 ;;
        2d3k) run $label 2d3k $Y/sls_2d_glx_3k.yaml ${D2_TO:-1200} 33 ;;
        1d) run $label 1d $Y/sls_1d_glx.yaml ${D1_TO:-1200} 10 ;;
        vc1) run $label vc1 $Y/sls_glx_vc1.yaml ${VC1_TO:-900} 49 ;;
        vc13k) run $label vc13k $Y/sls_glx_vc1_3k.yaml ${VC1_TO:-900} 49 ;;
        vc2k) run $label vc2k $Y/sls_vc2_glx_4k.yaml ${D2_TO:-1200} 4 $VC2 ;;
        vc23k) run $label vc23k $Y/sls_vc2_glx_3k.yaml ${D2_TO:-1200} 8 $VC2 ;;
        # Slot-need sweep (SLOT_ALLOCATOR.md section 12; knee/yamls from mk_knee_yamls.py, copied into the tree).
        kn2w) run $label kn2w $KY/g2w_4352.yaml ${D2_TO:-1200} 12 ;;
        kn2f) run $label kn2f $KY/g2f_4352.yaml ${D2_TO:-1200} 20 ;;
        vc1kn) run $label vc1kn $KY/gvc1_4352.yaml ${VC1_TO:-900} 56 ;;
        vc1knx44) run $label vc1knx44 $KY/gvc1x44_4352.yaml ${VC1_TO:-900} 44 ;;
        vc1x44san) run $label vc1x44san $KY/gvc1x44san_4352.yaml ${VC1_TO:-900} 6 ;;
        vc1kn3) run $label vc1kn3 $KY/gvc1_3072.yaml ${VC1_TO:-900} 56 ;;
        vc1knx443) run $label vc1knx443 $KY/gvc1x44_3072.yaml ${VC1_TO:-900} 44 ;;
        vc1x44san3) run $label vc1x44san3 $KY/gvc1x44san_3072.yaml ${VC1_TO:-900} 6 ;;
        knm) run $label knm $KY/merge_g8x4_bench.yaml ${D2_TO:-1200} 10 ;;
        knmval) run $label knmval $KY/merge_g8x4_val.yaml ${D2_TO:-1200} 5 ;;
    esac
    check_slots $label $kind $e
    if [[ $arm == row_g[124] ]]; then  # every receiver channel the host logged has groups of ${arm#row_g}
        grp=$(sed 's/\x1b\[[0-9;]*m//g' $J/results/$label.log | grep -aoE 'RX_CH_[0-9] G [0-9]+' | sort | uniq -c | tr -s ' \n' ' ')
        echo "   groups: ${grp:-none}"
        { [ -n "$grp" ] && ! sed 's/\x1b\[[0-9;]*m//g' $J/results/$label.log | grep -aoE 'RX_CH_[0-9] G [0-9]+' | grep -vq " G ${arm#row_g}\$"; } ||
            { echo "!!! $label: completion groups not ${arm#row_g}"; exit 1; }
    fi
    [ "${KEEP_CACHES:-1}" = 1 ] || rm -rf $CR/cache_$label   # kept for the code-size check (dyn_size.py)
done
echo "done $(date -u +%T)"
