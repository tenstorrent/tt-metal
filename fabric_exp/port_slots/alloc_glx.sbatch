#!/bin/bash
# The staged slot allocator's device A/B on one Blackhole Galaxy (SLOT_ALLOCATOR.md section 12.14; user 2026-11-05:
# VC1 split 2x2 and 4x4 and VC2 rows, staged against tier, one allocation, after the P150 job). A port copy of
# consolidation/slotalloc/dyn_glx.sbatch: the tree is trees/$TREE (default port_alloc_c53b: the port-alloc build,
# c53b5a5cbbd + alloc_tier_switch.patch, never committed, with fabric_exp/slotstudy's Galaxy yamls, rank bindings and
# mesh graph descriptors, fabric_exp/port_slots/yamls/alloc_glx_expected_rows.json, hang_capture.py and base_1102's
# python_env); the port's own settings (TT_METAL_CLEAR_L1=1, profiler off, BW telemetry off, compile commands logged),
# no study switches; TT_FABRIC_EXP_LOG_SLOTS=1.
#   STAGED  no switch (the staged allocator)     TIER  TT_FABRIC_EXP_TIER_SLOTS=1 (main's tier table)
# A step is KIND:PASS:ARM (label <kind>_<pass>_<arm>; results/<label>.log, .csv, _tests.csv). Default STEPS: pass a,
# per kind staged then tier; pass b, kinds reversed, tier then staged.
#   KIND  vc1     two 2x2 meshes, one MPI rank each (slotstudy/sls_glx_vc1.yaml, 4352 B; 49 tests per rank)
#         vc13k   the same at 3072 B (sls_glx_vc1_3k.yaml; 49)
#         vc1x44  two 4x4 meshes, one MPI rank each (sls_glx_vc1_4x4.yaml, 4352 B; 48)
#         vc2k    VC2 on (TT_METAL_ENABLE_FABRIC_VC2=1), the Galaxy as one 8x4 mesh (sls_vc2_glx_4k.yaml, 4352 B; 4)
#         vc23k   the same at 3072 B (sls_vc2_glx_3k.yaml; 8)
#   The ring-jam sanity (SLOT_ALLOCATOR.md section 12.15; RING_JAM_FIX.md 10.3's Galaxy 172983 rows), the Galaxy as one
#   8x4 mesh, yamls from fabric_exp/galaxy/frgap:
#         frgval     the data-checked full-ring flow-control validation (glx_frg_val.yaml: Ring and XY torus, full-ring
#                    multicast and all-to-all, unicast write and fused atomic inc, 4096 B; 10)
#         frgnsrx    8-ring multicast N/S at 3072 (control), 3584 and 4352 B (glx_frg_nsrx.yaml; 3)
#         frgew      4-ring multicast E/W at 4352 B (glx_frg_ew.yaml; 1)
#         frgshort4k ring 1-hop and half ring at 4352 B (glx_frg_short4k.yaml; 2)
#   Default STEPS for these: JAM=1 (frgval staged then tier; pass a per kind staged then tier; pass b reversed).
# After each run: exit 0 with every rank reporting every test passed, and the logged slot rows against the mock
# captures' (alloc_glx_expected_rows.json): every logged row is one the capture logged for this arm, and every captured
# row that tells this arm from the other is logged. Otherwise stop. A timeout is a hang: kill this job's processes
# (command lines naming this job's tree only), read-only capture of the step's chips (hang_capture.py), tt-smi -glx_reset, stop.
# Before the steps: the node check (bh-glx-110-a* only), the busy check (other users' device work: stop; IGNORE_BUSY=1
# skips), the tree copy (cp -al), the MPI and Python preflight, tt-smi -glx_reset. MPI: --oversubscribe, btl self,sm.
# Kernel caches node-local (/tmp, removed at the end) so the job spends no /data inodes on them.
#   sbatch --exclude=<testplan/manifest.yaml bh_galaxy excludes> fabric_exp/port_slots/alloc_glx.sbatch
#   dry run: G=<scratch> SLURM_JOB_ID=dry TREE=<stub tree> TSMI=<stub> MPI_BIN=<stub dir> IGNORE_BUSY=1 bash alloc_glx.sbatch
#SBATCH --partition=bh_sc36_6,bh_110a_sc16
#SBATCH --exclude=bh-glx-110-a01u14,bh-glx-110-a02u02,bh-glx-110-a02u08,bh-glx-110-a02u14,bh-glx-110-a03u14,bh-glx-110-a08u20,bh-glx-110-a09u02,bh-glx-110-a10u02
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:20:00
#SBATCH --job-name=alloc-glx
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
TREE=${TREE:-port_alloc_c53b}
J=$G/jobs/${SLURM_JOB_ID:-manual}
TSMI=${TSMI:-tt-smi}
MPI_BIN=${MPI_BIN:-/opt/openmpi-v5.0.7-ulfm/bin}
MPI_ARGS=${MPI_ARGS:---oversubscribe --mca btl self,sm}
KINDS=${KINDS:-vc1 vc13k vc1x44 vc2k vc23k}
if [ -z "${STEPS:-}" ] && [ "${JAM:-0}" = 1 ]; then
    STEPS="frgval:a:staged frgval:a:tier "
    for k in frgnsrx frgew frgshort4k; do STEPS+="$k:a:staged $k:a:tier "; done
    for k in frgshort4k frgew frgnsrx; do STEPS+="$k:b:tier $k:b:staged "; done
fi
if [ -z "${STEPS:-}" ]; then
    STEPS=""
    for k in $KINDS; do STEPS+="$k:a:staged $k:a:tier "; done
    for k in $(echo $KINDS | tr ' ' '\n' | tac); do STEPS+="$k:b:tier $k:b:staged "; done
fi
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) tree=$TREE"
case "${SLURMD_NODENAME:-}" in
    bh-glx-110-a*) ;;
    "") [ "${SLURM_JOB_ID:-}" = "" ] || [[ ${SLURM_JOB_ID} == dry* ]] || { echo "!!! no SLURMD_NODENAME: stopping"; exit 1; } ;;
    *) echo "!!! node ${SLURMD_NODENAME}: study Galaxy runs are on bh-glx-110-a* only (user, 2026-11-02): stopping"; exit 1 ;;
esac
case " bh-glx-110-a01u14 bh-glx-110-a02u02 bh-glx-110-a02u08 bh-glx-110-a02u14 bh-glx-110-a03u14 bh-glx-110-a08u20 bh-glx-110-a09u02 bh-glx-110-a10u02 " in
    *" ${SLURMD_NODENAME:-none} "*) echo "!!! node ${SLURMD_NODENAME} is in the standing excludes (testplan/manifest.yaml): stopping"; exit 1 ;;
esac
echo "steps=$STEPS"

busy=$(ps -eo user:20,etime,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && /pytest|python|test_|tt-metal|ttnn/' |
    grep -v -e tt-fabric-manager -e tt-metrics-exporter -e tt_telemetry | cut -c1-200 | head -5)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then
    echo "!!! $(hostname): other users' processes that may hold the device; stopping:"
    echo "$busy"
    exit 1
fi

# The tree: TREE_TAR (one .tar.zst on /data, 2026-11-05: /data short of inodes) unpacked on the node's /tmp with
# EXTRA_TAR (fabric_exp's Galaxy yamls, rank bindings, descriptors, the expected rows, hang_capture.py) and base_1102's
# python_env linked in (removed at the end with the kernel caches); or trees/$TREE hard-linked into the job directory.
if [ -n "${TREE_TAR:-}" ]; then
    TR=/tmp/$USER-allocglx-${SLURM_JOB_ID:-manual}/tree
    rm -rf "${TR:?}"; mkdir -p $TR
    tar -I zstd -xf $TREE_TAR -C $TR && tar -xf ${EXTRA_TAR:?} -C $TR || { echo "!!! unpack of $TREE_TAR failed"; exit 1; }
    ln -sfn ${PY_ENV:-$G/trees/base_1102/python_env} $TR/python_env
else
    TR=$J/tree
    cp -al $G/trees/$TREE $TR || { echo "!!! copy of $TREE failed"; exit 1; }
fi
echo "tree at $TR ($(cat $TR/PORT_COMMIT 2>/dev/null)) $(date -u +%T)"
(cd $TR && md5sum -c --quiet MD5SUMS) || { echo "!!! the tree's binaries differ from MD5SUMS"; exit 1; }
E=$TR/fabric_exp/port_slots/yamls/alloc_glx_expected_rows.json
[ -f $E ] || { echo "!!! no $E"; exit 1; }
export PATH=$MPI_BIN:$PATH
echo "mpirun-ulfm: $(command -v mpirun-ulfm || echo none); mpirun: $(command -v mpirun || echo none)"
if ! $TR/python_env/bin/python -c "import click, yaml, loguru, pydantic" 2>$J/results/preflight.log; then
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
    pkill -KILL -u "$USER" -f "$TR" 2>/dev/null
    sleep 5
}

CR=/tmp/$USER-allocglx-${SLURM_JOB_ID:-manual}; mkdir -p $CR || { echo "!!! no node-local cache dir $CR"; exit 1; }
trap 'rm -rf "$CR"' EXIT
echo "kernel caches node-local: $CR ($(df -h /tmp | tail -1 | awk '{print $4}') free on /tmp)"

capture() {  # out-name label chips cache: read-only capture of the routers and test workers
    echo "=== hang capture $1 ($(date -u +%T))"
    timeout ${CAPTURE_TIMEOUT:-420} $TR/python_env/bin/python $TR/fabric_exp/consolidation/hang_capture.py \
        $J/results/$1 --chips $3 --time-budget ${CAPTURE_BUDGET:-360} --meta cache=$4 \
        --meta label=$2 --meta node=$(hostname) > $J/results/$1.log 2>&1
    echo "hang capture: exit $? ($(du -sh $J/results/$1 2>/dev/null | cut -f1)); $(tail -n 1 $J/results/$1.log)"
}

run() {  # label kind arm yaml timeout expected-tests ranks [ENV=VAL...]
    local label=$1 kind=$2 arm=$3 yaml=$4 to=$5 want=$6 nrank=$7; shift 7
    local tree=$TR lib=$TR/build_Release log=$J/results/$label.log t0=$SECONDS busy=0 chips armenv=()
    local launch=(./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml)
    case $kind in
        vc1|vc13k) launch=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py
            --rank-binding fabric_exp/slotstudy/glx_2x2_split_rank_bindings.yaml --mpi-args "$MPI_ARGS" "${launch[@]}")
            chips=0,1,2,3,4,5,6,7 ;;
        vc1x44) launch=(./python_env/bin/python ttnn/ttnn/distributed/ttrun.py
            --rank-binding fabric_exp/slotstudy/glx_4x4_split_rank_bindings.yaml --mpi-args "$MPI_ARGS" "${launch[@]}")
            chips=$(seq -s, 0 31) ;;
        *) chips=$(seq -s, 0 31) ;;
    esac
    [ $arm = tier ] && armenv=(TT_FABRIC_EXP_TIER_SLOTS=1)
    echo "=== $label ($(date -u +%T)): $(basename $yaml), timeout $to s ${armenv[*]} $*"
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$CR/cache_${kind}_${arm} TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_LOG_SLOTS=1 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        "${armenv[@]}" "$@" timeout $to "${launch[@]}" > $log 2>&1) &
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
    echo "$label: exit $rc in $((SECONDS - t0)) s; per rank: ${res:-NO RESULT}"
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aE 'TT_FATAL|TT_THROW|FAILED' | head -5
    if [ $rc = 124 ]; then
        echo "!!! $label timed out (hang?): killing this job's processes, capturing, resetting the Galaxy, stopping"
        kill_mine
        local w; for w in $(seq 1 60); do pgrep -u "$USER" -f "$TR" > /dev/null || break; sleep 1; done
        [ "${CAPTURE:-1}" = 1 ] && capture ${label}_hangcap $label $chips $CR/cache_${kind}_${arm}
        glx_reset "after a timeout in $label"
        exit 1
    fi
    local ok; ok=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -acE "Tests: $want executed, 0 failed")
    if [ $rc != 0 ] || [ "$ok" -lt $nrank ]; then
        echo "!!! $label: exit $rc, $ok of $nrank ranks reported '$want executed, 0 failed': stopping"
        exit 1
    fi
    case $kind in frg*) awk -F, 'NR>1 {printf "   %s %s B %.1f cycles/packet (1.35 GHz)\n", $1, $7, 1.35e9/$10}' \
        $J/results/$label.csv 2>/dev/null ;; esac
    sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'VC[0-9] sender slots \{[^}]*\} receiver slots \{[^}]*\} \(channel buffer [0-9]+ B\)' |
        sort -u > $J/results/${label}_rows.txt
    $TR/python_env/bin/python - $E $kind $arm $J/results/${label}_rows.txt <<'PY' || { echo "!!! $label: slot rows not as captured: stopping"; exit 1; }
import json, sys
e, k, a, f = sys.argv[1:5]
exp = json.load(open(e))[k]
want = set(exp[a]); other = set().union(*(set(v) for x, v in exp.items() if x != a))
got = {l.strip() for l in open(f) if l.strip()}
extra, missing = got - want, (want - other) - got
if extra or missing:
    print("   rows: not captured for this arm", sorted(extra), "; missing", sorted(missing)); sys.exit(1)
print("   rows as captured: " + " | ".join(sorted(x.split(" (channel")[0] for x in got)))
PY
}

Y=$TR/fabric_exp/slotstudy  # absolute: the test binary reads a relative --test_config under its test_yamls directory
FRG=$TR/fabric_exp/galaxy/frgap
VC2=TT_METAL_ENABLE_FABRIC_VC2=1
for step in $STEPS; do
    kind=${step%%:*}; rest=${step#*:}; pass=${rest%%:*}; arm=${rest#*:}
    case $arm in staged|tier) ;; *) echo "!!! unknown arm in $step"; exit 1 ;; esac
    label=${kind}_${pass}_${arm}
    case $kind in
        vc1) run $label vc1 $arm $Y/sls_glx_vc1.yaml ${VC1_TO:-900} 49 2 ;;
        vc13k) run $label vc13k $arm $Y/sls_glx_vc1_3k.yaml ${VC1_TO:-900} 49 2 ;;
        vc1x44) run $label vc1x44 $arm $Y/sls_glx_vc1_4x4.yaml ${VC1_TO:-900} 48 2 ;;
        vc2k) run $label vc2k $arm $Y/sls_vc2_glx_4k.yaml ${D2_TO:-1200} 4 1 $VC2 ;;
        vc23k) run $label vc23k $arm $Y/sls_vc2_glx_3k.yaml ${D2_TO:-1200} 8 1 $VC2 ;;
        frgval) run $label frgval $arm $FRG/glx_frg_val.yaml ${VAL_TO:-900} 10 1 ;;
        frgnsrx) run $label frgnsrx $arm $FRG/glx_frg_nsrx.yaml ${FRG_TO:-1200} 3 1 ;;
        frgew) run $label frgew $arm $FRG/glx_frg_ew.yaml ${FRG_TO:-1200} 1 1 ;;
        frgshort4k) run $label frgshort4k $arm $FRG/glx_frg_short4k.yaml ${FRG_TO:-1200} 2 1 ;;
        *) echo "!!! unknown kind in $step"; exit 1 ;;
    esac
done
echo "done $(date -u +%T)"
