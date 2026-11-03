#!/bin/bash
# DeepSeek CCL trace perf matrix (models/demos/deepseek_v3_b1/scripts/ccl_trace_perf_matrix.py) per fabric router
# (user, 2026-11-02): main's router as shipped first (baseline), then the v4 lead, to see how router gains show in CCL
# op times; across max payload sizes 2048 4096 4352 8192 15232 (4352 is where the study's slot rows apply; other
# sizes use the tier tables for both routers). SDPA reduce-to-all: only its tuned (num_l_chunks, compute_block_size)
# per payload (4352 takes 4096's tuning, as the table has no 4352 entry).
#
# ccl_matrix_runs.sh ROUTER CCL [NUM_LINKS]
#   ROUTER  main_fw (kernel v1, flags 0, firmware keepalive) or a variant directory under fabric_exp/variants (swapped
#           into the tree for this job, kernel v2, the study's env as in the census)
#   CCL     all_gather | all_reduce | broadcast | reduce_to_one | sdpa_reduce_to_all
# Uses the ttnn build in build_ttnn (Python bindings; build_Release has none) through LD_LIBRARY_PATH. Summaries go to
# fabric_exp/results/ccl/<ROUTER>/. TT_FABRIC_NO_BW_TELEMETRY=1 always. Run inside a broker job.
set -u
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
BK=/home/aagarwal/.tt-metal-p1-swap-backup
R=${1:?router} C=${2:?ccl} L=${3:-}
SIZES=${SIZES:-"2048 4096 4352 8192 15232"}  # override for repeats of one payload
OUT=fabric_exp/results/ccl/$R/${PASS:-a}; mkdir -p $OUT  # PASS=a|b: two passes in reversed router order
PY=/home/aagarwal/tt-metal/python_env/bin/python
export LD_LIBRARY_PATH=/home/aagarwal/tt-metal/build_ttnn/lib:/home/aagarwal/tt-metal/build_ttnn/ttnn:/home/aagarwal/tt-metal/build_ttnn/tt_metal${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TT_METAL_HOME=/home/aagarwal/tt-metal PYTHONPATH=/home/aagarwal/tt-metal
export TT_FABRIC_NO_BW_TELEMETRY=1
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2 TT_METAL_SLOW_DISPATCH_MODE
SWAPPED=0
restore() { [ $SWAPPED = 1 ] && python3 $CON/swap.py restore $BK; }
trap restore EXIT
case $R in
    main_fw) export TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0; unset TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT ;;
    *) [ -d fabric_exp/variants/$R ] || { echo "no variant $R"; exit 1; }
       python3 $CON/swap.py apply fabric_exp/variants/$R $BK || { echo "swap failed"; exit 1; }
       SWAPPED=1
       export TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 ;;
esac
export TT_METAL_CACHE=/home/aagarwal/.tt-metal-cache-ccl-$R; mkdir -p $TT_METAL_CACHE
run() {  # label, args...
    local lab=$1; shift
    echo "=== $lab ($R) at ${SECONDS}s"
    timeout ${STEP_TIMEOUT:-600} $PY models/demos/deepseek_v3_b1/scripts/ccl_trace_perf_matrix.py "$@" \
        --output-dir $OUT > $OUT/$lab.log 2>&1
    local rc=$?
    echo "$lab: rc=$rc ($(grep -c -E 'PASSED|passed' $OUT/$lab.log) pass lines)"
    [ $rc = 0 ] || { echo "STOPPED at $lab"; tail -20 $OUT/$lab.log; exit 1; }
}
links=${L:+--num-links $L}
case $C in
    sdpa_reduce_to_all)
        for s in $SIZES; do
            case $s in 2048) t="4 8" ;; 4096|4352) t="2 4" ;; 8192|15232) t="2 8" ;; esac
            set -- $t
            run sdpa_$s --ccl sdpa_reduce_to_all --max-payload-sizes $s --sdpa-num-l-chunks $1 --sdpa-compute-block-sizes $2
        done ;;
    *) for s in $SIZES; do run ${C}${L:+_l$L}_$s --ccl $C --max-payload-sizes $s $links; done ;;
esac
echo "ALL_DONE ($R $C${L:+ links $L}) at ${SECONDS}s"
