#!/bin/bash
# Run all DM NOC perf tests (1/2/3/4 cores) one by one and regenerate results
set -e

BINARY="./build/test/tt_metal/unit_tests_legacy"
ENV="TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DPRINT_CORES=all"
LOG="dm_noc_perf_run_all.log"

TESTS=(
    # --- 1 core ---
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd
    DmDirectSramNoc_TwoPhase_BarrierEnd
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter
    DmDirectSramNoc_TwoPhase_BarrierPerIter
    DmCacheL2FlushNoc_PerIter_BarrierEnd
    DmDirectSramNoc_PerIter_BarrierEnd
    DmCacheL2FlushNoc_PerIter_BarrierPerIter
    DmDirectSramNoc_PerIter_BarrierPerIter
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr
    DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr
    DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr
    DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr
    DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr
    DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr
    DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr
    # --- 2 cores ---
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd_2Cores
    DmDirectSramNoc_TwoPhase_BarrierEnd_2Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_2Cores
    DmDirectSramNoc_TwoPhase_BarrierPerIter_2Cores
    DmCacheL2FlushNoc_PerIter_BarrierEnd_2Cores
    DmDirectSramNoc_PerIter_BarrierEnd_2Cores
    DmCacheL2FlushNoc_PerIter_BarrierPerIter_2Cores
    DmDirectSramNoc_PerIter_BarrierPerIter_2Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr_2Cores
    DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr_2Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_2Cores
    DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_2Cores
    DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr_2Cores
    DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr_2Cores
    DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr_2Cores
    DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr_2Cores
    # --- 3 cores ---
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd_3Cores
    DmDirectSramNoc_TwoPhase_BarrierEnd_3Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_3Cores
    DmDirectSramNoc_TwoPhase_BarrierPerIter_3Cores
    DmCacheL2FlushNoc_PerIter_BarrierEnd_3Cores
    DmDirectSramNoc_PerIter_BarrierEnd_3Cores
    DmCacheL2FlushNoc_PerIter_BarrierPerIter_3Cores
    DmDirectSramNoc_PerIter_BarrierPerIter_3Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr_3Cores
    DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr_3Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_3Cores
    DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_3Cores
    DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr_3Cores
    DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr_3Cores
    DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr_3Cores
    DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr_3Cores
    # --- 4 cores ---
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd_4Cores
    DmDirectSramNoc_TwoPhase_BarrierEnd_4Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_4Cores
    DmDirectSramNoc_TwoPhase_BarrierPerIter_4Cores
    DmCacheL2FlushNoc_PerIter_BarrierEnd_4Cores
    DmDirectSramNoc_PerIter_BarrierEnd_4Cores
    DmCacheL2FlushNoc_PerIter_BarrierPerIter_4Cores
    DmDirectSramNoc_PerIter_BarrierPerIter_4Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierEnd_UpdateNocAddr_4Cores
    DmDirectSramNoc_TwoPhase_BarrierEnd_UpdateNocAddr_4Cores
    DmCacheL2FlushNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_4Cores
    DmDirectSramNoc_TwoPhase_BarrierPerIter_UpdateNocAddr_4Cores
    DmCacheL2FlushNoc_PerIter_BarrierEnd_UpdateNocAddr_4Cores
    DmDirectSramNoc_PerIter_BarrierEnd_UpdateNocAddr_4Cores
    DmCacheL2FlushNoc_PerIter_BarrierPerIter_UpdateNocAddr_4Cores
    DmDirectSramNoc_PerIter_BarrierPerIter_UpdateNocAddr_4Cores
)

# Fresh results file
rm -f dm_noc_perf_results.txt
> "$LOG"

TOTAL=${#TESTS[@]}
for i in "${!TESTS[@]}"; do
    TEST="${TESTS[$i]}"
    echo "[$((i+1))/$TOTAL] Running $TEST ..." | tee -a "$LOG"
    env $ENV $BINARY --gtest_filter="MeshDeviceSingleCardFixture.$TEST" 2>&1 | tee -a "$LOG"
    echo "[$((i+1))/$TOTAL] Done: $TEST" | tee -a "$LOG"
    echo "---" | tee -a "$LOG"
done

echo "All $TOTAL tests complete. Results in dm_noc_perf_results.txt" | tee -a "$LOG"
