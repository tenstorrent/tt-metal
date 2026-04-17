#!/bin/bash
# Run DM NOC perf tests 6-16 individually (resuming after test 5)
BINARY="./build/test/tt_metal/unit_tests_legacy"
ENV="TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DPRINT_CORES=all"
LOG="dm_noc_perf_run_100iters_from6.log"

TESTS=(
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
)

> "$LOG"

TOTAL=${#TESTS[@]}
for i in "${!TESTS[@]}"; do
    TEST="${TESTS[$i]}"
    echo "[$((i+6))/16] Running $TEST ..." | tee -a "$LOG"
    env $ENV $BINARY --gtest_filter="MeshDeviceSingleCardFixture.$TEST" 2>&1 | tee -a "$LOG"
    echo "[$((i+6))/16] Done: $TEST" | tee -a "$LOG"
    echo "---" | tee -a "$LOG"
done

echo "All done." | tee -a "$LOG"
