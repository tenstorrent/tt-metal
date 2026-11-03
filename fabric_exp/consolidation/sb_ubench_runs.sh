#!/bin/bash
# Send-batch study (V4_SEND_BATCH.md, 2026-11-01): the ERISC load-overlap microbenchmark (RESULTS §17) with the
# send-batch patterns (header groups after the count read, parked-slot chains, one-ahead prefetch behind the TX-queue
# writes, the post-send group with the next header). ERISC0 of two Ethernet cores (index 0 and 3), fabric off, slow
# dispatch. Results: fabric_exp/results/sb_ubench_e<index>.csv (+ .log). No router, no tree swap.
cd /home/aagarwal/tt-metal
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
out=fabric_exp/results
for idx in 0 3; do
    echo "=== eth index $idx"
    env TT_METAL_SLOW_DISPATCH_MODE=1 ERISC_LOAD_ETH_INDEX=$idx ERISC_LOAD_CSV=$out/sb_ubench_e$idx.csv \
        TT_METAL_CACHE=/home/aagarwal/.tt-metal-cache-sb-ubench \
        timeout 600 ./build/test/tt_metal/tt_fabric/test_erisc_load_overlap --gtest_filter=EriscLoadOverlap.ReadLatencyHiding \
        2>&1 | tee $out/sb_ubench_e$idx.log | grep --line-buffered -E "load cost|count read|PASSED|FAILED|error|Error"
    rc=${PIPESTATUS[0]}
    [ $rc = 0 ] || { echo "STOPPED (exit $rc)"; exit 1; }
done
echo "ALL_DONE (job at ${SECONDS} s)"
