#!/bin/bash
# Temporary A/B benchmark for PR #57251: host-to-device DMA writes on Blackhole.
# Same build, toggled with TT_METAL_DISABLE_DMA_OPS. Slow dispatch exercises
# Cluster::write_core per page; fast dispatch is a control.
set -u

OUT=dma_ab
BENCH=./build/test/tt_metal/perf_microbenchmark/dispatch/benchmark_rw_buffer
FILTER='^(Write|WriteSharded)/'
mkdir -p "$OUT" generated/test_reports
rc=0

echo "Host: $(hostname)"
lspci -d 1e52: 2>/dev/null || true
ls /sys/kernel/iommu_groups 2>/dev/null | wc -l | xargs echo "iommu groups:"

run_bench() {
    local name=$1
    shift
    echo "=== $name ($*)"
    local start=$SECONDS
    if ! env "$@" "$BENCH" --benchmark_filter="$FILTER" --benchmark_repetitions=3 \
        --benchmark_out_format=json --benchmark_out="$OUT/$name.json" >"$OUT/$name.log" 2>&1; then
        echo "!!! $name FAILED"
        tail -60 "$OUT/$name.log"
        rc=1
    fi
    echo "    took $((SECONDS - start)) s"
}

run_bench sd_dma_on TT_METAL_SLOW_DISPATCH_MODE=1
run_bench sd_dma_off TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_DMA_OPS=1
run_bench fd_dma_on DMA_AB=on
run_bench fd_dma_off TT_METAL_DISABLE_DMA_OPS=1

# Device init + close wall time (firmware load goes through write_core).
INIT=./build/test/tt_metal/test_clean_init
if [ -x "$INIT" ]; then
    "$INIT" >/dev/null 2>&1
    : >"$OUT/init_times.txt"
    for i in 1 2 3 4 5; do
        for dma in on off; do
            if [ "$dma" = off ]; then e=TT_METAL_DISABLE_DMA_OPS=1; else e=DMA_AB=on; fi
            s=$(date +%s%N)
            env "$e" "$INIT" >/dev/null 2>&1
            echo "$dma $(( ($(date +%s%N) - s) / 1000000 ))" >>"$OUT/init_times.txt"
        done
    done
fi

python3 tests/scripts/bh_dma_ab/summarize.py "$OUT" | tee "$OUT/summary.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    { echo "## $(hostname)"; cat "$OUT/summary.md"; } >>"$GITHUB_STEP_SUMMARY" || true
fi
exit $rc
