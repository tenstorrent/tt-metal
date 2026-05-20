#!/bin/bash
# Sweep prefetcher matmul A/B across Llama-3.1-8B-relevant shapes.
# Usage: scripts/run_llama_matmul_sweep.sh [reps]
#
# Each shape edits out/bench_env and runs the matmul bench. Per-matmul us +
# TFLOP/s are extracted and tabulated. Use BENCH_REPEATS=1000 (default) for
# asymptotic numbers.
#
# Llama-3.1-8B production on single Blackhole (per
# models/tt_transformers/tt/prefetcher.py): num_receiver_cores=8 -> ring=64
# for all gather_in0 matmuls.

set -u
REPS=${1:-1000}
RESULTS_FILE="out/llama_matmul_sweep_results.txt"
{
    echo "# Llama-3.1-8B matmul A/B sweep on single Blackhole, BENCH_REPEATS=$REPS"
    echo "# Production ring=64 (8 banks x 8 recv/bank) for all shapes."
    echo "# Shape : DRAM-core (us/op, TFLOP/s) | Worker-core (us/op, TFLOP/s)"
    echo
} > "$RESULTS_FILE"

# Shapes: (label, K, N, dtype, recv_per_bank)
# Production ring=64 = 8 banks * 8 recv/bank. M (chunks) is auto-picked by the factory.
SHAPES=(
    "ff_widest_ref     512   1024   bfloat16  1"
    "kv_proj_r16       4096  1024   bfloat8_b 2"
    "kv_proj_r64       4096  1024   bfloat8_b 8"
    "o_proj_r64        4096  4096   bfloat8_b 8"
    "ff1_r64           4096  14336  bfloat8_b 8"
    "ff2_r64           14336 4096   bfloat8_b 8"
    "qkv_bf8_r64       4096  12288  bfloat8_b 8"
)

for shape in "${SHAPES[@]}"; do
    read -r label K N DTYPE RECV <<<"$shape"
    cat > out/bench_env <<EOF
export BENCH_REPEATS=$REPS
export BENCH_K=$K
export BENCH_N=$N
export BENCH_DTYPE=$DTYPE
export BENCH_RECV_PER_BANK=$RECV
export BENCH_LAYERS=$REPS
EOF
    ~/bin-metal/tt-smi-reset > /dev/null 2>&1
    timeout 300 out/run_pytest.sh tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bench.py > /dev/null 2>&1

    dram=$(grep "\[dram_core\]" out/pytest.log | tail -1 | sed -n 's/.*per_matmul=\([0-9.]*us\) -> \([0-9.]*\) TFLOP\/s.*/\1, \2 TFLOP\/s/p')
    worker=$(grep "\[workercore\]" out/pytest.log | tail -1 | sed -n 's/.*per_matmul=\([0-9.]*us\) -> \([0-9.]*\) TFLOP\/s.*/\1, \2 TFLOP\/s/p')
    [ -z "$dram" ]   && { dram="FAIL"; grep -q "Out of Memory" out/pytest.log && dram="OOM"; grep -q "Timeout (20000 ms)" out/pytest.log && dram="HANG"; grep -q "TT_FATAL" out/pytest.log && [ "$dram" = "FAIL" ] && dram="FATAL"; }
    [ -z "$worker" ] && { worker="FAIL"; grep -q "Out of Memory" out/pytest.log && worker="OOM"; grep -q "Timeout (20000 ms)" out/pytest.log && worker="HANG"; grep -q "TT_FATAL" out/pytest.log && [ "$worker" = "FAIL" ] && worker="FATAL"; }

    row=$(printf '%-18s K=%-5s N=%-6s %-9s ring=%2d | DRAM: %-26s | Worker: %s' \
        "$label" "$K" "$N" "$DTYPE" "$((8 * RECV))" "$dram" "$worker")
    echo "$row" | tee -a "$RESULTS_FILE"
done

echo | tee -a "$RESULTS_FILE"
echo "Results saved to $RESULTS_FILE"
