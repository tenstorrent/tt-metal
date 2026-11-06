#!/usr/bin/env bash
# The benchmark matrix on a 4x8 Blackhole Galaxy: dense, Sol (tau sweep, bf16 and LoFi / bf8 K/V),
# VSA (two sparsities, bf16 and LoFi / bf8 K/V) and cube sparse attention (four top-k ratios, bf16 and LoFi / bf8 K/V),
# at 5, 10 and 15 s of synthetic input. Sparse times are on-device scoring + attention.
#   LENGTHS="5 10" ./run_matrix.sh        # a subset of the lengths
#   INPUTS=/data/h3_qkv ./run_matrix.sh   # captured tensors instead of synthetic ones (one pass, no length loop)
#   OUT=results.jsonl ./run_matrix.sh     # results file (appended to)
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-results_tt.jsonl}
if [ -n "${INPUTS:-}" ]; then LENGTHS=input; fi
LENGTHS=${LENGTHS:-"5 10 15"}

run() {
    local src
    if [ "$len" = input ]; then src=(--inputs "$INPUTS"); else src=(--synthetic --seconds "$len"); fi
    python "$B/bench_tt.py" "${src[@]}" --output "$OUT" "$@"
}

for len in $LENGTHS; do
    run --attention-backend fa
    run --attention-backend fa --attention-backend-config '{"rung": "ladder"}'  # baseline at the VSA / cube rung
    for tau in 1.0 1.5 2.0; do
        run --attention-backend sol_attn --attention-backend-config "{\"tau\": $tau}"
        run --attention-backend sol_attn --attention-backend-config "{\"tau\": $tau}" --lofi --kv-dtype bf8
    done
    # VSA, exempt prefix tiles, on the shard-balanced tile layout
    for sparsity in 0.9 0.8; do
        run --attention-backend video_sparse_attn --attention-backend-config "{\"VSA_sparsity\": $sparsity}"
    done
    run --attention-backend video_sparse_attn --attention-backend-config '{"VSA_sparsity": 0.9}' --lofi --kv-dtype bf8
    # the ratios of SGLang's example schedule [1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5]; 1.0 steps are dense (fa)
    for ratio in 0.5 0.6 0.7 0.8; do
        run --attention-backend cube_sparse_attn --attention-backend-config "{\"topk_ratio\": $ratio}"
    done
    run --attention-backend cube_sparse_attn --attention-backend-config '{"topk_ratio": 0.5}' --lofi --kv-dtype bf8
done

python "$B/merge_results.py" "$OUT" --latest
