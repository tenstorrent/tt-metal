from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# strict correctness gate: per-core-DISTINCT partials on small boxes (a dropped or
# mis-slotted contribution is 12-25% of the sum there), plus the multi-block +
# rows_t>1 path that exercises slot reuse and the ag_free ack.
main(
    ["small_4x2_r4_b2", "small_3x3", "small_2x2", "col_1x8"],
    ["baseline", "allgather", "flat_allgather", "no_collective_ablation"],
    partial_kind="distinct",
)
