from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

main(
    ["focus_11x10", "wshard_8x1", "wshard_7x4"],
    ["baseline", "baseline_nohs", "allgather", "flat_allgather", "no_collective_ablation"],
)
