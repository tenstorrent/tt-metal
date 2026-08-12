from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# strict correctness (distinct per-core partials, small boxes, multi-block slot reuse)
main(["small_4x2_r4_b2", "small_3x3"], ["baseline", "allgather", "sum_mcast"], partial_kind="distinct")
# the focus shape's group, then the width-sharded perf geometries
main(["focus_11x10"], ["baseline", "baseline_nohs", "allgather", "sum_mcast", "no_collective_ablation"])
main(["wshard_8x1", "wshard_7x4"], ["baseline", "sum_mcast", "allgather", "no_collective_ablation"])
