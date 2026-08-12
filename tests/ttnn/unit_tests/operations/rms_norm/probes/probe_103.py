from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# the BLOCK-sharded geometry (8 concurrent 8-core groups, 16 tile-rows, 2 blocks)
main(["bshard_8x1_r16"], ["baseline", "sum_mcast", "allgather", "no_collective_ablation"])
main(["bshard_8x1_r16_b1"], ["baseline", "sum_mcast", "allgather", "flat_allgather"])
# slot reuse at the focus geometry + a second read of the focus triple (noise check)
main(["focus_11x10_b4"], ["baseline", "sum_mcast", "allgather"])
main(["focus_11x10"], ["baseline", "sum_mcast", "flat_allgather"])
