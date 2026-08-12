from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# does the sum_mcast win scale with the block count? (prefill stand-in)
main(["focus_11x10_b16"], ["baseline", "sum_mcast", "no_collective_ablation"])
# col_1x8 now uses the op's own tree choice for nx == 1 (flat, not a self-write level)
main(["col_1x8"], ["baseline", "sum_mcast", "allgather"], partial_kind="distinct")
# and the 7x4 box with 4 blocks, to separate "multi-block" from "110 cores"
main(["wshard_7x4_b4"], ["baseline", "sum_mcast"])
