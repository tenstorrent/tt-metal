from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# a 1-core-wide box: the op picks the FLAT tree there (leader == root), which the bench
# now mirrors. Correctness first (distinct partials), then perf.
main(["col_1x8"], ["baseline", "sum_mcast", "allgather", "flat_allgather"], partial_kind="distinct")
main(["wshard_8x1", "wshard_7x4_b4"], ["baseline", "sum_mcast"])
