from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# sum_mcast = MODE 4: the baseline's gather tree and its ONE broadcast, but the root
# broadcasts the raw SUM and every core finalizes locally.
main(["small_4x2_r4_b2", "small_3x3", "col_1x8"], ["sum_mcast"], partial_kind="distinct")
main(["focus_11x10", "wshard_8x1", "wshard_7x4", "bshard_8x1_r16"], ["baseline", "sum_mcast"])
