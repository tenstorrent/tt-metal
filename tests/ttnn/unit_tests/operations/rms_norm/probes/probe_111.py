from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# separates "multi-block" from "110 cores" for the sum_mcast reading
main(["wshard_7x4_b4"], ["baseline", "sum_mcast", "allgather"])
main(["small_4x2_r4_b2"], ["baseline", "sum_mcast"])
