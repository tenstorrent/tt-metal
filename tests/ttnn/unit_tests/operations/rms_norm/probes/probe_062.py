from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

main(["small_2x2"], ["baseline", "allgather", "flat_allgather"])
