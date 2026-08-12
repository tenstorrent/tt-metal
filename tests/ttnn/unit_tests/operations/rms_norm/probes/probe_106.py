from ttnn.operations.rms_norm.perf_experiments.allgather_combine.harness import main

# the one non-flat sum_mcast reading needs a repeat (11% > the ~5% noise band here)
for _ in range(3):
    main(["focus_11x10_b4"], ["baseline", "sum_mcast"])
# domain sweep around the perf geometries
main(["wshard_9x1", "wshard_8x4", "col_1x8"], ["baseline", "sum_mcast", "allgather"])
