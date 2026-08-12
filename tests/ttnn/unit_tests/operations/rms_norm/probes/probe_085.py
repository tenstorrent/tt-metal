from ttnn.operations.rms_norm.perf_experiments.cskip_finalize import bench

bench.main(tile_counts=(1, 16))
