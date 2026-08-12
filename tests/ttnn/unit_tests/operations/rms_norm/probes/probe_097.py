from ttnn.operations.rms_norm.perf_experiments.cskip_finalize import bench

# DOMAIN question: does the vector-level scope survive fp32 stat tiles
# (fp32_dest_acc_en=True, the op's other stat-pipeline precision)?
bench.main(tile_counts=(1, 16), fp32=True, tag_prefix="fp32_")
