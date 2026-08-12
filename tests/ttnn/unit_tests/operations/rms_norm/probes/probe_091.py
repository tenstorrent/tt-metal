from ttnn.operations.rms_norm.perf_experiments.cskip_finalize import bench

bench.main(tile_counts=(4, 32))
print("\n\n########## REPEAT of N=16 (noise check on the key variants) ##########")
bench.main(variants=["copy_only", "c_pair", "cskip_pair", "cskip_fused"], tile_counts=(16,), tag_prefix="rep_")
