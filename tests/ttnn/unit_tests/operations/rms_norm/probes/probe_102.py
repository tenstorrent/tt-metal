from ttnn.operations.rms_norm.perf_experiments.cskip_finalize import bench

# The FOCUS SHAPE (1,1,32,7168) has ONE tile-row, i.e. rows_t == 1, so N=1 decides it.
# Three fresh runs: at N=1 the whole-kernel number sits on the per-dispatch floor, so the
# math-thread zone is the only signal and one-shot variation has to be bounded.
V = ["copy_only", "c_pair", "cskip_pair", "c_fused", "cskip_fused", "cskip_fused_bitexact"]
for trial in (1, 2, 3):
    print(f"\n\n########## TRIAL {trial} (N=1,2) ##########")
    bench.main(variants=V, tile_counts=(1, 2), tag_prefix=f"t{trial}_")
