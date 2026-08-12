from ttnn.operations.rms_norm.perf_experiments.sumsq_reduce_merge.harness import main

main(
    variants=["baseline", "baseline_onechain", "merged_hoist", "merged_cvalid", "merged_noscale"],
    zone_pass=("focus_r1_c3", "bshard_r16_c4"),
)
