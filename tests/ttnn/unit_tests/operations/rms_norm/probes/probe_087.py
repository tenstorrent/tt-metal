from ttnn.operations.rms_norm.perf_experiments.sumsq_reduce_merge.harness import main

main(zone_pass=("focus_r1_c3", "bshard_r16_c4"))
