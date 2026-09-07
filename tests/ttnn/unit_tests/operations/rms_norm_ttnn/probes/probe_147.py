import sys

sys.path.insert(
    0,
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/gather_transport",
)
import bench_op as B

B.sweep(
    ["base", "flush"], names=["G_w5120_32c_gbr", "G_blk8192_64c", "G_blk7168_gbr", "N_int1024_prefill", "F_w7168_28c"]
)
