import sys

sys.path.insert(
    0,
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/gather_transport",
)
import bench_op as B

B.sweep(["base", "flush"], names=["F_w7168_28c", "G_w1024_8c", "G_w2304_9c", "G_w5120_32c"])
