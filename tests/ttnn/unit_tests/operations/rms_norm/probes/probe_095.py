import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/apply_fusion",
)
from bakeoff import main

opts = ["fused_rstd_blk4", "fused_rstd_blk5", "fused_rstd_blk6", "fused_rstd_blk8"]
main(shapes=[(1, 8), (1, 16)], options=opts, iters=(1, 11))
