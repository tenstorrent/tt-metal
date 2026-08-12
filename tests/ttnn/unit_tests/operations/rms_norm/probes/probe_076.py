import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/apply_fusion",
)
from bakeoff import main

opts = [
    "baseline",
    "baseline_rc2off",
    "baseline_blk8",
    "baseline_blk8_rc2off",
    "fused_rstd",
    "fused_rstd_blk8",
    "fused_gamma_blk8",
    "fused_sfpu_blk4",
    "fold_gamma_blk8",
]
main(shapes=[(1, 16), (3, 32), (16, 4), (1, 112)], options=opts, iters=(1, 11))
