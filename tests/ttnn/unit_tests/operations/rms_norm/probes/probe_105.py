import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/apply_fusion",
)
from bakeoff import main

# A) attribute the blk6 win: is it the DEST-lane blocking, or the bulk output lifecycle
#    that blocking forces?
main(
    shapes=[(1, 16), (3, 32), (16, 4), (1, 112)],
    options=["baseline_bulk", "baseline_blk2", "baseline_blk4"],
    iters=(1, 11),
)
# B) the FP32 corner (float32 activations + fp32_dest_acc_en=True), same kernel path.
main(
    shapes=[(1, 3), (1, 16)],
    options=[
        "baseline",
        "baseline_bulk",
        "baseline_blk2",
        "baseline_blk4",
        "fused_rstd",
        "fused_rstd_bulk",
        "fused_rstd_blk2",
        "fused_rstd_blk4",
        "fused_sfpu",
        "fold_gamma",
    ],
    iters=(1, 11),
    dtype="fp32",
)
