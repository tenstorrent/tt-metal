import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/apply_fusion",
)
from bakeoff import main

# C) fp32 at HiFi4 — the fidelity that actually resolves an fp32 operand. Does routing
#    DEST through a Src register (dest reuse) lose mantissa the L1 round trip keeps?
main(
    shapes=[(1, 16)],
    options=["baseline", "fused_rstd", "fused_rstd_blk2", "fused_sfpu"],
    iters=(1, 11),
    dtype="fp32hifi4",
)
# D) the focus geometry on a realistic grid (64 cores), to check the single-core
#    conclusion survives contention.
main(
    shapes=[(1, 3)],
    options=["baseline", "baseline_bulk", "baseline_blk2", "fused_rstd", "fused_rstd_blk2"],
    iters=(1, 21),
    grid=(8, 8),
)
