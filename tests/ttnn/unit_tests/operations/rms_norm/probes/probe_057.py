import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/worksplit_retune",
)
import sweep

# focus shape: 3 repeats of the top-4 G values, WITH per-zone dumps
sweep.main(["decode7168"], gs=[110, 55, 22, 11] * 3, default_first=False, zones=True)
# the two marginal decode points + the marginal prefill point: repeats vs the default pick
sweep.main(["decode1024", "decode2304"], gs=[11, 22, 55] * 3)
