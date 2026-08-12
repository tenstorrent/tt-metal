import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/worksplit_retune",
)
import sweep

# the ROWS axis of the rule's domain
sweep.main(["rows128_1024", "rows320_1024"], gs=["rule", 11, 22], rule=sweep.rule_two_stage_gain, reps=2)
