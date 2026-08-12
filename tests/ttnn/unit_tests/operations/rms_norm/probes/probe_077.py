import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/worksplit_retune",
)
import sweep

sweep.main(
    ["prefill1024", "prefill2304", "prefill5120", "prefill7168"], gs=["rule"], rule=sweep.rule_two_stage_gain, reps=2
)
