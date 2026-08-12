import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/worksplit_retune",
)
import sweep

# (a) why does G=11 beat G=22 at W=1024? per-zone
sweep.main(["decode1024"], gs=[11, 22], default_first=False, zones=True)
# (b) map the flat-vs-two-stage turn across decode widths
sweep.main(["decode1280", "decode1536", "decode3072"], gs=[11, 22, 55], reps=2)
# (c) price G=11 on the prefill end (the rule must not touch these)
sweep.main(["prefill1024", "prefill7168"], gs=[11], reps=2)
