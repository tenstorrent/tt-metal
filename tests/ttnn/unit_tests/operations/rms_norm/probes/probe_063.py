import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/apply_fusion",
)
from bakeoff import main, OPTIONS

main(shapes=[(1, 3)], options=list(OPTIONS), iters=(1, 21))
