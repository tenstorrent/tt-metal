import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/tests/ttnn/unit_tests/operations/rms_norm",
)
from perf_zone_harness import main

main(["decode7168"])
