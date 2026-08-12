import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/apply_lifecycle",
)
import ttnn
from bakeoff import main

device = ttnn.open_device(device_id=0)
opts = ["strided_blk1", "strided_blk4", "strided_blk8"]
try:
    main(device, [(32, 4), (16, 4), (1, 112)], opts, iters=(1, 21))
finally:
    ttnn.close_device(device)
