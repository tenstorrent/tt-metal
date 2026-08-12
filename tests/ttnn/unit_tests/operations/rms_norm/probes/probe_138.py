import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/apply_lifecycle",
)
import ttnn
from bakeoff import main

device = ttnn.open_device(device_id=0)
opts = ["baseline", "perchunk_both_blk2", "perchunk_both_blk3", "perchunk_both_blk4"]
try:
    main(device, [(1, 3), (1, 2)], opts, iters=(1, 21))
    main(device, [(1, 3)], opts, iters=(1, 21), grid=(8, 8))
    main(device, [(32, 4)], ["baseline", "perchunk_both_blk4"], iters=(1, 21), grid=(8, 8))
finally:
    ttnn.close_device(device)
