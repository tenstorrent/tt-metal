import ttnn
import ttnn.operations.mhc_pre.mhc_pre as op

op.SUPPORTED["dtype"] = [ttnn.float32, ttnn.bfloat16]
op.SUPPORTED["weight_dtype"] = [ttnn.float32, ttnn.bfloat16]
from eval.golden_tests.mhc_pre.helpers import run_mhc_pre

device = ttnn.open_device(device_id=0)
try:
    for xs in [(32, 128), (1, 100, 4096), (1, 1, 640, 7168), (1, 1, 640, 28672), (1, 28672)]:
        for dt in (ttnn.bfloat16, ttnn.float32):
            for wd in (ttnn.float32, ttnn.bfloat16):
                if dt == ttnn.float32 and wd == ttnn.float32:
                    continue
                try:
                    run_mhc_pre((xs, (xs[-1], 24)), dtype=dt, layout=ttnn.TILE_LAYOUT, weight_dtype=wd, device=device)
                    print("PROBE PASS", xs, dt, wd)
                except Exception as e:
                    print("PROBE FAIL", xs, dt, wd, str(e)[:300])
finally:
    ttnn.close_device(device)
