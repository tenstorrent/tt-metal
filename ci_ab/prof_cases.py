# Scratch, not for merge: device time of the ttnn ops the lever kernels reach, one device session; a signpost precedes
# the R3_NMEAS measured calls of every case, after 2 warm-up calls. usage: python -m tracy -r -p ... prof_cases.py <cases>
import gc
import os
import sys

import torch
import ttnn
from tracy import signpost

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prof_misc2 as P2  # noqa: E402
import prof_oc_misc as P1  # noqa: E402
import bits_oc_misc as B  # noqa: E402

CASES = dict(P2.CASES)
CASES.update(P1.CASES)


def _ptc(m, w, dt):
    def setup(d):
        x = torch.randn((1, 1, m, w), generator=torch.Generator().manual_seed(5)).to(torch.float32 if dt == ttnn.float32 else torch.bfloat16)
        return (ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG),)

    def run(d, x):
        return ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x)

    return setup, run


CASES["per_token_cast_bf16_2048x7168"] = _ptc(2048, 7168, ttnn.bfloat16)
CASES["per_token_cast_fp32_1024x7168"] = _ptc(1024, 7168, ttnn.float32)
N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        for name in sys.argv[1:]:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
