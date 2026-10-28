"""Kernel-time sweep for HW variance: main's composite, ttnn.var over H and W, and mean(square(y - mean(y))).

Run under `python -m tracy -r -p -v`; sum_signposts.py totals DEVICE KERNEL DURATION per signpost.
"""
import os

import torch
import ttnn
from tracy import signpost

SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 64),
    (1, 1, 128, 128),
    (1, 1, 256, 256),
    (1, 1, 256, 512),
    (1, 1, 512, 512),
    (1, 1, 1024, 1024),
    (1, 1, 2048, 2048),
    (1, 1, 32, 8192),
    (1, 1, 8192, 32),
    (1, 2, 256, 256),
    (1, 4, 256, 256),
    (1, 16, 256, 256),
    (1, 64, 256, 256),
    (1, 2, 512, 512),
    (1, 8, 512, 512),
    (1, 64, 64, 64),
    (1, 128, 64, 64),
    (1, 256, 32, 32),
    (4, 32, 128, 128),
    # Held out: not used to pick the 64-tile threshold.
    (1, 1, 4096, 64),
    (1, 1, 2048, 128),
    (1, 1, 1024, 32),
    (1, 1, 512, 128),
    (1, 1, 128, 2048),
    (1, 1, 96, 4096),
    (1, 3, 512, 256),
    (2, 3, 320, 320),
    (1, 32, 256, 256),
    (1, 96, 128, 128),
    (16, 1, 512, 512),
    (1, 1, 33, 33),
    (1, 2, 30, 70),
    (1, 1, 300, 700),
]
REPEATS = 3
SQUARE = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]


def mean_of_squared_deviation(x):
    mean = ttnn.mean(x, [2, 3], keepdim=True)
    return ttnn.mean(ttnn.subtract(x, mean, activations=SQUARE), [2, 3], keepdim=True)


VARIANTS = {
    "var_hw": lambda x: ttnn.var_hw(x),
    "std_hw": lambda x: ttnn.std_hw(x),
}
only = os.environ.get("VARIANTS")
if only:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in only.split(",")}
dtypes = os.environ.get("DTYPES", "bf16,fp32").split(",")

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
for dtype_name, dtype in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
    if dtype_name not in dtypes:
        continue
    for shape in SHAPES:
        x = ttnn.from_torch(torch.randn(shape) * 3 + 1, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        for name, op in VARIANTS.items():
            try:
                op(x)
                ttnn.synchronize_device(device)
            except Exception as e:
                print(f"SKIP {name} {dtype_name} {shape}: {str(e).splitlines()[0][:200]}", flush=True)
                continue
            tag = f"{name}|{dtype_name}|{'x'.join(map(str, shape))}"
            for _ in range(REPEATS):
                signpost(f"BEGIN {tag}")
                op(x)
                ttnn.synchronize_device(device)
                signpost(f"END {tag}")
        x.deallocate()
        ttnn.ReadDeviceProfiler(device)
ttnn.close_device(device)
