# Op-level device time and logical output bits of reduces over H (Qwen 3.6 MoE routing max and others), one side per process.
# usage: python -m tracy -r -p --no-web-server -o OUT prof2.py <side> <run>
import hashlib
import sys

import torch
import ttnn
from tracy import signpost

REPS = 6
CASES = [
    ("q36_prefill4096_max_1x128x32x64", (1, 128, 32, 64), "max", ttnn.bfloat16),
    ("q36_prefill2048_max_1x64x32x64", (1, 64, 32, 64), "max", ttnn.bfloat16),
    ("q36_decode_b32_max_1x1x32x64", (1, 1, 32, 64), "max", ttnn.bfloat16),
    ("colsum_1x1x32x8192", (1, 1, 32, 8192), "sum", ttnn.bfloat16),
    ("colsum_fp32_1x1x32x8192", (1, 1, 32, 8192), "sum", ttnn.float32),
    ("colsum_ht4_1x1x128x4096", (1, 1, 128, 4096), "sum", ttnn.bfloat16),
    ("colmean_1x1x1024x768", (1, 1, 1024, 768), "mean", ttnn.bfloat16),
]


def main():
    side, run = sys.argv[1], sys.argv[2]
    device = ttnn.open_device(device_id=0)
    for name, shape, op, dt in CASES:
        torch.manual_seed(58709)
        x = ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        fn = {"max": ttnn.max, "sum": ttnn.sum, "mean": ttnn.mean}[op]
        y = None
        for k in range(REPS):
            ttnn.synchronize_device(device)
            signpost(header=f"{name}#{k}")
            y = fn(x, dim=2, keepdim=True)
        ttnn.synchronize_device(device)
        o = ttnn.to_torch(y).contiguous().float()
        torch.save(o, f"/tmp/hr_out/{side}_{run}_{name}.pt")
        ttnn.ReadDeviceProfiler(device)
        print(f"CASE {name} ok BITS {side} r{run} {hashlib.sha256(o.numpy().tobytes()).hexdigest()[:16]} shape {tuple(o.shape)}", flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
