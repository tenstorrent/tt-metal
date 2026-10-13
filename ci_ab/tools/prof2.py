# Op-level device time of COL reduces with one input tile per output (Qwen 3.6 MoE routing max), one side per process.
# usage: python -m tracy -r -p --no-web-server -o OUT prof2.py <side> <run>
import hashlib
import sys

import torch
import ttnn
from tracy import signpost

REPS = 6
CASES = [
    ("q36_prefill4096_max_1x128x32x64", (1, 128, 32, 64), "max"),
    ("q36_prefill2048_max_1x64x32x64", (1, 64, 32, 64), "max"),
    ("q36_decode_b32_max_1x1x32x64", (1, 1, 32, 64), "max"),
    ("colsum_1x1x32x8192", (1, 1, 32, 8192), "sum"),
]


def main():
    side, run = sys.argv[1], sys.argv[2]
    device = ttnn.open_device(device_id=0)
    for name, shape, op in CASES:
        torch.manual_seed(58709)
        x = ttnn.from_torch(torch.rand(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        fn = ttnn.max if op == "max" else ttnn.sum
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
