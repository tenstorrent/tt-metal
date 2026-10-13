# Main against this head at op level: the ROW MAX callers of the PR body's device time table. One side per process.
# usage: python -m tracy -r -p --no-web-server -o OUT prof_mh.py <side> <run>
import hashlib
import sys

import torch
import ttnn
from tracy import signpost

import rcu
from prof7 import sdpa_prefill_case

REPS = 5
NAMES = [
    "pr_sbertsm_8x12x384x384", "smvit_10x11x224x224_bfp8", "smsh_8x4x384x384_bf16", "smst_1x8x1024x1024_bf16",
    "pr_t5sm_1x16x512x512", "pr_clipsm_1x12x77x77", "sm_1x16x256x256_bfp8", "msm_1x1x256x1024",
    "sdpad_b32_s2048_k512_hifi2fp32", "sdpad_b32_s2048_k512_hifi4fp32", "pr_sdpad_b32_s1024_hifi4fp32",
]


def main():
    side, run = sys.argv[1], sys.argv[2]
    cases = dict(rcu.CASES)
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    todo = [(n, lambda d, v, s=cases[n]: s(d, v)) for n in NAMES]
    todo.append(("sdpap_llama8b_s4096_hifi4fp32", lambda d, v: sdpa_prefill_case(32, 8, 4096, 128, 256, ttnn.MathFidelity.HiFi4, True)(d)))
    for name, setup in todo:
        torch.manual_seed(58707)
        try:
            fn, keep = setup(device, "dflt")
            y = None
            for k in range(REPS):
                ttnn.synchronize_device(device)
                signpost(header=f"{name}#{k}")
                y = fn()
            ttnn.synchronize_device(device)
            o = ttnn.to_torch(y if y is not None else keep[0]).contiguous().float()
            torch.save(o, f"/tmp/hr_out/{side}_{run}_{name}.pt")
            ttnn.ReadDeviceProfiler(device)
            print(f"CASE {name} ok BITS {side} r{run} {hashlib.sha256(o.numpy().tobytes()).hexdigest()[:16]}", flush=True)
            del y, keep, o
        except Exception as e:
            print(f"CASE {name} FAIL {str(e).splitlines()[0][:300]}", flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
