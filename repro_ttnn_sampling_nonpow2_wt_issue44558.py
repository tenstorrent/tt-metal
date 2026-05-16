# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
"""
Standalone TTNN repro: ttnn.sampling hangs when Wt = W/32 is not a power of 2.

  python repro_ttnn_sampling_nonpow2_wt.py --wt 5   # Qwen3-32B    -> HANG
  python repro_ttnn_sampling_nonpow2_wt.py --wt 7   # gpt-oss-120b -> HANG
  python repro_ttnn_sampling_nonpow2_wt.py --wt 4   # Llama-3.1-8B -> PASS

Hangs surface as `metal_context.cpp: Timeout detected` after ~5 minutes and
wedge the device; recover with `tt-smi -r`. Use ./run_repro.sh to sweep Wt
in {2..8} with per-shape timeouts + auto-reset between shapes.
"""

import argparse
import time

import torch
import ttnn

BATCH = 32  # ttnn.sampling requires exactly 32 users.
TILE_W = 32
K_PER_USER = 32

# Wt -> real model the tt-xla / vLLM device sampler hits at this Wt.
WT_NOTES = {
    2: "synthetic (pow2)",
    3: "synthetic (non-pow2)",
    4: "Llama-3.1-8B (pow2)",
    5: "Qwen3-32B (non-pow2)",
    6: "synthetic (non-pow2)",
    7: "gpt-oss-120b (non-pow2)",
    8: "synthetic (pow2)",
}


def run_one(wt: int, device):
    width = wt * TILE_W
    is_pow2 = (wt & (wt - 1)) == 0
    print(
        f"\n=== Wt={wt}  W={width}  {WT_NOTES.get(wt, '')}  " f"{'[expect PASS]' if is_pow2 else '[expect HANG]'}",
        flush=True,
    )

    torch.manual_seed(0)
    values = torch.randn([1, 1, BATCH, width], dtype=torch.bfloat16)
    indices = torch.arange(width, dtype=torch.int32).expand(1, 1, BATCH, width).contiguous()

    values_tt = ttnn.from_torch(values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    indices_tt = ttnn.from_torch(indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_tt = ttnn.from_torch(
        torch.full((BATCH,), K_PER_USER, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p_tt = ttnn.from_torch(
        torch.zeros(BATCH, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    temp_tt = ttnn.from_torch(
        torch.ones(BATCH, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    t0 = time.time()
    out = ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt, seed=12345)
    ttnn.to_torch(out)
    print(f"    returned in {(time.time() - t0) * 1000:.2f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wt", default="5", help="Wt to test (W = wt*32). 'all' walks 2..8. Default: 5 (Qwen3-32B).")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        wts = range(2, 9) if args.wt == "all" else [int(args.wt)]
        for wt in wts:
            run_one(wt, device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
