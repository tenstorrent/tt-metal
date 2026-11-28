# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Long single-sequence SDPA (32 q / 8 kv heads, head_dim 64): padding via dense bfp4 mask + block map
vs on-device windows (cu_window_seqlens [0, n, S]) vs no mask, over q/k chunk sizes and fidelity.

    python .../probe_long_sdpa.py --S 8192 --n 7000
"""

import argparse
import time

import torch

import ttnn


def timed(device, fn, reps=5):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(reps)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    t = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t) / reps * 1e6
    ttnn.release_trace(device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--n", type=int, default=7000)
    args = ap.parse_args()
    S, n = args.S, args.n
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    try:
        torch.manual_seed(0)
        q, k, v = torch.randn(1, 32, S, 64), torch.randn(1, 8, S, 64), torch.randn(1, 8, S, 64)
        tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for t in (q, k, v))
        kr, vr = k[:, :, :n].repeat_interleave(4, 1), v[:, :, :n].repeat_interleave(4, 1)
        ref = torch.softmax(q[:, :, :n] @ kr.transpose(-1, -2) * 0.125, -1) @ vr
        m = torch.zeros(1, 1, S, S)
        m[..., n:] = -1e9
        mask = ttnn.from_torch(m, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev)
        cu = ttnn.from_torch(
            torch.tensor([0, n, S], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )
        for fid in ("HiFi4", "HiFi2"):
            ckc = ttnn.init_device_compute_kernel_config(
                dev.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fid),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            for qc, kc in ((256, 256), (256, 512), (512, 256), (512, 512), (128, 512)):
                pc = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
                    q_chunk_size=qc,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                )
                bm = torch.zeros(1, 1, S // qc, S // kc, dtype=torch.int32)
                bm[0, 0, : -(-n // qc), : -(-n // kc)] = 1
                bmap = ttnn.from_torch(bm, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
                variants = {
                    "mask+map": dict(attn_mask=mask, attn_mask_block_map=bmap),
                    "windowed": dict(cu_window_seqlens=cu),
                    "nomask": dict(),
                }
                for name, kw in variants.items():
                    fn = lambda: ttnn.transformer.scaled_dot_product_attention(
                        tq, tk, tv, is_causal=False, scale=0.125, compute_kernel_config=ckc, program_config=pc, **kw
                    )
                    try:
                        out = ttnn.to_torch(fn()).float()[:, :, :n]
                        err = (out - ref).abs().max().item() if name != "nomask" else float("nan")
                        print(
                            f"S={S} n={n} {fid} q{qc} k{kc} {name:9s} {timed(dev, fn):9.1f} us  err {err:.3g}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"S={S} n={n} {fid} q{qc} k{kc} {name:9s} ERR {str(e)[:90]}", flush=True)
                ttnn.deallocate(bmap)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
