# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cost of the bidirectional padding mask in SDPA: mask dtype and attn_mask_block_map.

B sequences of random real length in [S/4, S] padded to S. For each variant:
max |err| on the real query rows vs torch per-sequence attention, and traced time.
"""

import argparse
import time

import torch

import ttnn


def timed(device, fn, reps=10):
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
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--q", type=int, default=128)
    ap.add_argument("--k", type=int, default=512)
    args = ap.parse_args()
    B, S, qc, kc = args.B, args.S, args.q, args.k
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    try:
        g = torch.Generator().manual_seed(0)
        lens = [int(x) for x in torch.randint(S // 4, S + 1, (B,), generator=g)]
        q, k, v = torch.randn(B, 32, S, 64), torch.randn(B, 8, S, 64), torch.randn(B, 8, S, 64)
        tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for t in (q, k, v))
        m = torch.zeros(B, 1, S, S)
        bm = torch.zeros(B, 1, (S + qc - 1) // qc, (S + kc - 1) // kc, dtype=torch.int32)
        for b, n in enumerate(lens):
            m[b, :, :, n:] = -1e9
            # A block is needed when its query rows hold a real token and its keys hold one.
            for i in range(bm.shape[2]):
                for j in range(bm.shape[3]):
                    bm[b, 0, i, j] = int(i * qc < n and j * kc < n)
        bmap = ttnn.from_torch(bm, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
        )
        kr, vr = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
        for mdt in ("bfloat16", "bfloat8_b", "bfloat4_b"):
            try:
                tm = ttnn.from_torch(m, dtype=getattr(ttnn, mdt), layout=ttnn.TILE_LAYOUT, device=dev)
            except Exception as e:
                print(f"mask {mdt}: cannot build ({str(e)[:80]})")
                continue
            for use_map in (False, True):
                fn = lambda: ttnn.transformer.scaled_dot_product_attention(
                    tq,
                    tk,
                    tv,
                    is_causal=False,
                    attn_mask=tm,
                    attn_mask_block_map=bmap if use_map else None,
                    scale=0.125,
                    compute_kernel_config=ckc,
                    program_config=pc,
                )
                try:
                    out = ttnn.to_torch(fn()).float()
                    worst = 0.0
                    for b, n in enumerate(lens):
                        ref = (
                            torch.softmax(q[b : b + 1, :, :n] @ kr[b : b + 1, :, :n].transpose(-1, -2) * 0.125, -1)
                            @ vr[b : b + 1, :, :n]
                        )
                        worst = max(worst, (out[b : b + 1, :, :n] - ref).abs().max().item())
                    print(
                        f"mask {mdt:9s} block_map={use_map!s:5s}  {timed(dev, fn):9.1f} us  max|err| {worst:.4f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"mask {mdt:9s} block_map={use_map!s:5s}  ERR {str(e)[:120]}", flush=True)
            ttnn.deallocate(tm)
        plain = lambda: ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, scale=0.125, compute_kernel_config=ckc, program_config=pc
        )
        print(
            f"no mask (wrong for padded rows, timing floor) {timed(dev, plain):9.1f} us   lens mean {sum(lens)/B:.0f}"
        )
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
