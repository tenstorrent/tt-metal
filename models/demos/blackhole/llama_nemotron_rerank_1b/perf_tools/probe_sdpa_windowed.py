# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Padded-batch bidirectional SDPA two ways, same pair lengths:

  masked:   Q [B, 32, S, 64], K / V [B, 8, S, 64], bfp4 padding mask [B, 1, S, S] + attn_mask_block_map
            (what the model runs today)
  windowed: the batch flattened to one sequence, Q [1, 32, B*S, 64], K / V [1, 8, B*S, 64], no mask tensor;
            cu_window_seqlens puts each pair's real tokens in one window and its padding in another, so a
            real token only sees its own pair's real tokens (stock ttnn windowed SDPA)

Prints traced time per call and the max error on real tokens against torch.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_sdpa_windowed.py --shape 32x512
"""
import argparse
import time

import torch

import ttnn

REPS = 10


def timed(dev, fn):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t0) / REPS * 1e6)
    ttnn.release_trace(dev, tid)
    out = ttnn.to_torch(outs[0]).float()
    for o in outs:
        ttnn.deallocate(o)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="32x512")
    ap.add_argument("--full", action="store_true", help="every pair fills the bucket (no padding)")
    ap.add_argument("--chunks", default="128x512,128x256,256x256,128x128", help="windowed q x k chunks")
    ap.add_argument("--masked-chunks", default="128x512")
    args = ap.parse_args()
    B, S = map(int, args.shape.split("x"))
    torch.manual_seed(0)
    lens = [S] * B if args.full else torch.randint(S // 4, S + 1, (B,)).tolist()
    lens[0] = S  # the bucket is set by the longest pair
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    grid = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
    )
    q = torch.randn(B, 32, S, 64)
    k = torch.randn(B, 8, S, 64)
    v = torch.randn(B, 8, S, 64)

    # torch reference on real tokens (scale 1, GQA 4 q heads per kv head)
    ref = torch.zeros(B, 32, S, 64)
    for b, n in enumerate(lens):
        kk = k[b, :, :n].repeat_interleave(4, 0)
        vv = v[b, :, :n].repeat_interleave(4, 0)
        ref[b, :, :n] = torch.softmax(q[b, :, :n] @ kk.transpose(-1, -2), -1) @ vv

    def err(out_bhsd):
        return max((out_bhsd[b, :, :n] - ref[b, :, :n]).abs().max().item() for b, n in enumerate(lens))

    def to_dev(x, dt=ttnn.bfloat16):
        return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    print(f"shape {B}x{S}, real tokens {sum(lens)} of {B * S}", flush=True)

    # masked (model today)
    tq, tk, tv = to_dev(q), to_dev(k), to_dev(v)
    for spec in args.masked_chunks.split(","):
        qc, kc = map(int, spec.split("x"))
        valid = torch.zeros(B, S)
        for b, n in enumerate(lens):
            valid[b, :n] = 1
        mask = ((1 - valid) * -1e9)[:, None, None, :].expand(B, 1, S, S).contiguous()
        tmask = to_dev(mask, ttnn.bfloat4_b)
        bm = torch.zeros(B, 1, (S + qc - 1) // qc, (S + kc - 1) // kc, dtype=torch.int32)
        for b, n in enumerate(lens):
            bm[b, 0, : (n + qc - 1) // qc, : (n + kc - 1) // kc] = 1
            bm[b, 0, : (n + qc - 1) // qc, : n // kc] = 2
        tbm = ttnn.from_torch(bm, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, max_cores_per_head_batch=16
        )
        us, out = timed(
            dev,
            lambda: ttnn.transformer.scaled_dot_product_attention(
                tq,
                tk,
                tv,
                is_causal=False,
                attn_mask=tmask,
                attn_mask_block_map=tbm,
                scale=1.0,
                compute_kernel_config=ckc,
                program_config=pc,
            ),
        )
        print(f"masked   q{qc} k{kc}: {us:8.1f} us  max err {err(out):.4f}", flush=True)
        ttnn.deallocate(tmask)
        ttnn.deallocate(tbm)
    for t in (tq, tk, tv):
        ttnn.deallocate(t)

    # windowed: flatten the batch into one sequence
    flat = lambda x: x.permute(1, 0, 2, 3).reshape(1, x.shape[1], B * S, 64)
    tq, tk, tv = to_dev(flat(q)), to_dev(flat(k)), to_dev(flat(v))
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)  # real tokens
        if n < S:
            cu.append(cu[-1] + S - n)  # padding: its own window
    tcu = ttnn.from_torch(
        torch.tensor(cu, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    for spec in args.chunks.split(","):
        qc, kc = map(int, spec.split("x"))
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, max_cores_per_head_batch=16
        )
        try:
            us, out = timed(
                dev,
                lambda: ttnn.transformer.scaled_dot_product_attention(
                    tq,
                    tk,
                    tv,
                    is_causal=False,
                    cu_window_seqlens=tcu,
                    scale=1.0,
                    compute_kernel_config=ckc,
                    program_config=pc,
                ),
            )
        except Exception as e:
            print(f"windowed q{qc} k{kc}: failed: {str(e).splitlines()[0][:200]}", flush=True)
            continue
        out = out.reshape(32, B, S, 64).permute(1, 0, 2, 3)
        print(f"windowed q{qc} k{kc}: {us:8.1f} us  max err {err(out):.4f}", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
