# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-local SDPA with the head-broadcast mask cache (tt/custom_ops/sdpa_mask_cache) vs ttnn's SDPA.

For every batch / sequence length the model runs masked (B 8 / 32, S 128 / 256 / 512), each output memory config
(L1, DRAM) and each Q / K / V placement (DRAM; L1 too where it fits): random Q / K / V, a packed-rows mask (each
batch row cut into 2-4 segments at random token boundaries: 0 inside a segment, -1e9 across, bfp4) and its block map
(1 where the Q chunk and K chunk share a segment). Checks the two outputs are bit-identical, then times both
(traced, 10 replays, best of 3) at 32x512 and 8x512.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_sdpa_mask_cache.py
"""
import argparse
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.sdpa_mask_cache import sdpa_mask_cache
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.sdpa_mask_cache.op import (
    build_kv_chains,
    global_q_range,
)
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import sdpa_chunks

REPS = 10
NQH, NKH, D = 32, 8, 64
MEMS = {"L1": ttnn.L1_MEMORY_CONFIG, "DRAM": ttnn.DRAM_MEMORY_CONFIG}


def packed_mask(B, S, qc, kc, gen):
    seg = torch.zeros(B, S, dtype=torch.int64)
    for b in range(B):
        n = int(torch.randint(2, 5, (1,), generator=gen))
        cuts = sorted(torch.randperm(S - 1, generator=gen)[: n - 1].add(1).tolist())
        for c in cuts:
            seg[b, c:] += 1
    same = seg[:, None, :, None] == seg[:, None, None, :]  # [B, 1, S, S]
    mask = torch.where(same, 0.0, -1e9)
    blocks = same.reshape(B, 1, S // qc, qc, S // kc, kc).any(dim=5).any(dim=3).to(torch.int32)
    return mask, blocks


def make_inputs(dev, B, S, in_mem, gen):
    qc, kc = sdpa_chunks(B, S)
    th = {
        "q": torch.randn(B, NQH, S, D, generator=gen),
        "k": torch.randn(B, NKH, S, D, generator=gen),
        "v": torch.randn(B, NKH, S, D, generator=gen),
    }
    t = {
        n: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=in_mem)
        for n, x in th.items()
    }
    mask, blocks = packed_mask(B, S, qc, kc, gen)
    t["mask"] = ttnn.from_torch(
        mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    t["map"] = ttnn.from_torch(
        blocks, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return t, qc, kc, int(blocks.sum()), blocks.numel(), (th, mask)


def reference(th, mask):
    """torch SDPA in fp32 on the same (bf16-rounded) inputs, heads concatenated: [B, 1, S, NQH * D]."""
    q, k, v = (th[n].to(torch.bfloat16).float() for n in ("q", "k", "v"))
    k, v = (x.repeat_interleave(NQH // NKH, dim=1) for x in (k, v))
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
    B, _, S, _ = o.shape
    return o.permute(0, 2, 1, 3).reshape(B, 1, S, NQH * D)


def chain_summary(dev, B, S, qc):
    """The K/V chains the op builds (B <= 8 only; above, K/V retention skips them)."""
    if B > 8:
        return "no chains (retention, B > 8)"
    g = dev.compute_with_storage_grid_size()
    n = g.x * g.y
    total = B * NQH * (S // qc)
    rng = lambda i: global_q_range(i, 0, total % n, total // n, 1, total)
    info, mcast = build_kv_chains(dev, g.x, g.y, B, NQH, NQH // NKH, S // qc, 1, True, rng)
    inj = sum(c.is_injector for c in info)
    part = sum(c.participates for c in info)
    return (
        f"{inj} chains over {part} cores, multicast {'on' if mcast else 'off'} (dropped if K/V in L1 and no multicast)"
    )


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def runners(dev, t, qc, kc, out_mem):
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
        k_splits=1,
        segmented_accumulation=False,
    )
    kw = dict(
        attn_mask=t["mask"],
        attn_mask_block_map=t["map"],
        is_causal=False,
        scale=1.0,
        output_concat_heads=True,
        compute_kernel_config=ckc,
        program_config=pc,
        memory_config=out_mem,
    )
    stock = lambda: ttnn.transformer.scaled_dot_product_attention(t["q"], t["k"], t["v"], **kw)
    local = lambda: sdpa_mask_cache(t["q"], t["k"], t["v"], **kw)
    return stock, local


def traced_us(dev, fn):
    ttnn.deallocate(fn())
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(REPS):  # each output freed before the next call, as in the model (10 x 64 MB do not fit L1)
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t0) / REPS * 1e6)
    ttnn.release_trace(dev, tid)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["8x128", "8x256", "8x512", "32x128", "32x256", "32x512"])
    ap.add_argument("--time-shapes", nargs="*", default=["32x512", "8x512"])
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000)
    gen = torch.Generator().manual_seed(0)
    fails = 0
    if not args.no_check:
        for shape in args.shapes:
            B, S = map(int, shape.split("x"))
            # Q / K / V in L1 only where they fit next to an L1 output (B 8).
            for in_name in ("DRAM", "L1") if B <= 8 else ("DRAM",):
                t, qc, kc, live, nblk, host = make_inputs(dev, B, S, MEMS[in_name], gen)
                ref = reference(*host)
                print(f"{shape} in {in_name}: {chain_summary(dev, B, S, qc)}", flush=True)
                for out_name, out_mem in MEMS.items():
                    stock, local = runners(dev, t, qc, kc, out_mem)
                    a = ttnn.to_torch(stock())
                    b_ = ttnn.to_torch(local())
                    # a second call reuses the cached program with new output addresses
                    c = ttnn.to_torch(local())
                    ok = torch.equal(a, b_) and torch.equal(a, c)
                    fails += not ok
                    diff = (a.float() - b_.float()).abs().max().item()
                    p = pcc(ref, b_.float())
                    print(
                        f"{shape} q{qc} k{kc} in {in_name:4s} out {out_name:4s} blocks {live}/{nblk}: "
                        f"{'BIT-EXACT' if ok else 'MISMATCH'} (max |diff| {diff:.3g}, local vs torch PCC {p:.5f})",
                        flush=True,
                    )
                for x in t.values():
                    ttnn.deallocate(x)
        print(f"check: {fails} mismatching case(s)", flush=True)
    for shape in args.time_shapes:
        B, S = map(int, shape.split("x"))
        t, qc, kc, live, nblk, _ = make_inputs(dev, B, S, ttnn.DRAM_MEMORY_CONFIG, gen)
        for out_name, out_mem in MEMS.items():
            stock, local = runners(dev, t, qc, kc, out_mem)
            ts, tl = traced_us(dev, stock), traced_us(dev, local)
            print(
                f"time {shape} out {out_name}: stock {ts:.1f} us, local {tl:.1f} us ({ts / tl:.2f}x), blocks {live}/{nblk}",
                flush=True,
            )
        for x in t.values():
            ttnn.deallocate(x)
    ttnn.close_device(dev)
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
