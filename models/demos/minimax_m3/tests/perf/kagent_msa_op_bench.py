# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip correctness + device-time harness for ``ttnn.transformer.sparse_sdpa_msa`` at the M3 4x4 prefill
shape (one SP-rank x TP-column shard: H=16 q heads, 1 KV head, S = chunk/4 query rows, causal, top-16 of
128-token blocks, bf8 K/V). Built for an autonomous kernel-optimizer loop over the op's JIT kernels
(ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/.../sparse_sdpa_msa_*): no host rebuild, no mesh, no
model weights.

Indices: ``--indices <msa_block_ids.pt>`` (captured from the real model by kagent_prefill_bench.py with
BENCH_CAPTURE_MSA; realistic locality between neighbouring query tokens) or synthetic (random causal, the op
unit tests' pattern, no locality).

Prints (one line each, machine-parsable):
  MSA_PCC min=<min over sampled tokens> mean=<..>      (vs the fp32 torch reference, causal, sampled tokens)
  MSA_OP_US median=<device kernel us> min=<> max=<> n=<iters>

Device time comes from the device profiler's C++ post-process (no tracy, no IOMMU needed):
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 are set below
before ttnn is imported.

Usage (prefill partition, the op runs on the first visible chip):
  tt-partition-run prefill --timeout 900 --log L -- python kagent_msa_op_bench.py \
      --S 1280 --T 56320 --chunk-start 51200 --rank 0 --indices <dump>/msa_block_ids.pt --layer 30 --iters 20
"""

import argparse
import os
import statistics
import sys

for _k, _v in {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
}.items():
    os.environ.setdefault(_k, _v)

import torch  # noqa: E402

import ttnn  # noqa: E402

BLK = 128


def build_indices(args, S, T, topk, gen):
    """[1, 1, S, topk] int32 natural block ids (sentinel -1 tail) for query rows chunk_start + rank*S + s."""
    q0 = args.chunk_start + args.rank * S
    if args.indices:
        cap = torch.load(args.indices)
        layer = args.layer if args.layer is not None else sorted(cap)[0]
        pos = cap[layer]["positions"]
        ids = cap[layer]["block_ids"][args.kv_head]  # [rows, topk]
        sel = (pos >= q0) & (pos < q0 + S)
        assert int(sel.sum()) == S, f"capture has {int(sel.sum())} rows in [{q0}, {q0 + S})"
        ids = ids[sel].clone()
        ids[ids >= 0xFFFFFFF0] = -1
        assert int(ids.max()) < T // BLK, f"captured block id {int(ids.max())} beyond T={T}"
        return ids.to(torch.int32).view(1, 1, S, topk), f"captured layer {layer} kv_head {args.kv_head}"
    idx = torch.full((1, 1, S, topk), -1, dtype=torch.int32)
    for s in range(S):
        p = q0 + s
        local = p // BLK
        pool = torch.randperm(local, generator=gen)[: topk - 1]
        chosen = torch.cat([pool, torch.tensor([local])]).sort().values
        idx[0, 0, s, : chosen.numel()] = chosen.to(torch.int32)
    return idx, "synthetic random causal"


def reference(q, k, v, idx, scale, tokens, chunk_start):
    out = []
    for s in tokens:
        row = idx[0, 0, s]
        blocks = row[row >= 0].long()
        keys = torch.cat([torch.arange(b * BLK, (b + 1) * BLK) for b in blocks.tolist()])
        ks, vs = k[0, 0, keys].float(), v[0, 0, keys].float()
        sc = (q[0, :, s].float() * scale) @ ks.T
        sc = sc.masked_fill(keys.view(1, -1) > chunk_start + s, float("-inf"))
        out.append(sc.softmax(-1) @ vs)
    return torch.stack(out, dim=1)  # [H, n, d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1280)
    ap.add_argument("--T", type=int, default=56320)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--chunk-start", type=int, default=51200)
    ap.add_argument("--rank", type=int, default=0, help="SP rank: query rows start at chunk_start + rank*S")
    ap.add_argument("--indices")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--kv-head", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pcc-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    S, T, H, d = a.S, a.T, a.H, 128
    gen = torch.Generator().manual_seed(a.seed)
    q = torch.randn(1, H, S, d, generator=gen)
    k = torch.randn(1, 1, T, d, generator=gen)
    v = torch.randn(1, 1, T, d, generator=gen)
    idx, src = build_indices(a, S, T, a.topk, gen)
    q_start = a.chunk_start + a.rank * S
    uniq = [len(set(idx[0, 0, s : s + 32].flatten().tolist()) - {-1}) for s in range(0, S, 32)]
    print(
        f"[msa-op] S={S} T={T} H={H} topk={a.topk} q_start={q_start} indices={src}; distinct blocks per 32-row "
        f"tile: mean {statistics.mean(uniq):.1f} (vs {32 * a.topk} if no reuse)",
        flush=True,
    )

    dev = ttnn.open_device(device_id=0)
    try:

        def rm(t, dt):
            return ttnn.from_torch(
                t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        tq = rm(q.float(), ttnn.bfloat16)
        tk = ttnn.from_torch(
            k.bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tv = ttnn.from_torch(
            v.bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ti = rm(idx, ttnn.uint32)
        scale = d**-0.5

        def run():
            return ttnn.transformer.sparse_sdpa_msa(
                tq, tk, tv, ti, scale=scale, block_size=BLK, chunk_start_idx=q_start
            )

        out = ttnn.to_torch(run())[:, :H].float()
        toks = sorted(set(torch.linspace(0, S - 1, a.pcc_tokens).long().tolist()))
        # reference on the bf8-rounded K/V the device actually used
        kq = ttnn.to_torch(tk).float()
        vq = ttnn.to_torch(tv).float()
        ref = reference(q.bfloat16(), kq, vq, idx, scale, toks, q_start)
        pccs = []
        for j, s in enumerate(toks):
            x, y = out[0, :, s].flatten(), ref[:, j].flatten()
            pccs.append(torch.corrcoef(torch.stack([x, y]))[0, 1].item())
        print(f"MSA_PCC min={min(pccs):.6f} mean={statistics.mean(pccs):.6f} tokens={len(toks)}", flush=True)

        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        durs = []
        for _ in range(a.iters):
            o = run()
            ttnn.synchronize_device(dev)
            ttnn.deallocate(o)
            ttnn.ReadDeviceProfiler(dev)
            data = ttnn.get_latest_programs_perf_data()
            progs = [p for lst in data.values() for p in lst]
            best = None
            for p in progs:
                r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
                if r is not None and (best is None or r.duration > best):
                    best = r.duration
            if best is not None:
                durs.append(best / 1e3)
        if durs:
            print(
                f"MSA_OP_US median={statistics.median(durs):.2f} min={min(durs):.2f} max={max(durs):.2f} n={len(durs)}",
                flush=True,
            )
        else:
            print("MSA_OP_US unavailable (profiler post-process returned nothing)", flush=True)
    finally:
        ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
