# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone sweep: paged SDPA-decode ``k_chunk_size`` x ``max_cores_per_head_batch``.

Stage 06's lever analysis picked ``k_chunk_size=512, max_cores_per_head_batch=32``
as a *first guess*. This probe sweeps both, at several decode positions, and
scores every leg on **PCC against a float32 torch reference computed from the
exact same quantised cache the device reads** -- not against the shipped
``program_config=None`` leg, which is itself an approximation and cannot be the
accuracy datum.

Shapes are the shipped per-die decode shapes (TP=4): 8 Q heads, 1 KV head,
head_dim 128, page size 32, batch 1, bfloat16 Q.

**Cache dtype is bfloat16 and that is not a detail.** Every stage-06 SDPA probe
before this one -- ``sdpa_depth_probe``, ``sdpa_curpos_probe``,
``sdpa_progcfg_probe``, ``sdpa_crossover_probe``, and the first run of this file
-- allocated the cache as ``bfloat8_b``, while ``create_mesh_kv_cache``
allocates ``ttnn.bfloat16`` (``tt/multichip_decoder.py:1167``). At half the
bytes per element a ``k_chunk`` of 512 fits in L1 and at the real dtype it does
not, so the bfloat8_b probes measured -- and recommended -- a configuration the
model cannot run. Pass ``--cache-dtype bfloat8_b`` to reproduce the old,
misleading numbers.

    python sdpa_sweep_probe.py [--depth 65536] [--cache-dtype bfloat16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).resolve().parent

MESH_SHAPE = (1, 4)
N_Q_HEADS = 8
N_KV_HEADS = 1
HEAD_DIM = 128
PAGE = 32
BATCH = 1
ITERS = 20
#: what ``create_mesh_kv_cache`` actually allocates
CACHE_DTYPE = ttnn.bfloat16

#: decode positions to sweep. ctx128 is where the model is profiled today;
#: 32767 is the deepest that fits in the allocated cache below.
CUR_POS = [127, 1023, 4095, 16383, 32767]
K_CHUNKS = [32, 64, 128, 256, 512, 1024]
MAX_CORES = [8, 16, 32, 64]


def make(mesh, depth, cur_pos, seed=0, cache_dtype=None):
    torch.manual_seed(seed)
    pages = depth // PAGE
    shape = (pages, N_KV_HEADS, PAGE, HEAD_DIM)
    kt, vt = torch.randn(shape).float(), torch.randn(shape).float()
    k, v = (
        ttnn.from_torch(
            t,
            dtype=cache_dtype or CACHE_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for t in (kt, vt)
    )
    qt = torch.randn((1, BATCH, N_Q_HEADS, HEAD_DIM)).float()
    q = ttnn.from_torch(
        qt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pt = ttnn.from_torch(
        torch.arange(BATCH * pages, dtype=torch.int32).reshape(BATCH, pages),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pos = ttnn.from_torch(
        torch.tensor([cur_pos] * BATCH, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return q, qt, (k, v), pt, pos


def reference(mesh, q_t, k_dev, v_dev, cur_pos):
    """float32 attention over the *dequantised* device cache, positions 0..cur_pos."""
    # Read back device 0's copy so the reference sees exactly the values the
    # kernel reads -- otherwise the cache's own rounding swamps the chunking error.
    k = ttnn.to_torch(ttnn.get_device_tensors(k_dev)[0]).float()
    v = ttnn.to_torch(ttnn.get_device_tensors(v_dev)[0]).float()
    n = cur_pos + 1
    # page table is identity, so pages concatenate in order -> [seq, head_dim]
    k = k.permute(1, 0, 2, 3).reshape(N_KV_HEADS, -1, HEAD_DIM)[:, :n, :]
    v = v.permute(1, 0, 2, 3).reshape(N_KV_HEADS, -1, HEAD_DIM)[:, :n, :]
    q = q_t.reshape(BATCH, N_Q_HEADS, HEAD_DIM)[0]  # [NQH, HD]
    rep = N_Q_HEADS // N_KV_HEADS
    kk = k.repeat_interleave(rep, dim=0)  # [NQH, n, HD]
    vv = v.repeat_interleave(rep, dim=0)
    scores = torch.einsum("hd,hnd->hn", q, kk) * (HEAD_DIM**-0.5)
    return torch.einsum("hn,hnd->hd", scores.softmax(-1), vv)  # [NQH, HD]


def run(q, caches, pt, pos, pc):
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        caches[0],
        caches[1],
        page_table_tensor=pt,
        cur_pos_tensor=pos,
        scale=HEAD_DIM**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
    )


def pcc_of(mesh, out, ref):
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(-1)
    exp = ref.reshape(-1)
    p = torch.corrcoef(torch.stack([got, exp]))[0, 1].item()
    return p, (got - exp).abs().max().item()


def bench(mesh, q, caches, pt, pos, pc, iters=ITERS):
    o = run(q, caches, pt, pos, pc)
    ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        o = run(q, caches, pt, pos, pc)
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    return 1e6 * (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=65536)
    ap.add_argument("--cache-dtype", default="bfloat16", choices=["bfloat16", "bfloat8_b"])
    ap.add_argument("--out", default=str(HERE / "sdpa_sweep_probe.json"))
    args = ap.parse_args()

    cache_dtype = getattr(ttnn, args.cache_dtype)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
    grid = mesh.compute_with_storage_grid_size()
    print(f"grid {grid}  allocated depth {args.depth}  cache dtype {args.cache_dtype}", flush=True)
    results = []
    try:
        for cur in CUR_POS:
            q, qt, caches, pt, pos = make(mesh, args.depth, cur, seed=7, cache_dtype=cache_dtype)
            ref = reference(mesh, qt, caches[0], caches[1], cur)
            legs = [("None", None)] + [
                (
                    f"k{k}/c{c}",
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid,
                        q_chunk_size=32,
                        k_chunk_size=k,
                        max_cores_per_head_batch=c,
                    ),
                )
                for k in K_CHUNKS
                for c in MAX_CORES
            ]
            base = None
            print(f"\n=== cur_pos {cur} ===", flush=True)
            for name, pc in legs:
                try:
                    out = run(q, caches, pt, pos, pc)
                    p, md = pcc_of(mesh, out, ref)
                    ttnn.deallocate(out)
                    us = bench(mesh, q, caches, pt, pos, pc)
                except Exception as exc:  # noqa: BLE001 - recorded
                    print(f"  {name:<12} FAILED: {str(exc).splitlines()[0][:100]}", flush=True)
                    results.append({"cur_pos": cur, "cfg": name, "error": str(exc).splitlines()[0][:200]})
                    continue
                if base is None:
                    base = us
                print(
                    f"  {name:<12} {us:9.2f} us  {base/us:6.2f}x  PCC {p:.6f}  max|d| {md:.2e}",
                    flush=True,
                )
                results.append(
                    {"cur_pos": cur, "cfg": name, "us": us, "speedup": base / us, "pcc": p, "max_abs_diff": md}
                )
            for x in (q, pt, pos, *caches):
                ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
