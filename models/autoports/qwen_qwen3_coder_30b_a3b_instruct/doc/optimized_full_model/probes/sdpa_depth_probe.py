# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone probe: does paged SDPA-decode cost track ``cur_pos`` or the *allocated* cache?

**This is a measurement, not a change.** Nothing in ``tt/`` is touched.

Why it exists. Stage 04 profiled its layer with a **1024**-position paged cache
(``in1 = 32 pages x 1 x 32 x 128`` in ``doc/optimized_multichip_decoder/window_decode.txt``)
and read ``SdpaDecode`` at **9.816 us**. The stage-06 48-layer in-model profile,
built at ``max_context_len=4096`` (``in1 = 128 pages``), reads **20.70 us** for
the same op at the same decode position (~130) -- and stage 05's independent
2-layer capture, also at 4096, reads 20.3-21.4 us. Two captures agree, so the op
more than doubled between two runs that decode the same token at the same
position. The only thing that differs is how deep the cache is *allocated*.

If that is right it is a scaling fact about the whole model, not a detail:
the shipped contract advertises 262144 tokens of context, and a term that grows
with allocated depth rather than with position would dominate decode there.

The legs hold ``cur_pos`` fixed at 128 and vary only the allocated depth, then
one extra leg varies ``cur_pos`` at a fixed depth to separate the two effects.
Shapes are the shipped per-die decode shapes: 8 Q heads, 1 KV head, head_dim
128, page size 32, ``program_config=None`` (what the paged branch of
``attention_decode_optimized`` passes).

    python sdpa_depth_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import ttnn

MESH_SHAPE = (1, 4)
N_Q_HEADS = 8
N_KV_HEADS = 1
HEAD_DIM = 128
PAGE = 32
BATCH = 1
ITERS = 50
#: allocated context depths, and the (depth, cur_pos) pairs to run
DEPTHS = [1024, 4096, 16384, 65536]
CUR_POS_SWEEP = [(65536, p) for p in (128, 1024, 8192, 32768)]


def bench(mesh, depth, cur_pos):
    pages = depth // PAGE
    cache_shape = (pages, N_KV_HEADS, PAGE, HEAD_DIM)
    caches = [
        ttnn.from_torch(
            torch.randn(cache_shape, dtype=torch.bfloat16).float(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for _ in range(2)
    ]
    q = ttnn.from_torch(
        torch.randn((1, BATCH, N_Q_HEADS, HEAD_DIM)).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    page_table = ttnn.from_torch(
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

    def call():
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            caches[0],
            caches[1],
            page_table_tensor=page_table,
            cur_pos_tensor=pos,
            scale=HEAD_DIM**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )
        ttnn.deallocate(out)

    call()  # compile
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        call()
    ttnn.synchronize_device(mesh)
    ms = 1e3 * (time.perf_counter() - t0) / ITERS
    for t in (q, page_table, pos, *caches):
        ttnn.deallocate(t)
    return ms


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
    results = []
    try:
        print("leg                                        ms/call (eager, mesh of 4, 50 iters)")
        for depth in DEPTHS:
            ms = bench(mesh, depth, 128)
            results.append({"allocated_context": depth, "cur_pos": 128, "ms": ms})
            print(f"  allocated {depth:>6}, cur_pos    128    {ms:8.4f}")
        for depth, pos in CUR_POS_SWEEP:
            ms = bench(mesh, depth, pos)
            results.append({"allocated_context": depth, "cur_pos": pos, "ms": ms})
            print(f"  allocated {depth:>6}, cur_pos {pos:>6}    {ms:8.4f}")
    finally:
        ttnn.close_mesh_device(mesh)
    Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
