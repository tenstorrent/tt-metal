# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_EMBED_SHARD=1 (tt/device_head.py ShardedEmbeddingTable): hidden-dim sharded embedding + all_gather vs the replicated table, on the 4x8 mesh.

Checks bit-identity of the lookup on all 32 devices (decode T = 1, 16, 24 per row; prefill 2048 tokens per row), traced us per call (rep vs dshard), DRAM MiB/bank of both tables.
Prints EDS lines. Real table: embed.weight of the checkpoint (DSV41_CKPT, index json)."""
import json
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import ShardedEmbeddingTable, embed_lookup

ROWS, COLS, VOCAB, DIM = 4, 8, 129280, 5120


def load_table():
    from safetensors import safe_open

    ck = os.environ.get("DSV41_CKPT", "/ckpt")
    idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(ck, idx["embed.weight"]), framework="pt") as f:
        return f.get_tensor("embed.weight").to(torch.bfloat16)


def dram_mib(md):
    ttnn.synchronize_device(md)
    return ttnn.get_memory_view(md, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank / 2**20


def upload(md, ids):
    return ttnn.from_torch(
        ids.reshape(-1, 1).to(torch.int32),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
    )


def traced_us(md, fn, n=20, reps=10):
    def body():
        for _ in range(reps):
            fn()

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


MESH = [pytest.param((4, 8), id="4x8")]
DP = [
    pytest.param(
        {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000},
        id="ring",
    )
]


@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
@pytest.mark.parametrize("device_params", DP, indirect=True)
@torch.no_grad()
def test_embedding_dshard(mesh_device):
    md = mesh_device
    w = load_table()
    d0 = dram_mib(md)
    rep = ttnn.from_torch(
        w,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    d1 = dram_mib(md)
    sh = ShardedEmbeddingTable(md, w)
    d2 = dram_mib(md)
    print(
        f"EDS dram MiB/bank: replicated {d1 - d0:.1f}  dim-sharded {d2 - d1:.1f}  saved {d1 - d0 - (d2 - d1):.1f}",
        flush=True,
    )
    g = torch.Generator().manual_seed(0)
    for n_per_row in (1, 16, 24, 2048):
        ids = torch.randint(0, VOCAB, (ROWS * n_per_row,), generator=g)
        ids[0], ids[-1] = 0, VOCAB - 1
        tok = upload(md, ids)
        a = embed_lookup(rep, tok, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        b = embed_lookup(sh, tok, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ref = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(a)]
        got = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(b)]
        ok = all(torch.equal(r, q) for r, q in zip(ref, got))
        # row-major request (drafter path)
        b2 = embed_lookup(sh, tok, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        ok2 = torch.equal(ttnn.to_torch(ttnn.get_device_tensors(b2)[0]).reshape(-1, DIM), ref[0].reshape(-1, DIM))
        reps = 10 if n_per_row <= 24 else 3
        t_rep = traced_us(md, lambda: embed_lookup(rep, tok), reps=reps)
        t_sh = traced_us(md, lambda: embed_lookup(sh, tok), reps=reps)
        print(
            f"EDS n/row={n_per_row}: bit-identical all 32 dev {ok}  row-major {ok2} | us/call rep {t_rep:.1f}  dshard+all_gather {t_sh:.1f}  added {t_sh - t_rep:+.1f}",
            flush=True,
        )
        assert ok and ok2
