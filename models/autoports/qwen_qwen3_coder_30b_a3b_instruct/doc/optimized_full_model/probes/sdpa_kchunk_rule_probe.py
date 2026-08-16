# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The rule ``k_chunk_size`` has to obey, found in-model and then pinned here.

``test_multichip_decode_batch`` allocates a **128-position** paged cache and
decodes at ``cur_pos=32``. Sweeping the adopted config through that *real* test
(``logs/sdpa_kchunk_grid.log``) gives an unambiguous split:

    k_chunk  32  64  128  256  ->  4 passed, at every max_cores in {16,32,64}
    k_chunk 512                 ->  2-3 of 4 failed, PCC -0.04 to -0.17

so ``max_cores_per_head_batch`` is innocent and ``k_chunk_size`` is the whole
effect. Note the failures are **nondeterministic** in which batch sizes trip,
which is the signature of reading memory that is not part of the cache rather
than of an arithmetic difference.

The first standalone shallow-cache probe *missed* this: it read PCC 0.9998 at
depth 128 / k512, because it used **batch 1 and an identity page table**. This
one uses a multi-user page table laid out the way ``create_mesh_kv_cache`` does
-- which is the only difference -- and sweeps allocated depth against k_chunk to
find where the boundary actually is.

    python sdpa_kchunk_rule_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sdpa_sweep_probe import HEAD_DIM, N_KV_HEADS, N_Q_HEADS, PAGE  # noqa: E402

DEPTHS = [128, 256, 512, 1024, 2048, 4096]
K_CHUNKS = [32, 64, 128, 256, 512, 1024]
BATCH = 8


def build(mesh, depth, cur_pos, seed=13):
    """A per-user paged cache: ``BATCH`` users x ``depth`` positions each."""
    torch.manual_seed(seed)
    per_user_pages = depth // PAGE
    total_pages = per_user_pages * BATCH
    shape = (total_pages, N_KV_HEADS, PAGE, HEAD_DIM)
    kt, vt = torch.randn(shape).float(), torch.randn(shape).float()
    k, v = (
        ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,  # what create_mesh_kv_cache allocates
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
    page_host = torch.arange(total_pages, dtype=torch.int32).reshape(BATCH, per_user_pages)
    pt = ttnn.from_torch(
        page_host,
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
    return q, qt, (k, v), pt, pos, page_host


def ref(mesh, qt, k_dev, v_dev, page_host, cur_pos):
    kd = ttnn.to_torch(ttnn.get_device_tensors(k_dev)[0]).float()
    vd = ttnn.to_torch(ttnn.get_device_tensors(v_dev)[0]).float()
    n = cur_pos + 1
    outs = []
    for u in range(BATCH):
        pages = page_host[u].tolist()
        kk = torch.cat([kd[p, 0] for p in pages], 0)[:n]  # [n, HD]
        vv = torch.cat([vd[p, 0] for p in pages], 0)[:n]
        q = qt.reshape(BATCH, N_Q_HEADS, HEAD_DIM)[u]  # [NQH, HD]
        s = (q @ kk.T) * (HEAD_DIM**-0.5)
        outs.append(s.softmax(-1) @ vv)
    return torch.stack(outs)  # [B, NQH, HD]


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    grid = mesh.compute_with_storage_grid_size()
    results = []
    try:
        for depth in DEPTHS:
            cur = 32
            q, qt, caches, pt, pos, page_host = build(mesh, depth, cur)
            expect = ref(mesh, qt, caches[0], caches[1], page_host, cur)
            row = f"depth {depth:5d} (per user, {depth//PAGE:4d} pages)  "
            for kc in [None] + K_CHUNKS:
                pc = (
                    None
                    if kc is None
                    else ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid,
                        q_chunk_size=32,
                        k_chunk_size=kc,
                        max_cores_per_head_batch=16,
                    )
                )
                out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q,
                    caches[0],
                    caches[1],
                    page_table_tensor=pt,
                    cur_pos_tensor=pos,
                    scale=HEAD_DIM**-0.5,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=pc,
                )
                got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(BATCH, N_Q_HEADS, HEAD_DIM)
                ttnn.deallocate(out)
                worst = min(
                    torch.corrcoef(torch.stack([got[u].reshape(-1), expect[u].reshape(-1)]))[0, 1].item()
                    for u in range(BATCH)
                )
                row += f"{'None' if kc is None else 'k'+str(kc)}={worst:9.5f} "
                results.append({"depth": depth, "k_chunk": kc, "worst_user_pcc": worst})
            print(row, flush=True)
            for x in (q, pt, pos, *caches):
                ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)
    (HERE / "sdpa_kchunk_rule_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
