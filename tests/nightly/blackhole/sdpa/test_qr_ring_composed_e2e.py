# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qr-ring composed path — FULL multi-device pipeline on the LoudBox SP ring (Blackhole).

Puts the whole thing together on a real 8-chip SP ring (FABRIC_1D_RING, which trains on this box):
  - KV latent cache block-cyclic SP-sharded and STATIONARY (chip s holds stripe s, T/sp tokens) — NOT gathered.
  - Q + top-k indices replicated (the post-all-gather state).
  - each chip: sparse_sdpa_stats_shard_local -> its stripe's partial (O, m, l).  [per-device shard_id]
  - cross-SP online-softmax MERGE via all_gather of the partials + local eltwise combine -> full attention.
Compared to the full-T sparse golden. This is the qr-ring win realized with standard collectives: the O(T) KV
gather is gone; only the fixed-size Q/partials move.
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import MASKED_INDEX, golden, pcc
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import create_global_semaphores

K_DIM = 576
V_DIM = 512


def _stripe_natural_ids(shard, chunk_local, sp, T):
    num_slabs = T // (chunk_local * sp)
    slab = torch.arange(num_slabs).view(-1, 1)
    r = torch.arange(chunk_local).view(1, -1)
    return (slab * (chunk_local * sp) + shard * chunk_local + r).reshape(-1)


def _build_indices(S, T, TOPK, sp, chunk_local, gen):
    idx = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
    stoks = [_stripe_natural_ids(s, chunk_local, sp, T) for s in range(sp)]
    for s in range(S):
        if s % 3 == 2:
            tgt = stoks[s % sp]
            perm = tgt[torch.randperm(tgt.numel(), generator=gen)[: min(TOPK, tgt.numel())]]
        else:
            nv = TOPK if s % 3 == 0 else max(1, TOPK // 2)
            perm = torch.randperm(T, generator=gen)[:nv]
        idx[0, 0, s, : perm.numel()] = perm
    return idx


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["sp8"])
@pytest.mark.parametrize("H,S,T,TOPK,chunk_local", [(32, 96, 2048, 256, 64)], ids=["h32s96t2048k256cl64"])
def test_qr_ring_composed_e2e(mesh_device, H, S, T, TOPK, chunk_local):
    ca = 1  # SP is mesh axis 1 on the (1,8) ring
    sp = tuple(mesh_device.shape)[ca]
    mesh_shape = tuple(mesh_device.shape)
    assert T % (chunk_local * sp) == 0
    shard_len = T // sp
    scale = K_DIM**-0.5

    gen = torch.Generator().manual_seed(2027)
    q = torch.randn(1, H, S, K_DIM, generator=gen)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen)
    indices = _build_indices(S, T, TOPK, sp, chunk_local, gen)
    ref = golden(q, kv, indices, scale, V_DIM)  # [1,H,S,V]

    bc = torch.empty_like(kv)  # block-cyclic physical order (stripe-major)
    for s in range(sp):
        bc[:, :, s * shard_len : (s + 1) * shard_len, :] = kv[:, :, _stripe_natural_ids(s, chunk_local, sp, T), :]

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    sem = create_global_semaphores(mesh_device, sp, crs, 0)

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kv_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_shape)  # stripe s -> chip s
    id_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_shape)  # stripe id s -> chip s
    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_q = ttnn.from_torch(q.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=repl, **common)
    tt_kv = ttnn.from_torch(bc.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=kv_shard, **common)
    tt_idx = ttnn.from_torch(indices.to(torch.int32), dtype=ttnn.uint32, mesh_mapper=repl, **common)
    tt_shard = ttnn.from_torch(
        torch.arange(sp, dtype=torch.int32).reshape(sp, 1, 1, 1), dtype=ttnn.uint32, mesh_mapper=id_shard, **common
    )

    outs = ttnn.transformer.sparse_sdpa_stats_shard_local(
        tt_q, tt_kv, tt_idx, tt_shard, V_DIM, sp=sp, chunk_local=chunk_local, scale=scale, k_chunk_size=32
    )

    def _ag(t, dim):  # all-gather the per-device partial along `dim` over the SP ring
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=sem,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=ca,
            subdevice_id=sub_id,
        )

    # Gather every chip's (O, m, l) partial onto all chips (stacked along dim 0), then flash-merge locally.
    O_all = _ag(outs[0], 0)  # [sp,H,S,V]
    m_all = _ag(outs[1], 0)  # [sp,H,S,32]
    l_all = _ag(outs[2], 0)
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])

    tl = ttnn.TILE_LAYOUT
    scaled_m, l_sum, O_tile, M = [], [], [], None
    for s in range(sp):
        sm0 = ttnn.multiply(ttnn.to_layout(ttnn.slice(m_all, [s, 0, 0, 0], [s + 1, H, S, 1]), tl), scale)
        ls = ttnn.sum(ttnn.to_layout(ttnn.slice(l_all, [s, 0, 0, 0], [s + 1, H, S, 32]), tl), dim=-1, keepdim=True)
        Ot = ttnn.to_layout(ttnn.slice(O_all, [s, 0, 0, 0], [s + 1, H, S, V_DIM]), tl)
        scaled_m.append(sm0)
        l_sum.append(ls)
        O_tile.append(Ot)
        M = sm0 if M is None else ttnn.maximum(M, sm0)
    num, den = None, None
    for sm0, ls, Ot in zip(scaled_m, l_sum, O_tile):
        w = ttnn.multiply(ttnn.exp(ttnn.subtract(sm0, M)), ls)
        term = ttnn.multiply(Ot, w)
        num = term if num is None else ttnn.add(num, term)
        den = w if den is None else ttnn.add(den, w)
    out = ttnn.multiply(num, ttnn.reciprocal(den))  # [1,H,S,V] (same on every chip)

    p = pcc(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float(), ref)
    assert p >= 0.99, f"qr-ring composed e2e PCC {p:.5f} (sp={sp}) < 0.99"
