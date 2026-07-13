# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qr-ring SHARD-LOCAL sparse attention on a real SP MESH (LoudBox Blackhole), no fabric collective.

The production layout for the fused Q-gather: KV latent cache stays block-cyclic SP-sharded and STATIONARY
(chip s holds stripe s = T/sp rows), queries are gathered (here already replicated), and each chip computes
its stripe's sparse partial (O, m, l) via ttnn.transformer.sparse_sdpa_stats_shard_local with
sp_axis=<SP mesh axis> — so my_shard is derived PER DEVICE from the mesh coordinate (patched by
get_dynamic_runtime_args), NOT passed explicitly. The partials come home and flash-merge (online softmax) to
the full-attention golden.

This proves the per-device my_shard mechanism + block-cyclic SP sharding on hardware. Coverage forces empty
stripes (a query whose top-k all land in one shard) to exercise the identity-partial path across devices.
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import MASKED_INDEX, golden, pcc

K_DIM = 576
V_DIM = 512


def _stripe_natural_ids(shard, chunk_local, sp, T):
    num_slabs = T // (chunk_local * sp)
    slab = torch.arange(num_slabs).view(-1, 1)
    r = torch.arange(chunk_local).view(1, -1)
    return (slab * (chunk_local * sp) + shard * chunk_local + r).reshape(-1)


def _build_indices(S, T, TOPK, sp, chunk_local, gen):
    idx = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
    shard_tokens = [_stripe_natural_ids(s, chunk_local, sp, T) for s in range(sp)]
    for s in range(S):
        kind = s % 3
        if kind == 0:
            perm = torch.randperm(T, generator=gen)[:TOPK]
        elif kind == 1:
            nv = max(1, TOPK // 2 - (s % 5))
            perm = torch.randperm(T, generator=gen)[:nv]
        else:
            tgt = shard_tokens[s % sp]
            perm = tgt[torch.randperm(tgt.numel(), generator=gen)[: min(TOPK, tgt.numel())]]
        idx[0, 0, s, : perm.numel()] = perm
    return idx


def _merge(partials, scale):
    ms = torch.stack([scale * p[1] for p in partials])
    M = ms.max(dim=0).values
    num, den = None, 0.0
    for O, m, l in partials:
        w = torch.exp(scale * m - M) * l
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        num = (w * O) if num is None else num + w * O
        den = den + w
    return num / den.clamp_min(1e-30)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(4, 1), (2, 1)], indirect=True, ids=["sp4", "sp2"])
@pytest.mark.parametrize("H,S,T,TOPK,chunk_local", [(32, 96, 1024, 128, 64)], ids=["h32s96t1024k128cl64"])
def test_shard_local_mesh(mesh_device, H, S, T, TOPK, chunk_local):
    sp = tuple(mesh_device.shape)[0]  # SP is mesh axis 0
    mesh_shape = tuple(mesh_device.shape)
    assert T % (chunk_local * sp) == 0
    shard_len = T // sp
    scale = K_DIM**-0.5

    gen = torch.Generator().manual_seed(1000 * sp + T)
    q = torch.randn(1, H, S, K_DIM, generator=gen)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen)
    indices = _build_indices(S, T, TOPK, sp, chunk_local, gen)
    ref = golden(q, kv, indices, scale, V_DIM)  # [1,H,S,V]

    # Reorder KV into block-cyclic PHYSICAL order (stripe-major): rows [s*shard_len:(s+1)*shard_len] hold
    # stripe s's tokens. Sharding dim 2 across SP then places stripe s on chip s.
    bc = torch.empty_like(kv)
    for s in range(sp):
        bc[:, :, s * shard_len : (s + 1) * shard_len, :] = kv[:, :, _stripe_natural_ids(s, chunk_local, sp, T), :]

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kv_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=mesh_shape)  # stripe s -> chip s
    id_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape)  # stripe id s -> chip s
    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_q = ttnn.from_torch(q.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=repl, **common)
    tt_kv = ttnn.from_torch(bc.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=kv_shard, **common)
    tt_idx = ttnn.from_torch(indices.to(torch.int32), dtype=ttnn.uint32, mesh_mapper=repl, **common)
    # Per-device stripe id: [sp,1,1,1] sharded on axis 0 -> chip s reads value s. One broadcast program, correct
    # across the mesh (the reader's shard_id accessor resolves each device's own value).
    shard_ids = torch.arange(sp, dtype=torch.int32).reshape(sp, 1, 1, 1)
    tt_shard = ttnn.from_torch(shard_ids, dtype=ttnn.uint32, mesh_mapper=id_shard, **common)

    outs = ttnn.transformer.sparse_sdpa_stats_shard_local(
        tt_q, tt_kv, tt_idx, tt_shard, V_DIM, sp=sp, chunk_local=chunk_local, scale=scale, k_chunk_size=32
    )

    comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_shape)
    O_all = ttnn.to_torch(outs[0], mesh_composer=comp).float()  # [sp,H,S,V]
    m_all = ttnn.to_torch(outs[1], mesh_composer=comp).float()  # [sp,H,S,32]
    l_all = ttnn.to_torch(outs[2], mesh_composer=comp).float()

    partials = [
        (O_all[c : c + 1], m_all[c : c + 1, ..., 0:1], l_all[c : c + 1].sum(dim=-1, keepdim=True)) for c in range(sp)
    ]
    out = _merge(partials, scale)
    p = pcc(out, ref)
    assert p >= 0.99, f"shard-local mesh merge PCC {p:.5f} (sp={sp}) < 0.99"
