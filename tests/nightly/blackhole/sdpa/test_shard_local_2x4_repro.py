# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate the qr shard-local hang seen in the GLM (2,4)=sp2/tp4 chunked run.

Reproduces the EXACT op inputs from that run — q_rm=[1,64,256,576], kvpe_stripe=[1,1,2560,576] (T=5120, sp=2),
idx=[1,1,256,1024] natural, shard_id per-device, sp=2, chunk_local=512 — on a (2,4) mesh with CONTROLLED data
and NO fabric/reshard. If this hangs, the bug is the op on a 2D mesh (independent of the real cache/indices);
if it passes, the trigger is the real data (index values / cache layout). Just checks it COMPLETES.
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import MASKED_INDEX

K_DIM, V_DIM = 576, 512


def _stripe_ids(shard, chunk_local, sp, T):
    ns = T // (chunk_local * sp)
    slab = torch.arange(ns).view(-1, 1)
    r = torch.arange(chunk_local).view(1, -1)
    return (slab * (chunk_local * sp) + shard * chunk_local + r).reshape(-1)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True, ids=["sp2tp4"])
def test_shard_local_2x4(mesh_device):
    # Faithful to the real reshard path: q_rm and idx are seq-sharded over TP (each tp-device a DIFFERENT
    # 256-query slice), KV stripe sharded over SP. S_full = per-chunk queries (1024) split over tp=4 -> 256.
    sp, tp, H, T, TOPK, chunk_local = 2, 4, 64, 5120, 1024, 512
    S_full = 1024
    mesh_shape = tuple(mesh_device.shape)
    shard_len = T // sp
    scale = K_DIM**-0.5
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, H, S_full, K_DIM, generator=gen)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen)
    bc = torch.empty_like(kv)
    for s in range(sp):
        bc[:, :, s * shard_len : (s + 1) * shard_len, :] = kv[:, :, _stripe_ids(s, chunk_local, sp, T), :]
    # indices in the small real range; per TP-group (256 queries), force ALL-empty-on-one-shard patterns so
    # some device sees every query empty on a stripe (the untested edge case).
    RNG = 640
    idx = torch.full((1, 1, S_full, TOPK), MASKED_INDEX, dtype=torch.int64)
    for qi in range(S_full):
        grp = qi // 256  # which tp-group (0..3)
        if grp == 0:
            pool = torch.arange(0, 512)  # tp-group 0: only shard-0 tokens -> shard 1 ALL empty
        elif grp == 1:
            pool = torch.arange(512, RNG)  # tp-group 1: only shard-1 tokens -> shard 0 ALL empty
        else:
            pool = torch.arange(0, RNG)
        nv = min(320, pool.numel())
        perm = pool[torch.randperm(pool.numel(), generator=gen)[:nv]]
        idx[0, 0, qi, : perm.numel()] = perm

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tp_seq = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_shape)  # seq over TP, repl SP
    tt_q = ttnn.from_torch(q.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=tp_seq, **common)
    tt_kv = ttnn.from_torch(
        bc.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=mesh_shape),
        **common,
    )
    tt_idx = ttnn.from_torch(idx.to(torch.int32), dtype=ttnn.uint32, mesh_mapper=tp_seq, **common)
    tt_shard = ttnn.from_torch(
        torch.arange(sp, dtype=torch.int32).reshape(sp, 1, 1, 1),
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape),
        **common,
    )
    outs = ttnn.transformer.sparse_sdpa_stats_shard_local(
        tt_q, tt_kv, tt_idx, tt_shard, V_DIM, sp=sp, chunk_local=chunk_local, scale=scale, k_chunk_size=128
    )
    ttnn.synchronize_device(mesh_device)
    assert len(outs) == 3, "op did not return [O,m,l]"
