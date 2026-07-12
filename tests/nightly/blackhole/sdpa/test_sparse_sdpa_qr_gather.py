# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qr-ring Q-gather sparse-MLA — MULTI-DEVICE accuracy on the LoudBox (Blackhole).

Demonstrates the qr-ring dataflow on a real SP mesh WITHOUT any fabric collective:
  - KV cache stays STATIONARY, contiguously sharded across the SP ring (each chip holds T/sp tokens).
  - Q is REPLICATED on every chip (the post-all-gather state).
  - each chip scores only ITS shard's selected tokens (host-compacted local indices) and emits the
    normalized partial + softmax stats via ttnn.transformer.sparse_sdpa_stats.
  - the per-shard (O, m, l) partials are flash-merged (online softmax) on the host -> full-T result.

The KV cache is never gathered — only the fixed-size queries "travel" (here already replicated) and the
small partials come home. Correctness is checked against the full-T golden and against a plain single-device
sparse_sdpa over all top-k. The merge math + device stat export are the same ones validated single-chip in
tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_qr_stats.py.

Run: scripts/run_safe_pytest.sh --run-all tests/nightly/blackhole/sdpa/test_sparse_sdpa_qr_gather.py
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import MASKED_INDEX, golden, pcc

K_DIM = 576
V_DIM = 512


def _make_covered_inputs(H, S, T, topk, sp, seed=0):
    """Build (q, kv_natural, indices) where every SP shard holds exactly topk/sp of each row's selected
    tokens — so no shard is empty for any row (the op requires >=1 valid key per row). Contiguous shards:
    shard c owns tokens [c*T/sp, (c+1)*T/sp)."""
    assert topk % sp == 0 and T % sp == 0
    g, tl = topk // sp, T // sp
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, K_DIM, generator=gen)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen)
    indices = torch.full((1, 1, S, topk), MASKED_INDEX, dtype=torch.int64)
    for s in range(S):
        slot = 0
        for c in range(sp):
            picks = torch.randperm(tl, generator=gen)[:g] + c * tl  # g distinct tokens in shard c
            indices[0, 0, s, slot : slot + g] = picks
            slot += g
    return q, kv, indices


def _local_shard_indices(indices, sp, T):
    """[1,1,S,TOPK] natural -> [1,sp,S,TOPK] per-shard LOCAL indices (token - c*T/sp), compacted to the front
    with a masked tail (the op needs sentinels contiguous at the end). Rows keep the same shard coverage."""
    S, topk = indices.shape[2], indices.shape[3]
    tl = T // sp
    out = torch.full((1, sp, S, topk), MASKED_INDEX, dtype=indices.dtype)
    idx = indices[0, 0]
    for c in range(sp):
        lo, hi = c * tl, (c + 1) * tl
        for s in range(S):
            locals_ = [int(p) - lo for p in idx[s].tolist() if p != MASKED_INDEX and lo <= int(p) < hi]
            for k, lp in enumerate(locals_):
                out[0, c, s, k] = lp
    return out


def _merge(partials, scale):
    """Online-softmax merge of per-shard [(O_norm, m_raw, l)] -> normalized out. m in col 0, l summed over cols."""
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
@pytest.mark.parametrize("H,S,T,TOPK", [(32, 64, 1024, 64)], ids=["h32s64t1024k64"])
def test_qr_gather_multidevice(mesh_device, H, S, T, TOPK):
    sp = tuple(mesh_device.shape)[0]  # SP ring size (mesh axis 0)
    mesh_shape = tuple(mesh_device.shape)
    scale = K_DIM**-0.5

    q, kv_nat, indices = _make_covered_inputs(H, S, T, TOPK, sp, seed=sp)
    ref = golden(q, kv_nat, indices, scale, V_DIM)  # full-T normalized golden [1,H,S,V]

    # Device placement: KV contiguously sharded on seq (dim 2) across the SP axis -> chip c holds tokens
    # [c*T/sp,(c+1)*T/sp). Q replicated. Per-shard local indices sharded on dim 1 (one shard per chip).
    local_idx = _local_shard_indices(indices, sp, T)  # [1,sp,S,TOPK]
    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kv_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=mesh_shape)  # seq-shard KV
    idx_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(1, None), mesh_shape=mesh_shape)  # shard-per-chip idx
    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_q = ttnn.from_torch(q.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=repl, **common)
    tt_kv = ttnn.from_torch(kv_nat.to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=kv_shard, **common)
    tt_idx = ttnn.from_torch(local_idx.to(torch.int32), dtype=ttnn.uint32, mesh_mapper=idx_shard, **common)

    outs = ttnn.transformer.sparse_sdpa_stats(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=32)

    # Gather each chip's partial home (concat the SP-axis shards along tensor dim 0).
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_shape)
    O_all = ttnn.to_torch(outs[0], mesh_composer=comp).float()  # [sp, H, S, V]
    m_all = ttnn.to_torch(outs[1], mesh_composer=comp).float()  # [sp, H, S, 32]
    l_all = ttnn.to_torch(outs[2], mesh_composer=comp).float()

    partials = []
    for c in range(sp):
        O = O_all[c : c + 1]  # [1,H,S,V]
        m = m_all[c : c + 1, ..., 0:1]  # col 0
        l = l_all[c : c + 1].sum(dim=-1, keepdim=True)  # sum over the 32 tile cols
        partials.append((O, m, l))
    out = _merge(partials, scale)

    p = pcc(out, ref)
    assert p >= 0.99, f"qr-gather multidevice PCC {p:.5f} (sp={sp}) < 0.99"
