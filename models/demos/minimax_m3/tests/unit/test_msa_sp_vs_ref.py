# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MSA sparse attention under SP=8 × TP=4 — the cross-device KV AllGather (the real integration risk).

sparse_sdpa_msa is a pure dense-context kernel (no cache-read), so under SP each device holds only a
sequence shard of K/V/index_k and must AllGather them across the SP axis to assemble the full context
before the indexer + sparse op — a query reaches other devices' (and cached) tokens *only* through
that gather. This test exercises exactly that on a real (8,4) mesh at the real M3 chunk (640/chip ×
SP=8 = 5120, 40 blocks, top-16), per-device heads 16q/1kv/1index + 1 shared index-k, num_groups=1.

Validation is differential against the SAME model function run on the full (un-sharded) context: if
the AllGather reorders or drops any cross-device shard, the gathered context differs, block selection
+ sparse output diverge, and PCC collapses. (The full-context chain itself is golden-validated by
test_msa_prefill_vs_ref.) SP layout is CONTIGUOUS — no zigzag/balancing (chunked prefill).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.attention.msa import msa_indexer_sparse
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..msa_golden import msa_sp_attention_gather_all
from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, NIDX, HEAD_DIM = 64, 4, 4, 128


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("chunk_local", [640], ids=["chunk640"])  # 5120 chunk / SP=8
def test_msa_sp(mesh_device, device_params, chunk_local, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis, tp_axis = rows, cols, 0, 1
    S = sp * chunk_local  # 5120
    G = NQ // NKV  # 16 q heads per group
    scale = HEAD_DIM**-0.5

    torch.manual_seed(0)
    q = torch.randn(1, NQ, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    iq = torch.randn(1, NIDX, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ik = torch.randn(1, 1, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1  # single shared index-k head

    mesh_config = MeshConfig((rows, cols), decode=ModeConfig(tp=tp, ep=sp))  # prefill -> sp=rows, tp=cols
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    # --- SP path: shard seq across SP (dim2), heads across TP (dim1); index_k shared -> replicate on TP ---
    def shard(t, split_heads):
        dims = [None, None]
        dims[sp_axis] = 2
        dims[tp_axis] = 1 if split_heads else None
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    out = msa_sp_attention_gather_all(
        shard(q, True),
        shard(k, True),
        shard(v, True),
        shard(iq, True),
        shard(ik, False),
        mesh_config=mesh_config,
        ccl_manager=ccl,
        cached_len=0,
        scale=scale,
        num_groups=1,
    )
    # Per device [1, G, S, HD] = that column's group, replicated across SP rows (q was gathered). Take
    # row 0; concat the 4 columns -> [1, NQ, S, HD] in natural head order (col c = group c).
    dts = ttnn.get_device_tensors(out)
    out_sp = torch.cat([ttnn.to_torch(dts[c]).float()[:, :G] for c in range(cols)], dim=1)

    # --- reference: the SAME function on the full (un-sharded) context, per group, replicated on the
    # mesh (every device computes the identical full-context result; read device 0). ---
    def full_tile(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    refs = []
    for g in range(NKV):
        out_ref_g = msa_indexer_sparse(
            full_tile(iq[:, g : g + 1]),
            full_tile(ik),
            full_tile(q[:, g * G : (g + 1) * G]),
            full_tile(k[:, g : g + 1]),
            full_tile(v[:, g : g + 1]),
            chunk_start_idx=0,
            scale=scale,
            num_groups=1,
            device=mesh_device,
        )
        refs.append(ttnn.to_torch(ttnn.get_device_tensors(out_ref_g)[0]).float()[:, :G])
    out_ref = torch.cat(refs, dim=1)  # [1, NQ, S, HD]

    passing, pcc = comp_pcc(out_ref, out_sp, 0.99)
    logger.info(f"MSA SP=8 x TP=4 (AllGather full context) vs full-context ref: pcc={pcc}")
    assert passing, f"MSA SP PCC fail: {pcc}"
