# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sharded-query MSA under SP=8 × TP=4 with per-device chunk_offset — the deployed path.

`msa_sp_attention_nocache` keeps the query SP-sharded (S/sp = 640 rows/device) and gives each device a
per-device causal `chunk_offset` (rank r -> r*640), so the output stays SP-sharded (no replication, no
reshard) — what the SP residual stream + EP MoE need. This validates it against the already-golden
`msa_sp_attention_gather_all` (gather-everything, uniform chunk_start; golden-validated in test_msa_sp_vs_ref at
PCC 1.0): for the same single chunk, device r's query at global r*640+row must score/attend identically
whether the indexer ran on the full query (chunk_start=0) or the local 640-row shard (chunk_offset=r*640).
Equal outputs ⇒ the per-device chunk_offset SP path is correct.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.attention.msa import msa_sp_attention_nocache
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..msa_golden import msa_sp_attention_gather_all
from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, NIDX, HEAD_DIM = 64, 4, 4, 128


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("chunk_local", [640], ids=["chunk640"])  # 5120 / SP=8
def test_msa_sp_sharded(mesh_device, device_params, chunk_local, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    S = sp * chunk_local  # 5120
    G = NQ // NKV  # 16 q/group
    scale = HEAD_DIM**-0.5

    torch.manual_seed(0)
    q = torch.randn(1, NQ, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    iq = torch.randn(1, NIDX, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ik = torch.randn(1, 1, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    mesh_config = MeshConfig((rows, cols), decode=ModeConfig(tp=tp, ep=sp))
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    def shard(t, split_heads):
        dims = [None, None]
        dims[sp_axis] = 2  # seq across SP rows
        dims[1] = 1 if split_heads else None  # heads across TP cols (index_k shared -> replicate)
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    common = dict(mesh_config=mesh_config, ccl_manager=ccl, cached_len=0, scale=scale, num_groups=1)

    # A) gather-everything (golden path): full-seq output, replicated across SP rows.
    out_a = msa_sp_attention_gather_all(
        shard(q, True), shard(k, True), shard(v, True), shard(iq, True), shard(ik, False), **common
    )
    dts_a = ttnn.get_device_tensors(out_a)
    ref = torch.cat([ttnn.to_torch(dts_a[c]).float()[:, :G] for c in range(cols)], dim=1)  # row0, [1,NQ,S,HD]

    # B) sharded-query (deployed path): per-device chunk_offset, SP-sharded output [1,G,640,HD]/device.
    out_b = msa_sp_attention_nocache(
        shard(q, True), shard(k, True), shard(v, True), shard(iq, True), shard(ik, False), s_local=chunk_local, **common
    )
    dts_b = ttnn.get_device_tensors(out_b)
    # reassemble: per col(group) concat the 8 SP rows' 640-row shards along seq; then concat groups (heads)
    groups = [
        torch.cat([ttnn.to_torch(dts_b[r * cols + c]).float()[:, :G] for r in range(rows)], dim=2) for c in range(cols)
    ]
    out = torch.cat(groups, dim=1)  # [1, NQ, S, HD]

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"MSA sharded-query SP=8xTP=4 (per-device chunk_offset) vs gather-everything: pcc={pcc}")
    assert passing, f"sharded-query MSA PCC fail: {pcc}"
