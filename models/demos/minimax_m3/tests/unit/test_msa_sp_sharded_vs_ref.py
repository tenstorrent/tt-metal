# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sharded-query MSA under SP=8 × TP=4 — the deployed path's per-device causality via the merged
indexer's NATIVE mesh-coord chunk_start (cluster_axis), which replaced our old host-built chunk_offset.

`msa_sp_attention_nocache` keeps the query SP-sharded (S/sp = 640 rows/device) and lets the op derive
each device's causal start from its mesh coordinate: chunk_start = chunk_start_idx(=cached_len=0) +
rank·640 via cluster_axis=sp_axis. Output stays SP-sharded — what the SP residual stream + EP-MoE need.

The OLD gather-everything golden (`msa_sp_attention_gather_all`) is INCOMPATIBLE with the merged op:
cluster_axis=None linearizes the whole mesh, so a full-T (5120-row) query OOBs on every device r>0
(`max_cs + Sq <= T` TT_FATAL). So we verify the NEW thing directly:

  (A) cluster_axis CAUSALITY — the decisive check. Device r owns global query positions [r·640 ..]. The
      indexer masks future blocks to -inf, so the causal frontier (count of FINITE blocks for a device's
      LAST query row) must equal (r·640 + 639)//128 + 1 — i.e. grow 5 → 40 across the 8 SP rows. A broken
      / missing per-device offset would give a flat 5 on every device. Matching the per-row global frontier
      proves cluster_axis applies rank·640.
  (B) the full deployed path runs end-to-end at SP=8×TP=4 and yields finite, non-degenerate SP-sharded
      output of the right shape.

MSA *compute* correctness (index branch + indexer + sparse_sdpa_msa) is validated on REAL layer-3 weights
in test_msa_layer_vs_ref (PCC 0.9994); the op's per-rank causality in main's indexer_score tests.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.attention.msa import msa_sp_attention_nocache
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, NIDX, HEAD_DIM = 64, 4, 4, 128
BLOCK = 128


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("chunk_local", [640], ids=["chunk640"])  # 5120 / SP=8
def test_msa_sp_sharded(mesh_device, device_params, chunk_local, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    S = sp * chunk_local  # 5120
    G = NQ // NKV  # 16 q/group
    nblk = S // BLOCK  # 40
    scale = HEAD_DIM**-0.5

    torch.manual_seed(0)
    q = torch.randn(1, NQ, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    iq = torch.randn(1, NIDX, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ik = torch.randn(1, 1, S, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    mesh_config = MeshConfig((rows, cols), tp=tp)
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

    iq_t, ik_t = shard(iq, True), shard(ik, False)

    # --- (A) cluster_axis causality: raw indexer with the deployed per-device mesh-coord chunk_start ---
    # AllGather index_k across SP (the deployed path does this), keep index_q sharded; cluster_axis=sp_axis
    # makes device r score its 640 rows starting at global r*640.
    ik_full = mesh_config.allgather(ik_t, ccl, axis=sp_axis, dim=2)
    block_scores = ttnn.experimental.indexer_score_msa(
        iq_t,
        ik_full,
        num_groups=1,
        chunk_start_idx=0,
        scale=scale,
        block_size=BLOCK,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0),
        seq_shard_axes=[sp_axis],
    )
    bs_dev = ttnn.get_device_tensors(block_scores)
    # Per SP row r, the LAST local query (row 639) is at global r*640+639. Its causal frontier = the count
    # of VISIBLE blocks = blocks that are NOT -inf (future-masked). That's the past blocks (finite scores)
    # PLUS the query's own/current block, which the op force-locals to +inf (so we count "not -inf", not
    # "isfinite"). Frontier must grow by 640/128 = 5 per SP row -> cluster_axis applied rank*640.
    frontiers, expected = [], []
    for r in range(rows):
        bs = ttnn.to_torch(bs_dev[r * cols]).float()  # device (r, col0): [1, 1, 640, nblk]
        last = bs[0, 0, chunk_local - 1, :]  # last query row's block scores
        visible = int((~torch.isneginf(last)).sum().item())  # past (finite) + current (+inf force-local)
        frontiers.append(visible)
        expected.append((r * chunk_local + chunk_local - 1) // BLOCK + 1)
    logger.info(f"MSA SP cluster_axis causal frontier per SP row: got={frontiers} expected={expected}")
    assert frontiers == expected, (
        f"cluster_axis per-device causality WRONG: frontier {frontiers} != {expected}. "
        f"A flat ~{expected[0]} on all rows means the mesh-coord per-device chunk_start did not apply."
    )

    # --- (B) full deployed path runs at SP=8xTP=4: finite, right shape, non-degenerate ---
    out = msa_sp_attention_nocache(
        shard(q, True),
        shard(k, True),
        shard(v, True),
        iq_t,
        ik_t,
        mesh_config=mesh_config,
        ccl_manager=ccl,
        cached_len=0,
        s_local=chunk_local,
        scale=scale,
        num_groups=1,
        block_size=128,
        topk_blocks=16,
    )
    dts = ttnn.get_device_tensors(out)
    # reassemble: per col(group) concat the 8 SP rows' 640-row shards along seq; then concat groups (heads)
    groups = [
        torch.cat([ttnn.to_torch(dts[r * cols + c]).float()[:, :G] for r in range(rows)], dim=2) for c in range(cols)
    ]
    full = torch.cat(groups, dim=1)  # [1, NQ, S, HD]
    assert full.shape == (1, NQ, S, HEAD_DIM), f"bad output shape {tuple(full.shape)}"
    assert bool(torch.isfinite(full).all()), "MSA SP output has non-finite values"
    assert full.std().item() > 1e-3, f"MSA SP output degenerate (std={full.std().item():.2e})"
    logger.info(f"MSA SP deployed path OK: shape={tuple(full.shape)} std={full.std().item():.4f}")
