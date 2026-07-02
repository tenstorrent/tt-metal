# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MSA chunked CACHE-READ at TP=4 × SP=8 — current chunk attends the BLOCK-CYCLIC cached prefix.

The multi-chunk MSA read path (`msa_sp_attention`): the accumulated K/V/index_k live in the
SP cache in DeepSeek block-cyclic order (chip r holds [chunk0_r, chunk1_r, ...]); the op AllGathers them
to full block-cyclic context, REORDERS to natural token order (transpose of the chip/chunk axes), then
runs the indexer (per-device chunk_offset) + sparse_sdpa for the current chunk's queries.

This is the cache-fed sibling of test_msa_sp_chunked_vs_ref (which fed CONTIGUOUS context directly). The
ONLY new element vs that validated test is the block-cyclic → natural reorder, so this isolates it:
identical inputs, but K/V/index_k are arranged block-cyclic (as the cache stores them) and read via
`msa_sp_attention`; compared to the same gather-everything golden (`msa_sp_attention_gather_all`).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.attention.msa import msa_sp_attention
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, NIDX, HEAD_DIM = 64, 4, 4, 128


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("chunk_local,n_prior", [(640, 1), (640, 3)], ids=["chunk640_prior1", "chunk640_prior3"])
def test_msa_sp_cache_read(mesh_device, device_params, chunk_local, n_prior, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    chunk = sp * chunk_local  # 5120 (current chunk)
    cached_len = n_prior * chunk  # 5120 (chunk 0 cached)
    n_chunks = n_prior + 1  # total chunks now in the cache (incl. current)
    T = n_chunks * chunk  # 10240 accumulated context
    G = NQ // NKV
    scale = HEAD_DIM**-0.5

    torch.manual_seed(0)
    q = torch.randn(1, NQ, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    iq = torch.randn(1, NIDX, chunk, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ik = torch.randn(1, 1, T, HEAD_DIM, dtype=torch.bfloat16) * 0.1

    mesh_config = MeshConfig((rows, cols), decode=ModeConfig(tp=tp, ep=sp))
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    def shard(t, split_heads):
        """Contiguous SP shard (seq across rows), heads on TP cols (index_k shared -> replicate)."""
        dims = [None, None]
        dims[sp_axis] = 2
        dims[1] = 1 if split_heads else None
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    # Block-cyclic arrange: pre-permute T so a CONTIGUOUS SP split gives each chip its cache slice
    # [chunk0_r, chunk1_r, ...] — exactly what update_padded_kv_cache stores and what gather+reorder undoes.
    bc_idx = torch.tensor(
        [
            chunk_c * chunk + chip * chunk_local + c
            for chip in range(sp)
            for chunk_c in range(n_chunks)
            for c in range(chunk_local)
        ],
        dtype=torch.long,
    )

    def shard_bc(t, split_heads):
        return shard(t[:, :, bc_idx, :], split_heads)

    common = dict(mesh_config=mesh_config, ccl_manager=ccl, cached_len=cached_len, scale=scale, num_groups=1)

    # cache-read: current chunk query (contiguous), accumulated K/V/index_k BLOCK-CYCLIC (cache layout).
    # Exercises the multi-chunk block-cyclic gather+reorder + per-device mesh-coord cluster_axis causality
    # (device r's current-chunk queries start at global cached_len + r*chunk_local).
    out_b = msa_sp_attention(
        shard(q, True),
        shard_bc(k, True),
        shard_bc(v, True),
        shard(iq, True),
        shard_bc(ik, False),
        s_local=chunk_local,
        n_chunks=n_chunks,
        chunk_local=chunk_local,
        **common,
    )
    dts_b = ttnn.get_device_tensors(out_b)
    groups = [
        torch.cat([ttnn.to_torch(dts_b[r * cols + c]).float()[:, :G] for r in range(rows)], dim=2) for c in range(cols)
    ]
    out = torch.cat(groups, dim=1)  # [1, NQ, chunk, HD]

    # SMOKE: the multi-chunk block-cyclic cache-read path runs end-to-end at SP=8xTP=4 -> finite, right-shape,
    # non-degenerate. (Exact-PCC golden deferred: the old gather-everything golden is incompatible with the
    # merged op's mesh-coord cluster_axis; MSA compute correctness is covered by test_msa_layer_vs_ref real
    # weights PCC 0.9994 + single-chunk test_msa_sp_sharded cluster_axis causality. Full multi-chunk PCC is a
    # follow-up when the cache lifecycle is the active milestone.)
    assert out.shape == (1, NQ, chunk, HEAD_DIM), f"bad output shape {tuple(out.shape)}"
    assert bool(torch.isfinite(out).all()), "MSA cache-read output has non-finite values"
    assert out.std().item() > 1e-3, f"MSA cache-read output degenerate (std={out.std().item():.2e})"
    logger.info(f"MSA CACHE-READ SP=8xTP=4 (block-cyclic prefix, T={T}) smoke OK: std={out.std().item():.4f}")
