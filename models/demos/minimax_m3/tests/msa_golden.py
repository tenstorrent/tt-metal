# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test-only MSA golden — the gather-everything SP decomposition.

``msa_sp_attention_gather_all`` AllGathers BOTH the query and the context across the SP axis and runs the
indexer + sparse_sdpa over the full chunk with a single uniform ``chunk_start_idx`` — the simplest,
obviously-correct formulation, with a replicated full-chunk output. It is the differential golden the
deployed sharded / cache-read paths (``msa_sp_attention_nocache`` / ``msa_sp_attention``) are validated
against; it is NOT used in the model. It lives here, not in ``tt/attention/msa.py``, so the production
module carries only deployed code. ``msa_indexer_sparse`` (the shared op chain) stays in the model module.
"""

from models.demos.minimax_m3.tt.attention.msa import msa_indexer_sparse


def msa_sp_attention_gather_all(
    q, k, v, index_q, index_k, *, mesh_config, ccl_manager, cached_len, scale, num_groups=1
):
    """MSA sparse attention under SP, gather-everything golden: AllGather K/V/index_k AND q/index_q.

    Inputs are per-device CONTIGUOUS sequence shards on the mesh:
      q        [1, Hq_local, S_local, head_dim]    (TP slices heads; SP slices the query seq)
      k, v     [1, n_kv_local, S_local, head_dim]  (this device's KV sequence shard, TILE)
      index_q  [1, num_groups, S_local, INDEX_DIM]
      index_k  [1, 1, S_local, INDEX_DIM]          (the single shared index-k head, SP-sharded, TILE)

    Every tensor (query included) is gathered to the full context [.., T, ..] (T = S_local * SP) on every
    device, so the indexer scores the whole chunk under one uniform ``chunk_start_idx=cached_len``. The
    output is the full chunk, replicated across the SP devices.
    """
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device

    # AllGather the KEYS across SP (seq dim=2): each device now holds the full context locally.
    k_full = mesh_config.allgather(k, ccl_manager, axis=sp_axis, dim=2)
    v_full = mesh_config.allgather(v, ccl_manager, axis=sp_axis, dim=2)
    index_k_full = mesh_config.allgather(index_k, ccl_manager, axis=sp_axis, dim=2)
    # Gather index-q + q too so the (lightweight) indexer scores the whole chunk under one uniform
    # chunk_start; the deployed paths instead keep the query sharded with a per-device chunk_offset.
    index_q_full = mesh_config.allgather(index_q, ccl_manager, axis=sp_axis, dim=2)
    q_full = mesh_config.allgather(q, ccl_manager, axis=sp_axis, dim=2)

    return msa_indexer_sparse(
        index_q_full,
        index_k_full,
        q_full,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        device=device,
    )
