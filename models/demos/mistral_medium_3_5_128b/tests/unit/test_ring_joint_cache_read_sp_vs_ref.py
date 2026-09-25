# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ring-joint SDPA reading K/V out of the block-cyclic KV cache: the Q of chunk ``i`` attends the
accumulated prefix [0, (i+1)*C) written chunk by chunk — the mechanism chunked prefill depends on.
The 4-chunk-capacity case checks the ring-gather buffer spans the whole cache, not just the prefix."""

import pytest

from models.demos.mistral_medium_3_5_128b.reference.model import causal_attention
from models.demos.mistral_medium_3_5_128b.tt.attention import ring_sdpa_cache_read
from models.demos.mistral_medium_3_5_128b.tt.kv_cache import allocate_kv_cache, write_kv_chunk

from .common import CFG, assert_pcc, heads_to_torch, randn, to_heads


@pytest.mark.parametrize(
    "capacity, chunk, q_chunk_idx",
    [(10240, 5120, 1), (20480, 5120, 2)],
    ids=["chunk1_of_2", "chunk2_of_4"],
)
def test_ring_joint_cache_read_sp_vs_ref(galaxy_mesh, mesh_config, ccl_manager, capacity, chunk, q_chunk_idx):
    d, nkv = CFG.head_dim, CFG.num_key_value_heads
    prefix = (q_chunk_idx + 1) * chunk
    k = randn(1, nkv, prefix, d, seed=51)
    v = randn(1, nkv, prefix, d, seed=52)
    q = randn(1, CFG.num_attention_heads, chunk, d, seed=53)
    ref = causal_attention(q, k, v, q_offset=q_chunk_idx * chunk)

    user_id, layer_idx = 1, 1
    cache = allocate_kv_cache(
        galaxy_mesh,
        mesh_config,
        num_layers=2,
        max_seq_len=capacity,
        num_users=2,
        num_local_kv_heads=nkv // mesh_config.tp,
        head_dim=d,
    )
    for c in range(q_chunk_idx + 1):
        sl = slice(c * chunk, (c + 1) * chunk)
        write_kv_chunk(
            cache,
            to_heads(k[:, :, sl], galaxy_mesh, mesh_config),
            to_heads(v[:, :, sl], galaxy_mesh, mesh_config),
            user_id=user_id,
            layer_idx=layer_idx,
            kv_actual=c * chunk,
            sp_axis=mesh_config.sp_axis,
        )
    out = ring_sdpa_cache_read(
        to_heads(q, galaxy_mesh, mesh_config),
        cache,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        user_id=user_id,
        layer_idx=layer_idx,
        kv_actual=q_chunk_idx * chunk,
        chunk_global=chunk,
        scale=d**-0.5,
    )
    assert_pcc(f"ring_joint_cache_read[chunk {q_chunk_idx}]", ref, heads_to_torch(out, galaxy_mesh, mesh_config))
