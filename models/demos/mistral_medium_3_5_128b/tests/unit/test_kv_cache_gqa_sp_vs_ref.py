# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write and read-back of the model's own cache shape (2 KV heads per chip, head_dim 128, bf8) on the
chunked-KV substrate at SP=8 x TP=4: per-chunk block-cyclic writes into user-major slots land where the
host inverse (``naturalize``) expects them, and untouched slots stay zero."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.tt.kv_cache import allocate_kv_cache, naturalize, read_slot_kv, write_kv_chunk

from .common import CFG, assert_pcc, randn, to_heads


@pytest.mark.parametrize("capacity, chunk", [(10240, 5120), (10240, 10240)], ids=["two_chunks", "one_shot"])
def test_kv_cache_gqa_sp_write_read(galaxy_mesh, mesh_config, capacity, chunk):
    d, nkv = CFG.head_dim, CFG.num_key_value_heads
    num_layers, num_users = 4, 2
    cache = allocate_kv_cache(
        galaxy_mesh,
        mesh_config,
        num_layers=num_layers,
        max_seq_len=capacity,
        num_users=num_users,
        num_local_kv_heads=nkv // mesh_config.tp,
        head_dim=d,
    )
    assert list(cache.k.shape) == [num_users * num_layers, nkv // mesh_config.tp, capacity // mesh_config.sp, d]

    written = {}
    for user_id, layer_idx, seed in ((1, 2, 61), (0, 3, 62)):
        k = randn(1, nkv, capacity, d, seed=seed)
        v = randn(1, nkv, capacity, d, seed=seed + 100)
        for c in range(capacity // chunk):
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
        written[(user_id, layer_idx)] = (k[0], v[0])

    for user_id in range(num_users):
        k_blk, v_blk = read_slot_kv(galaxy_mesh, cache, user_id)
        for layer_idx in range(num_layers):
            got_k = naturalize(k_blk[layer_idx], capacity, mesh_config.sp, chunk, capacity)
            got_v = naturalize(v_blk[layer_idx], capacity, mesh_config.sp, chunk, capacity)
            if (user_id, layer_idx) in written:
                ref_k, ref_v = written[(user_id, layer_idx)]
                assert_pcc(f"kv_cache_k[u{user_id} l{layer_idx}]", ref_k, got_k, lower=0.999)
                assert_pcc(f"kv_cache_v[u{user_id} l{layer_idx}]", ref_v, got_v, lower=0.999)
            else:
                assert torch.count_nonzero(got_k) == 0 and torch.count_nonzero(got_v) == 0, (user_id, layer_idx)
