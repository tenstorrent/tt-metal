# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.paged_update_cache`` (decode KV cache update).

Model call site (modules/attention/attention_1d.py):
  * L1033-1034  _kv_update_decode_nonfused — writes the current decode step's K and
                V head tensors into the KV cache at each user's current position:

            ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(values, v_heads, update_idxs_tensor=current_pos, page_table=page_table)

Decode inputs are ``[1, batch, n_kv_heads, head_dim]`` (head axis padded to 32,
matching tests/ttnn/.../test_paged_update_cache.py) and ``current_pos`` is a
per-user position tensor. The op's device validate requires the new-KV input to be
HEIGHT_SHARDED, one user per core: shard width == head_dim, shard grid num_cores ==
num_users, ROW_MAJOR (paged_update_cache_device_operation.cpp:231-266). This matches
the model, where k_heads/v_heads come out of nlp_create_qkv_heads_decode already
sharded (decode_create_qkv_head_memcfg, attention_1d.py:685-691). This test uses the
contiguous (non-paged, page_table=None) cache ``[batch, n_kv_heads, max_seq_len,
head_dim]``, updates one token per user, reads back and verifies each user's written
slice matches the input.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

MAX_SEQ_LEN = 128  # cache depth for the contiguous decode cache


@U.with_default_mesh()
@pytest.mark.parametrize(
    "batch",
    [pytest.param(batch, id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_paged_update_cache(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device
    n_kv = U.N_KV_HEADS

    # Contiguous decode cache: [batch, n_kv_heads, max_seq_len, head_dim].
    cache_torch = torch.zeros((batch, n_kv, MAX_SEQ_LEN, U.HEAD_DIM))
    cache_tt = U.to_tt(cache_torch, mesh)

    # Decode K heads: [1, batch, n_kv_heads, head_dim], head axis padded to 32.
    k_torch = U.torch_rand((1, batch, n_kv, U.HEAD_DIM))
    k_pad = torch.nn.functional.pad(k_torch, (0, 0, 0, U.TILE - n_kv), "constant", 0.0)
    # HEIGHT_SHARDED one user per core: shard (padded_heads=32, head_dim=64) over
    # `batch` cores (num_cores==num_users) — the op's canonical decode layout
    # (paged_update_cache_device_operation.cpp:258-266).
    k_memcfg = U.height_sharded_batch_memcfg(mesh, batch, (U.TILE, U.HEAD_DIM))
    k_heads = U.to_tt(k_pad, mesh, memory_config=k_memcfg)

    # One update position per user (all within MAX_SEQ_LEN).
    positions = [3 + i for i in range(batch)]
    current_pos = ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )

    result = ttnn.experimental.paged_update_cache(cache_tt, k_heads, update_idxs_tensor=current_pos, page_table=None)
    cache_out = result if result is not None else cache_tt

    # Reference: write each user's head vector at its position.
    expected = cache_torch.clone()
    for i in range(batch):
        # x_view: [n_kv_heads, 1, head_dim] (permute(1,2,0,3) then index user i on the unpadded input).
        x_view = k_torch.permute(1, 2, 0, 3)[i]  # [n_kv, 1, head_dim]
        pos = positions[i]
        expected[i, 0:n_kv, pos : pos + 1, :] = x_view

    U.assert_pcc(expected, cache_out, pcc=0.99, mesh_device=mesh)
