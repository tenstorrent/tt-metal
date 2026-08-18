# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.paged_fused_update_cache`` (fused decode KV update).

Model call site (modules/attention/attention_1d.py):
  * L1019-1021  _kv_update_decode_fused — writes the current decode step's K and V
                heads into their caches in a single fused kernel (use_qk_fused=True):

            ttnn.experimental.paged_fused_update_cache(
                keys, k_heads, values, v_heads, update_idxs_tensor=current_pos, page_table=page_table
            )

Argument order is (keys_cache, k_heads, values_cache, v_heads). Decode inputs are
``[1, batch, n_kv_heads, head_dim]`` (head axis padded to 32) and ``current_pos``
is a per-user position tensor. Both new-KV inputs must be HEIGHT_SHARDED one user per
core (shard width == head_dim, num_cores == num_users, K and V grids same core count;
paged_fused_update_cache_device_operation.cpp:145-195) — the fused sibling of
paged_update_cache. This test uses contiguous (non-paged, page_table=None) K and V
caches ``[batch, n_kv_heads, max_seq_len, head_dim]``, updates one token per user,
reads both caches back and verifies each user's written K and V slice matches the input.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

MAX_SEQ_LEN = 128  # cache depth for the contiguous decode cache


@U.with_default_mesh()
@pytest.mark.parametrize(
    "batch",
    [pytest.param(batch, id=f"decode-batch{batch}") for batch in U.DECODE_BATCHES],
)
def test_paged_fused_update_cache(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device
    n_kv = U.N_KV_HEADS

    # Contiguous decode caches for K and V: [batch, n_kv_heads, max_seq_len, head_dim].
    keys_torch = torch.zeros((batch, n_kv, MAX_SEQ_LEN, U.HEAD_DIM))
    values_torch = torch.zeros((batch, n_kv, MAX_SEQ_LEN, U.HEAD_DIM))
    keys_tt = U.to_tt(keys_torch, mesh)
    values_tt = U.to_tt(values_torch, mesh)

    # Decode K/V heads: [1, batch, n_kv_heads, head_dim], head axis padded to 32.
    k_torch = U.torch_rand((1, batch, n_kv, U.HEAD_DIM))
    v_torch = U.torch_rand((1, batch, n_kv, U.HEAD_DIM))
    # HEIGHT_SHARDED one user per core: shard (32, head_dim) over `batch` cores.
    # The fused op requires K and V on NON-OVERLAPPING grids
    # (paged_fused_update_cache_device_operation.cpp:227 "!is_overlap"), so V is placed on the
    # cores after K — same non-overlap pattern as the fused-QK Q/K split.
    k_memcfg = U.height_sharded_batch_memcfg(mesh, batch, (U.TILE, U.HEAD_DIM))
    v_start = ttnn.CoreCoord(batch % 8, batch // 8)
    v_memcfg = U.height_sharded_batch_memcfg(mesh, batch, (U.TILE, U.HEAD_DIM), start_core=v_start)
    k_heads = U.to_tt(
        torch.nn.functional.pad(k_torch, (0, 0, 0, U.TILE - n_kv), "constant", 0.0), mesh, memory_config=k_memcfg
    )
    v_heads = U.to_tt(
        torch.nn.functional.pad(v_torch, (0, 0, 0, U.TILE - n_kv), "constant", 0.0), mesh, memory_config=v_memcfg
    )

    positions = [3 + i for i in range(batch)]
    current_pos = ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )

    result = ttnn.experimental.paged_fused_update_cache(
        keys_tt, k_heads, values_tt, v_heads, update_idxs_tensor=current_pos, page_table=None
    )
    if result is not None:
        keys_out, values_out = result
    else:
        keys_out, values_out = keys_tt, values_tt

    # Reference: write each user's K/V head vector at its position.
    keys_expected = keys_torch.clone()
    values_expected = values_torch.clone()
    for i in range(batch):
        pos = positions[i]
        keys_expected[i, 0:n_kv, pos : pos + 1, :] = k_torch.permute(1, 2, 0, 3)[i]
        values_expected[i, 0:n_kv, pos : pos + 1, :] = v_torch.permute(1, 2, 0, 3)[i]

    U.assert_pcc(keys_expected, keys_out, pcc=0.99, mesh_device=mesh)
    U.assert_pcc(values_expected, values_out, pcc=0.99, mesh_device=mesh)
