# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.paged_fill_cache`` (prefill KV cache fill).

Model call site (modules/attention/attention_1d.py):
  * L877-878  _kv_fill_prefill_paged — fills the paged K and V caches for one
              user from the prefill head tensors:

            ttnn.experimental.paged_fill_cache(keys, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values, v_fill_sliced, fill_page_table, batch_idx=user_id)

  * L911-912  _kv_fill_prefill_batched — same op, one call per user in a loop.

Cache layout is paged: ``[num_blocks, n_kv_heads, block_size, head_dim]``. The
fill input is a per-user head tensor ``[1, n_kv_heads, seq, head_dim]`` and the
page table maps the user's logical blocks to physical blocks. This test fills a
zero cache for a single user, reads it back, and verifies the written blocks
match the input (via PCC over the whole cache — unwritten blocks stay zero in
both reference and result).
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

BLOCK_SIZE = 64  # paged KV block size (tile-aligned)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "seq",
    [pytest.param(seq, id=f"prefill-seq{seq}") for seq in U.PREFILL_SEQ_LENS],
)
def test_paged_fill_cache(ttnn_mesh_device, reset_seeds, seq):
    mesh = ttnn_mesh_device
    num_users = 1
    batch_idx = 0
    num_blocks_per_seq = seq // BLOCK_SIZE
    num_blocks = num_users * num_blocks_per_seq

    # Zero paged cache: [num_blocks, n_kv_heads, block_size, head_dim].
    cache_torch = torch.zeros((num_blocks, U.N_KV_HEADS, BLOCK_SIZE, U.HEAD_DIM))
    cache_tt = U.to_tt(cache_torch, mesh)

    # Per-user fill input: [1, n_kv_heads, seq, head_dim].
    fill_torch = U.torch_rand((1, U.N_KV_HEADS, seq, U.HEAD_DIM))
    fill_tt = U.to_tt(fill_torch, mesh)

    # Page table [num_users, num_blocks_per_seq]; arange -> logical block i == physical block i.
    page_table_torch = torch.arange(num_blocks, dtype=torch.int32).reshape(num_users, num_blocks_per_seq)
    page_table_tt = ttnn.from_torch(
        page_table_torch,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )

    ttnn.experimental.paged_fill_cache(cache_tt, fill_tt, page_table_tt, batch_idx=batch_idx)

    # Expected cache: reshape the seq axis into (num_blocks, block_size) and move blocks to axis 0.
    expected = fill_torch[0].reshape(U.N_KV_HEADS, num_blocks, BLOCK_SIZE, U.HEAD_DIM).permute(1, 0, 2, 3).contiguous()

    U.assert_pcc(expected, cache_tt, pcc=0.99, mesh_device=mesh)
