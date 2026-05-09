# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-side helpers for MLA paged KV layout (no DeepSeek / fragile HF imports)."""

from __future__ import annotations

import torch

from models.demos.mistral_small_4_119B.tt_utils.config_helpers import even_int_div
from models.tt_transformers.tt.common import PagedAttentionConfig


def paged_cache_from_torch(
    torch_cache: torch.Tensor,
    mesh_shape: tuple[int, int],
    paged_config: PagedAttentionConfig,
    user_id: int | None,
    mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a torch KVPE cache into the flat paged layout expected by Mistral MLA device code.

    Same algorithm as DeepSeek ``paged_cache_from_torch``; kept here so Mistral tests do not import
    ``models.demos.deepseek_v3.utils.test_utils`` (which pulls optional / version-mismatched HF symbols).

    Args:
        torch_cache: Shape ``(batch_size, num_heads, seq_len, dim)`` with ``num_heads == 1``.
        mesh_shape: Mesh ``(rows, cols)``.
        paged_config: Block size and max block count.
        user_id: Optional row placement (see DeepSeek doc).
        mapping: Optional fixed page map; if ``None``, a random permutation is used.
    """
    if user_id is not None:
        torch_cache_line = torch_cache
        batch_size_per_row = torch_cache_line.shape[0]
        row_start = (user_id // batch_size_per_row) * batch_size_per_row
        row_end = row_start + batch_size_per_row
        torch_cache = torch.zeros(
            (mesh_shape[0] * batch_size_per_row, *torch_cache_line.shape[1:]),
            dtype=torch_cache_line.dtype,
        )
        torch_cache[row_start:row_end] = torch_cache_line

    batch_size, num_heads, seq_len, dim = torch_cache.shape
    batches_per_device = even_int_div(batch_size, mesh_shape[0] * mesh_shape[1])
    blocks_per_batch = even_int_div(paged_config.max_num_blocks, batches_per_device)
    assert num_heads == 1, "Expected the kvpe cache to have only one head"

    if mapping is None:
        mapping = torch.randperm(batches_per_device * blocks_per_batch).reshape(batches_per_device, blocks_per_batch)
    assert mapping.shape == (batches_per_device, blocks_per_batch)

    assert paged_config.block_size * blocks_per_batch >= seq_len
    torch_cache = torch.nn.functional.pad(torch_cache, (0, 0, 0, paged_config.block_size * blocks_per_batch - seq_len))

    torch_cache = torch_cache.reshape(
        mesh_shape[0] * mesh_shape[1], batches_per_device, num_heads, blocks_per_batch, paged_config.block_size, dim
    )
    torch_cache = torch_cache.transpose(2, 3)

    paged_cache = torch.empty(
        (mesh_shape[0] * mesh_shape[1], batches_per_device * blocks_per_batch, num_heads, paged_config.block_size, dim),
        dtype=torch_cache.dtype,
    )
    paged_cache[:, mapping] = torch_cache
    paged_cache = paged_cache.reshape(
        mesh_shape[0] * mesh_shape[1] * batches_per_device * blocks_per_batch, num_heads, paged_config.block_size, dim
    )

    return paged_cache, mapping
