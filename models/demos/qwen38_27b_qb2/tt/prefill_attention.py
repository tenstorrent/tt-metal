# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Equal-length paged prefill, with an opt-in batched cache/attention boundary."""

import ttnn


def make_batch_indices(mesh, maximum=32):
    """Allocate local-row maps during model setup, before any trace reserves scratch."""
    import torch

    return {
        batch: ttnn.from_torch(
            torch.arange(batch, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for batch in range(2, maximum + 1)
    }


def fill_and_attend(
    q, k, v, key, value, page_table, start_pos, *, page_size, scale, program_config, batch_indices=None
):
    """Keep physical page ownership and causal offsets identical in both paths.

    batch_indices maps input rows to this already-sliced page table, not to
    global request slots or physical cache blocks. Its lifetime belongs to the
    caller; allocate it before capturing a trace that uses this boundary.
    """
    batch, _, length, _ = q.shape
    if page_table.shape[0] != batch or k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError("Prefill Q/K/V and page-table batches must match")
    first_page = start_pos // page_size
    end_page = (start_pos + length + page_size - 1) // page_size
    if start_pos < 0 or start_pos % page_size or end_page > page_table.shape[-1]:
        raise ValueError("Prefill requires a page-aligned, covered continuation")
    if batch_indices is not None:
        if tuple(batch_indices.shape) != (batch,):
            raise ValueError("Prefill batch indices must contain one local page-table row per input")
        chunk_table = page_table[:, first_page:end_page]
        for cache, update in ((key, k), (value, v)):
            if update.dtype != cache.dtype:
                update = ttnn.typecast(update, cache.dtype)
            ttnn.experimental.paged_fill_cache(cache, update, chunk_table, batch_idx_tensor=batch_indices, batch_idx=0)
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            q, key, value, page_table, start_pos, scale=scale, program_config=program_config
        )
    outputs = []
    for user in range(batch):
        table = page_table[user : user + 1, :]
        chunk_table = table[:, first_page:end_page]
        for cache, update in ((key, k), (value, v)):
            part = update[user : user + 1, :, :, :]
            if part.dtype != cache.dtype:
                part = ttnn.typecast(part, cache.dtype)
            ttnn.experimental.paged_fill_cache(cache, part, chunk_table, batch_idx=0)
        outputs.append(
            ttnn.transformer.chunked_scaled_dot_product_attention(
                q[user : user + 1, :, :, :],
                key,
                value,
                table,
                start_pos,
                scale=scale,
                program_config=program_config,
            )
        )
    return outputs[0] if batch == 1 else ttnn.concat(outputs, dim=0)
