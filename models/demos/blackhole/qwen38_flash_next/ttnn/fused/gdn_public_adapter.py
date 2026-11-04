# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Public ``chunk_gated_delta_rule`` adapter for the model's pre-normalized, pre-scaled head-major producer.

Keep rank four for q/k: the public rank-three form normalizes raw q/k inside the kernel.
"""

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import same_buffer


def _token_major(head_major, shape, dims):
    """``head_major`` reshaped to ``shape`` and permuted by ``dims``: a new buffer the caller frees (a reshape that
    is not a view of ``head_major`` is freed here)."""

    reshaped = ttnn.reshape(head_major, shape)
    permuted = ttnn.permute(reshaped, dims)
    if not same_buffer(reshaped, head_major):
        ttnn.deallocate(reshaped)
    return permuted


def chunk_public(q_c, k_c, v, beta_c, g_c, initial_state, chunk_tiles, *, rows_total, program_config=None):
    heads, _, chunk, dim = tuple(q_c.shape)
    if rows_total % chunk or tuple(k_c.shape) != tuple(q_c.shape):
        raise ValueError("q/k must have matching complete head-major chunks")
    if tuple(q_c.shape) != (heads, rows_total // chunk, chunk, dim):
        raise ValueError("rows_total does not match the head-major chunks")
    q, k = (_token_major(x, (1, heads, rows_total, dim), (0, 2, 1, 3)) for x in (q_c, k_c))
    # gdn_pre_rows writes beta and g head-major per chunk ([12, NC, 32, 1]); the op takes them token-major.
    g, beta = (_token_major(x, (1, heads, rows_total), (0, 2, 1)) for x in (g_c, beta_c))
    eye, tril, ones, masks = chunk_tiles
    try:
        return ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            ttnn.reshape(v, (1, rows_total, heads * dim)),
            g,
            beta,
            # gdn_pre_rows has already applied both q scale folds.
            scale=1.0,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk,
            output_head_major=True,
            program_config=program_config,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
    finally:
        for tensor in (q, k, g, beta):
            ttnn.deallocate(tensor)
