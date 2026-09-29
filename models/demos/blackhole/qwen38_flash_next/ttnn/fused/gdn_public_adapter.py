# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Public-op adapter for the merged owner prerequisite 96cc4a7937.

The input contract is Samuel Jett's pre-normalized, pre-scaled head-major
producer at cd9a11771107ea2c27da3303a0556ff7343e4af5. Keep rank four for q/k:
the public rank-three form normalizes raw q/k inside the kernel.
"""

import ttnn


def chunk_public(q_c, k_c, v, beta_c, g_c, initial_state, chunk_tiles, *, rows_total, program_config=None):
    heads, _, chunk, dim = tuple(q_c.shape)
    if rows_total % chunk or tuple(k_c.shape) != tuple(q_c.shape):
        raise ValueError("q/k must have matching complete head-major chunks")
    if tuple(q_c.shape) != (heads, rows_total // chunk, chunk, dim):
        raise ValueError("rows_total does not match the head-major chunks")
    q = ttnn.permute(ttnn.reshape(q_c, (1, heads, rows_total, dim)), (0, 2, 1, 3))
    k = ttnn.permute(ttnn.reshape(k_c, (1, heads, rows_total, dim)), (0, 2, 1, 3))
    # Invert the composite's headvec_split_tile and per-chunk column reshape.
    g = ttnn.permute(ttnn.reshape(g_c, (1, heads, rows_total)), (0, 2, 1))
    beta = ttnn.permute(ttnn.reshape(beta_c, (1, heads, rows_total)), (0, 2, 1))
    eye, tril, ones, masks = chunk_tiles
    return ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        ttnn.reshape(v, (1, rows_total, heads * dim)),
        g,
        beta,
        # The source producer has already applied both of its q scale folds.
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
