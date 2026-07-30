# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fully device-resident KDA prefill adapter."""

from __future__ import annotations

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn


def kda_prefill(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    const_tiles: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
    summary_group_chunks: int = 8,
    sequence_parallel_axis: int | None = None,
    affine_identity: ttnn.Tensor | None = None,
    affine_zero: ttnn.Tensor | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Execute chunk-parallel prefill with persistent KDA state."""
    key_dim = q.shape[-1] // beta.shape[-1] if len(q.shape) == 3 else q.shape[-1]
    if len(q.shape) == 4:
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)
    output, final_state = ttnn.transformer.chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        scale=key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        output_head_major=len(q.shape) == 3,
        chunk_size=32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        eye=const_tiles[0],
        tril=const_tiles[1],
        ones=const_tiles[2],
        masks=const_tiles[3],
        summary_group_chunks=summary_group_chunks,
        sequence_parallel_axis=sequence_parallel_axis,
        affine_identity=affine_identity,
        affine_zero=affine_zero,
    )
    assert final_state is not None
    if len(q.shape) == 4:
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return output, final_state
