# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pinned Blackhole standard, causal paged-SDPA FP32 accumulation footprint.

Matches sdpa_program_factory.cpp's Q scheduling and circular buffers. This is
the Gemma4 full-attention path, without sinks, masks or flexible tensor offsets.
The model's logical context and outer prefill chunks are independent of this
per-core compute-Q admission check. See doc/datatype_sweep/autofix_canonical.
"""

BLACKHOLE_L1_BYTES = 1572864
BLACKHOLE_CB_BASE = 111616


def full_sdpa_fp32_l1_end(
    *, rows, q_chunk, k_chunk, heads, cores, head_dim, q_tile_bytes, kv_tile_bytes, page_table_bytes
):
    assert rows % q_chunk == q_chunk % 32 == k_chunk % 32 == head_dim % 32 == 0
    chunks = rows // q_chunk
    scheduling_unit = 2 if chunks % 2 == 0 else 1
    units = heads * chunks // scheduling_unit
    max_chunks_per_core = ((units + cores - 1) // cores) * scheduling_unit
    q_buffers = 2 if max_chunks_per_core > 1 else 1
    a, b, d = q_chunk // 32, k_chunk // 32, head_dim // 32
    scalar_tile_bytes = 4096 if q_tile_bytes == 4096 else 2048
    payload = (
        a * d * q_buffers * q_tile_bytes  # Q
        + 4 * b * d * kv_tile_bytes  # double-buffered K and V
        + 2 * 2048  # lightweight causal mask
        + 2 * scalar_tile_bytes
        + page_table_bytes
        + a * b * 4096  # FP32 QK
        + 2 * a * d * 2048  # BF16 output intermediates
        + 3 * a * 2048  # maxima and exp difference
        + 2 * a * 4096  # FP32 sums
        + a * d * q_tile_bytes  # output uses Q dtype
    )
    return BLACKHOLE_CB_BASE + payload
