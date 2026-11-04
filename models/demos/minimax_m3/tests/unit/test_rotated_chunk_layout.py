# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the rotated per-chip chunk layout a mid-slab (multi-turn resume) chunk must be fed in.

make_chunk_input / the shared producer rotate a chunk's tokens (rotate_chunk_tokens) and the MoE padding config
counts each chip's real rows (rotated_chunk_real_counts) from rotated_chunk_positions. These must agree with
  - the KV writer's placement (DeepSeek's rotated_chip_positions, which mirrors update_padded_kv_cache), and
  - the engine's H2D prefill connector, which rotates the chunk on the host before the socket
    (tt-llm-engine ring_sdpa_reshuffle, ported line by line below),
or the tokens are written / roped / MoE-masked at the wrong positions.
"""

import pytest

from models.demos.common.prefill.chunk_layout import (
    rotate_chunk_tokens,
    rotated_chunk_positions,
    rotated_chunk_real_counts,
)
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions, rotated_chip_real_token_counts


def ring_sdpa_reshuffle(tokens, kv_offset, n_c, w):
    """tt-llm-engine include/tt_llm_engine/pipeline/ring_sdpa_reshuffle.hpp, applied with
    c_start = (kv_offset / W) % N_C and intra = kv_offset % W by H2DPrefillPipeline."""
    c_start, intra = (kv_offset // w) % n_c, kv_offset % w
    volume = n_c * w
    out = [None] * volume
    for i in range(w - intra):
        out[c_start * w + i] = tokens[i]
    for i in range(intra):
        out[c_start * w + (w - intra) + i] = tokens[volume - intra + i]
    for k in range(1, n_c):
        col = (c_start + k) % n_c
        s = (w - intra) + (k - 1) * w
        for i in range(w):
            out[col * w + i] = tokens[s + i]
    return out


CASES = [(2, 256), (4, 256), (8, 640)]


@pytest.mark.parametrize("sp,chunk_local", CASES)
def test_rotated_layout_matches_writer_and_engine(sp, chunk_local):
    chunk_global = sp * chunk_local
    tokens = list(range(chunk_global))
    for start in range(0, 2 * chunk_global + 1, 32):
        assert rotated_chunk_positions(start, sp, chunk_local) == rotated_chip_positions(start, sp, chunk_local)
        assert rotate_chunk_tokens(tokens, start, sp) == ring_sdpa_reshuffle(tokens, start, sp, chunk_local)
        for actual_isl in (1, 31, chunk_local, chunk_global // 2 + 7, chunk_global - 1, chunk_global):
            assert rotated_chunk_real_counts(start, actual_isl, sp, chunk_local) == rotated_chip_real_token_counts(
                start, actual_isl, sp, chunk_local
            )
        if start % chunk_global == 0:
            assert rotate_chunk_tokens(tokens, start, sp) == tokens  # chunk-aligned: identity
