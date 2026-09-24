# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TtSWA``: the sliding-window attention of V4-Flash layers 0 and 1 = ``TtHCA`` without a compressor.

Same q/kv stems, per-head sinks, 128-token window carry across chunks, "-i" un-rope and grouped o-projection;
no compressed entries, so the keys are ``[carry | chunk | pad]`` and Sk is ``128 + chunk`` (cheaper than HCA).
Rope type is ``"main"`` (theta 10000, no yarn) -- the model-level ``DeepseekV4RotaryEmbedding`` is passed in,
since a sliding layer has no compressor to borrow one from. Block I/O ``[1, 1, S/sp, hidden/tp]``.

Chunking: ``prepare_input(hidden, sp_factor, TILE)`` pads the slab to a multiple of ``32 * sp``; a non-final
chunk must be >= one window (128) and a multiple of 32 (the engine's 5120 is); only the final chunk may be
ragged. ``alloc_state(max_seq_len, chunk_tokens=...)`` once per layer, then ``forward(x, seq_len_actual, state=)``.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA, TtHCAState

__all__ = ["TtSWA", "TtHCAState"]


class TtSWA(TtHCA):
    def __init__(self, device, **kwargs):
        kwargs.setdefault("compressor", None)
        kwargs.setdefault("rope_layer_type", "main")
        assert kwargs["compressor"] is None, "TtSWA has no compressor; use TtHCA for compressed layers"
        super().__init__(device, **kwargs)

    @staticmethod
    def prepare_input(hidden, sp_factor: int, compress_rate: int = ttnn.TILE_SIZE):
        """Pad the slab so every SP shard is tile-aligned (there are no compression windows to honour)."""
        return TtHCA.prepare_input(hidden, sp_factor, ttnn.TILE_SIZE)

    @classmethod
    def from_reference(cls, device, reference, config, rotary_emb=None, **kwargs) -> "TtSWA":
        assert reference.compressor is None, "TtSWA.from_reference expects a sliding_attention layer"
        assert rotary_emb is not None, "pass the model-level DeepseekV4RotaryEmbedding (rotated with 'main')"
        return super().from_reference(device, reference, config, rotary_emb=rotary_emb, **kwargs)
