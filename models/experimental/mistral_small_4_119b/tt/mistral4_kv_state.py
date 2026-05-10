# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-layer K/V state for Mistral4 TTNN self-attention decode (bring-up).

Holds device tensors shaped ``[1, num_heads, T, qk_head_dim]`` matching HF ``DynamicCache``
after prefill / decode steps. See :class:`Mistral4DecoderStackKvState` for one object per
decoder block in :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderSequence`.
"""

from __future__ import annotations

import ttnn


class Mistral4AttentionKvState:
    """K/V for a single decoder layer's self-attention."""

    __slots__ = ("key_states", "value_states")

    def __init__(self) -> None:
        self.key_states: ttnn.Tensor | None = None
        self.value_states: ttnn.Tensor | None = None

    @property
    def populated(self) -> bool:
        return self.key_states is not None and self.value_states is not None

    def clear(self) -> None:
        if self.key_states is not None:
            ttnn.deallocate(self.key_states)
            self.key_states = None
        if self.value_states is not None:
            ttnn.deallocate(self.value_states)
            self.value_states = None

    def replace(self, key_states: ttnn.Tensor, value_states: ttnn.Tensor) -> None:
        """Take ownership of ``key_states`` / ``value_states``; frees any prior buffers."""
        self.clear()
        self.key_states = key_states
        self.value_states = value_states


class Mistral4DecoderStackKvState:
    """
    Fixed-depth tuple of :class:`Mistral4AttentionKvState`, index ``i`` matches ``sequence.blocks[i]``.
    """

    __slots__ = ("layers",)

    def __init__(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = tuple(Mistral4AttentionKvState() for _ in range(num_layers))

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Mistral4AttentionKvState:
        return self.layers[idx]

    def clear(self) -> None:
        for s in self.layers:
            s.clear()
