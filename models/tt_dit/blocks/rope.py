# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import ttnn
from models.tt_dit.layers.module import Module


@dataclass
class RopeConfig:
    theta: float
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RopeConfig:
        return cls(
            theta=data["theta"],
            mrope_section=data.get("mrope_section"),
            mrope_interleaved=data.get("mrope_interleaved", False),
        )


class RotaryEmbedding(Module):
    def __init__(self, *, head_size: int, config: RopeConfig) -> None:
        super().__init__()

        self.head_size = head_size
        self.config = config

        if config.mrope_section is not None and sum(config.mrope_section) != head_size // 2:
            msg = f"mrope_section {config.mrope_section} must sum to half the head size {head_size}"
            raise ValueError(msg)

    def forward(
        self, positions: ttnn.Tensor | Sequence[ttnn.Tensor], *, dtype: ttnn.DataType
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Returns cos and sin of shape (batch, sequence, head_size).

        `positions` is a (batch, sequence) float32 tensor, or one such tensor per multimodal rope
        axis.
        """
        section = self.config.mrope_section

        if isinstance(positions, ttnn.Tensor):
            positions = [positions]
        elif section is None:
            msg = "this rope has no multimodal sections, so it takes a single position tensor"
            raise ValueError(msg)
        elif len(positions) != len(section):
            msg = f"expected {len(section)} position tensors, got {len(positions)}"
            raise ValueError(msg)

        for axis_positions in positions:
            assert axis_positions.dtype == ttnn.float32
            assert len(axis_positions.shape) == 2

        k = self._inverse_frequencies(positions[0].device(), num_axes=len(positions))

        freqs = None
        for axis_positions, axis_k in zip(positions, k, strict=True):
            # Use multiply instead of matmul for improved accuracy.
            axis_freqs = ttnn.unsqueeze(axis_positions, 2) * ttnn.unsqueeze(axis_k, 0)  # outer product
            freqs = axis_freqs if freqs is None else freqs + axis_freqs

        emb = ttnn.concat([freqs, freqs], dim=-1)
        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)

        return ttnn.typecast(cos, dtype), ttnn.typecast(sin, dtype)

    def _inverse_frequencies(self, device: ttnn.MeshDevice, *, num_axes: int) -> list[ttnn.Tensor]:
        """The inverse frequency of every rotary pair, one (head_size // 2) tensor per position axis.

        Each frequency is owned by exactly one axis; the other axes hold zero there, so the frequencies
        of a multimodal position are the sum over the axes. A single axis gets the whole table.
        """
        size = self.head_size

        index = ttnn.arange(0, size // 2, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        k = ttnn.pow(self.config.theta, index * (-2 / size))

        if num_axes == 1:
            return [k]

        section = self.config.mrope_section
        if self.config.mrope_interleaved:
            slot = ttnn.remainder(index, 3.0)
            spatial = [
                ttnn.logical_and(ttnn.eq(slot, float(axis)), ttnn.lt(index, 3.0 * section[axis])) for axis in (1, 2)
            ]
            owned = [ttnn.rsub(spatial[0] + spatial[1], 1.0), *spatial]
        else:
            bounds = list(itertools.accumulate(section, initial=0))
            owned = [
                ttnn.logical_and(ttnn.ge(index, float(start)), ttnn.lt(index, float(end)))
                for start, end in itertools.pairwise(bounds)
            ]

        return [k * mask for mask in owned]
