# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import ttnn

from ..layers.module import Module


@dataclass
class RopeConfig:
    theta: float
    mrope_section: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RopeConfig:
        return cls(
            theta=data["theta"],
            mrope_section=data.get("mrope_section"),
        )


class RotaryEmbedding(Module):
    def __init__(self, *, head_size: int, config: RopeConfig) -> None:
        super().__init__()

        self.head_size = head_size
        self.config = config

    def forward(self, positions: ttnn.Tensor, *, dtype: ttnn.DataType) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        assert len(positions.shape) == 2
        assert positions.dtype == ttnn.float32

        device = positions.device()
        size = self.head_size
        theta = self.config.theta

        # https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/models/llama/modeling_llama.py#L73
        k = ttnn.pow(theta, ttnn.arange(0, size, 2, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) / -size)

        if self.config.mrope_section is not None:
            warnings.warn("mrope_section is not implemented yet", stacklevel=2)
            # this only seems to affect decode mode
            # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1577
            # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L513
            # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L577-L583

        freqs = ttnn.unsqueeze(positions, 2) @ ttnn.unsqueeze(k, 0)  # outer product
        emb = ttnn.concat([freqs, freqs], dim=-1)
        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)

        return ttnn.clone(cos, dtype=dtype), ttnn.clone(sin, dtype=dtype)
