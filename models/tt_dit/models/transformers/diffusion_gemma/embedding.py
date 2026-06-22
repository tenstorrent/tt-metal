# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scaled token embedding used by both DiffusionGemma encoder and decoder.

Mirrors ``Gemma4TextScaledWordEmbedding`` (inherited by
``DiffusionGemmaTextScaledWordEmbedding``): a regular embedding table whose
output is scaled by ``sqrt(hidden_size)``.

The scale lives as a learned ``embed_scale`` parameter (scalar) in HF — its
state-dict value is ``sqrt(hidden_size)`` after init. We match that layout so
``load_state_dict`` works directly.
"""

from __future__ import annotations

import math

import torch

import ttnn

from ....layers.module import Module, Parameter


class DiffusionGemmaScaledWordEmbedding(Module):
    """Embedding table + sqrt(hidden_size) scaling.

    Weights are kept replicated across the mesh (no TP on the vocab axis for
    now — vocab=262144 × hidden=2816 in bf16 ≈ 1.4 GB which fits per-device).
    Future optimization: row-shard the vocab axis (EP-style).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int | None = 0,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.weight = Parameter(
            total_shape=[vocab_size, hidden_size],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # HF stores embed_scale as a scalar buffer with value sqrt(hidden_size).
        # We allocate it as a tile-aligned [1, 1] parameter for ttnn.multiply.
        self.embed_scale = Parameter(total_shape=[1, 1], device=mesh_device, dtype=ttnn.bfloat16)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # HF embed_scale is a 0-d or 1-d buffer; reshape to [1, 1] for ttnn.
        if "embed_scale" in state:
            state["embed_scale"] = state["embed_scale"].reshape(1, 1)
        elif "weight" in state:
            # Some HF model variants don't materialize embed_scale as a tensor (only the
            # `embed_scale` python attribute). Synthesize sqrt(hidden_size) for those.
            state["embed_scale"] = torch.tensor([[math.sqrt(self.hidden_size)]], dtype=state["weight"].dtype)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """input_ids: int [B, S] on device → embedded [B, S, hidden_size] bf16 replicated."""
        embedded = ttnn.embedding(input_ids, self.weight.data, layout=ttnn.TILE_LAYOUT)
        return ttnn.multiply(embedded, self.embed_scale.data)
