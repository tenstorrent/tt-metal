# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 sinusoidal positional embedding.

PyTorch reference is
`models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::sinusoidal_positional_embedding_forward`
which is a padding-aware index lookup into a pre-computed
``[num_embeddings, hidden_size]`` sinusoidal weight table (the NLLB-style
``sin || cos`` table, with row ``padding_idx`` zeroed). The HuggingFace block
this mirrors is ``SeamlessM4Tv2SinusoidalPositionalEmbedding``.

Forward strategy:

1. Compute ``position_ids`` from the host integer ``input_ids`` (or a simple
   ``arange`` when ``inputs_embeds`` is provided). This step is index-only
   arithmetic on small integer tensors -- doing it on host is a fair call and
   matches the loading pattern in
   ``models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py``
   for absolute position lookups.
2. Run the lookup itself on device via ``ttnn.embedding`` against the
   pre-loaded sinusoidal weight table.

The sinusoidal weight table is fixed for a given ``(num_embeddings,
hidden_size, padding_idx)`` so we materialize it once at construction (via
``build_sinusoidal_positional_embedding_weights`` from the reference) and
upload it to DRAM in ``ROW_MAJOR_LAYOUT`` -- the layout ``ttnn.embedding``
expects for the gather table.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _create_position_ids_from_input_ids(
    input_ids: torch.Tensor,
    padding_idx: int,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Host-side padding-aware position id computation.

    Mirrors the reference ``_create_position_ids_from_input_ids`` helper:
    non-padding tokens are replaced by their per-row cumulative index
    (offset by ``padding_idx + 1`` and ``past_key_values_length``); padding
    positions keep ``padding_idx``.
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class SinusoidalPositionalEmbedding(LightweightModule):
    """SeamlessM4T-v2 sinusoidal (NLLB-style) positional embedding.

    Args:
        device: ttnn device or mesh device.
        weights: torch.Tensor of shape ``(num_embeddings, hidden_size)``
            holding the pre-computed sinusoidal table. Row ``padding_idx``
            is expected to already be zero (HF semantics).
        padding_idx: padding token id (default 1, matching SeamlessM4T-v2).
        weight_dtype: storage dtype for the sinusoidal table on device.
        weight_memory_config: where to place the table (DRAM by default).
    """

    def __init__(
        self,
        device,
        weights: torch.Tensor,
        padding_idx: int = 1,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.padding_idx = int(padding_idx)
        self.num_embeddings, self.hidden_size = int(weights.shape[0]), int(weights.shape[1])

        # ttnn.embedding's gather table must be ROW_MAJOR.
        self.weight = ttnn.from_torch(
            weights.to(torch.float32),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds_shape: Optional[tuple] = None,
        past_key_values_length: int = 0,
        precomputed_position_ids_tt: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Gather sinusoidal rows for the requested positions.

        Args:
            input_ids: optional integer (host-side) torch tensor of shape
                ``(batch, seq_len)``. When provided, positions are derived
                from the non-padding mask (padding-aware path).
            inputs_embeds_shape: optional ``(batch, seq_len)`` tuple used
                when only embeddings are available -- positions are then
                sequential ``padding_idx + 1 .. padding_idx + seq_len``.
            past_key_values_length: number of cached tokens to offset by
                during incremental decoding.
            precomputed_position_ids_tt: optional pre-allocated device
                tensor of shape ``(batch, seq_len)`` uint32 ROW_MAJOR
                holding the position ids the caller wants gathered. When
                supplied, the host-side position computation + H2D
                upload is skipped entirely (lets the gather op be
                captured into a metal trace). The caller is responsible
                for writing the correct ids into this buffer before each
                call (via ``ttnn.copy_host_to_device_tensor``).

        Returns:
            ttnn TILE_LAYOUT tensor of shape
            ``(batch, seq_len, hidden_size)`` placed in DRAM.
        """
        if precomputed_position_ids_tt is not None:
            return ttnn.embedding(
                precomputed_position_ids_tt,
                self.weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if (input_ids is None) == (inputs_embeds_shape is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds_shape must be provided.")

        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            position_ids = _create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        else:
            bsz, seq_len = inputs_embeds_shape
            position_ids = torch.arange(
                self.padding_idx + 1,
                seq_len + self.padding_idx + 1,
                dtype=torch.long,
            )
            position_ids = position_ids.unsqueeze(0).expand((bsz, seq_len)).contiguous() + past_key_values_length

        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.num_embeddings:
            raise ValueError(
                f"Sinusoidal weight table too small: have {self.num_embeddings} rows, "
                f"need at least {max_pos} for seq_len={seq_len}, "
                f"padding_idx={self.padding_idx}, past_key_values_length={past_key_values_length}."
            )

        # Upload the (small) integer position ids to device for the gather.
        # ttnn.embedding expects uint32 ROW_MAJOR ids of shape (batch, seq).
        tt_position_ids = ttnn.from_torch(
            position_ids.to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return ttnn.embedding(
            tt_position_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
