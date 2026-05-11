# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro PL-BERT on TTNN: full `AlbertModel` + `bert_encoder` linear (projection).

`TtKokoroPlBertHybrid` keeps the legacy CPU ALBERT path for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import ttnn
from models.experimental.kokoro.reference.kokoro_plbert import KokoroPlBert
from models.experimental.kokoro.tt.preprocess_kokoro_albert import preprocess_kokoro_albert_for_ttnn
from models.experimental.kokoro.tt.preprocessing import preprocess_bert_encoder_linear
from models.experimental.kokoro.tt.ttnn_kokoro_albert import TtKokoroAlbert
from models.experimental.kokoro.tt.ttnn_kokoro_plbert_projection import TtKokoroPlBertProjection


@dataclass(frozen=True)
class TtKokoroPlBertOutput:
    """Mirror of reference outputs; `d_en` comes from TTNN projection."""

    bert_dur: torch.Tensor
    d_en: torch.Tensor
    text_mask: torch.Tensor
    input_lengths: torch.LongTensor


class TtKokoroPlBert(nn.Module):
    """PL-BERT entirely on device: TTNN Albert + TTNN `bert_encoder` projection."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        reference_plbert: KokoroPlBert,
        *,
        activation_dtype=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.reference = reference_plbert
        if activation_dtype is None:
            activation_dtype = ttnn.bfloat16
        self.tt_albert = TtKokoroAlbert(
            mesh_device,
            reference_plbert.albert.config,
            preprocess_kokoro_albert_for_ttnn(reference_plbert.albert, mesh_device, weights_dtype=activation_dtype),
        )
        enc_params = preprocess_bert_encoder_linear(
            reference_plbert.bert_encoder,
            device=mesh_device,
            weights_dtype=activation_dtype,
        )
        self.projection = TtKokoroPlBertProjection(mesh_device, parameters=enc_params)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: Optional[torch.LongTensor] = None,
    ) -> TtKokoroPlBertOutput:
        if input_lengths is None:
            input_lengths = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[-1],
                device=input_ids.device,
                dtype=torch.long,
            )

        text_mask = (
            torch.arange(input_lengths.max(), device=input_ids.device)
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(input_ids.device)

        attn_hf = (~text_mask).int()
        bert_tt = self.tt_albert(input_ids.to(torch.device("cpu")), attn_hf.float())
        d_en_tt = self.projection(bert_tt)
        bert_dur = ttnn.to_torch(bert_tt).to(torch.float32)
        d_en = ttnn.to_torch(d_en_tt).to(torch.float32)
        ttnn.deallocate(bert_tt)
        ttnn.deallocate(d_en_tt)

        return TtKokoroPlBertOutput(
            bert_dur=bert_dur,
            d_en=d_en,
            text_mask=text_mask,
            input_lengths=input_lengths,
        )

    @torch.no_grad()
    def forward_d_en_ttnn(
        self,
        input_ids: torch.LongTensor,
        input_lengths: Optional[torch.LongTensor] = None,
    ) -> tuple[ttnn.Tensor, torch.LongTensor, torch.Tensor]:
        """
        Run Albert + projection on device and return ``d_en`` as a TTNN tensor (no host round-trip).

        Returns:
            ``(d_en_bf16_bct, input_lengths, text_mask)``. Caller must ``ttnn.deallocate(d_en)`` when done.
        """
        if input_lengths is None:
            input_lengths = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[-1],
                device=input_ids.device,
                dtype=torch.long,
            )

        text_mask = (
            torch.arange(input_lengths.max(), device=input_ids.device)
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(input_ids.device)

        attn_hf = (~text_mask).int()
        bert_tt = self.tt_albert(input_ids.to(torch.device("cpu")), attn_hf.float())
        d_en_tt = self.projection(bert_tt)
        ttnn.deallocate(bert_tt)
        return d_en_tt, input_lengths, text_mask


class TtKokoroPlBertHybrid(nn.Module):
    """
    Wraps `KokoroPlBert.albert` (torch) + `TtKokoroPlBertProjection` (TTNN).

    Drop-in shape compatibility with `reference/kokoro_plbert.KokoroPlBert.forward`.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        reference_plbert: KokoroPlBert,
        *,
        activation_dtype=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.reference = reference_plbert
        if activation_dtype is None:
            activation_dtype = ttnn.bfloat16
        params = preprocess_bert_encoder_linear(
            reference_plbert.bert_encoder,
            device=mesh_device,
            weights_dtype=activation_dtype,
        )
        self.projection = TtKokoroPlBertProjection(mesh_device, parameters=params)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: Optional[torch.LongTensor] = None,
    ) -> TtKokoroPlBertOutput:
        if input_lengths is None:
            input_lengths = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[-1],
                device=input_ids.device,
                dtype=torch.long,
            )

        text_mask = (
            torch.arange(input_lengths.max(), device=input_ids.device)
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(input_ids.device)

        outputs = self.reference.albert(input_ids=input_ids, attention_mask=(~text_mask).int())
        bert_dur = outputs.last_hidden_state

        tt_hidden = ttnn.from_torch(
            bert_dur,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        d_en_tt = self.projection(tt_hidden)
        d_en = ttnn.to_torch(d_en_tt).to(torch.float32)

        return TtKokoroPlBertOutput(
            bert_dur=bert_dur,
            d_en=d_en,
            text_mask=text_mask,
            input_lengths=input_lengths,
        )
