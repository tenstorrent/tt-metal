# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro PL-BERT on TTNN: full `AlbertModel` + `bert_encoder` linear (projection).

Padding masks are built without a torch ``arange``/compare graph; Albert uses device
``ttnn.arange`` / ``ttnn.zeros`` and a TTNN-only extended attention bias (no HF
``get_extended_attention_mask`` torch path).

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


def _padding_token_mask(input_lengths: torch.LongTensor, max_len: int) -> torch.Tensor:
    """
    ``text_mask[b, j]`` is True past valid length (padding), matching the prior ``torch.arange`` logic.

    Built with plain Python + a single ``torch.tensor`` so PL-BERT does not depend on a torch
    ``arange``/compare subgraph for masking (host I/O still returns ``torch.Tensor`` for callers).
    """
    bsz = int(input_lengths.shape[0])
    rows: list[list[bool]] = []
    for bi in range(bsz):
        L = int(input_lengths[bi].item())
        rows.append([(j + 1) > L for j in range(max_len)])
    return torch.tensor(rows, dtype=torch.bool, device=input_lengths.device)


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

        max_t = int(input_lengths.max())
        text_mask = _padding_token_mask(input_lengths, max_t)

        attn_hf = (~text_mask).int()
        ids_cpu = input_ids.detach().contiguous()
        if ids_cpu.device.type != "cpu":
            ids_cpu = ids_cpu.cpu()
        bert_tt = self.tt_albert(ids_cpu, attn_hf.float())
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

        max_t = int(input_lengths.max())
        text_mask = _padding_token_mask(input_lengths, max_t)

        attn_hf = (~text_mask).int()
        ids_cpu = input_ids.detach().contiguous()
        if ids_cpu.device.type != "cpu":
            ids_cpu = ids_cpu.cpu()
        bert_tt = self.tt_albert(ids_cpu, attn_hf.float())
        d_en_tt = self.projection(bert_tt)
        ttnn.deallocate(bert_tt)
        return d_en_tt, input_lengths, text_mask
