# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M on TT + PyTorch: full PL-BERT (TTNN Albert + projection); predictor and ISTFT decoder on PyTorch.

Predictor/vocoder TTNN ports use `conv1d`, `conv_transpose2d` (1D), packed LSTM loops, and AdaIN — see
`ttnn_kokoro_albert.py` / `TtKokoroPlBert` for the completed text encoder front-end.
"""

from __future__ import annotations

import json
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import ttnn
from models.demos.kokoro.reference.kokoro_config import KokoroConfig
from models.demos.kokoro.reference.kokoro_full_model import KokoroFullOutput
from models.demos.kokoro.reference.kokoro_istftnet import KokoroIstftNet, load_decoder_from_huggingface
from models.demos.kokoro.reference.kokoro_plbert import load_plbert_from_huggingface
from models.demos.kokoro.reference.kokoro_predictor import KokoroPredictor, load_predictor_from_huggingface
from models.demos.kokoro.tt.ttnn_kokoro_plbert import TtKokoroPlBert


class KokoroTtHybridFull(nn.Module):
    """
    Kokoro inference with PL-BERT on TTNN (`TtKokoroPlBert`); acoustic stack and vocoder on PyTorch (`torch_device`).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
        torch_device: Optional[str] = None,
        disable_complex: bool = False,
    ):
        super().__init__()
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mesh_device = mesh_device
        self.repo_id = repo_id
        self.torch_device = torch_device

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.vocab: dict[str, int] = cfg["vocab"]
        self.context_length: int = int(cfg["plbert"]["max_position_embeddings"])

        plbert_cpu = load_plbert_from_huggingface(repo_id=repo_id, device="cpu")
        self.tt_plbert = TtKokoroPlBert(mesh_device, plbert_cpu)
        self.predictor: KokoroPredictor = load_predictor_from_huggingface(repo_id=repo_id, device=torch_device)
        self.decoder: KokoroIstftNet = load_decoder_from_huggingface(
            repo_id=repo_id, device=torch_device, disable_complex=disable_complex
        )

    def phonemes_to_input_ids(self, phonemes: str) -> torch.LongTensor:
        ids = [self.vocab.get(p) for p in phonemes]
        ids = [i for i in ids if i is not None]
        if len(ids) + 2 > self.context_length:
            raise ValueError(f"Too many tokens: {len(ids)+2} > context_length={self.context_length}")
        return torch.tensor([[0, *ids, 0]], dtype=torch.long, device="cpu")

    @torch.no_grad()
    def forward(
        self,
        *,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_intermediates: bool = False,
    ) -> KokoroFullOutput:
        input_ids = self.phonemes_to_input_ids(phonemes)
        return self.forward_with_tokens(input_ids, ref_s, speed=speed, return_intermediates=return_intermediates)

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_intermediates: bool = False,
    ) -> KokoroFullOutput:
        dev = torch.device(self.torch_device)
        input_ids = input_ids.to(torch.device("cpu"))
        ref_s = ref_s.to(dev)

        plbert_out = self.tt_plbert(input_ids)
        pred_out = self.predictor(
            d_en=plbert_out.d_en.to(dev),
            ref_s=ref_s,
            input_ids=input_ids.to(dev),
            input_lengths=plbert_out.input_lengths.to(dev),
            text_mask=plbert_out.text_mask.to(dev),
            speed=speed,
        )
        dec_out = self.decoder(asr=pred_out.asr, F0_pred=pred_out.F0_pred, N_pred=pred_out.N_pred, ref_s=ref_s)
        audio = dec_out.audio

        if not return_intermediates:
            return KokoroFullOutput(audio=audio.squeeze().cpu(), pred_dur=pred_out.pred_dur.detach().cpu())

        return KokoroFullOutput(
            audio=audio.squeeze().cpu(),
            pred_dur=pred_out.pred_dur.detach().cpu(),
            bert_dur=plbert_out.bert_dur.detach().cpu(),
            d_en=plbert_out.d_en.detach().cpu(),
            d=pred_out.d.detach().cpu(),
            duration=pred_out.duration.detach().cpu(),
            pred_aln_trg=pred_out.pred_aln_trg.detach().cpu(),
            en=pred_out.en.detach().cpu(),
            F0_pred=pred_out.F0_pred.detach().cpu(),
            N_pred=pred_out.N_pred.detach().cpu(),
            t_en=pred_out.t_en.detach().cpu(),
            asr=pred_out.asr.detach().cpu(),
        )
