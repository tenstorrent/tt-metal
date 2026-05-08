# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Complete PyTorch reference for Kokoro-82M (repo-owned blocks, HF weights).

Analogous to `models/experimental/speecht5_tts/reference/speecht5_full_model.py`:
- stable full-model entrypoint for PCC and TT bring-up
- composes op-by-op torch modules loaded from Hugging Face (`hexgrad/Kokoro-82M`)

Components:
- KokoroPlBert
- KokoroPredictor
- KokoroIstftNet
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .kokoro_config import KokoroConfig
from .kokoro_istftnet import KokoroIstftNet, load_decoder_from_huggingface
from .kokoro_plbert import KokoroPlBert, load_plbert_from_huggingface
from .kokoro_predictor import KokoroPredictor, load_predictor_from_huggingface


@dataclass(frozen=True)
class KokoroFullOutput:
    audio: torch.FloatTensor
    pred_dur: torch.LongTensor

    bert_dur: Optional[torch.Tensor] = None
    d_en: Optional[torch.Tensor] = None
    d: Optional[torch.Tensor] = None
    duration: Optional[torch.Tensor] = None
    pred_aln_trg: Optional[torch.Tensor] = None
    en: Optional[torch.Tensor] = None
    F0_pred: Optional[torch.Tensor] = None
    N_pred: Optional[torch.Tensor] = None
    t_en: Optional[torch.Tensor] = None
    asr: Optional[torch.Tensor] = None


class KokoroFullReference(nn.Module):
    """
    Full Kokoro reference (repo-owned torch, weights from HF).

    Use `forward_with_tokens` when you already have `input_ids`, or call the module
    with `phonemes=` for the same path as upstream `KModel(phonemes, ...)`.
    """

    def __init__(
        self,
        *,
        repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
        device: Optional[str] = None,
        disable_complex: bool = False,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.repo_id = repo_id
        self.device_name = device

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.vocab: dict[str, int] = cfg["vocab"]
        self.context_length: int = int(cfg["plbert"]["max_position_embeddings"])

        self.plbert: KokoroPlBert = load_plbert_from_huggingface(repo_id=repo_id, device=device)
        self.predictor: KokoroPredictor = load_predictor_from_huggingface(repo_id=repo_id, device=device)
        self.decoder: KokoroIstftNet = load_decoder_from_huggingface(
            repo_id=repo_id, device=device, disable_complex=disable_complex
        )

    def phonemes_to_input_ids(self, phonemes: str) -> torch.LongTensor:
        ids = [self.vocab.get(p) for p in phonemes]
        ids = [i for i in ids if i is not None]
        if len(ids) + 2 > self.context_length:
            raise ValueError(f"Too many tokens: {len(ids)+2} > context_length={self.context_length}")
        dev = next(self.plbert.parameters()).device
        return torch.tensor([[0, *ids, 0]], dtype=torch.long, device=dev)

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
        dev = next(self.plbert.parameters()).device
        input_ids = input_ids.to(dev)
        ref_s = ref_s.to(dev)

        plbert_out = self.plbert(input_ids=input_ids)
        pred_out = self.predictor(
            d_en=plbert_out.d_en,
            ref_s=ref_s,
            input_ids=input_ids,
            input_lengths=plbert_out.input_lengths,
            text_mask=plbert_out.text_mask,
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


def load_full_reference_from_huggingface(
    repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
    device: Optional[str] = None,
    disable_complex: bool = False,
) -> KokoroFullReference:
    model = KokoroFullReference(repo_id=repo_id, device=device, disable_complex=disable_complex)
    # print(model)
    return model


load_full_reference_model = load_full_reference_from_huggingface
