# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro Model — uses upstream `kokoro.model.KModel` as reference for PCC validation.

Analogous to `models/experimental/speecht5_tts/reference/speecht5_model.py`:
wraps the official implementation so tests and demos share one stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from .kokoro_config import KokoroConfig


class KokoroModelReference(nn.Module):
    """
    Wrapper around upstream `kokoro.model.KModel` for a consistent interface.

    Use `get_*` accessors for submodules (same role as `SpeechT5ModelReference.get_encoder`, etc.).
    """

    def __init__(
        self,
        repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
        device: Optional[str] = None,
        disable_complex: bool = False,
    ):
        super().__init__()
        from kokoro.model import KModel  # upstream

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.repo_id = repo_id
        self.device_name = device
        self.kmodel = KModel(repo_id=repo_id, disable_complex=disable_complex).to(device).eval()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.kmodel.parameters())

    def get_bert(self) -> nn.Module:
        return self.kmodel.bert

    def get_bert_encoder(self) -> nn.Module:
        return self.kmodel.bert_encoder

    def get_predictor(self) -> nn.Module:
        return self.kmodel.predictor

    def get_text_encoder(self) -> nn.Module:
        return self.kmodel.text_encoder

    def get_decoder(self) -> nn.Module:
        return self.kmodel.decoder

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_output: bool = False,
    ) -> Union[torch.FloatTensor, "KModelOutput"]:
        out = self.kmodel(phonemes, ref_s, speed=speed, return_output=True)
        if return_output:
            return KModelOutput(audio=out.audio, pred_dur=out.pred_dur)
        return out.audio

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        return self.kmodel.forward_with_tokens(input_ids=input_ids, ref_s=ref_s, speed=speed)

    @property
    def bert(self) -> nn.Module:
        return self.kmodel.bert

    @property
    def bert_encoder(self) -> nn.Module:
        return self.kmodel.bert_encoder

    @property
    def predictor(self) -> nn.Module:
        return self.kmodel.predictor

    @property
    def text_encoder(self) -> nn.Module:
        return self.kmodel.text_encoder

    @property
    def decoder(self) -> nn.Module:
        return self.kmodel.decoder


@dataclass(frozen=True)
class KModelOutput:
    audio: torch.FloatTensor
    pred_dur: Optional[torch.LongTensor] = None


def load_reference_model(
    repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
    device: Optional[str] = None,
    disable_complex: bool = False,
) -> KokoroModelReference:
    """Load the upstream Kokoro reference (`KModel`), mirroring `load_reference_model` for SpeechT5."""
    return KokoroModelReference(repo_id=repo_id, device=device, disable_complex=disable_complex)


load_reference_kmodel = load_reference_model
