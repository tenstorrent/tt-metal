# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Repo-owned PyTorch implementation of Kokoro PLBERT + projection.

This is the first "true reference" step toward SpeechT5-style bring-up:
- we instantiate the PLBERT module ourselves from `config.json`
- we load weights directly from the HF checkpoint (`kokoro-v1_0.pth`)
- we do not depend on upstream `kokoro` Python module code for this block

Model source:
- https://huggingface.co/hexgrad/Kokoro-82M
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel

from .kokoro_config import KokoroConfig


@dataclass(frozen=True)
class KokoroPlBertOutput:
    bert_dur: torch.Tensor
    d_en: torch.Tensor
    text_mask: torch.Tensor
    input_lengths: torch.LongTensor


class KokoroPlBert(nn.Module):
    def __init__(self, albert: AlbertModel, bert_encoder: nn.Linear):
        super().__init__()
        self.albert = albert
        self.bert_encoder = bert_encoder

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: Optional[torch.LongTensor] = None,
    ) -> KokoroPlBertOutput:
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

        outputs = self.albert(input_ids=input_ids, attention_mask=(~text_mask).int())
        bert_dur = outputs.last_hidden_state
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        return KokoroPlBertOutput(bert_dur=bert_dur, d_en=d_en, text_mask=text_mask, input_lengths=input_lengths)


def load_plbert_from_huggingface(
    repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
    device: Optional[str] = None,
) -> KokoroPlBert:
    """
    Load PLBERT + projection weights from the official Kokoro checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Upstream creates: CustomAlbert(AlbertConfig(vocab_size=n_token, **plbert))
    albert_cfg = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    albert = AlbertModel(albert_cfg)
    bert_encoder = nn.Linear(albert.config.hidden_size, cfg["hidden_dim"])

    # Load weights from checkpoint dict (contains keys: bert, bert_encoder, predictor, ...)
    ckpt_name = "kokoro-v1_0.pth"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "bert" not in state or "bert_encoder" not in state:
        raise RuntimeError(f"Unexpected checkpoint format; missing keys in {ckpt_name}: {list(state.keys())[:20]}")

    bert_state = state["bert"]
    # Kokoro checkpoints may store PLBERT weights with a leading "module." prefix.
    if any(k.startswith("module.") for k in bert_state.keys()):
        bert_state = {k[len("module.") :]: v for k, v in bert_state.items()}
    albert.load_state_dict(bert_state, strict=True)
    enc_state = state["bert_encoder"]
    if any(k.startswith("module.") for k in enc_state.keys()):
        enc_state = {k[len("module.") :]: v for k, v in enc_state.items()}
    bert_encoder.load_state_dict(enc_state, strict=True)

    model = KokoroPlBert(albert=albert, bert_encoder=bert_encoder).to(device).eval()
    return model
