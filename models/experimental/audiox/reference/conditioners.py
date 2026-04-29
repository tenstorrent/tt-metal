# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioX conditioner stack (reference / CPU).

The AudioX HF config (HKUSTAudio/AudioX) wires three cross-attention
conditioners — CLIP (video), T5 (text), AudioAutoencoder (audio) — into a
``MultiConditioner`` dispatcher. Conditioners run **once per generation**,
producing tensors of shape [B, S_k, 768] that get concatenated along the
sequence dim into ``cross_attn_cond`` for the DiT (which runs ~50 steps × 24
blocks per call). Conditioner compute is <0.1% of total — porting them to
TTNN would not move the needle, so this layer stays on CPU and reuses the
HuggingFace models directly. The TTNN side ingests the concatenated tensor
via ``ttnn.from_torch`` at the DiT boundary.

This module currently ports ``Conditioner``, ``T5Conditioner`` (the only
heavy conditioner with no zero-input fast-path), and the
``MultiConditioner`` dispatcher. CLIP and AudioAutoencoder land in
follow-up chunks alongside their dependencies (CLIP visual encoder +
SA temporal transformer, oobleck VAE encoder).
"""

import typing as tp

import torch
from torch import nn


class Conditioner(nn.Module):
    """Base class: optional output projection so each conditioner emits the
    shared ``cond_token_dim`` (768 for AudioX) regardless of its native dim."""

    def __init__(self, dim: int, output_dim: int, project_out: bool = False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any, device: tp.Any) -> tp.Any:
        raise NotImplementedError()


class T5Conditioner(Conditioner):
    """Wraps a HuggingFace T5 encoder + an output projection. AudioX uses
    ``t5-base`` (dim 768) with ``max_length=128``, padded to fixed length so
    the cross-attn sequence shape is constant across generations.

    The HF model is held outside ``self`` (via ``__dict__``) so it does not
    appear in ``state_dict`` — matches upstream and keeps checkpoint loading
    of the surrounding pipeline clean."""

    T5_MODELS = ("t5-small", "t5-base", "t5-large", "google/flan-t5-base", "google/flan-t5-large")
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
    }

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        max_length: int = 128,
        project_out: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        from transformers import AutoTokenizer, T5EncoderModel

        self.t5_model_name = t5_model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        # Held outside state_dict; loaded weights stay on the HF model side.
        self.__dict__["model"] = T5EncoderModel.from_pretrained(t5_model_name).eval().requires_grad_(False)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.to(device)
        self.proj_out.to(device)

        with torch.no_grad():
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        embeddings = self.proj_out(embeddings.float())
        # Zero-out padded positions so they contribute nothing through cross-attn.
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask


class MultiConditioner(nn.Module):
    """Dispatcher: maps a list of per-sample metadata dicts into a dict of
    (cond_tensor, mask) tuples keyed by conditioner id. Mirrors upstream
    ``audiox/models/conditioners.py:MultiConditioner`` for inference (no
    drop-out, no negative-prompt handling here — the diffusion wrapper owns
    that)."""

    def __init__(
        self,
        conditioners: tp.Dict[str, Conditioner],
        default_keys: tp.Optional[tp.Dict[str, str]] = None,
    ):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys or {}

    def forward(
        self,
        batch_metadata: tp.List[tp.Dict[str, tp.Any]],
        device: tp.Union[torch.device, str],
    ) -> tp.Dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            inputs = []
            for sample in batch_metadata:
                ck = key if key in sample else self.default_keys.get(key, key)
                if ck not in sample:
                    raise ValueError(f"Conditioner key '{ck}' not found in batch metadata")
                value = sample[ck]
                # Upstream unwraps single-element list/tuple but keeps multi-element ones.
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                inputs.append(value)
            output[key] = conditioner(inputs, device)
        return output
