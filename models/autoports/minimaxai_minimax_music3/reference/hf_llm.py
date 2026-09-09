# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The audio-frame embedding formula and the teacher-forced loop below are transcribed from
# diffusers ``src/diffusers/modular_pipelines/minimax_music3/encoders.py``
# (``_embed_audio_frame`` and ``MiniMaxMusic3AutoregressiveStep.__call__``), Apache-2.0,
# Copyright 2025 The HuggingFace Team. Only the language-model side is kept here; the RVQ depth
# decoder and the samplers belong to later stages.
"""Host (CPU, torch) reference pieces for the MiniMax-Music3 Qwen3 backbone.

Everything here is test-side: loading ``Qwen3ForCausalLM`` in bf16, loading the depth decoder's
``audio_embeddings`` table, the exact frame-embedding formula the AR loop feeds back into the
backbone, and a teacher-forced HF run that yields per-step normed hidden states and logits.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CODE_OFFSET, AUDIO_VOCAB_SIZE, NUM_CODEBOOKS


def hf_model_dir() -> Path:
    """The Qwen3 backbone directory (``HF_MODEL``), as tt_transformers also reads it."""
    path = os.environ.get("HF_MODEL")
    if not path:
        raise RuntimeError("HF_MODEL is not set; source ~/mm3-bringup/common.sh")
    return Path(path)


def weights_dir() -> Path:
    """The MiniMax-Music3 snapshot directory (``MM3_WEIGHTS``)."""
    path = os.environ.get("MM3_WEIGHTS")
    if path:
        return Path(path)
    return hf_model_dir().parent


def reference_dir() -> Path:
    """Golden CPU outputs from the diffusers run (stage 01)."""
    return Path(os.environ.get("MM3_REF", str(Path.home() / "mm3-bringup" / "reference")))


def load_hf_qwen3(dtype: torch.dtype = torch.bfloat16):
    """Load the full ``Qwen3ForCausalLM`` backbone on CPU (about 16 GB in bf16)."""
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained(str(hf_model_dir()), dtype=dtype)
    model.eval()
    return model


def load_embed_weight() -> torch.Tensor:
    """Only ``model.embed_tokens.weight`` (bf16) straight from the safetensors shards."""
    import json

    from safetensors import safe_open

    root = hf_model_dir()
    index = json.loads((root / "model.safetensors.index.json").read_text())
    key = "model.embed_tokens.weight"
    shard = index["weight_map"][key]
    with safe_open(str(root / shard), "pt") as f:
        return f.get_tensor(key)


def load_audio_embeddings() -> torch.Tensor:
    """The depth decoder's ``audio_embeddings.weight`` [(NUM_CODEBOOKS-1)*AUDIO_VOCAB_SIZE, 4096] bf16."""
    from safetensors import safe_open

    path = weights_dir() / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors"
    with safe_open(str(path), "pt") as f:
        table = f.get_tensor("audio_embeddings.weight")
    assert table.shape == ((NUM_CODEBOOKS - 1) * AUDIO_VOCAB_SIZE, 4096), table.shape
    return table


def residual_embedding_sum(audio_embeddings: torch.Tensor, frame_codes: torch.Tensor) -> torch.Tensor:
    """Sum of the 7 residual-code embeddings, un-scaled: ``sum_k audio_embeddings[c_k + (k-1)*1024]``.

    ``frame_codes`` is [B, NUM_CODEBOOKS]; column 0 (the semantic code) is ignored here.
    """
    offsets = (torch.arange(NUM_CODEBOOKS - 1) * AUDIO_VOCAB_SIZE).unsqueeze(0)
    idx = frame_codes[:, 1:].to(torch.long) + offsets
    return audio_embeddings[idx].sum(dim=1)  # [B, 4096]


def embed_audio_frame(
    embed_weight: torch.Tensor, audio_embeddings: torch.Tensor, frame_codes: torch.Tensor
) -> torch.Tensor:
    """diffusers ``_embed_audio_frame``: ``(embed_tokens(c0 + OFFSET) + sum_k audio_emb(c_k + (k-1)*1024)) * 8**-0.5``.

    Returns [B, 4096] in the embedding table's dtype (bf16 for the checkpoint), computed exactly as
    HF does it: the residual sum is added in the embedding dtype before the scale is applied.
    """
    embeds = embed_weight[frame_codes[:, 0].to(torch.long) + AUDIO_CODE_OFFSET]
    extra = residual_embedding_sum(audio_embeddings, frame_codes)
    embeds = embeds + extra.to(embeds.dtype)
    return embeds * NUM_CODEBOOKS**-0.5


@torch.no_grad()
def hf_prefill(model, inputs_embeds: torch.Tensor):
    """HF prefill from embeddings. Returns (past_key_values, normed last hidden [B, 4096], logits [B, vocab] fp32)."""
    out = model.model(inputs_embeds=inputs_embeds, use_cache=True)
    last_hidden = out.last_hidden_state[:, -1]
    logits = model.lm_head(last_hidden).float()
    return out.past_key_values, last_hidden, logits


@torch.no_grad()
def hf_decode_step(model, past_key_values, inputs_embeds: torch.Tensor):
    """One HF decode step from [B, 1, 4096] embeddings. Returns (past, normed hidden [B, 4096], logits fp32)."""
    out = model.model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
    last_hidden = out.last_hidden_state[:, -1]
    logits = model.lm_head(last_hidden).float()
    return out.past_key_values, last_hidden, logits


@torch.no_grad()
def hf_teacher_forced(model, text_ids: torch.Tensor, frame_codes: torch.Tensor, audio_embeddings: torch.Tensor):
    """Prefill ``text_ids`` [B, L] then feed ``frame_codes`` [N, NUM_CODEBOOKS], one frame per step.

    Every frame is fed to both batch rows (the AR loop repeats the sampled codes so the
    conditional / unconditional rows stay aligned). Returns ``(prefill_hidden, prefill_logits, steps)`` with ``steps`` a list of (hidden, logits)
    per teacher-forced frame; all tensors fp32 on CPU.
    """
    embed_weight = model.model.embed_tokens.weight
    batch = text_ids.shape[0]
    past, hidden0, logits0 = hf_prefill(model, model.model.embed_tokens(text_ids))
    steps = []
    for codes in frame_codes:
        feedback = embed_audio_frame(embed_weight, audio_embeddings, codes.unsqueeze(0).expand(batch, -1))
        past, hidden, logits = hf_decode_step(model, past, feedback.unsqueeze(1))
        steps.append((hidden.float(), logits))
    return hidden0.float(), logits0, steps
