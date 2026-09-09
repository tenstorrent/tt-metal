# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# Copyright 2026 Tenstorrent USA, Inc. (torch-only transcription)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch reference of ``MiniMaxMusic3RVQDepthDecoder`` (host side of the stage-03 tests).

Transcribed from diffusers ``src/diffusers/models/transformers/minimax_music3_rvq_depth_decoder.py``
(diffusers main, 040c7cde) with the ``ModelMixin`` / ``AttentionModuleMixin`` scaffolding removed
so it imports in tt-metal's ``python_env`` (its diffusers 0.38 predates the class). Semantics are
identical: causal SDPA over the depth sequence, RMSNorm eps 1e-6, SwiGLU, learned position
embedding, and the seven residual-codebook heads. Also holds the teacher-forced depth loop of
``modular_pipelines/minimax_music3/encoders.py::_generate_depth_codes`` (sampling replaced by the
given codes) that the golden-frame test replays.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS

DEPTH_HIDDEN = 4096
DEPTH_LAYERS = 4
DEPTH_HEADS = 16
DEPTH_INTERMEDIATE = 6144
DEPTH_MAX_POSITIONS = 16
DEPTH_NORM_EPS = 1e-6


class RMSNorm(nn.Module):
    """diffusers ``RMSNorm(dim, eps, elementwise_affine=True)``: fp32 statistics, weight applied in the weight dtype."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.to_q(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.to_out(o.transpose(1, 2).reshape(b, s, -1))


class DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate: int):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, DEPTH_NORM_EPS)
        self.attn = DepthAttention(dim, heads)
        self.post_attention_layernorm = RMSNorm(dim, DEPTH_NORM_EPS)
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x))
        n = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(n)) * self.up_proj(n))


class RVQDepthDecoderRef(nn.Module):
    """``MiniMaxMusic3RVQDepthDecoder`` with the checkpoint's fixed geometry."""

    def __init__(self):
        super().__init__()
        self.audio_embeddings = nn.Embedding(AUDIO_VOCAB_SIZE * (NUM_CODEBOOKS - 1), DEPTH_HIDDEN)
        self.projection = nn.Linear(DEPTH_HIDDEN, DEPTH_HIDDEN, bias=False)
        self.pos_embedding = nn.Embedding(DEPTH_MAX_POSITIONS, DEPTH_HIDDEN)
        self.layers = nn.ModuleList(
            [DepthDecoderBlock(DEPTH_HIDDEN, DEPTH_HEADS, DEPTH_INTERMEDIATE) for _ in range(DEPTH_LAYERS)]
        )
        self.norm = RMSNorm(DEPTH_HIDDEN, DEPTH_NORM_EPS)
        self.audio_heads = nn.ModuleList(
            [nn.Linear(DEPTH_HIDDEN, AUDIO_VOCAB_SIZE, bias=False) for _ in range(NUM_CODEBOOKS - 1)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        h = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


def load_state_dict(weights_dir: Path) -> dict:
    """The bf16 safetensors of ``rvq_depth_decoder`` as a plain ``{name: tensor}`` dict."""
    from safetensors.torch import load_file

    return load_file(str(Path(weights_dir) / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors"))


def load_reference(weights_dir: Path, dtype: torch.dtype = torch.float32) -> RVQDepthDecoderRef:
    model = RVQDepthDecoderRef()
    state = {k: v.to(dtype) for k, v in load_state_dict(weights_dir).items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return model.to(dtype).eval()


@torch.no_grad()
def teacher_forced_depth_loop(
    model: RVQDepthDecoderRef, global_hidden: torch.Tensor, semantic_embed: torch.Tensor, residual_codes: torch.Tensor
):
    """``_generate_depth_codes`` with the sampled codes replaced by ``residual_codes``.

    ``global_hidden`` [B, 4096] is the backbone's last normed hidden, ``semantic_embed`` [B, 4096] the
    backbone's ``embed_tokens(c0 + AUDIO_CODE_OFFSET)``, ``residual_codes`` [B, 7] the codes c1..c7.
    Returns ``(hiddens, logits)``: seven [B, 4096] normed last-step hiddens (step k = the hidden that
    feeds ``audio_heads[k]``) and the seven [B, 1024] head logits.
    """
    sequence = [model.projection(global_hidden).unsqueeze(1), model.projection(semantic_embed).unsqueeze(1)]
    hiddens, logits = [], []
    for index in range(1, NUM_CODEBOOKS):
        hidden = model(torch.cat(sequence, dim=1))[:, -1]
        hiddens.append(hidden)
        logits.append(model.audio_heads[index - 1](hidden))
        if index < NUM_CODEBOOKS - 1:
            code = residual_codes[:, index - 1]
            embed = model.audio_embeddings(code + (index - 1) * AUDIO_VOCAB_SIZE)
            sequence.append(model.projection(embed).unsqueeze(1))
    return hiddens, logits
