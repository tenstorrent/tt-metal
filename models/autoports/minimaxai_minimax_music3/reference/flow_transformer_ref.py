# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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
#
# Vendored from diffusers (main, 040c7cde626504d14caf63b13b8b25b6a9f62120):
#   src/diffusers/models/transformers/transformer_minimax_music3.py   (MiniMaxMusic3Transformer1DModel)
#   src/diffusers/models/condition_embedders/condition_embedder_minimax_music3.py (MiniMaxMusic3ConditionEncoder)
#   src/diffusers/models/embeddings.py                                 (TimestepEmbedding, silu variant only)
# Trimmed to plain torch (no ConfigMixin / attention dispatch / gradient checkpointing) so the
# stage-05 tests can run the fp32 reference inside tt-metal's python_env, which has no diffusers.

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMaxMusic3FourierEmbedding(nn.Module):
    """Random Fourier features over the flow-matching time in `[0, 1]`. The projection is a trained checkpoint weight."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * timestep.unsqueeze(-1) @ self.weight.T
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class TimestepEmbedding(nn.Module):
    """diffusers ``TimestepEmbedding`` with ``act_fn="silu"`` and no conditioning projection."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class MiniMaxMusic3RotaryEmbedding(nn.Module):
    """Partial rotary embedding: only the first `rotary_dim` dimensions of each head rotate."""

    def __init__(self, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / self.rotary_dim))
        steps = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(steps, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos().contiguous(), freqs.sin().contiguous()


def _apply_partial_rotary_emb(
    hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    # hidden_states: [batch, seq, heads, head_dim]; only the leading rotary dims rotate.
    cos, sin = rotary_emb
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :].to(hidden_states.dtype)
    sin = sin[:, None, :].to(hidden_states.dtype)
    rotated = hidden_states[..., :rotary_dim]
    half_first, half_second = rotated.chunk(2, dim=-1)
    rotate_half = torch.cat((-half_second, half_first), dim=-1)
    rotated = rotated * cos + rotate_half * sin
    return torch.cat((rotated, hidden_states[..., rotary_dim:]), dim=-1)


class MiniMaxMusic3Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        query = _apply_partial_rotary_emb(query, rotary_emb)
        key = _apply_partial_rotary_emb(key, rotary_emb)
        # dispatch_attention_fn(native backend): [B, S, H, D] -> permute -> F.scaled_dot_product_attention -> [B, S, H, D]
        out = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)
        out = out.flatten(2, 3).to(query.dtype)
        out = self.to_out[0](out)
        return self.to_out[1](out)


class MiniMaxMusic3TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniMaxMusic3Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb)
        gate_states, gate = self.ff_in(self.norm2(hidden_states)).chunk(2, dim=-1)
        hidden_states = hidden_states + self.ff_out(gate_states * torch.nn.functional.silu(gate))
        return hidden_states


class MiniMaxMusic3Transformer1DModel(nn.Module):
    r"""
    The flow-matching diffusion transformer of MiniMax Music 3. It denoises Flow-VAE audio latents conditioned on
    per-frame hidden states produced by the autoregressive language-model stage.

    Inputs are 1D latent sequences of shape `(batch, in_channels, length)`. The conditioning signal
    (`encoder_hidden_states`, shape `(batch, length, condition_dim)`) must already be aligned to the latent timeline.
    The flow-matching `timestep` runs from 0 (noise) to 1 (data).
    """

    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_embedding_dim: int = 256,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        # The transformer input concatenates [latent, zeros(in_channels), condition] along channels.
        concat_channels = 2 * in_channels + condition_dim

        self.time_proj = MiniMaxMusic3FourierEmbedding(fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(fourier_embedding_dim, inner_dim)

        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.rotary_emb = MiniMaxMusic3RotaryEmbedding(rotary_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxMusic3TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim)
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        zeros = torch.zeros_like(hidden_states)
        hidden_states = torch.cat((hidden_states, zeros, encoder_hidden_states.transpose(1, 2)), dim=1)
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)

        temb = self.time_embed(self.time_proj(timestep))

        hidden_states = self.proj_in(hidden_states)
        # The timestep embedding is prepended as one extra token and removed after the blocks.
        hidden_states = torch.cat((temb.unsqueeze(1), hidden_states), dim=1)
        rotary_emb = self.rotary_emb(hidden_states.shape[1], hidden_states.device)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(hidden_states[:, 1:])
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states
        return hidden_states


class MiniMaxMusic3ConditionEncoder(nn.Module):
    r"""
    Projects the per-frame hidden states of the autoregressive stage onto the Flow-VAE latent timeline.

    Each generated frame carries `num_condition_layers` hidden states of size `condition_hidden_dim` (one from the
    language model and one per residual codebook step). They are mixed with learned softmax weights, projected, and
    resampled from the language-model frame rate to the latent frame rate with nearest-neighbor interpolation.
    """

    def __init__(
        self,
        condition_hidden_dim: int = 4096,
        num_condition_layers: int = 8,
        out_dim: int = 2048,
        input_sampling_rate: int = 24000,
        input_hop_length: int = 960,
        output_sampling_rate: int = 44100,
        output_hop_length: int = 512,
    ):
        super().__init__()
        self.condition_hidden_dim = condition_hidden_dim
        self.num_condition_layers = num_condition_layers
        self.input_sampling_rate = input_sampling_rate
        self.input_hop_length = input_hop_length
        self.output_sampling_rate = output_sampling_rate
        self.output_hop_length = output_hop_length
        self.layer_weight_logits = nn.Parameter(torch.zeros(num_condition_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)

    def latent_length(self, num_frames: int) -> int:
        return max(
            1,
            int(
                num_frames
                * self.output_sampling_rate
                / self.input_sampling_rate
                * self.input_hop_length
                / self.output_hop_length
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = hidden_states.shape
        num_layers = self.num_condition_layers
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, num_layers, self.condition_hidden_dim, num_frames)
        layer_weights = torch.softmax(self.layer_weight_logits, dim=0).to(hidden_states.dtype)
        hidden_states = torch.einsum("blht,l->bht", hidden_states, layer_weights)
        hidden_states = self.layer_scale.to(hidden_states.dtype) * hidden_states
        hidden_states = self.proj(hidden_states)
        hidden_states = F.interpolate(hidden_states, size=self.latent_length(num_frames), mode="nearest")
        return hidden_states.transpose(1, 2)


# ----------------------------------------------------------------------------- loaders (Tenstorrent additions)
def weights_root() -> Path:
    import os

    root = os.environ.get("MM3_WEIGHTS")
    if not root:
        raise RuntimeError("MM3_WEIGHTS is not set (source ~/mm3-bringup/common.sh)")
    return Path(root)


def load_transformer_state_dict(weights_dir: Optional[Path] = None, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """The fp32 ``transformer/`` safetensors shards as one state dict."""
    from safetensors.torch import load_file

    d = Path(weights_dir or weights_root()) / "transformer"
    sd: Dict[str, torch.Tensor] = {}
    for shard in sorted(d.glob("diffusion_pytorch_model*.safetensors")):
        sd.update(load_file(str(shard)))
    return {k: v.to(dtype) for k, v in sd.items()}


def load_transformer(weights_dir: Optional[Path] = None, num_layers: int = 36) -> MiniMaxMusic3Transformer1DModel:
    """fp32 reference DiT (``num_layers`` < 36 keeps only the first blocks, for layer-wise debugging)."""
    sd = load_transformer_state_dict(weights_dir)
    model = MiniMaxMusic3Transformer1DModel(num_layers=num_layers)
    if num_layers < 36:
        sd = {
            k: v for k, v in sd.items() if not k.startswith("transformer_blocks.") or int(k.split(".")[1]) < num_layers
        }
    model.load_state_dict(sd, strict=True)
    return model.eval()


def load_condition_encoder_state_dict(weights_dir: Optional[Path] = None) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = Path(weights_dir or weights_root()) / "condition_encoder" / "diffusion_pytorch_model.safetensors"
    return {k: v.float() for k, v in load_file(str(path)).items()}


def load_condition_encoder(weights_dir: Optional[Path] = None) -> MiniMaxMusic3ConditionEncoder:
    model = MiniMaxMusic3ConditionEncoder()
    model.load_state_dict(load_condition_encoder_state_dict(weights_dir), strict=True)
    return model.eval()
