# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional PyTorch reference blocks for Voxtral TTS.

Each function is standalone and takes tensors, weight dictionaries, and config
objects as arguments. This mirrors the Qwen3-TTS reference style used for TTNN
bring-up tests.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from models.experimental.voxtraltts.reference.voxtral_config import VoxtralConfig, load_voxtral_config


@dataclass(frozen=True)
class VoxtralTextConfig:
    hidden_size: int = 3072
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    vocab_size: int = 131072
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-5


@dataclass(frozen=True)
class VoxtralAcousticConfig:
    input_dim: int = 3072
    hidden_size: int = 3072
    num_hidden_layers: int = 3
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    num_acoustic_codebooks: int = 36
    cfg_alpha: float = 1.2
    noise_scale: float = 1.0
    acoustic_decode_iters: int = 8


def get_default_text_config(model_name_or_path: str | None = None) -> VoxtralTextConfig:
    if model_name_or_path is None:
        return VoxtralTextConfig()
    config = load_voxtral_config(model_name_or_path)
    return VoxtralTextConfig(
        hidden_size=config.dim,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        intermediate_size=config.hidden_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.norm_eps,
    )


def get_default_acoustic_config(model_name_or_path: str | None = None) -> VoxtralAcousticConfig:
    if model_name_or_path is None:
        return VoxtralAcousticConfig()
    config: VoxtralConfig = load_voxtral_config(model_name_or_path)
    acoustic = config.audio_model_args.acoustic_transformer_args
    return VoxtralAcousticConfig(
        input_dim=acoustic.input_dim,
        hidden_size=acoustic.dim,
        num_hidden_layers=acoustic.n_layers,
        num_attention_heads=acoustic.n_heads,
        num_key_value_heads=acoustic.n_kv_heads,
        head_dim=acoustic.head_dim,
        intermediate_size=acoustic.hidden_dim,
        rope_theta=acoustic.rope_theta,
        rms_norm_eps=acoustic.sigma,
        semantic_codebook_size=config.audio_model_args.semantic_codebook_size,
        acoustic_codebook_size=config.audio_model_args.acoustic_codebook_size,
        num_acoustic_codebooks=config.audio_model_args.n_acoustic_codebook,
    )


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (hidden_states * weight.float()).to(input_dtype)


def swiglu_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    return F.linear(F.silu(F.linear(hidden_states, w1)) * F.linear(hidden_states, w3), w2)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(position_ids, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    cos = cos.to(orig_dtype)
    sin = sin.to(orig_dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def text_attention(
    hidden_states: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: VoxtralTextConfig,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, seq_len, _ = hidden_states.shape
    q = F.linear(hidden_states, layer_weights["attention.wq.weight"])
    k = F.linear(hidden_states, layer_weights["attention.wk.weight"])
    v = F.linear(hidden_states, layer_weights["attention.wv.weight"])

    q = q.view(batch, seq_len, config.num_attention_heads, config.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos[:, :seq_len], sin[:, :seq_len])

    n_rep = config.num_attention_heads // config.num_key_value_heads
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (config.head_dim**0.5)
    if attention_mask is not None:
        scores = scores + attention_mask
    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
    return F.linear(output, layer_weights["attention.wo.weight"])


def text_decoder_layer(
    hidden_states: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: VoxtralTextConfig,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights["attention_norm.weight"], config.rms_norm_eps)
    hidden_states = residual + text_attention(hidden_states, layer_weights, cos, sin, config, attention_mask)

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights["ffn_norm.weight"], config.rms_norm_eps)
    hidden_states = residual + swiglu_mlp(
        hidden_states,
        layer_weights["feed_forward.w1.weight"],
        layer_weights["feed_forward.w2.weight"],
        layer_weights["feed_forward.w3.weight"],
    )
    return hidden_states


def acoustic_attention(
    hidden_states: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    config: VoxtralAcousticConfig,
) -> torch.Tensor:
    batch, seq_len, _ = hidden_states.shape
    q = F.linear(hidden_states, layer_weights["attention.wq.weight"])
    k = F.linear(hidden_states, layer_weights["attention.wk.weight"])
    v = F.linear(hidden_states, layer_weights["attention.wv.weight"])

    q = q.view(batch, seq_len, config.num_attention_heads, config.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)

    n_rep = config.num_attention_heads // config.num_key_value_heads
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (config.head_dim**0.5)
    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
    return F.linear(output, layer_weights["attention.wo.weight"])


def acoustic_transformer_layer(
    hidden_states: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    config: VoxtralAcousticConfig,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights["attention_norm.weight"], config.rms_norm_eps)
    hidden_states = residual + acoustic_attention(hidden_states, layer_weights, config)

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights["ffn_norm.weight"], config.rms_norm_eps)
    hidden_states = residual + swiglu_mlp(
        hidden_states,
        layer_weights["feed_forward.w1.weight"],
        layer_weights["feed_forward.w2.weight"],
        layer_weights["feed_forward.w3.weight"],
    )
    return hidden_states


def extract_layer_weights(state_dict: dict[str, torch.Tensor], layer_idx: int, prefix: str = "layers.") -> dict:
    layer_prefix = f"{prefix}{layer_idx}."
    return {k.removeprefix(layer_prefix): v for k, v in state_dict.items() if k.startswith(layer_prefix)}


def extract_acoustic_layer_weights(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict:
    return extract_layer_weights(state_dict, layer_idx, prefix="acoustic_transformer.layers.")
