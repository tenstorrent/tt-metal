# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone pure-PyTorch reference of the Muse-Glimmer DFlash2 drafter.

Adapted from z-lab/dflash `model.py` (Apache-2.0), `DFlash2DraftModel`, greedy path only (no sampling, no KV cache).
Depends only on torch, safetensors and the standard library (no `transformers` import).
"""

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class DFlash2Config:
    vocab_size: int = 202_048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    sliding_window: int = 2048
    block_size: int = 16
    mask_token_id: int = 201_818
    target_layer_ids: list = field(default_factory=lambda: [1, 13, 25, 37, 49])
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0

    @classmethod
    def from_json(cls, path: str) -> "DFlash2Config":
        with open(path) as f:
            raw = json.load(f)
        merged = dict(raw)
        merged.update(raw.get("dflash_config", {}))
        if "rope_parameters" in raw:
            merged["rope_theta"] = raw["rope_parameters"]["rope_theta"]
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: merged[k] for k in names if k in merged})


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """Default RoPE cos/sin for explicit integer positions [B, T] -> two [B, T, head_dim] tensors."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    freqs = position_ids[:, :, None].float() * inv_freq[None, None, :].to(position_ids.device)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class MLP(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlash2Attention(nn.Module):
    """Non-causal sliding-window attention: queries from the block, keys/values from [context, block]."""

    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.scaling = cfg.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, hidden, target_hidden, cos, sin, position_ids):
        bsz, q_len = hidden.shape[:2]
        ctx_len = target_hidden.shape[1]
        q = self.q_norm(self.q_proj(hidden).view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        k = torch.cat([self.k_proj(target_hidden), self.k_proj(hidden)], dim=1)
        v = torch.cat([self.v_proj(target_hidden), self.v_proj(hidden)], dim=1)
        k = self.k_norm(k.view(bsz, ctx_len + q_len, -1, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, ctx_len + q_len, -1, self.head_dim).transpose(1, 2)

        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q = q * c[..., -q_len:, :] + rotate_half(q) * s[..., -q_len:, :]
        k = k * c + rotate_half(k) * s

        groups = self.n_heads // self.n_kv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        # visibility by position: |q_pos - k_pos| < sliding_window
        q_pos = position_ids[:, -q_len:, None]
        k_pos = position_ids[:, None, :]
        w = self.cfg.sliding_window
        visible = ((q_pos - k_pos) < w) & ((k_pos - q_pos) < w)  # [B, L, K]

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        scores = scores.masked_fill(~visible[:, None], float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(out)


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, self.kernel_size, groups)
        return (
            _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size)


class DFlash2DecoderLayer(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.self_attn = DFlash2Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attention_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)
        self.mlp_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)

    def forward(self, hidden_states, target_hidden, cos, sin, position_ids):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cos, sin, position_ids)
        hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return residual + hidden_states


class CandidateSelector(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.top_k = cfg.selector_top_k
        self.predecessor_codebook = nn.Embedding(cfg.vocab_size, cfg.selector_rank)
        self.successor_codebook = nn.Embedding(cfg.vocab_size, cfg.selector_rank)
        self.hidden_projection = nn.Linear(cfg.hidden_size, cfg.selector_rank, bias=False)

    def select(self, hidden, logits, anchor_ids):
        """Greedy (temperature 0) path selection. Returns (path [B, L], candidates [B, L, top_k])."""
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * hidden[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1), candidates


# --------------------------------------------------------------------------------------
# Drafter
# --------------------------------------------------------------------------------------


class DFlash2Drafter(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([DFlash2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.candidate_selector = CandidateSelector(cfg)

    def context_features(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """target_hidden: [B, C, 5*H] raw concatenated target taps -> [B, C, H]."""
        return self.hidden_norm(self.fc(target_hidden))

    def forward(self, noise_embedding, target_hidden, position_ids):
        """noise_embedding [B, L, H], target_hidden [B, C, 5H], position_ids [B, C+L] -> [B, L, H]."""
        hidden_states = noise_embedding
        ctx = self.context_features(target_hidden)
        cos, sin = rope_cos_sin(position_ids, self.cfg.head_dim, self.cfg.rope_theta, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, ctx, cos, sin, position_ids)
        return self.norm(hidden_states)

    def compute_logits(self, hidden, lm_head_weight):
        logits = F.linear(hidden, lm_head_weight) * self.cfg.output_multiplier
        softcap = self.cfg.final_logit_softcapping
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits

    def propose_greedy(self, hidden, logits, anchor_ids):
        return self.candidate_selector.select(hidden, logits, anchor_ids)


def load_dflash2(ckpt_dir: str, dtype: torch.dtype = torch.bfloat16):
    from safetensors.torch import load_file

    cfg = DFlash2Config.from_json(os.path.join(ckpt_dir, "config.json"))
    with torch.device("meta"):
        model = DFlash2Drafter(cfg)
    state = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        if key in state:
            state[key + ".weight"] = state.pop(key)
    state = {k: v.to(dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval(), cfg


# --------------------------------------------------------------------------------------
# Helpers shared with the target model
# --------------------------------------------------------------------------------------


def extract_context_feature(hidden_states_list, layer_ids):
    """Concatenate the target taps (hidden_states[0] is the embedding output, hence offset 1)."""
    return torch.cat([hidden_states_list[i + 1] for i in layer_ids], dim=-1)


def greedy_acceptance(block_ids: torch.Tensor, target_argmax: torch.Tensor):
    """block_ids, target_argmax: [1, L]. Returns (acceptance_length, bonus_token)."""
    acc = int((block_ids[:, 1:] == target_argmax[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item())
    return acc, int(target_argmax[0, acc].item())
