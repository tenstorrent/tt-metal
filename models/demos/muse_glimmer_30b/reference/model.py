# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone pure-PyTorch reference of the Muse-Glimmer text decoder.

Adapted from HF transformers `modeling_muse_glimmer.py` (Apache-2.0, Copyright 2026 the HuggingFace Team),
text model only. Depends only on torch, safetensors and the standard library (no `transformers` import).
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class MuseGlimmerTextConfig:
    vocab_size: int = 202_048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131_072
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    rope_theta: float = 500000.0
    sliding_window: Optional[int] = 2048
    layer_types: Optional[list] = None
    layer_rope_theta: Optional[list] = None
    final_logit_softcapping: float = 20.0
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    tie_word_embeddings: bool = False
    bos_token_id: Optional[int] = 200_000
    eos_token_id: Optional[object] = 200_001
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (self.num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if (self.num_hidden_layers - 1 - i) % 4 == 0 else self.rope_theta
                for i in range(self.num_hidden_layers)
            ]

    def is_nope(self, layer_idx: int) -> bool:
        return self.layer_rope_theta[layer_idx] == 0

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    @classmethod
    def from_json(cls, path_or_dir):
        path = path_or_dir
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path) as f:
            d = json.load(f)
        if "text_config" in d:
            d = d["text_config"]
        d = dict(d)
        rp = d.pop("rope_parameters", None)
        if rp is not None and "rope_theta" in rp:
            d["rope_theta"] = float(rp["rope_theta"])
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


# --------------------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: Optional[int] = None, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, x):
        out = self._norm(x.float())
        if self.with_scale:
            out = out * self.weight.float()
        return out.type_as(x)


class CenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float())
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


# --------------------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------


def build_attention_mask(q_len, kv_len, q_offset, sliding_window, dtype, device=None):
    """Additive [1,1,q_len,kv_len] mask.

    HF semantics (masking_utils.py): causal: kv_idx <= q_idx; sliding overlay: kv_idx > q_idx - sliding_window.
    Absolute q_idx = q_offset + i, kv_idx = 0..kv_len-1.
    """
    q_idx = (torch.arange(q_len, device=device) + q_offset)[:, None]
    kv_idx = torch.arange(kv_len, device=device)[None, :]
    allowed = kv_idx <= q_idx
    if sliding_window is not None:
        allowed = allowed & (kv_idx > q_idx - sliding_window)
    mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


# --------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        H, KV, D = config.hidden_size, config.num_key_value_heads, self.head_dim
        self.q_proj = nn.Linear(H, config.num_attention_heads * D, bias=False)
        self.k_proj = nn.Linear(H, KV * D, bias=False)
        self.v_proj = nn.Linear(H, KV * D, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * D, H, bias=False)
        self.gate_proj = nn.Linear(H, config.num_attention_heads * D, bias=False)
        self.qk_norm = RMSNorm(eps=config.rms_norm_eps, with_scale=False)
        self.qk_scale_factor = config.qk_scale_factor
        self.sliding_window = config.sliding_window if config.is_sliding(layer_idx) else None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_kv=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q = self.qk_norm(q) * self.qk_scale_factor
        k = self.qk_norm(k)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        w = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            w = w + attention_mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, value_states).transpose(1, 2).contiguous()

        out = out.reshape(*input_shape, -1).contiguous()
        out = out * torch.sigmoid(self.gate_proj(hidden_states))
        out = self.o_proj(out)
        return out, new_kv


class MLP(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        assert config.hidden_activation == "silu"
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = CenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(config.hidden_size, eps=config.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = CenteredRMSNorm(config.hidden_size, eps=config.post_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_kv=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, new_kv = self.self_attn(h, position_embeddings, attention_mask, past_kv)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        hidden_states = residual + h
        return hidden_states, new_kv


class TextModel(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_norm = RMSNorm(eps=config.rms_norm_eps, with_scale=False)  # weightless: no state-dict entry
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, input_ids=None, inputs_embeds=None, start_pos=0, past_kvs=None, return_hidden_states=False):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))
        h = inputs_embeds
        seq = h.shape[1]
        position_ids = (torch.arange(seq, device=h.device) + start_pos).unsqueeze(0)
        pos_emb = self.rotary_emb(h, position_ids)
        kv_len = start_pos + seq
        masks = {
            None: build_attention_mask(seq, kv_len, start_pos, None, h.dtype, h.device),
            self.config.sliding_window: build_attention_mask(
                seq, kv_len, start_pos, self.config.sliding_window, h.dtype, h.device
            ),
        }
        hidden_states = [h] if return_hidden_states else None
        new_kvs = []
        for i, layer in enumerate(self.layers):
            mask = masks[self.config.sliding_window] if self.config.is_sliding(i) else masks[None]
            pe = None if self.config.is_nope(i) else pos_emb
            h, kv = layer(h, pe, mask, None if past_kvs is None else past_kvs[i])
            new_kvs.append(kv)
            if return_hidden_states:
                hidden_states.append(h)
        h = self.norm(h)
        return {"last_hidden_state": h, "past_kvs": new_kvs, "hidden_states": hidden_states}


class CausalLM(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None, start_pos=0, past_kvs=None, return_hidden_states=False):
        out = self.model(input_ids, inputs_embeds, start_pos, past_kvs, return_hidden_states)
        logits = self.lm_head(out["last_hidden_state"])
        logits = logits * self.config.output_multiplier
        logits = logits / self.config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.final_logit_softcapping
        out["logits"] = logits
        return out

    @torch.no_grad()
    def generate_greedy(self, input_ids, max_new_tokens):
        out = self(input_ids)
        past, pos = out["past_kvs"], input_ids.shape[1]
        nxt = out["logits"][:, -1].argmax(-1, keepdim=True)
        toks = [nxt]
        for _ in range(max_new_tokens - 1):
            out = self(nxt, start_pos=pos, past_kvs=past)
            past, pos = out["past_kvs"], pos + 1
            nxt = out["logits"][:, -1].argmax(-1, keepdim=True)
            toks.append(nxt)
        return torch.cat([input_ids] + toks, dim=1)

    @classmethod
    def from_checkpoint(cls, ckpt_dir, num_layers=None, dtype=torch.bfloat16):
        config = MuseGlimmerTextConfig.from_json(ckpt_dir)
        if num_layers is not None and num_layers < config.num_hidden_layers:
            config.num_hidden_layers = num_layers
            config.layer_types = config.layer_types[:num_layers]
            config.layer_rope_theta = config.layer_rope_theta[:num_layers]
        model = cls(config).to(dtype)
        sd = load_checkpoint_state_dict(ckpt_dir, layer_indices=range(config.num_hidden_layers), dtype=dtype)
        sd = {("lm_head.weight" if k == "lm_head.weight" else f"model.{k}"): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        return model.eval()


# --------------------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------------------


def load_checkpoint_state_dict(
    ckpt_dir,
    prefixes=None,
    layer_indices=None,
    include_lm_head=True,
    include_embed=True,
    include_norm=True,
    dtype=torch.bfloat16,
):
    """Load text-model tensors from the HF sharded safetensors checkpoint.

    `prefixes`: source-key prefixes to strip (default ["model.language_model."]). Returns keys in the
    TextModel layout (`embed_tokens.weight`, `layers.{i}...`, `norm.weight`) plus `lm_head.weight`.
    Vision tensors (and anything not matching) are skipped.
    """
    from safetensors import safe_open

    prefixes = prefixes or ["model.language_model."]
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    layer_set = None if layer_indices is None else set(layer_indices)

    wanted = {}  # source key -> (file, output key)
    for key, fname in weight_map.items():
        if key == "lm_head.weight":
            if include_lm_head:
                wanted[key] = (fname, "lm_head.weight")
            continue
        for p in prefixes:
            if key.startswith(p):
                k = key[len(p) :]
                break
        else:
            continue
        if k.startswith("layers."):
            if layer_set is not None and int(k.split(".")[1]) not in layer_set:
                continue
        elif k == "embed_tokens.weight":
            if not include_embed:
                continue
        elif k == "norm.weight":
            if not include_norm:
                continue
        else:
            continue
        wanted[key] = (fname, k)

    by_file = {}
    for key, (fname, k) in wanted.items():
        by_file.setdefault(fname, []).append((key, k))
    out = {}
    for fname, items in by_file.items():
        with safe_open(os.path.join(ckpt_dir, fname), framework="pt", device="cpu") as sf:
            for key, k in items:
                out[k] = sf.get_tensor(key).to(dtype)
    return out
