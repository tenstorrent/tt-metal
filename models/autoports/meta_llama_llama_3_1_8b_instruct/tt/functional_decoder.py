# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = ttnn.TILE_LAYOUT
BF16 = ttnn.bfloat16


@dataclass(frozen=True)
class LlamaDecoderConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float


def _layer_prefixes(layer_idx: int) -> tuple[str, ...]:
    return (
        f"model.layers.{layer_idx}.",
        f"model.language_model.layers.{layer_idx}.",
        "",
    )


def _lookup(state_dict: dict[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    for prefix in _layer_prefixes(layer_idx):
        key = prefix + suffix
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"missing layer {layer_idx} weight: {suffix}")


def _to_tt_weight(tensor: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().to(torch.bfloat16),
        dtype=BF16,
        layout=TILE,
        device=mesh_device,
        memory_config=DRAM,
    )


def _to_tt_hidden(tensor: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().to(torch.bfloat16),
        dtype=BF16,
        layout=TILE,
        device=mesh_device,
        memory_config=DRAM,
    )


def _require_config(hf_config) -> LlamaDecoderConfig:
    hidden_size = int(hf_config.hidden_size)
    num_attention_heads = int(hf_config.num_attention_heads)
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_attention_heads))
    num_key_value_heads = int(getattr(hf_config, "num_key_value_heads", num_attention_heads))
    return LlamaDecoderConfig(
        hidden_size=hidden_size,
        intermediate_size=int(hf_config.intermediate_size),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=float(hf_config.rms_norm_eps),
    )


class FunctionalDecoder(LightweightModule):
    """Single Llama-3.1 decoder layer translated from the forge-emitted TTNN graph.

    Host conversion is intentionally limited to ``from_state_dict`` and helper setup.
    ``prefill_forward`` is a TTNN-only runtime path. Decode/paged KV is not present in
    the supplied forge emit and is left as an explicit pending-version stub.
    """

    def __init__(self, *, config: LlamaDecoderConfig, layer_idx: int, mesh_device, weights: dict, batch: int):
        self.config = config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.weights = weights
        self.batch = batch
        self.scale = 1.0 / math.sqrt(config.head_dim)

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, batch=32, **kwargs):
        del kwargs
        config = _require_config(hf_config)

        q_proj = _lookup(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_proj = _lookup(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_proj = _lookup(state_dict, layer_idx, "self_attn.v_proj.weight")

        weights = {
            "input_layernorm": _to_tt_weight(_lookup(state_dict, layer_idx, "input_layernorm.weight"), mesh_device),
            "post_attention_layernorm": _to_tt_weight(
                _lookup(state_dict, layer_idx, "post_attention_layernorm.weight"), mesh_device
            ),
            # The Llama forge emit fuses layers 0..30 in [Q, V, K] order.
            "qkv": _to_tt_weight(torch.cat([q_proj.T, v_proj.T, k_proj.T], dim=1), mesh_device),
            "o_proj": _to_tt_weight(_lookup(state_dict, layer_idx, "self_attn.o_proj.weight").T, mesh_device),
            "gate_proj": _to_tt_weight(_lookup(state_dict, layer_idx, "mlp.gate_proj.weight").T, mesh_device),
            "up_proj": _to_tt_weight(_lookup(state_dict, layer_idx, "mlp.up_proj.weight").T, mesh_device),
            "down_proj": _to_tt_weight(_lookup(state_dict, layer_idx, "mlp.down_proj.weight").T, mesh_device),
        }
        return cls(config=config, layer_idx=layer_idx, mesh_device=mesh_device, weights=weights, batch=int(batch))

    @staticmethod
    def prepare_inputs(hidden_states: torch.Tensor, mesh_device) -> ttnn.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        return _to_tt_hidden(hidden_states.unsqueeze(0), mesh_device)

    @staticmethod
    def prepare_rope(
        position_cos: torch.Tensor, position_sin: torch.Tensor, mesh_device
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if position_cos.ndim == 3:
            position_cos = position_cos[:, None, :, :]
        if position_sin.ndim == 3:
            position_sin = position_sin[:, None, :, :]
        if position_cos.shape[0] != 1:
            position_cos = position_cos[:1]
            position_sin = position_sin[:1]
        return _to_tt_hidden(position_cos, mesh_device), _to_tt_hidden(position_sin, mesh_device)

    @staticmethod
    def build_causal_mask(batch: int, seq_len: int, mesh_device) -> ttnn.Tensor:
        del batch
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1).to(torch.bfloat16)
        return _to_tt_hidden(mask, mesh_device)

    def _apply_rotary(self, tensor, position_cos, position_sin, num_heads: int, seq_len: int):
        half = self.config.head_dim // 2
        x1 = ttnn.slice(tensor, [0, 0, 0, 0], [self.batch, num_heads, seq_len, half], [1, 1, 1, 1], memory_config=DRAM)
        x2 = ttnn.slice(
            tensor,
            [0, 0, 0, half],
            [self.batch, num_heads, seq_len, self.config.head_dim],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )
        rotated = ttnn.concat([ttnn.neg(x2, memory_config=DRAM), x1], dim=3, memory_config=DRAM)
        return ttnn.add(
            ttnn.multiply(tensor, position_cos, memory_config=DRAM),
            ttnn.multiply(rotated, position_sin, memory_config=DRAM),
            dtype=BF16,
            memory_config=DRAM,
        )

    def prefill_forward(self, hidden_states, *, position_cos, position_sin, attn_mask=None):
        cfg = self.config
        batch = hidden_states.shape[1]
        seq_len = hidden_states.shape[2]
        if hidden_states.shape[0] != 1 or hidden_states.shape[3] != cfg.hidden_size:
            raise ValueError("hidden_states must have shape [1, batch, seq, hidden_size]")
        if batch != self.batch:
            raise ValueError(f"runtime batch {batch} does not match configured batch {self.batch}")

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["input_layernorm"],
            memory_config=DRAM,
        )
        qkv = ttnn.matmul(normed, self.weights["qkv"], dtype=BF16, memory_config=DRAM)

        kv_width = cfg.num_key_value_heads * cfg.head_dim
        q_width = cfg.num_attention_heads * cfg.head_dim
        query = ttnn.slice(qkv, [0, 0, 0, 0], [1, batch, seq_len, q_width], [1, 1, 1, 1], memory_config=DRAM)
        value = ttnn.slice(
            qkv,
            [0, 0, 0, q_width],
            [1, batch, seq_len, q_width + kv_width],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )
        key = ttnn.slice(
            qkv,
            [0, 0, 0, q_width + kv_width],
            [1, batch, seq_len, q_width + kv_width + kv_width],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )

        query = ttnn.reshape(query, [batch, seq_len, cfg.num_attention_heads, cfg.head_dim])
        key = ttnn.reshape(key, [batch, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        value = ttnn.reshape(value, [batch, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        query = ttnn.permute(query, [0, 2, 1, 3], memory_config=DRAM)
        key = ttnn.permute(key, [0, 2, 1, 3], memory_config=DRAM)
        value = ttnn.permute(value, [0, 2, 1, 3], memory_config=DRAM)

        query = self._apply_rotary(query, position_cos, position_sin, cfg.num_attention_heads, seq_len)
        key = self._apply_rotary(key, position_cos, position_sin, cfg.num_key_value_heads, seq_len)

        if attn_mask is None:
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=self.scale,
                memory_config=DRAM,
            )
        else:
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=False,
                scale=self.scale,
                memory_config=DRAM,
            )

        attn = ttnn.permute(attn, [0, 2, 1, 3], memory_config=DRAM)
        attn = ttnn.reshape(attn, [1, batch, seq_len, cfg.hidden_size])
        attn = ttnn.matmul(attn, self.weights["o_proj"], dtype=BF16, memory_config=DRAM)
        hidden = ttnn.add(attn, residual, dtype=BF16, memory_config=DRAM)

        mlp_input = ttnn.rms_norm(
            hidden,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["post_attention_layernorm"],
            memory_config=DRAM,
        )
        gate = ttnn.matmul(mlp_input, self.weights["gate_proj"], dtype=BF16, memory_config=DRAM)
        up = ttnn.matmul(mlp_input, self.weights["up_proj"], dtype=BF16, memory_config=DRAM)
        gated = ttnn.multiply(ttnn.silu(gate, memory_config=DRAM), up, memory_config=DRAM)
        mlp = ttnn.matmul(gated, self.weights["down_proj"], dtype=BF16, memory_config=DRAM)
        return ttnn.add(mlp, hidden, dtype=BF16, memory_config=DRAM)

    def decode_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "decode path pending emitted-decode forge version; current artifact is prefill-only and has no paged KV path"
        )

    def forward(self, hidden_states, *, mode="prefill", **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported FunctionalDecoder mode: {mode}")
