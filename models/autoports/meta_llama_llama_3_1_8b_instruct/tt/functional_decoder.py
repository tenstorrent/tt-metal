# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.common.lightweightmodule import LightweightModule

HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PENDING_DECODE_MESSAGE = "decode path pending emitted-decode forge version"


@dataclass(frozen=True)
class Llama31DecoderConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float

    @classmethod
    def from_hf_config(cls, hf_config) -> "Llama31DecoderConfig":
        hidden_size = int(hf_config.hidden_size)
        num_attention_heads = int(hf_config.num_attention_heads)
        num_key_value_heads = int(hf_config.num_key_value_heads)
        head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_attention_heads))
        intermediate_size = int(hf_config.intermediate_size)
        max_position_embeddings = int(hf_config.max_position_embeddings)
        rope_parameters = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
        rope_theta = float(getattr(hf_config, "rope_theta", None) or rope_parameters.get("rope_theta", 500000.0))
        rms_norm_eps = float(getattr(hf_config, "rms_norm_eps", 1.0e-5))

        if hidden_size != 4096:
            raise ValueError(f"{HF_MODEL_ID} functional decoder expects hidden_size=4096, got {hidden_size}")
        if num_attention_heads != 32 or num_key_value_heads != 8:
            raise ValueError(
                f"{HF_MODEL_ID} functional decoder expects 32 Q heads and 8 KV heads, got "
                f"{num_attention_heads} and {num_key_value_heads}"
            )
        if head_dim != 128 or intermediate_size != 14336:
            raise ValueError(
                f"{HF_MODEL_ID} functional decoder expects head_dim=128 and intermediate_size=14336, got "
                f"{head_dim} and {intermediate_size}"
            )
        if max_position_embeddings != 131072:
            raise ValueError(
                f"{HF_MODEL_ID} functional decoder expects max_position_embeddings=131072, got "
                f"{max_position_embeddings}"
            )

        return cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )


def _canonical_key(layer_idx: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.language_model.layers.{layer_idx}.{suffix}",
        suffix,
    )


def _get_layer_tensor(state_dict: dict[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    for key in _canonical_key(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"missing Llama 3.1 layer {layer_idx} tensor for {suffix}")


def _to_device_tensor(
    tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _fused_qkv_weight(state_dict: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    q_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
    k_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
    v_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
    if layer_idx == 31:
        return torch.cat((q_proj.transpose(0, 1), k_proj.transpose(0, 1), v_proj.transpose(0, 1)), dim=1)
    return torch.cat((q_proj.transpose(0, 1), v_proj.transpose(0, 1), k_proj.transpose(0, 1)), dim=1)


def build_rope_tables(hf_config, seq_len: int, mesh_device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    cfg = Llama31DecoderConfig.from_hf_config(hf_config)
    rotary = LlamaRotaryEmbedding(hf_config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    dummy = torch.empty(1, seq_len, cfg.head_dim, dtype=torch.bfloat16)
    cos, sin = rotary(dummy, position_ids)
    return (
        _to_device_tensor(cos.reshape(1, 1, seq_len, cfg.head_dim), mesh_device),
        _to_device_tensor(sin.reshape(1, 1, seq_len, cfg.head_dim), mesh_device),
    )


def build_causal_mask(seq_len: int, mesh_device) -> ttnn.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    return _to_device_tensor(mask, mesh_device)


class FunctionalDecoder(LightweightModule):
    """Single Llama-3.1-8B-Instruct decoder layer translated from the forge emit."""

    def __init__(
        self,
        *,
        cfg: Llama31DecoderConfig,
        layer_idx: int,
        mesh_device,
        qkv_proj_weight: ttnn.Tensor,
        o_proj_weight: ttnn.Tensor,
        gate_proj_weight: ttnn.Tensor,
        up_proj_weight: ttnn.Tensor,
        down_proj_weight: ttnn.Tensor,
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        max_seq_len: int,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.qkv_proj_weight = qkv_proj_weight
        self.o_proj_weight = o_proj_weight
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.down_proj_weight = down_proj_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.position_cos = position_cos
        self.position_sin = position_sin
        self.attention_mask = attention_mask
        self.max_seq_len = max_seq_len

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_seq_len: int = 128,
        **kwargs,
    ) -> "FunctionalDecoder":
        if kwargs:
            raise TypeError(f"unsupported FunctionalDecoder kwargs: {sorted(kwargs)}")
        cfg = Llama31DecoderConfig.from_hf_config(hf_config)
        if max_seq_len <= 0 or max_seq_len > cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {cfg.max_position_embeddings}], got {max_seq_len}")

        position_cos, position_sin = build_rope_tables(hf_config, max_seq_len, mesh_device)
        attention_mask = build_causal_mask(max_seq_len, mesh_device)

        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            qkv_proj_weight=_to_device_tensor(_fused_qkv_weight(state_dict, layer_idx), mesh_device),
            o_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight"), mesh_device
            ),
            gate_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight"), mesh_device
            ),
            up_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight"), mesh_device
            ),
            down_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight"), mesh_device
            ),
            input_layernorm_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "input_layernorm.weight"), mesh_device
            ),
            post_attention_layernorm_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "post_attention_layernorm.weight"), mesh_device
            ),
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
            max_seq_len=max_seq_len,
        )

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor | None = None,
        position_sin: ttnn.Tensor | None = None,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        if seq_len > self.max_seq_len and (position_cos is None or position_sin is None):
            raise ValueError(
                f"prefill seq_len {seq_len} exceeds setup max_seq_len {self.max_seq_len}; "
                "provide matching RoPE tables or rebuild the decoder"
            )
        cos = position_cos if position_cos is not None else self.position_cos
        sin = position_sin if position_sin is not None else self.position_sin
        mask = attention_mask

        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.input_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv = ttnn.matmul(
            normed,
            self.qkv_proj_weight,
            transpose_a=False,
            transpose_b=False,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, seq_len, 4096], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.layer_idx == 31:
            k = ttnn.slice(
                qkv, [0, 0, 0, 4096], [1, 1, seq_len, 5120], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            v = ttnn.slice(
                qkv, [0, 0, 0, 5120], [1, 1, seq_len, 6144], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            v = ttnn.slice(
                qkv, [0, 0, 0, 4096], [1, 1, seq_len, 5120], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            k = ttnn.slice(
                qkv, [0, 0, 0, 5120], [1, 1, seq_len, 6144], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        q = ttnn.reshape(
            q, [1, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(
            q,
            [0, 0, 0, 0],
            [1, self.cfg.num_attention_heads, seq_len, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        k = ttnn.reshape(
            k, [1, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(
            k,
            [0, 0, 0, 0],
            [1, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        v = ttnn.reshape(
            v, [1, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(
            attn,
            [1, 1, seq_len, self.cfg.num_attention_heads * self.cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = ttnn.matmul(
            attn,
            self.o_proj_weight,
            transpose_a=False,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        gate = ttnn.matmul(
            post_norm,
            self.gate_proj_weight,
            transpose_a=False,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.matmul(
            post_norm,
            self.up_proj_weight,
            transpose_a=False,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gated = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mlp_out = ttnn.matmul(
            gated,
            self.down_proj_weight,
            transpose_a=False,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.add(mlp_out, attn_residual, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(self, *args, **kwargs):
        """Incremental decode is intentionally absent until a forge emitted-decode graph exists."""
        raise NotImplementedError(PENDING_DECODE_MESSAGE)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str = "prefill", **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported FunctionalDecoder mode: {mode!r}")
