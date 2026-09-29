# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact Qwen Sparse Attention checkpoint contract and TP4 placement.

This module supplies the CPU value oracle and the host-side weight packing for
the TTNN port.  The dense matrix used by :class:`Qwen38QSA` is intentionally a
short-context oracle only; token selection is still the exact QSA indexer.  The
final device path must execute only selected complete blocks plus the causal
tail at long context.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.reference import qsa_selected_token_mask, zero_centered_rms_norm

TP_SIZE = 4


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _partial_rope(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_width = cos.shape[-1]
    rotary, passthrough = tensor[..., :rotary_width], tensor[..., rotary_width:]
    rotated = rotary * cos + _rotate_half(rotary) * sin
    return torch.cat((rotated, passthrough), dim=-1)


@dataclass(frozen=True)
class Qwen38QSADimensions:
    hidden_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    rope_dim: int
    index_query_heads: int
    index_kv_heads: int
    index_head_dim: int
    token_budget: int
    compress_ratio: int
    tp_size: int = TP_SIZE

    def __post_init__(self) -> None:
        if self.tp_size != TP_SIZE:
            raise ValueError(f"Qwen3.8-Flash-Next QSA requires TP={TP_SIZE}, got {self.tp_size}")
        if self.query_heads % self.tp_size or self.index_query_heads % self.tp_size:
            raise ValueError("QSA query/index-query heads must divide exactly over four devices")
        if self.query_heads % self.kv_heads:
            raise ValueError("QSA query heads must group exactly over KV heads")
        if self.kv_heads != 2 or self.index_kv_heads != 1:
            raise ValueError("the pinned QSA grouped-KV contract is exactly 2 main KV heads and 1 index KV head")

    @property
    def query_heads_per_device(self) -> int:
        return self.query_heads // self.tp_size

    @property
    def index_query_heads_per_device(self) -> int:
        return self.index_query_heads // self.tp_size

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def query_width_per_device(self) -> int:
        return self.query_heads_per_device * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def kv_repeat(self) -> int:
        return self.query_heads // self.kv_heads


@dataclass(frozen=True)
class Qwen38QSADeviceShard:
    device_index: int
    kv_head_index: int
    dimensions: Qwen38QSADimensions
    qg: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    index_q: torch.Tensor
    index_k: torch.Tensor
    index_q_norm: torch.Tensor
    index_k_norm: torch.Tensor

    def split_query_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        heads = self.dimensions.query_heads_per_device
        shaped = self.qg.reshape(heads, 2 * self.dimensions.head_dim, self.dimensions.hidden_size)
        return shaped[:, : self.dimensions.head_dim].flatten(0, 1), shaped[:, self.dimensions.head_dim :].flatten(0, 1)


@dataclass(frozen=True)
class Qwen38QSAWeights:
    layer_idx: int
    dimensions: Qwen38QSADimensions
    rms_norm_eps: float
    qg: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    index_qk: torch.Tensor
    index_q_norm: torch.Tensor
    index_k_norm: torch.Tensor

    @classmethod
    def from_checkpoint(cls, checkpoint: Qwen38Checkpoint, layer_idx: int) -> "Qwen38QSAWeights":
        config = checkpoint.config
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError(f"QSA layer index is outside [0, {config.num_hidden_layers}): {layer_idx}")
        if config.layer_types[layer_idx] != "full_attention":
            raise ValueError(f"layer {layer_idx} is {config.layer_types[layer_idx]!r}, not full_attention")
        dimensions = Qwen38QSADimensions(
            hidden_size=config.hidden_size,
            query_heads=config.qsa_query_heads,
            kv_heads=config.qsa_kv_heads,
            head_dim=config.qsa_head_dim,
            rope_dim=config.qsa_rope_dim,
            index_query_heads=config.index_query_heads,
            index_kv_heads=config.index_kv_heads,
            index_head_dim=config.index_head_dim,
            token_budget=config.index_budget,
            compress_ratio=config.index_compress_ratio,
        )
        prefix = f"model.language_model.layers.{layer_idx}.self_attn."
        tensors = {
            field: checkpoint.tensor(prefix + checkpoint_name)
            for field, checkpoint_name in {
                "qg": "q_proj.weight",
                "k": "k_proj.weight",
                "v": "v_proj.weight",
                "out": "o_proj.weight",
                "q_norm": "q_norm.weight",
                "k_norm": "k_norm.weight",
                "index_qk": "indexer.index_qk_proj.weight",
                "index_q_norm": "indexer.q_layernorm.weight",
                "index_k_norm": "indexer.k_layernorm.weight",
            }.items()
        }
        expected_shapes = {
            "qg": (2 * dimensions.query_width, dimensions.hidden_size),
            "k": (dimensions.kv_width, dimensions.hidden_size),
            "v": (dimensions.kv_width, dimensions.hidden_size),
            "out": (dimensions.hidden_size, dimensions.query_width),
            "q_norm": (dimensions.head_dim,),
            "k_norm": (dimensions.head_dim,),
            "index_qk": (
                (dimensions.index_query_heads + dimensions.index_kv_heads) * dimensions.index_head_dim,
                dimensions.hidden_size,
            ),
            "index_q_norm": (dimensions.index_head_dim,),
            "index_k_norm": (dimensions.index_head_dim,),
        }
        for name, expected in expected_shapes.items():
            tensor = tensors[name]
            if tuple(tensor.shape) != expected:
                raise ValueError(f"layer {layer_idx} QSA {name} shape must be {expected}, got {tuple(tensor.shape)}")
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"layer {layer_idx} QSA {name} must be BF16, got {tensor.dtype}")
        return cls(layer_idx=layer_idx, dimensions=dimensions, rms_norm_eps=config.rms_norm_eps, **tensors)

    def split_query_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        dims = self.dimensions
        shaped = self.qg.reshape(dims.query_heads, 2 * dims.head_dim, dims.hidden_size)
        return shaped[:, : dims.head_dim].flatten(0, 1), shaped[:, dims.head_dim :].flatten(0, 1)

    def device_shard(self, device_index: int) -> Qwen38QSADeviceShard:
        if not 0 <= device_index < self.dimensions.tp_size:
            raise ValueError(f"device_index must be in [0, 4), got {device_index}")
        dims = self.dimensions
        qg_rows = 2 * dims.query_width_per_device
        qg_start = device_index * qg_rows
        query_start = device_index * dims.query_width_per_device
        query_end = query_start + dims.query_width_per_device
        # KV head 0 serves query devices 0/1; KV head 1 serves 2/3.  This is
        # deliberate pair-local replication, never a four-way KV replica.
        kv_head_index = device_index // 2
        kv_start = kv_head_index * dims.head_dim
        kv_end = kv_start + dims.head_dim
        index_q_start = device_index * dims.index_head_dim
        index_q_end = index_q_start + dims.index_head_dim
        index_k_start = dims.index_query_heads * dims.index_head_dim
        return Qwen38QSADeviceShard(
            device_index=device_index,
            kv_head_index=kv_head_index,
            dimensions=dims,
            qg=self.qg[qg_start : qg_start + qg_rows].contiguous(),
            k=self.k[kv_start:kv_end].contiguous(),
            v=self.v[kv_start:kv_end].contiguous(),
            out=self.out[:, query_start:query_end].contiguous(),
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            index_q=self.index_qk[index_q_start:index_q_end].contiguous(),
            index_k=self.index_qk[index_k_start:].contiguous(),
            index_q_norm=self.index_q_norm,
            index_k_norm=self.index_k_norm,
        )

    def transformers_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "q_proj.weight": self.qg,
            "k_proj.weight": self.k,
            "v_proj.weight": self.v,
            "o_proj.weight": self.out,
            "q_norm.weight": self.q_norm,
            "k_norm.weight": self.k_norm,
            "indexer.index_qk_proj.weight": self.index_qk,
            "indexer.q_layernorm.weight": self.index_q_norm,
            "indexer.k_layernorm.weight": self.index_k_norm,
        }


@dataclass(frozen=True)
class Qwen38QSAState:
    """Append-only QSA cache with indexer and main K/V histories separated."""

    raw_index_keys: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.raw_index_keys.ndim != 3:
            raise ValueError("raw QSA index keys must be [batch, sequence, index_head_dim]")
        if self.keys.ndim != 4 or self.values.ndim != 4:
            raise ValueError("QSA keys and values must be [batch, kv_heads, sequence, head_dim]")
        if self.keys.shape != self.values.shape:
            raise ValueError("QSA key/value cache shapes differ")
        if self.raw_index_keys.shape[0] != self.keys.shape[0]:
            raise ValueError("QSA index and value cache batch dimensions differ")
        if self.raw_index_keys.shape[1] != self.keys.shape[-2]:
            raise ValueError("QSA index and value cache lengths differ")
        if not (self.raw_index_keys.dtype == self.keys.dtype == self.values.dtype == torch.bfloat16):
            raise ValueError("QSA cache tensors must remain BF16")
        if not (self.raw_index_keys.device == self.keys.device == self.values.device):
            raise ValueError("QSA cache tensors must share one device")

    @property
    def length(self) -> int:
        return self.raw_index_keys.shape[1]

    def prefix(self, length: int) -> "Qwen38QSAState":
        """Return a non-mutating rollback view of the first ``length`` positions."""

        if not 0 <= length <= self.length:
            raise ValueError(f"QSA prefix {length} is outside cached QSA length {self.length}")
        return Qwen38QSAState(
            raw_index_keys=self.raw_index_keys[:, :length],
            keys=self.keys[:, :, :length],
            values=self.values[:, :, :length],
        )


class Qwen38QSA:
    """Exact indexer plus dense short-context attention value oracle."""

    def __init__(self, weights: Qwen38QSAWeights):
        self.weights = weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        state: Qwen38QSAState | None = None,
    ) -> tuple[torch.Tensor, Qwen38QSAState, torch.Tensor]:
        dims = self.weights.dimensions
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != dims.hidden_size:
            raise ValueError(f"hidden_states must be [batch, sequence, {dims.hidden_size}]")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"QSA oracle requires BF16 activations, got {hidden_states.dtype}")
        batch, sequence, _ = hidden_states.shape
        previous_length = 0 if state is None else state.length
        key_length = previous_length + sequence
        if state is not None:
            if state.raw_index_keys.shape != (batch, previous_length, dims.index_head_dim):
                raise ValueError("cached raw QSA index keys do not match the current batch/dimensions")
            if state.keys.shape != (batch, dims.kv_heads, previous_length, dims.head_dim):
                raise ValueError("cached QSA keys do not match the pinned KV dimensions")
        if attention_mask.shape != (batch, 1, sequence, key_length):
            raise ValueError(f"QSA mask must be [B,1,{sequence},{key_length}] for the supplied cache")
        cos, sin = position_embeddings
        if cos.shape != (batch, key_length, dims.rope_dim) or sin.shape != cos.shape:
            raise ValueError(f"position embeddings must both cover the full cache as [B,{key_length},{dims.rope_dim}]")
        current_cos = cos[:, -sequence:]
        current_sin = sin[:, -sequence:]

        index_qk = F.linear(hidden_states, self.weights.index_qk)
        index_query, raw_keys = torch.split(
            index_qk,
            (dims.index_query_heads * dims.index_head_dim, dims.index_kv_heads * dims.index_head_dim),
            dim=-1,
        )
        index_query = index_query.reshape(batch, sequence, dims.index_query_heads, dims.index_head_dim)
        current_raw_keys = raw_keys.reshape(batch, sequence, dims.index_head_dim)
        raw_keys = current_raw_keys if state is None else torch.cat((state.raw_index_keys, current_raw_keys), dim=1)
        selected = qsa_selected_token_mask(
            index_query,
            raw_keys,
            cos,
            sin,
            attention_mask,
            q_norm_weight=self.weights.index_q_norm,
            k_norm_weight=self.weights.index_k_norm,
            token_budget=dims.token_budget,
            compress_ratio=dims.compress_ratio,
            eps=self.weights.rms_norm_eps,
        )
        if attention_mask.is_floating_point():
            combined_mask = attention_mask + selected
            selected_visible = selected == 0
        else:
            combined_mask = attention_mask & selected
            selected_visible = selected

        qg = F.linear(hidden_states, self.weights.qg).view(batch, sequence, dims.query_heads, 2 * dims.head_dim)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(batch, sequence, dims.query_width)
        query = zero_centered_rms_norm(query, self.weights.q_norm, self.weights.rms_norm_eps)
        key = F.linear(hidden_states, self.weights.k).view(batch, sequence, dims.kv_heads, dims.head_dim)
        key = zero_centered_rms_norm(key, self.weights.k_norm, self.weights.rms_norm_eps)
        value = F.linear(hidden_states, self.weights.v).view(batch, sequence, dims.kv_heads, dims.head_dim)

        query = _partial_rope(query.transpose(1, 2), current_cos, current_sin, unsqueeze_dim=1)
        current_key = _partial_rope(key.transpose(1, 2), current_cos, current_sin, unsqueeze_dim=1)
        current_value = value.transpose(1, 2)
        if state is None:
            key, value = current_key, current_value
        else:
            key = torch.cat((state.keys, current_key), dim=-2)
            value = torch.cat((state.values, current_value), dim=-2)
        repeated_key = key.repeat_interleave(dims.kv_repeat, dim=1)
        repeated_value = value.repeat_interleave(dims.kv_repeat, dim=1)
        scores = torch.matmul(query, repeated_key.transpose(2, 3)) * (dims.head_dim**-0.5)
        if combined_mask.is_floating_point():
            scores = scores + combined_mask
        else:
            scores = scores.masked_fill(~combined_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = (
            torch.matmul(probabilities, repeated_value)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch, sequence, dims.query_width)
        )
        output = output * torch.sigmoid(gate)
        next_state = Qwen38QSAState(raw_index_keys=raw_keys, keys=key, values=value)
        return F.linear(output, self.weights.out), next_state, selected_visible
