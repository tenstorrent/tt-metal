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
EMITTED_BATCH_SIZE = 32
EMITTED_DECODE_CACHE_LEN = 128


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

        expected = {
            "hidden_size": (hidden_size, 4096),
            "num_attention_heads": (num_attention_heads, 32),
            "num_key_value_heads": (num_key_value_heads, 8),
            "head_dim": (head_dim, 128),
            "intermediate_size": (intermediate_size, 14336),
            "max_position_embeddings": (max_position_embeddings, 131072),
        }
        wrong = [f"{name}={actual} (expected {want})" for name, (actual, want) in expected.items() if actual != want]
        if wrong:
            raise ValueError(f"{HF_MODEL_ID} functional decoder config mismatch: {', '.join(wrong)}")

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


def _optional_layer_tensor(state_dict: dict[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor | None:
    for key in _canonical_key(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    return None


def _get_layer_tensor(state_dict: dict[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    tensor = _optional_layer_tensor(state_dict, layer_idx, suffix)
    if tensor is None:
        raise KeyError(f"missing Llama 3.1 layer {layer_idx} tensor for {suffix}")
    return tensor


def _to_device_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
    )


def _fused_emit_weight(state_dict: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    q_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
    k_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
    v_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
    if layer_idx == 31:
        return torch.cat((q_proj.transpose(0, 1), k_proj.transpose(0, 1), v_proj.transpose(0, 1)), dim=1)
    return torch.cat((q_proj.transpose(0, 1), v_proj.transpose(0, 1), k_proj.transpose(0, 1)), dim=1)


def build_rope_tables(
    hf_config,
    seq_len: int,
    mesh_device,
    *,
    start_pos: int = 0,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    cfg = Llama31DecoderConfig.from_hf_config(hf_config)
    rotary = LlamaRotaryEmbedding(hf_config)
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
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


def build_decode_mask(batch: int, cache_position: int, cache_len: int, mesh_device) -> ttnn.Tensor:
    mask = torch.full((1, 1, batch, cache_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask[:, :, :, : cache_position + 1] = 0.0
    return _to_device_tensor(mask, mesh_device)


def build_update_indices(batch: int, cache_position: int, mesh_device) -> ttnn.Tensor:
    update_idxs = torch.full((batch,), cache_position, dtype=torch.int32)
    return _to_device_tensor(update_idxs, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)


def _decode_height_sharded_memory_config() -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
                ]
            ),
            [32, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


class FunctionalDecoder(LightweightModule):
    """Single Llama-3.1-8B-Instruct decoder layer translated from the forge emit."""

    def __init__(
        self,
        *,
        cfg: Llama31DecoderConfig,
        layer_idx: int,
        mesh_device,
        batch: int,
        qkv_proj_weight: ttnn.Tensor,
        o_proj_weight: ttnn.Tensor,
        gate_proj_weight: ttnn.Tensor,
        up_proj_weight: ttnn.Tensor,
        down_proj_weight: ttnn.Tensor,
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
        q_norm_weight: ttnn.Tensor | None,
        k_norm_weight: ttnn.Tensor | None,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None,
        max_seq_len: int,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.qkv_proj_weight = qkv_proj_weight
        self.o_proj_weight = o_proj_weight
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.down_proj_weight = down_proj_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
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
        batch: int = EMITTED_BATCH_SIZE,
        max_seq_len: int = EMITTED_DECODE_CACHE_LEN,
        **kwargs,
    ) -> "FunctionalDecoder":
        if kwargs:
            raise TypeError(f"unsupported FunctionalDecoder kwargs: {sorted(kwargs)}")
        cfg = Llama31DecoderConfig.from_hf_config(hf_config)
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        if max_seq_len <= 0 or max_seq_len > cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {cfg.max_position_embeddings}], got {max_seq_len}")

        position_cos, position_sin = build_rope_tables(hf_config, max_seq_len, mesh_device)
        q_norm = _optional_layer_tensor(state_dict, layer_idx, "self_attn.q_norm.weight")
        k_norm = _optional_layer_tensor(state_dict, layer_idx, "self_attn.k_norm.weight")

        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            qkv_proj_weight=_to_device_tensor(_fused_emit_weight(state_dict, layer_idx), mesh_device),
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
            q_norm_weight=_to_device_tensor(q_norm, mesh_device) if q_norm is not None else None,
            k_norm_weight=_to_device_tensor(k_norm, mesh_device) if k_norm is not None else None,
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=None,
            max_seq_len=max_seq_len,
        )

    def _split_emit_qkv(self, qkv: ttnn.Tensor, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        cfg = self.cfg
        v_width = cfg.num_key_value_heads * cfg.head_dim
        q_width = cfg.num_attention_heads * cfg.head_dim
        q = ttnn.slice(
            qkv,
            [0, 0, 0, 0],
            [1, self.batch, seq_len, q_width],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.layer_idx == 31:
            k = ttnn.slice(
                qkv,
                [0, 0, 0, q_width],
                [1, self.batch, seq_len, q_width + v_width],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v = ttnn.slice(
                qkv,
                [0, 0, 0, q_width + v_width],
                [1, self.batch, seq_len, q_width + v_width + v_width],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            v = ttnn.slice(
                qkv,
                [0, 0, 0, q_width],
                [1, self.batch, seq_len, q_width + v_width],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k = ttnn.slice(
                qkv,
                [0, 0, 0, q_width + v_width],
                [1, self.batch, seq_len, q_width + v_width + v_width],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return v, q, k

    def _prepare_qkv(
        self,
        qkv: ttnn.Tensor,
        seq_len: int,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        cfg = self.cfg
        v, q, k = self._split_emit_qkv(qkv, seq_len)

        q = ttnn.reshape(
            q, [self.batch, seq_len, cfg.num_attention_heads, cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.q_norm_weight is not None:
            q = ttnn.rms_norm(q, epsilon=1.0e-6, weight=self.q_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(
            q,
            [0, 0, 0, 0],
            [self.batch, cfg.num_attention_heads, seq_len, cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        k = ttnn.reshape(
            k, [self.batch, seq_len, cfg.num_key_value_heads, cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.k_norm_weight is not None:
            k = ttnn.rms_norm(k, epsilon=1.0e-6, weight=self.k_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(
            k,
            [0, 0, 0, 0],
            [self.batch, cfg.num_key_value_heads, seq_len, cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        v = ttnn.reshape(
            v, [self.batch, seq_len, cfg.num_key_value_heads, cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return q, k, v

    def _attention_mlp(self, hidden_states: ttnn.Tensor, attn: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        cfg = self.cfg
        attn = ttnn.reshape(
            attn,
            [1, self.batch, seq_len, cfg.num_attention_heads * cfg.head_dim],
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

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor | None = None,
        position_sin: ttnn.Tensor | None = None,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        if hidden_states.shape[-3] != self.batch:
            raise ValueError(
                f"hidden_states batch {hidden_states.shape[-3]} does not match configured batch {self.batch}"
            )
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
        q, k, v = self._prepare_qkv(qkv, seq_len, cos, sin)
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
        return self._attention_mlp(hidden_states, attn, seq_len)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        update_idxs_tensor: ttnn.Tensor,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        if seq_len != 1:
            raise ValueError(f"decode_forward is the emitted single-token decode path; got seq_len={seq_len}")
        if hidden_states.shape[-3] != self.batch:
            raise ValueError(
                f"hidden_states batch {hidden_states.shape[-3]} does not match configured batch {self.batch}"
            )

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
        q, k, v = self._prepare_qkv(qkv, 1, position_cos, position_sin)
        k_update = ttnn.reshape(
            k, [1, self.batch, self.cfg.num_key_value_heads, self.cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v_update = ttnn.reshape(
            v, [1, self.batch, self.cfg.num_key_value_heads, self.cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_update = ttnn.to_memory_config(k_update, _decode_height_sharded_memory_config())
        v_update = ttnn.to_memory_config(v_update, _decode_height_sharded_memory_config())
        ttnn.experimental.paged_update_cache(
            key_cache,
            k_update,
            update_idxs_tensor=update_idxs_tensor,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            v_update,
            update_idxs_tensor=update_idxs_tensor,
            share_cache=False,
            page_table=None,
        )
        q = ttnn.reshape(
            q, [1, self.batch, self.cfg.num_attention_heads, self.cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            key_cache,
            value_cache,
            is_causal=False,
            attn_mask=attention_mask,
            cur_pos_tensor=None,
            attention_sink=None,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn = ttnn.to_memory_config(attn, _decode_height_sharded_memory_config())
        attn = ttnn.experimental.nlp_concat_heads_decode(
            attn,
            sub_core_grids=attn.memory_config().shard_spec.grid,
            num_heads=self.cfg.num_attention_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._attention_mlp(hidden_states, attn, 1)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str = "prefill", **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported FunctionalDecoder mode: {mode!r}")
