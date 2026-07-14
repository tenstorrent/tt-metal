# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Forge-translated functional decoder for meta-llama/Llama-3.1-8B-Instruct.

The source emit is decode-only: it consumes one token per batch slot plus a
persistent per-layer KV cache and appends at the supplied cache position. This
module keeps that contract. Host conversions and constant construction are
restricted to ``from_state_dict`` and helper builders; runtime forwards are TTNN
only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EMITTED_BATCH_SIZE = 32
EMITTED_CACHE_LEN = 128
PREFILL_NOT_EMITTED_MESSAGE = (
    "The tt-forge emit for meta-llama/Llama-3.1-8B-Instruct did not ship a "
    "prefill graph; this functional stage translates the emitted single-token "
    "decode graph only."
)


@dataclass(frozen=True)
class Llama31DecoderConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "Llama31DecoderConfig":
        hidden_size = int(hf_config.hidden_size)
        num_attention_heads = int(hf_config.num_attention_heads)
        num_key_value_heads = int(hf_config.num_key_value_heads)
        head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_attention_heads))
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        rope_theta = float(getattr(hf_config, "rope_theta", None) or rope_scaling.get("rope_theta", 500000.0))

        expected = {
            "hidden_size": (hidden_size, 4096),
            "num_attention_heads": (num_attention_heads, 32),
            "num_key_value_heads": (num_key_value_heads, 8),
            "head_dim": (head_dim, 128),
            "intermediate_size": (int(hf_config.intermediate_size), 14336),
        }
        for name, (actual, want) in expected.items():
            if actual != want:
                raise ValueError(f"{MODEL_ID} functional decoder expects {name}={want}, got {actual}")

        return cls(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=int(hf_config.intermediate_size),
            max_position_embeddings=int(hf_config.max_position_embeddings),
            rms_norm_eps=float(hf_config.rms_norm_eps),
            rope_theta=rope_theta,
        )


def _canonical_keys(layer_idx: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.language_model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )


def _state_tensor(state_dict: dict[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    for key in _canonical_keys(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"missing decoder state tensor for layer {layer_idx}: {suffix}")


def _device_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_qvk_weight(state_dict: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").transpose(-2, -1)
    v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").transpose(-2, -1)
    k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").transpose(-2, -1)
    if layer_idx == 31:
        return torch.cat([q, k, v], dim=-1).unsqueeze(0).unsqueeze(0)
    return torch.cat([q, v, k], dim=-1).unsqueeze(0).unsqueeze(0)


def build_decode_rope(
    hf_config: Any, cache_position: int | torch.Tensor, mesh_device
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Build RoPE tables matching the emit's inv_freq matmul/cos/sin const path."""
    cfg = Llama31DecoderConfig.from_hf_config(hf_config)
    if isinstance(cache_position, torch.Tensor):
        pos = cache_position.detach().to(torch.float32).reshape(-1)
        if pos.numel() != 1:
            raise ValueError("decode RoPE builder expects one shared cache position")
        position = pos[0]
    else:
        position = torch.tensor(float(cache_position), dtype=torch.float32)
    inv_freq = build_llama31_inv_freq(hf_config).to(torch.float32)
    angles = inv_freq * position
    emb = torch.cat([angles, angles], dim=0).reshape(1, 1, 1, cfg.head_dim)
    return (
        _device_tensor(torch.cos(emb).to(torch.bfloat16), mesh_device),
        _device_tensor(torch.sin(emb).to(torch.bfloat16), mesh_device),
    )


def build_llama31_inv_freq(hf_config: Any) -> torch.Tensor:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary = LlamaRotaryEmbedding(hf_config)
    return rotary.inv_freq.detach().cpu()


def build_decode_rope_torch(hf_config: Any, cache_position: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = Llama31DecoderConfig.from_hf_config(hf_config)
    if isinstance(cache_position, torch.Tensor):
        position = cache_position.detach().to(torch.float32).reshape(-1)[0]
    else:
        position = torch.tensor(float(cache_position), dtype=torch.float32)
    angles = build_llama31_inv_freq(hf_config).to(torch.float32) * position
    emb = torch.cat([angles, angles], dim=0).reshape(1, 1, 1, cfg.head_dim)
    return torch.cos(emb), torch.sin(emb)


def build_decode_attention_mask(cache_position: int | torch.Tensor, cache_len: int, mesh_device) -> ttnn.Tensor:
    if isinstance(cache_position, torch.Tensor):
        pos = int(cache_position.reshape(-1)[0].item())
    else:
        pos = int(cache_position)
    arange = torch.arange(cache_len, dtype=torch.int64)
    mask = torch.where(arange <= pos, torch.tensor(0.0), torch.tensor(float("-inf")))
    mask = mask.reshape(1, 1, 1, cache_len).repeat(1, 1, EMITTED_BATCH_SIZE, 1).to(torch.bfloat16)
    return _device_tensor(mask, mesh_device)


def _decode_height_sharded_memcfg(batch: int) -> ttnn.MemoryConfig:
    if batch != EMITTED_BATCH_SIZE:
        raise ValueError(
            f"decode L1 sharding is workload-derived for emitted batch {EMITTED_BATCH_SIZE}; got batch={batch}"
        )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
                ]
            ),
            [batch, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


class FunctionalDecoder(LightweightModule):
    """Single decoder layer translated from the forge single-token decode emit."""

    def __init__(
        self,
        *,
        cfg: Llama31DecoderConfig,
        layer_idx: int,
        mesh_device,
        batch: int,
        cache_len: int,
        qvk_proj_weight: ttnn.Tensor,
        o_proj_weight: ttnn.Tensor,
        gate_proj_weight: ttnn.Tensor,
        up_proj_weight: ttnn.Tensor,
        down_proj_weight: ttnn.Tensor,
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.cache_len = cache_len
        self.qvk_proj_weight = qvk_proj_weight
        self.o_proj_weight = o_proj_weight
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.down_proj_weight = down_proj_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH_SIZE,
        cache_len: int = EMITTED_CACHE_LEN,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        **kwargs,
    ) -> "FunctionalDecoder":
        if kwargs:
            raise TypeError(f"unsupported FunctionalDecoder kwargs: {sorted(kwargs)}")
        cfg = Llama31DecoderConfig.from_hf_config(hf_config)
        if batch != EMITTED_BATCH_SIZE:
            raise ValueError(f"forge decode emit preserves batch={EMITTED_BATCH_SIZE}; got batch={batch}")
        if cache_len <= 0 or cache_len > cfg.max_position_embeddings:
            raise ValueError(f"cache_len must be in [1, {cfg.max_position_embeddings}], got {cache_len}")

        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            cache_len=cache_len,
            qvk_proj_weight=_device_tensor(_build_qvk_weight(state_dict, layer_idx), mesh_device, dtype=weight_dtype),
            o_proj_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
                .transpose(-2, -1)
                .unsqueeze(0)
                .unsqueeze(0),
                mesh_device,
                dtype=weight_dtype,
            ),
            gate_proj_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
                .transpose(-2, -1)
                .unsqueeze(0)
                .unsqueeze(0),
                mesh_device,
                dtype=weight_dtype,
            ),
            up_proj_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0),
                mesh_device,
                dtype=weight_dtype,
            ),
            down_proj_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
                .transpose(-2, -1)
                .unsqueeze(0)
                .unsqueeze(0),
                mesh_device,
                dtype=weight_dtype,
            ),
            input_layernorm_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "input_layernorm.weight").reshape(1, 1, 1, -1),
                mesh_device,
            ),
            post_attention_layernorm_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").reshape(1, 1, 1, -1),
                mesh_device,
            ),
        )

    def prefill_forward(self, *args, **kwargs) -> ttnn.Tensor:
        raise NotImplementedError(PREFILL_NOT_EMITTED_MESSAGE)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        cache_position: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        key_cache_update_idxs: ttnn.Tensor | None = None,
        value_cache_update_idxs: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.input_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        qvk = ttnn.matmul(
            normed,
            self.qvk_proj_weight,
            transpose_a=False,
            transpose_b=False,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qvk = ttnn.reshape(qvk, [self.batch, 1, 6144], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        split_0, split_1, split_2 = ttnn.transformer.split_query_key_value_and_split_heads(
            qvk,
            None,
            num_heads=self.cfg.num_attention_heads,
            num_kv_heads=self.cfg.num_key_value_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.layer_idx == 31:
            query, key, value = split_0, split_1, split_2
        else:
            query, value, key = split_0, split_1, split_2

        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.cfg.num_key_value_heads, 1, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.reshape(
            key,
            [1, self.batch, self.cfg.num_key_value_heads, self.cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.to_memory_config(key, _decode_height_sharded_memcfg(self.batch))
        key_idxs = (
            key_cache_update_idxs
            if key_cache_update_idxs is not None
            else ttnn.repeat(cache_position, ttnn.Shape([self.batch]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=key_idxs,
            share_cache=False,
            page_table=None,
        )

        value = ttnn.reshape(
            value,
            [1, self.batch, self.cfg.num_key_value_heads, self.cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value = ttnn.to_memory_config(value, _decode_height_sharded_memcfg(self.batch))
        value_idxs = (
            value_cache_update_idxs
            if value_cache_update_idxs is not None
            else ttnn.repeat(cache_position, ttnn.Shape([self.batch]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=value_idxs,
            share_cache=False,
            page_table=None,
        )

        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.cfg.num_attention_heads, 1, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query = ttnn.reshape(
            query,
            [1, self.batch, self.cfg.num_attention_heads, self.cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
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
        attn_out = ttnn.to_memory_config(attn_out, _decode_height_sharded_memcfg(self.batch))
        attn_out = ttnn.experimental.nlp_concat_heads_decode(
            attn_out,
            sub_core_grids=attn_out.memory_config().shard_spec.grid,
            num_heads=self.cfg.num_attention_heads,
            memory_config=_decode_height_sharded_memcfg(self.batch),
        )
        attn_out = ttnn.to_memory_config(attn_out, ttnn.DRAM_MEMORY_CONFIG)
        attn_out = ttnn.reshape(
            attn_out,
            [1, 1, self.batch, self.cfg.hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = ttnn.matmul(
            attn_out,
            self.o_proj_weight,
            transpose_a=False,
            transpose_b=False,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(attn_out, residual, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mlp_in = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        gate = ttnn.matmul(
            mlp_in,
            self.gate_proj_weight,
            transpose_a=False,
            transpose_b=False,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        up = ttnn.matmul(
            mlp_in,
            self.up_proj_weight,
            transpose_a=False,
            transpose_b=False,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mlp = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mlp = ttnn.matmul(
            mlp,
            self.down_proj_weight,
            transpose_a=False,
            transpose_b=False,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.add(mlp, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs) -> ttnn.Tensor:
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported FunctionalDecoder mode {mode!r}; expected 'decode' or 'prefill'")
