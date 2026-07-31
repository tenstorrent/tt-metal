# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized TTNN decoder layer for microsoft/Phi-3.5-mini-instruct.

Runtime contract
----------------
``from_state_dict`` is the weight-loading boundary. It accepts a HuggingFace
decoder-layer state dict using canonical Phi-3 keys:

* ``self_attn.qkv_proj.weight``
* ``self_attn.o_proj.weight``
* ``mlp.gate_up_proj.weight``
* ``mlp.down_proj.weight``
* ``input_layernorm.weight``
* ``post_attention_layernorm.weight``

The two forward methods are TTNN-only hot paths. Tests may create inputs and
compare outputs with torch at explicit boundaries, but a measured forward pass
does not call torch, ``ttnn.from_torch``, or ``ttnn.to_torch``.

Prefill signature::

    prefill_forward(
        hidden_states,
        *,
        page_table,
        kv_cache,
        user_id=0,
        start_pos=0,
        rope_sequence_length=None,
        cache_position_modulo=None,
    )

``hidden_states`` is a TILE-layout TTNN tensor of shape ``[1, 1, seq_len, 3072]``.
``page_table`` is a ROW_MAJOR int32 TTNN tensor of shape
``[max_batch, ceil(seq_len / block_size)]`` or wider. ``kv_cache`` is a pair
``(k_cache, v_cache)`` of paged TTNN tensors shaped
``[num_blocks, 32, block_size, 96]``. The method fills the paged cache and
returns ``[1, 1, seq_len, 3072]``.

Decode signature::

    decode_forward(
        hidden_states,
        *,
        current_pos,
        position_ids=None,
        page_table,
        kv_cache,
        rope_sequence_length=None,
        cache_position_modulo=None,
    )

``hidden_states`` is a TILE-layout TTNN tensor of shape ``[1, 1, batch, 3072]``.
``current_pos`` is an int32 TTNN tensor of shape ``[batch]`` and is used for
paged cache updates and paged SDPA, making the decode pass trace-safe for fixed
short-vs-long RoPE selection. ``page_table`` is the paged-attention mapping for
the same batch. The method updates ``kv_cache`` and returns
``[1, 1, batch, 3072]``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import torch
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.precision import load_precision_policy
from models.common.lightweightmodule import LightweightModule


MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 8192
NUM_HEADS = 32
NUM_KV_HEADS = 32
HEAD_DIM = 96
QKV_SIZE = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
DEFAULT_BLOCK_SIZE = 32
TILE_SIZE = 32
PREFILL_QKV_CHUNK_SIZE = 2048
PREFILL_MATMUL_CHUNK_SIZE = 1024


@dataclass(frozen=True)
class Phi35MiniOptimizedDecoderConfig:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    original_max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    block_size: int = DEFAULT_BLOCK_SIZE
    dtype: ttnn.DataType = ttnn.bfloat16
    attention_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    mlp_prefill_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    @classmethod
    def from_hf_config(cls, hf_config, *, block_size: int = DEFAULT_BLOCK_SIZE) -> "Phi35MiniOptimizedDecoderConfig":
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        if hf_config.hidden_size != HIDDEN_SIZE:
            raise ValueError(f"Phi-3.5 mini hidden_size must be {HIDDEN_SIZE}, got {hf_config.hidden_size}")
        if hf_config.intermediate_size != INTERMEDIATE_SIZE:
            raise ValueError(
                f"Phi-3.5 mini intermediate_size must be {INTERMEDIATE_SIZE}, got {hf_config.intermediate_size}"
            )
        if hf_config.num_attention_heads != NUM_HEADS or hf_config.num_key_value_heads != NUM_KV_HEADS:
            raise ValueError(
                "Phi-3.5 mini optimized decoder expects dense 32Q/32KV attention, got "
                f"{hf_config.num_attention_heads}Q/{hf_config.num_key_value_heads}KV"
            )
        if head_dim != HEAD_DIM:
            raise ValueError(f"Phi-3.5 mini head_dim must be {HEAD_DIM}, got {head_dim}")
        if getattr(hf_config, "attention_bias", False):
            raise ValueError("Phi-3.5 mini attention_bias=True is not supported by this optimized decoder")

        precision = load_precision_policy()
        return cls(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=hf_config.max_position_embeddings,
            original_max_position_embeddings=hf_config.original_max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            rms_norm_eps=hf_config.rms_norm_eps,
            block_size=block_size,
            dtype=precision.residual_dtype,
            attention_weight_dtype=precision.weight_dtype("attention.qkv"),
            mlp_weight_dtype=precision.weight_dtype("mlp.gate_up"),
            mlp_prefill_weight_dtype=precision.weight_dtype("mlp.gate_up", prefill=True),
            cache_dtype=precision.kv_cache_dtype,
        )


class OptimizedDecoder(LightweightModule):
    """Single dense Phi-3.5-mini decoder layer with TTNN decoder-stage optimizations."""

    def __init__(
        self,
        *,
        config: Phi35MiniOptimizedDecoderConfig,
        mesh_device: ttnn.MeshDevice,
        layer_idx: int,
        input_norm_weight: ttnn.Tensor,
        post_norm_weight: ttnn.Tensor,
        qkv_weight: ttnn.Tensor,
        o_weight: ttnn.Tensor,
        gate_up_weight: ttnn.Tensor,
        down_weight: ttnn.Tensor,
        qkv_weight_prefill: ttnn.Tensor,
        o_weight_prefill: ttnn.Tensor,
        gate_up_weight_prefill: ttnn.Tensor,
        down_weight_prefill: ttnn.Tensor,
        rope_tables: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]],
        max_decode_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.input_norm_weight = input_norm_weight
        self.post_norm_weight = post_norm_weight
        self.qkv_weight = qkv_weight
        self.o_weight = o_weight
        self.gate_up_weight = gate_up_weight
        self.down_weight = down_weight
        self.qkv_weight_prefill = qkv_weight_prefill
        self.o_weight_prefill = o_weight_prefill
        self.gate_up_weight_prefill = gate_up_weight_prefill
        self.down_weight_prefill = down_weight_prefill
        self.rope_tables = rope_tables
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.max_decode_batch_size = max_decode_batch_size
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config = self.compute_kernel_config_hifi2
        self.decode_hidden_mem_config = _width_sharded_decode_mem_config(mesh_device, config.hidden_size)
        self.decode_qkv_mem_config = _width_sharded_decode_mem_config(mesh_device, QKV_SIZE)
        self.decode_gate_up_mem_config = _width_sharded_decode_mem_config(mesh_device, 2 * config.intermediate_size)
        self.decode_mlp_intermediate_mem_config = _width_sharded_decode_mem_config(
            mesh_device, config.intermediate_size
        )
        self.decode_kv_mem_config = _height_sharded_decode_mem_config(
            mesh_device, config.num_kv_heads, config.head_dim, max_batch_size=max_decode_batch_size
        )
        self.decode_q_mem_config = _height_sharded_decode_mem_config(
            mesh_device, config.num_heads, config.head_dim, max_batch_size=max_decode_batch_size
        )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=_sdpa_grid(mesh_device),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )
        self.decode_qkv_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=config.hidden_size,
            n=QKV_SIZE,
            num_cores=_dram_shard_core_grid(config.hidden_size).num_cores,
        )
        self.decode_o_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=config.hidden_size,
            n=config.hidden_size,
            num_cores=_dram_shard_core_grid(config.hidden_size).num_cores,
        )
        self.decode_gate_up_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=config.hidden_size,
            n=2 * config.intermediate_size,
            num_cores=_dram_shard_core_grid(config.hidden_size).num_cores,
        )
        self.decode_down_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=config.intermediate_size,
            n=config.hidden_size,
            num_cores=_dram_shard_core_grid(config.intermediate_size).num_cores,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_position_embeddings: int | None = None,
        batch: int = 1,
        **_: object,
    ) -> "OptimizedDecoder":
        """Create a decoder from a HF layer state dict.

        Weight conversion is intentionally eager and explicit here so runtime
        forwards have no hidden host fallback.
        """

        config = Phi35MiniOptimizedDecoderConfig.from_hf_config(hf_config, block_size=block_size)
        if max_position_embeddings is not None:
            config = Phi35MiniOptimizedDecoderConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                max_position_embeddings=max_position_embeddings,
                original_max_position_embeddings=config.original_max_position_embeddings,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                block_size=config.block_size,
                dtype=config.dtype,
                attention_weight_dtype=config.attention_weight_dtype,
                mlp_weight_dtype=config.mlp_weight_dtype,
                mlp_prefill_weight_dtype=config.mlp_prefill_weight_dtype,
                cache_dtype=config.cache_dtype,
            )

        required = {
            "self_attn.qkv_proj.weight": (QKV_SIZE, HIDDEN_SIZE),
            "self_attn.o_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
            "mlp.gate_up_proj.weight": (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "mlp.down_proj.weight": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            "input_layernorm.weight": (HIDDEN_SIZE,),
            "post_attention_layernorm.weight": (HIDDEN_SIZE,),
        }
        for name, shape in required.items():
            if name not in state_dict:
                raise KeyError(f"missing Phi decoder weight: {name}")
            if tuple(state_dict[name].shape) != shape:
                raise ValueError(f"{name} shape {tuple(state_dict[name].shape)} != expected {shape}")

        qkv_weight = _dram_sharded_weight_to_device(
            state_dict["self_attn.qkv_proj.weight"].T,
            mesh_device,
            dtype=config.attention_weight_dtype,
        )
        o_weight = _dram_sharded_weight_to_device(
            state_dict["self_attn.o_proj.weight"].T,
            mesh_device,
            dtype=config.attention_weight_dtype,
        )
        gate_up_weight = _dram_sharded_weight_to_device(
            state_dict["mlp.gate_up_proj.weight"].T,
            mesh_device,
            dtype=config.mlp_weight_dtype,
        )
        down_weight = _dram_sharded_weight_to_device(
            state_dict["mlp.down_proj.weight"].T,
            mesh_device,
            dtype=config.mlp_weight_dtype,
        )
        qkv_weight_prefill = _weight_to_device(
            state_dict["self_attn.qkv_proj.weight"].T, mesh_device, dtype=config.attention_weight_dtype
        )
        o_weight_prefill = _weight_to_device(
            state_dict["self_attn.o_proj.weight"].T, mesh_device, dtype=config.attention_weight_dtype
        )
        gate_up_weight_prefill = _weight_to_device(
            state_dict["mlp.gate_up_proj.weight"].T, mesh_device, dtype=config.mlp_prefill_weight_dtype
        )
        down_weight_prefill = _weight_to_device(
            state_dict["mlp.down_proj.weight"].T, mesh_device, dtype=config.mlp_prefill_weight_dtype
        )
        input_norm_weight = _norm_weight_to_device(state_dict["input_layernorm.weight"], mesh_device)
        post_norm_weight = _norm_weight_to_device(state_dict["post_attention_layernorm.weight"], mesh_device)
        rope_tables = _build_rope_tables(hf_config, config, mesh_device)

        return cls(
            config=config,
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            input_norm_weight=input_norm_weight,
            post_norm_weight=post_norm_weight,
            qkv_weight=qkv_weight,
            o_weight=o_weight,
            gate_up_weight=gate_up_weight,
            down_weight=down_weight,
            qkv_weight_prefill=qkv_weight_prefill,
            o_weight_prefill=o_weight_prefill,
            gate_up_weight_prefill=gate_up_weight_prefill,
            down_weight_prefill=down_weight_prefill,
            rope_tables=rope_tables,
            max_decode_batch_size=batch,
        )

    @staticmethod
    def allocate_paged_kv_cache(
        *,
        hf_config,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        dtype: ttnn.DataType = ttnn.bfloat8_b,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Allocate empty paged K/V cache tensors for this decoder layer."""

        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        num_blocks_per_seq = math.ceil(max_seq_len / block_size)
        num_blocks = max_batch_size * num_blocks_per_seq
        cache_shape = (num_blocks, hf_config.num_key_value_heads, block_size, head_dim)
        zero_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
        k_cache = _host_to_device(zero_cache, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        v_cache = _host_to_device(zero_cache, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return k_cache, v_cache

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported mode {mode!r}; expected 'prefill' or 'decode'")

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        start_pos: int = 0,
        rope_sequence_length: int | None = None,
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Run paged prefill for one user and return layer output."""

        cfg = self.config
        seq_len = int(hidden_states.shape[-2])
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError(f"hidden width must be {cfg.hidden_size}, got {hidden_states.shape[-1]}")
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1")
        if seq_len % cfg.block_size != 0:
            raise ValueError(f"prefill seq_len must be a multiple of block_size={cfg.block_size}, got {seq_len}")

        residual = hidden_states
        attn_in = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.input_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        qkv_m = seq_len
        if seq_len > PREFILL_QKV_CHUNK_SIZE:
            if seq_len % PREFILL_QKV_CHUNK_SIZE != 0:
                raise ValueError(f"prefill seq_len {seq_len} must be divisible by {PREFILL_QKV_CHUNK_SIZE}")
            attn_in = ttnn.reshape(attn_in, [1, seq_len // PREFILL_QKV_CHUNK_SIZE, PREFILL_QKV_CHUNK_SIZE, -1])
            qkv_m = PREFILL_QKV_CHUNK_SIZE
        qkv = ttnn.linear(
            attn_in,
            self.qkv_weight_prefill,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_matmul_config(qkv_m, cfg.hidden_size, QKV_SIZE),
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_in)
        if seq_len > PREFILL_QKV_CHUNK_SIZE:
            qkv = ttnn.reshape(qkv, [1, 1, seq_len, -1])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        cos, sin = self._prefill_rope_tables(start_pos, seq_len, rope_sequence_length)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        k_cache, v_cache = kv_cache
        k_for_cache = _typecast_if_needed(k, k_cache.dtype)
        v_for_cache = _typecast_if_needed(v, v_cache.dtype)
        fill_kwargs = {}
        if cache_position_modulo is not None:
            fill_kwargs["cache_position_modulo"] = cache_position_modulo
        ttnn.experimental.paged_fill_cache(k_cache, k_for_cache, page_table, batch_idx=user_id, **fill_kwargs)
        ttnn.experimental.paged_fill_cache(v_cache, v_for_cache, page_table, batch_idx=user_id, **fill_kwargs)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        if k_for_cache is not k:
            ttnn.deallocate(k_for_cache)
        if v_for_cache is not v:
            ttnn.deallocate(v_for_cache)

        attn_cat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        attn_cat_for_proj = attn_cat
        o_m = seq_len
        if seq_len > PREFILL_MATMUL_CHUNK_SIZE:
            if seq_len % PREFILL_MATMUL_CHUNK_SIZE != 0:
                raise ValueError(f"prefill seq_len {seq_len} must be divisible by {PREFILL_MATMUL_CHUNK_SIZE}")
            attn_cat_for_proj = ttnn.reshape(
                attn_cat_for_proj, [1, seq_len // PREFILL_MATMUL_CHUNK_SIZE, PREFILL_MATMUL_CHUNK_SIZE, -1]
            )
            o_m = PREFILL_MATMUL_CHUNK_SIZE
        elif seq_len <= TILE_SIZE:
            attn_cat_for_proj = ttnn.to_memory_config(attn_cat, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_cat)
        attn_proj = ttnn.linear(
            attn_cat_for_proj,
            self.o_weight_prefill,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_matmul_config(o_m, cfg.hidden_size, cfg.hidden_size),
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        if seq_len > PREFILL_MATMUL_CHUNK_SIZE:
            attn_proj = ttnn.reshape(attn_proj, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_cat_for_proj)
        hidden_states = ttnn.add(residual, attn_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=cfg.dtype)
        ttnn.deallocate(attn_proj)

        mlp_in = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.post_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        mlp_out = self._mlp_forward(mlp_in)
        ttnn.deallocate(mlp_in)
        out = ttnn.add(hidden_states, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=cfg.dtype)
        ttnn.deallocate(mlp_out)
        return out

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        position_ids: ttnn.Tensor | None = None,
        rope_sequence_length: int | None = None,
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Run traced-safe paged decode and return layer output."""

        cfg = self.config
        batch_size = int(hidden_states.shape[-2])
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError(f"hidden width must be {cfg.hidden_size}, got {hidden_states.shape[-1]}")
        if int(hidden_states.shape[-3]) != 1:
            raise ValueError(f"decode hidden_states must have seq_len=1, got shape {hidden_states.shape}")
        if batch_size > self.max_decode_batch_size:
            raise ValueError(
                f"decode batch_size={batch_size} exceeds configured maximum {self.max_decode_batch_size}"
            )

        residual = ttnn.to_memory_config(hidden_states, self.decode_hidden_mem_config)
        attn_in = ttnn.rms_norm(
            residual,
            epsilon=cfg.rms_norm_eps,
            weight=self.input_norm_weight,
            memory_config=self.decode_hidden_mem_config,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        qkv = ttnn.linear(
            attn_in,
            self.qkv_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.decode_qkv_program_config,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_in)
        qkv_interleaved = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.deallocate(qkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_interleaved,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_interleaved)

        cos, sin = self._decode_rope_tables(
            position_ids if position_ids is not None else current_pos, batch_size, rope_sequence_length
        )
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        q = ttnn.to_memory_config(q, self.decode_q_mem_config)
        k = ttnn.to_memory_config(k, self.decode_kv_mem_config)
        v = ttnn.to_memory_config(v, self.decode_kv_mem_config)

        update_kwargs = {}
        if cache_position_modulo is not None:
            update_kwargs["cache_position_modulo"] = cache_position_modulo
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_pos, page_table=page_table, **update_kwargs)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_pos, page_table=page_table, **update_kwargs)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        sdpa_kwargs = {}
        if cache_position_modulo is not None:
            sdpa_kwargs["cache_position_modulo"] = cache_position_modulo
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=(
                self.decode_q_mem_config
                if os.getenv("PHI35_ADVISOR_SDPA_L1", "0") == "1"
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            **sdpa_kwargs,
        )
        ttnn.deallocate(q)
        if os.getenv("PHI35_ADVISOR_SDPA_L1", "0") != "1":
            attn_out = ttnn.to_memory_config(attn_out, self.decode_q_mem_config)
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=cfg.num_heads)
        ttnn.deallocate(attn_out)
        attn_cat = ttnn.to_memory_config(attn_cat, self.decode_hidden_mem_config)
        attn_proj = ttnn.linear(
            attn_cat,
            self.o_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.decode_o_program_config,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_cat)
        if int(attn_proj.shape[-2]) != batch_size:
            attn_proj_full = attn_proj
            attn_proj = ttnn.slice(attn_proj_full, (0, 0, 0, 0), (1, 1, batch_size, cfg.hidden_size))
            ttnn.deallocate(attn_proj_full)
        hidden_states = ttnn.add(residual, attn_proj, memory_config=self.decode_hidden_mem_config, dtype=cfg.dtype)
        ttnn.deallocate(attn_proj)

        mlp_in = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.post_norm_weight,
            memory_config=self.decode_hidden_mem_config,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        mlp_out = self._mlp_forward(mlp_in)
        ttnn.deallocate(mlp_in)
        out = ttnn.add(hidden_states, mlp_out, memory_config=self.decode_hidden_mem_config, dtype=cfg.dtype)
        ttnn.deallocate(mlp_out)
        return out

    def _mlp_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        prefill_seq_len = int(hidden_states.shape[-2])
        reshape_prefill = (not _is_decode_tensor(hidden_states)) and prefill_seq_len > PREFILL_MATMUL_CHUNK_SIZE
        if reshape_prefill:
            if prefill_seq_len % PREFILL_MATMUL_CHUNK_SIZE != 0:
                raise ValueError(
                    f"prefill seq_len {prefill_seq_len} must be divisible by {PREFILL_MATMUL_CHUNK_SIZE}"
                )
            hidden_states = ttnn.reshape(
                hidden_states, [1, prefill_seq_len // PREFILL_MATMUL_CHUNK_SIZE, PREFILL_MATMUL_CHUNK_SIZE, -1]
            )
        gate_up = ttnn.linear(
            hidden_states,
            self.gate_up_weight if _is_decode_tensor(hidden_states) else self.gate_up_weight_prefill,
            dtype=cfg.dtype,
            memory_config=_linear_output_mem_config(hidden_states, self.decode_gate_up_mem_config),
            program_config=_mlp_gate_up_program_config(self, hidden_states),
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        gate_up_shape = _shape_tuple(gate_up)
        gate = ttnn.slice(gate_up, (0, 0, 0, 0), (*gate_up_shape[:-1], cfg.intermediate_size))
        up = ttnn.slice(
            gate_up,
            (0, 0, 0, cfg.intermediate_size),
            (*gate_up_shape[:-1], 2 * cfg.intermediate_size),
        )
        ttnn.deallocate(gate_up)
        down_in = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=cfg.dtype,
            memory_config=(
                self.decode_gate_up_mem_config if _is_decode_tensor(hidden_states) else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        if _is_decode_tensor(hidden_states):
            down_in = ttnn.to_memory_config(down_in, self.decode_mlp_intermediate_mem_config)
        elif int(hidden_states.shape[-2]) <= TILE_SIZE:
            down_in = ttnn.to_memory_config(down_in, ttnn.L1_MEMORY_CONFIG)
        down = ttnn.linear(
            down_in,
            self.down_weight if _is_decode_tensor(hidden_states) else self.down_weight_prefill,
            dtype=cfg.dtype,
            memory_config=_linear_output_mem_config(hidden_states, self.decode_hidden_mem_config),
            program_config=_mlp_down_program_config(self, hidden_states),
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(down_in)
        if reshape_prefill:
            down = ttnn.reshape(down, [1, 1, prefill_seq_len, cfg.hidden_size])
        return down

    def _prefill_rope_tables(
        self, start_pos: int, seq_len: int, rope_sequence_length: int | None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cfg = self.config
        if rope_sequence_length is None:
            rope_sequence_length = start_pos + seq_len
        table_key = "long" if rope_sequence_length > cfg.original_max_position_embeddings else "short"
        cos_table, sin_table = self.rope_tables[table_key]
        end_pos = start_pos + seq_len
        if end_pos > cfg.max_position_embeddings:
            raise ValueError(f"RoPE request [{start_pos}, {end_pos}) exceeds {cfg.max_position_embeddings}")
        return cos_table[:, :, start_pos:end_pos, :], sin_table[:, :, start_pos:end_pos, :]

    def _decode_rope_tables(
        self, current_pos: ttnn.Tensor, batch_size: int, rope_sequence_length: int | None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cfg = self.config
        if rope_sequence_length is None:
            raise ValueError("decode_forward requires rope_sequence_length for trace-stable short/long RoPE selection")
        table_key = "long" if rope_sequence_length > cfg.original_max_position_embeddings else "short"
        cos_table, sin_table = self.rope_tables[table_key]
        if current_pos.dtype != ttnn.uint32:
            current_pos = ttnn.typecast(current_pos, dtype=ttnn.uint32)
        rot_idxs = ttnn.reshape(current_pos, (1, batch_size))
        cos = ttnn.embedding(rot_idxs, cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, sin_table, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)
        return cos, sin


def _host_to_device(
    tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.Tensor(tensor.detach().contiguous(), dtype).to(layout).to(mesh_device, memory_config)


def _weight_to_device(
    weight: torch.Tensor, mesh_device: ttnn.MeshDevice, *, dtype: ttnn.DataType = ttnn.bfloat16
) -> ttnn.Tensor:
    return _host_to_device(weight.to(torch.bfloat16), mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _dram_sharded_weight_to_device(
    weight: torch.Tensor, mesh_device: ttnn.MeshDevice, *, dtype: ttnn.DataType
) -> ttnn.Tensor:
    memory_config = _dram_sharded_weight_mem_config(mesh_device, int(weight.shape[-2]), int(weight.shape[-1]))
    return _host_to_device(
        weight.to(torch.bfloat16),
        mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _norm_weight_to_device(weight: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return _host_to_device(
        weight.reshape(1, 1, 1, -1).to(torch.bfloat16),
        mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _build_rope_tables(hf_config, config: Phi35MiniOptimizedDecoderConfig, mesh_device: ttnn.MeshDevice):
    rope_scaling = hf_config.rope_scaling or {}
    short_factor = torch.tensor(rope_scaling.get("short_factor", [1.0] * (config.head_dim // 2)), dtype=torch.float32)
    long_factor = torch.tensor(rope_scaling.get("long_factor", [1.0] * (config.head_dim // 2)), dtype=torch.float32)
    if short_factor.numel() != config.head_dim // 2 or long_factor.numel() != config.head_dim // 2:
        raise ValueError("Phi LongRoPE factor length must equal head_dim / 2")

    def make_tables(factors: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        inv_shape = torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim
        inv_freq = 1.0 / (factors * (config.rope_theta**inv_shape))
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        scale = hf_config.max_position_embeddings / hf_config.original_max_position_embeddings
        scaling_factor = 1.0 if scale <= 1.0 else math.sqrt(1 + math.log(scale) / math.log(config.original_max_position_embeddings))
        cos = (emb.cos() * scaling_factor).reshape(1, 1, config.max_position_embeddings, config.head_dim)
        sin = (emb.sin() * scaling_factor).reshape(1, 1, config.max_position_embeddings, config.head_dim)
        return (
            _host_to_device(cos.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            _host_to_device(sin.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        )

    return {"short": make_tables(short_factor), "long": make_tables(long_factor)}


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    rotated = _rotate_half(x)
    # Advisor-challenger losing experiments remain reproducible but default-off.
    rope_memory_config = (
        ttnn.L1_MEMORY_CONFIG
        if os.getenv("PHI35_ADVISOR_ROPE_L1_TAIL", "0") == "1"
        else ttnn.DRAM_MEMORY_CONFIG
    )
    x_cos = ttnn.mul(x, cos, dtype=ttnn.bfloat16, memory_config=rope_memory_config)
    rot_sin = ttnn.mul(rotated, sin, dtype=ttnn.bfloat16, memory_config=rope_memory_config)
    ttnn.deallocate(rotated)
    out = ttnn.add(x_cos, rot_sin, memory_config=rope_memory_config, dtype=ttnn.bfloat16)
    ttnn.deallocate(x_cos)
    ttnn.deallocate(rot_sin)
    return out


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    shape = _shape_tuple(x)
    half = shape[-1] // 2
    x1 = ttnn.slice(x, (0, 0, 0, 0), (*shape[:-1], half))
    x2 = ttnn.slice(x, (0, 0, 0, half), (*shape[:-1], shape[-1]))
    rotate_memory_config = (
        ttnn.L1_MEMORY_CONFIG
        if os.getenv("PHI35_ADVISOR_ROPE_FULL_L1", "0") == "1"
        else ttnn.DRAM_MEMORY_CONFIG
    )
    neg_x2 = ttnn.neg(x2, memory_config=rotate_memory_config)
    ttnn.deallocate(x2)
    out = ttnn.concat([neg_x2, x1], dim=3, memory_config=rotate_memory_config)
    ttnn.deallocate(neg_x2)
    ttnn.deallocate(x1)
    return out


def _shape_tuple(tensor: ttnn.Tensor) -> tuple[int, ...]:
    return tuple(int(tensor.shape[i]) for i in range(len(tensor.shape)))


def _typecast_if_needed(tensor: ttnn.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
    if tensor.dtype == dtype:
        return tensor
    return ttnn.typecast(tensor, dtype=dtype)


def _is_decode_tensor(tensor: ttnn.Tensor) -> bool:
    return int(tensor.shape[-2]) == 1


def _linear_output_mem_config(hidden_states: ttnn.Tensor, decode_mem_config: ttnn.MemoryConfig) -> ttnn.MemoryConfig:
    return decode_mem_config if _is_decode_tensor(hidden_states) else ttnn.DRAM_MEMORY_CONFIG


def _mlp_gate_up_program_config(self: OptimizedDecoder, hidden_states: ttnn.Tensor):
    if _is_decode_tensor(hidden_states):
        return self.decode_gate_up_program_config
    return _prefill_matmul_config(int(hidden_states.shape[-2]), self.config.hidden_size, 2 * self.config.intermediate_size)


def _mlp_down_program_config(self: OptimizedDecoder, hidden_states: ttnn.Tensor):
    if _is_decode_tensor(hidden_states):
        return self.decode_down_program_config
    return _prefill_matmul_config(int(hidden_states.shape[-2]), self.config.intermediate_size, self.config.hidden_size)


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _find_grid(n_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    max_cores = max_rows * max_cols
    target = max_cores // 2
    possible_cores = [k for k in range(1, max_cores + 1) if n_tiles % k == 0]
    possible_cores.sort(key=lambda x: abs(x - target))
    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"Cannot find grid for {n_tiles} tiles within {max_rows}x{max_cols}")


def _dram_shard_core_grid(k: int) -> ttnn.CoreGrid:
    rows, cols = _find_grid(k // TILE_SIZE)
    return ttnn.CoreGrid(x=cols, y=rows)


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int = 1) -> int:
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_n % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def _dram_matmul_config(
    *, m: int, k: int, n: int, num_cores: int
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(k // (TILE_SIZE * num_cores)),
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=math.ceil(n / (TILE_SIZE * num_cores)),
        fused_activation=None,
    )


@lru_cache(maxsize=None)
def _prefill_matmul_config(
    m: int, k: int, n: int, grid_size: tuple[int, int] = (8, 8)
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None:
    if m <= TILE_SIZE:
        return None
    per_core_m = max(1, math.ceil(m / (TILE_SIZE * grid_size[1])))
    per_core_n = math.ceil(n / (TILE_SIZE * grid_size[0]))
    max_in0_block_w = 4 if per_core_m >= 4 and per_core_n >= 64 else 8
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=_find_largest_divisor(k // (TILE_SIZE * grid_size[1]), max_divisor=max_in0_block_w),
        out_subblock_h=1,
        out_subblock_w=_get_out_subblock_w(per_core_n),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=m <= TILE_SIZE,
    )


def _dram_sharded_weight_mem_config(mesh_device: ttnn.MeshDevice, k: int, n: int) -> ttnn.MemoryConfig:
    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
    )
    padded_n = math.ceil(n / (TILE_SIZE * dram_grid_size.x)) * (TILE_SIZE * dram_grid_size.x)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_grid_size.x), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _sdpa_grid(mesh_device: ttnn.MeshDevice) -> ttnn.CoreCoord:
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(8, grid.x), min(4, grid.y))


def _width_sharded_decode_mem_config(mesh_device: ttnn.MeshDevice, width: int) -> ttnn.MemoryConfig:
    core_grid = _dram_shard_core_grid(width)
    return ttnn.create_sharded_memory_config(
        (TILE_SIZE, width // core_grid.num_cores),
        core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _height_sharded_decode_mem_config(
    mesh_device: ttnn.MeshDevice, num_heads: int, head_dim: int, *, max_batch_size: int
) -> ttnn.MemoryConfig:
    grid = mesh_device.compute_with_storage_grid_size()
    grid_x = min(max_batch_size, grid.x)
    if max_batch_size >= grid_x and max_batch_size % grid_x != 0:
        grid_x = max(
            x for x in range(grid_x, 0, -1) if max_batch_size % x == 0 and max_batch_size // x <= grid.y
        )
    grid_y = math.ceil(max_batch_size / grid_x)
    shard_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
    padded_heads = math.ceil(num_heads / 32) * 32
    return ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
