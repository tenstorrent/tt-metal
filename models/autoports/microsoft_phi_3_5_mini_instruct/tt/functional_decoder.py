# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional TTNN decoder layer for microsoft/Phi-3.5-mini-instruct.

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
from dataclasses import dataclass
from typing import Mapping

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 8192
NUM_HEADS = 32
NUM_KV_HEADS = 32
HEAD_DIM = 96
QKV_SIZE = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
DEFAULT_BLOCK_SIZE = 32


@dataclass(frozen=True)
class Phi35MiniFunctionalDecoderConfig:
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
    cache_dtype: ttnn.DataType = ttnn.bfloat16

    @classmethod
    def from_hf_config(cls, hf_config, *, block_size: int = DEFAULT_BLOCK_SIZE) -> "Phi35MiniFunctionalDecoderConfig":
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        if hf_config.hidden_size != HIDDEN_SIZE:
            raise ValueError(f"Phi-3.5 mini hidden_size must be {HIDDEN_SIZE}, got {hf_config.hidden_size}")
        if hf_config.intermediate_size != INTERMEDIATE_SIZE:
            raise ValueError(
                f"Phi-3.5 mini intermediate_size must be {INTERMEDIATE_SIZE}, got {hf_config.intermediate_size}"
            )
        if hf_config.num_attention_heads != NUM_HEADS or hf_config.num_key_value_heads != NUM_KV_HEADS:
            raise ValueError(
                "Phi-3.5 mini functional decoder expects dense 32Q/32KV attention, got "
                f"{hf_config.num_attention_heads}Q/{hf_config.num_key_value_heads}KV"
            )
        if head_dim != HEAD_DIM:
            raise ValueError(f"Phi-3.5 mini head_dim must be {HEAD_DIM}, got {head_dim}")
        if getattr(hf_config, "attention_bias", False):
            raise ValueError("Phi-3.5 mini attention_bias=True is not supported by this functional decoder")

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
        )


class FunctionalDecoder(LightweightModule):
    """Single dense Phi-3.5-mini decoder layer in TTNN."""

    def __init__(
        self,
        *,
        config: Phi35MiniFunctionalDecoderConfig,
        mesh_device: ttnn.MeshDevice,
        layer_idx: int,
        input_norm_weight: ttnn.Tensor,
        post_norm_weight: ttnn.Tensor,
        qkv_weight: ttnn.Tensor,
        o_weight: ttnn.Tensor,
        gate_up_weight: ttnn.Tensor,
        down_weight: ttnn.Tensor,
        rope_tables: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]],
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
        self.rope_tables = rope_tables
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.decode_kv_mem_config = _height_sharded_decode_mem_config(
            mesh_device, config.num_kv_heads, config.head_dim, max_batch_size=1
        )
        self.decode_q_mem_config = _height_sharded_decode_mem_config(
            mesh_device, config.num_heads, config.head_dim, max_batch_size=1
        )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=_sdpa_grid(mesh_device),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
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
        **_: object,
    ) -> "FunctionalDecoder":
        """Create a decoder from a HF layer state dict.

        Weight conversion is intentionally eager and explicit here so runtime
        forwards have no hidden host fallback.
        """

        config = Phi35MiniFunctionalDecoderConfig.from_hf_config(hf_config, block_size=block_size)
        if max_position_embeddings is not None:
            config = Phi35MiniFunctionalDecoderConfig(
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

        qkv_weight = _weight_to_device(state_dict["self_attn.qkv_proj.weight"].T, mesh_device)
        o_weight = _weight_to_device(state_dict["self_attn.o_proj.weight"].T, mesh_device)
        gate_up_weight = _weight_to_device(state_dict["mlp.gate_up_proj.weight"].T, mesh_device)
        down_weight = _weight_to_device(state_dict["mlp.down_proj.weight"].T, mesh_device)
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
            rope_tables=rope_tables,
        )

    @staticmethod
    def allocate_paged_kv_cache(
        *,
        hf_config,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        dtype: ttnn.DataType = ttnn.bfloat16,
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
            compute_kernel_config=self.compute_kernel_config,
        )

        qkv = ttnn.linear(
            attn_in,
            self.qkv_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_in)
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
            compute_kernel_config=self.compute_kernel_config,
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
        attn_proj = ttnn.linear(
            attn_cat,
            self.o_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_cat)
        hidden_states = ttnn.add(residual, attn_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=cfg.dtype)
        ttnn.deallocate(attn_proj)

        mlp_in = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.post_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
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
        if batch_size != 1:
            raise ValueError("this functional decoder currently supports batch_size=1 for decode")

        residual = hidden_states
        attn_in = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.input_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        qkv = ttnn.linear(
            attn_in,
            self.qkv_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_in)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **sdpa_kwargs,
        )
        ttnn.deallocate(q)
        attn_out = ttnn.to_memory_config(attn_out, self.decode_q_mem_config)
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=cfg.num_heads)
        ttnn.deallocate(attn_out)
        attn_proj = ttnn.linear(
            attn_cat,
            self.o_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_cat)
        if int(attn_proj.shape[-2]) != batch_size:
            attn_proj_full = attn_proj
            attn_proj = ttnn.slice(attn_proj_full, (0, 0, 0, 0), (1, 1, batch_size, cfg.hidden_size))
            ttnn.deallocate(attn_proj_full)
        hidden_states = ttnn.add(residual, attn_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=cfg.dtype)
        ttnn.deallocate(attn_proj)

        mlp_in = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.post_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        mlp_out = self._mlp_forward(mlp_in)
        ttnn.deallocate(mlp_in)
        out = ttnn.add(hidden_states, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=cfg.dtype)
        ttnn.deallocate(mlp_out)
        return out

    def _mlp_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        gate_up = ttnn.linear(
            hidden_states,
            self.gate_up_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        down = ttnn.linear(
            down_in,
            self.down_weight,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(down_in)
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


def _weight_to_device(weight: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return _host_to_device(weight.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _norm_weight_to_device(weight: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return _host_to_device(
        weight.reshape(1, 1, 1, -1).to(torch.bfloat16),
        mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _build_rope_tables(hf_config, config: Phi35MiniFunctionalDecoderConfig, mesh_device: ttnn.MeshDevice):
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
    x_cos = ttnn.mul(x, cos, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rot_sin = ttnn.mul(rotated, sin, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(rotated)
    out = ttnn.add(x_cos, rot_sin, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(x_cos)
    ttnn.deallocate(rot_sin)
    return out


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    shape = _shape_tuple(x)
    half = shape[-1] // 2
    x1 = ttnn.slice(x, (0, 0, 0, 0), (*shape[:-1], half))
    x2 = ttnn.slice(x, (0, 0, 0, half), (*shape[:-1], shape[-1]))
    neg_x2 = ttnn.neg(x2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x2)
    out = ttnn.concat([neg_x2, x1], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(neg_x2)
    ttnn.deallocate(x1)
    return out


def _shape_tuple(tensor: ttnn.Tensor) -> tuple[int, ...]:
    return tuple(int(tensor.shape[i]) for i in range(len(tensor.shape)))


def _typecast_if_needed(tensor: ttnn.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
    if tensor.dtype == dtype:
        return tensor
    return ttnn.typecast(tensor, dtype=dtype)


def _sdpa_grid(mesh_device: ttnn.MeshDevice) -> ttnn.CoreCoord:
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(8, grid.x), min(4, grid.y))


def _height_sharded_decode_mem_config(
    mesh_device: ttnn.MeshDevice, num_heads: int, head_dim: int, *, max_batch_size: int
) -> ttnn.MemoryConfig:
    grid = mesh_device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(max_batch_size, grid, True)
    padded_heads = math.ceil(num_heads / 32) * 32
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [padded_heads, head_dim],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
