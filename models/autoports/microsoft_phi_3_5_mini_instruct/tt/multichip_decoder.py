# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""T3K multichip TTNN decoder layer for microsoft/Phi-3.5-mini-instruct.

This module uses :class:`OptimizedDecoder` as the single-chip baseline and
keeps the same public prefill/decode cache contract. The target mesh is a
specialized T3K 1x8 ring with tensor-parallel factor 8.

Runtime forwards are TTNN-only. Weight conversion and tensor reordering are
performed at ``from_state_dict`` load time.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

import torch
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.precision import (
    Phi35MiniPrecisionPolicy,
    dtype_name,
    load_precision_policy,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
    DEFAULT_BLOCK_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MODEL_ID,
    NUM_HEADS,
    NUM_KV_HEADS,
    PREFILL_MATMUL_CHUNK_SIZE,
    PREFILL_QKV_CHUNK_SIZE,
    QKV_SIZE,
    TILE_SIZE,
    OptimizedDecoder,
    Phi35MiniOptimizedDecoderConfig,
    _apply_rope,
    _dram_matmul_config,
    _dram_shard_core_grid,
    _dram_sharded_weight_mem_config,
    _height_sharded_decode_mem_config,
    _prefill_matmul_config,
    _sdpa_grid,
    _shape_tuple,
    _typecast_if_needed,
    _width_sharded_decode_mem_config,
)
from models.common.lightweightmodule import LightweightModule


TARGET_MESH_SHAPE = (1, 8)
TP_FACTOR = 8
TP_AXIS = 1
TOPOLOGY = ttnn.Topology.Ring

LOCAL_NUM_HEADS = NUM_HEADS // TP_FACTOR
LOCAL_NUM_KV_HEADS = NUM_KV_HEADS // TP_FACTOR
LOCAL_HIDDEN_SIZE = LOCAL_NUM_HEADS * HEAD_DIM
LOCAL_QKV_SIZE = (LOCAL_NUM_HEADS + 2 * LOCAL_NUM_KV_HEADS) * HEAD_DIM
LOCAL_INTERMEDIATE_SIZE = INTERMEDIATE_SIZE // TP_FACTOR
LOCAL_GATE_UP_SIZE = 2 * LOCAL_INTERMEDIATE_SIZE


@dataclass(frozen=True)
class Phi35MiniMultichipDecoderConfig:
    """Resolved multichip config for the fixed T3K TP=8 target."""

    single_chip: Phi35MiniOptimizedDecoderConfig
    tp_factor: int = TP_FACTOR
    topology: ttnn.Topology = TOPOLOGY
    tp_axis: int = TP_AXIS
    num_links: int = 1

    @property
    def hidden_size(self) -> int:
        return self.single_chip.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.single_chip.intermediate_size

    @property
    def num_heads(self) -> int:
        return self.single_chip.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.single_chip.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.single_chip.head_dim

    @property
    def max_position_embeddings(self) -> int:
        return self.single_chip.max_position_embeddings

    @property
    def original_max_position_embeddings(self) -> int:
        return self.single_chip.original_max_position_embeddings

    @property
    def rope_theta(self) -> float:
        return self.single_chip.rope_theta

    @property
    def rms_norm_eps(self) -> float:
        return self.single_chip.rms_norm_eps

    @property
    def block_size(self) -> int:
        return self.single_chip.block_size

    @property
    def dtype(self) -> ttnn.DataType:
        return self.single_chip.dtype

    @property
    def attention_weight_dtype(self) -> ttnn.DataType:
        return self.single_chip.attention_weight_dtype

    @property
    def mlp_weight_dtype(self) -> ttnn.DataType:
        return self.single_chip.mlp_weight_dtype

    @property
    def mlp_prefill_weight_dtype(self) -> ttnn.DataType:
        return self.single_chip.mlp_prefill_weight_dtype

    @property
    def cache_dtype(self) -> ttnn.DataType:
        return self.single_chip.cache_dtype

    @classmethod
    def from_hf_config(
        cls,
        hf_config,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_position_embeddings: int | None = None,
    ) -> "Phi35MiniMultichipDecoderConfig":
        base = Phi35MiniOptimizedDecoderConfig.from_hf_config(hf_config, block_size=block_size)
        if max_position_embeddings is not None:
            base = Phi35MiniOptimizedDecoderConfig(
                hidden_size=base.hidden_size,
                intermediate_size=base.intermediate_size,
                num_heads=base.num_heads,
                num_kv_heads=base.num_kv_heads,
                head_dim=base.head_dim,
                max_position_embeddings=max_position_embeddings,
                original_max_position_embeddings=base.original_max_position_embeddings,
                rope_theta=base.rope_theta,
                rms_norm_eps=base.rms_norm_eps,
                block_size=base.block_size,
                dtype=base.dtype,
                attention_weight_dtype=base.attention_weight_dtype,
                mlp_weight_dtype=base.mlp_weight_dtype,
                mlp_prefill_weight_dtype=base.mlp_prefill_weight_dtype,
                cache_dtype=base.cache_dtype,
            )
        return cls(single_chip=base)


class MultichipDecoder(LightweightModule):
    """Dense Phi-3.5-mini decoder layer specialized for T3K 1x8 tensor parallelism."""

    single_chip_baseline_cls = OptimizedDecoder

    def __init__(
        self,
        *,
        config: Phi35MiniMultichipDecoderConfig,
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
        precision_policy: Phi35MiniPrecisionPolicy | None = None,
    ) -> None:
        super().__init__()
        _validate_target_mesh(mesh_device)
        self.config = config
        self.precision_policy = precision_policy or load_precision_policy()
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
        self.decode_matmul_math_fidelity = _math_fidelity_from_name(
            os.getenv(
                "PHI35_MULTICHIP_DECODE_MATMUL_FIDELITY",
                os.getenv(
                    "PHI35_MULTICHIP_MATMUL_FIDELITY",
                    self.precision_policy.compute_fidelity("decode_matmul", "lofi"),
                ),
            )
        )
        self.prefill_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_math_fidelity_from_name(self.precision_policy.compute_fidelity("prefill_matmul", "hifi2")),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.decode_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.decode_matmul_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_math_fidelity_from_name(self.precision_policy.compute_fidelity("norm", "hifi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi2 = self.decode_compute_kernel_config
        self.compute_kernel_config = self.decode_compute_kernel_config
        self.ccl_policy = os.getenv("PHI35_MULTICHIP_CCL", self.precision_policy.ccl_policy)
        if self.ccl_policy not in {"sync_all_reduce", "async_all_reduce"}:
            raise ValueError(
                f"unsupported PHI35_MULTICHIP_CCL={self.ccl_policy!r}; "
                "expected 'sync_all_reduce' or 'async_all_reduce'"
            )
        self.decode_ccl_dtype = _ccl_dtype_from_name(
            os.getenv("PHI35_MULTICHIP_CCL_DTYPE", dtype_name(self.precision_policy.decode_ccl_dtype))
        )
        self.prefill_ccl_dtype = _ccl_dtype_from_name(
            os.getenv("PHI35_MULTICHIP_PREFILL_CCL_DTYPE", dtype_name(self.precision_policy.prefill_ccl_dtype))
        )
        self._ccl_semaphore_sets = _create_async_all_reduce_semaphore_sets(mesh_device) if self.ccl_policy == "async_all_reduce" else []
        self._ccl_semaphore_idx = 0
        self.decode_local_matmul_min_in0_block_w = int(os.getenv("PHI35_MULTICHIP_LOCAL_MATMUL_MIN_IN0_BLOCK_W", "2"))
        self.decode_o_num_cores = _select_dram_matmul_num_cores(
            LOCAL_HIDDEN_SIZE,
            _dram_shard_core_grid(LOCAL_HIDDEN_SIZE).num_cores,
            self.decode_local_matmul_min_in0_block_w,
        )
        self.decode_down_num_cores = _select_dram_matmul_num_cores(
            LOCAL_INTERMEDIATE_SIZE,
            _dram_shard_core_grid(LOCAL_INTERMEDIATE_SIZE).num_cores,
            self.decode_local_matmul_min_in0_block_w,
        )
        self.decode_hidden_mem_config = _width_sharded_decode_mem_config(mesh_device, config.hidden_size)
        self.decode_local_hidden_mem_config = _width_sharded_decode_mem_config_for_num_cores(
            LOCAL_HIDDEN_SIZE, self.decode_o_num_cores
        )
        self.decode_o_output_mem_config = (
            _width_sharded_decode_mem_config_for_num_cores(config.hidden_size, self.decode_o_num_cores)
            if self.decode_local_matmul_min_in0_block_w > 1
            else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )
        self.decode_qkv_mem_config = _width_sharded_decode_mem_config(mesh_device, LOCAL_QKV_SIZE)
        self.decode_gate_up_mem_config = _width_sharded_decode_mem_config(mesh_device, LOCAL_GATE_UP_SIZE)
        self.decode_mlp_intermediate_mem_config = _width_sharded_decode_mem_config_for_num_cores(
            LOCAL_INTERMEDIATE_SIZE, self.decode_down_num_cores
        )
        self.decode_down_output_mem_config = (
            _width_sharded_decode_mem_config_for_num_cores(config.hidden_size, self.decode_down_num_cores)
            if self.decode_local_matmul_min_in0_block_w > 1
            else self.decode_hidden_mem_config
        )
        self.decode_kv_mem_config = _height_sharded_decode_mem_config(
            mesh_device, LOCAL_NUM_KV_HEADS, config.head_dim, max_batch_size=1
        )
        self.decode_q_mem_config = _height_sharded_decode_mem_config(
            mesh_device, LOCAL_NUM_HEADS, config.head_dim, max_batch_size=1
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
            n=LOCAL_QKV_SIZE,
            num_cores=_dram_shard_core_grid(config.hidden_size).num_cores,
        )
        self.decode_o_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=LOCAL_HIDDEN_SIZE,
            n=config.hidden_size,
            num_cores=self.decode_o_num_cores,
        )
        self.decode_gate_up_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=config.hidden_size,
            n=LOCAL_GATE_UP_SIZE,
            num_cores=_dram_shard_core_grid(config.hidden_size).num_cores,
        )
        self.decode_down_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=LOCAL_INTERMEDIATE_SIZE,
            n=config.hidden_size,
            num_cores=self.decode_down_num_cores,
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
        precision_policy: Phi35MiniPrecisionPolicy | None = None,
        **_: object,
    ) -> "MultichipDecoder":
        """Create a T3K tensor-parallel decoder from a HF layer state dict."""

        _validate_target_mesh(mesh_device)
        precision = precision_policy or load_precision_policy()
        config = Phi35MiniMultichipDecoderConfig.from_hf_config(
            hf_config, block_size=block_size, max_position_embeddings=max_position_embeddings
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

        qkv_weight_host = _reorder_qkv_for_tp(state_dict["self_attn.qkv_proj.weight"].T)
        gate_up_weight_host = _reorder_gate_up_for_tp(state_dict["mlp.gate_up_proj.weight"].T)
        o_weight_host = state_dict["self_attn.o_proj.weight"].T
        down_weight_host = state_dict["mlp.down_proj.weight"].T

        qkv_weight = _mesh_sharded_weight_to_device(
            qkv_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("attention.qkv", layer_idx=layer_idx),
            mesh_shard_dim=-1,
            local_k=config.hidden_size,
            local_n=LOCAL_QKV_SIZE,
        )
        o_weight = _mesh_sharded_weight_to_device(
            o_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("attention.o", layer_idx=layer_idx),
            mesh_shard_dim=-2,
            local_k=LOCAL_HIDDEN_SIZE,
            local_n=config.hidden_size,
        )
        gate_up_weight = _mesh_sharded_weight_to_device(
            gate_up_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("mlp.gate_up", layer_idx=layer_idx),
            mesh_shard_dim=-1,
            local_k=config.hidden_size,
            local_n=LOCAL_GATE_UP_SIZE,
        )
        down_weight = _mesh_sharded_weight_to_device(
            down_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("mlp.down", layer_idx=layer_idx),
            mesh_shard_dim=-2,
            local_k=LOCAL_INTERMEDIATE_SIZE,
            local_n=config.hidden_size,
        )

        qkv_weight_prefill = _mesh_sharded_weight_to_device(
            qkv_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("attention.qkv", layer_idx=layer_idx, prefill=True),
            mesh_shard_dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        o_weight_prefill = _mesh_sharded_weight_to_device(
            o_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("attention.o", layer_idx=layer_idx, prefill=True),
            mesh_shard_dim=-2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up_weight_prefill = _mesh_sharded_weight_to_device(
            gate_up_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("mlp.gate_up", layer_idx=layer_idx, prefill=True),
            mesh_shard_dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_weight_prefill = _mesh_sharded_weight_to_device(
            down_weight_host,
            mesh_device,
            dtype=precision.weight_dtype("mlp.down", layer_idx=layer_idx, prefill=True),
            mesh_shard_dim=-2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_norm_weight = _replicated_norm_weight_to_device(state_dict["input_layernorm.weight"], mesh_device)
        post_norm_weight = _replicated_norm_weight_to_device(state_dict["post_attention_layernorm.weight"], mesh_device)
        rope_tables = _build_replicated_rope_tables(hf_config, config, mesh_device)

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
            precision_policy=precision,
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
        """Allocate local-head paged K/V cache tensors for the TP=8 decoder."""

        _validate_target_mesh(mesh_device)
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        if hf_config.num_key_value_heads != NUM_KV_HEADS or head_dim != HEAD_DIM:
            raise ValueError("Phi-3.5 mini multichip cache expects 32 KV heads and head_dim 96")
        num_blocks_per_seq = math.ceil(max_seq_len / block_size)
        num_blocks = max_batch_size * num_blocks_per_seq
        cache_shape = (num_blocks, LOCAL_NUM_KV_HEADS, block_size, head_dim)
        zero_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
        k_cache = _host_to_mesh(
            zero_cache,
            mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        v_cache = _host_to_mesh(
            zero_cache,
            mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
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
        """Run paged prefill for one user and return a replicated residual tensor."""

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
            program_config=_prefill_matmul_config(qkv_m, cfg.hidden_size, LOCAL_QKV_SIZE),
            compute_kernel_config=self.prefill_compute_kernel_config,
        )
        ttnn.deallocate(attn_in)
        if seq_len > PREFILL_QKV_CHUNK_SIZE:
            qkv = ttnn.reshape(qkv, [1, 1, seq_len, -1])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=LOCAL_NUM_HEADS,
            num_kv_heads=LOCAL_NUM_KV_HEADS,
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
        attn_partial = ttnn.linear(
            attn_cat_for_proj,
            self.o_weight_prefill,
            dtype=cfg.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_prefill_matmul_config(o_m, LOCAL_HIDDEN_SIZE, cfg.hidden_size),
            compute_kernel_config=self.prefill_compute_kernel_config,
        )
        if seq_len > PREFILL_MATMUL_CHUNK_SIZE:
            attn_partial = ttnn.reshape(attn_partial, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_cat_for_proj)
        attn_proj = self._all_reduce(
            attn_partial,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ccl_dtype=self.prefill_ccl_dtype,
        )
        ttnn.deallocate(attn_partial)
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
        """Run traced-safe paged decode and return a replicated residual tensor."""

        cfg = self.config
        batch_size = int(hidden_states.shape[-2])
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError(f"hidden width must be {cfg.hidden_size}, got {hidden_states.shape[-1]}")
        if int(hidden_states.shape[-3]) != 1:
            raise ValueError(f"decode hidden_states must have seq_len=1, got shape {hidden_states.shape}")
        if batch_size != 1:
            raise ValueError("this multichip decoder currently supports batch_size=1 for decode")

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
            num_heads=LOCAL_NUM_HEADS,
            num_kv_heads=LOCAL_NUM_KV_HEADS,
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **sdpa_kwargs,
        )
        ttnn.deallocate(q)
        attn_out = ttnn.to_memory_config(attn_out, self.decode_q_mem_config)
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=LOCAL_NUM_HEADS)
        ttnn.deallocate(attn_out)
        attn_cat = ttnn.to_memory_config(attn_cat, self.decode_local_hidden_mem_config)
        attn_partial = ttnn.linear(
            attn_cat,
            self.o_weight,
            dtype=cfg.dtype,
            memory_config=self.decode_o_output_mem_config,
            program_config=self.decode_o_program_config,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_cat)
        attn_proj = self._all_reduce(attn_partial, memory_config=self.decode_hidden_mem_config)
        ttnn.deallocate(attn_partial)
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
        is_decode = _is_decode_tensor(hidden_states)
        reshape_prefill = (not is_decode) and prefill_seq_len > PREFILL_MATMUL_CHUNK_SIZE
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
            self.gate_up_weight if is_decode else self.gate_up_weight_prefill,
            dtype=cfg.dtype,
            memory_config=self.decode_gate_up_mem_config if is_decode else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_gate_up_program_config
            if is_decode
            else _prefill_matmul_config(int(hidden_states.shape[-2]), cfg.hidden_size, LOCAL_GATE_UP_SIZE),
            compute_kernel_config=self.decode_compute_kernel_config if is_decode else self.prefill_compute_kernel_config,
        )
        gate_up_shape = _shape_tuple(gate_up)
        gate = ttnn.slice(gate_up, (0, 0, 0, 0), (*gate_up_shape[:-1], LOCAL_INTERMEDIATE_SIZE))
        up = ttnn.slice(
            gate_up,
            (0, 0, 0, LOCAL_INTERMEDIATE_SIZE),
            (*gate_up_shape[:-1], LOCAL_GATE_UP_SIZE),
        )
        ttnn.deallocate(gate_up)
        down_in = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=cfg.dtype,
            memory_config=gate.memory_config(),
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        if is_decode:
            down_in = ttnn.to_memory_config(down_in, self.decode_mlp_intermediate_mem_config)
        elif int(hidden_states.shape[-2]) <= TILE_SIZE:
            down_in = ttnn.to_memory_config(down_in, ttnn.L1_MEMORY_CONFIG)
        down_partial = ttnn.linear(
            down_in,
            self.down_weight if is_decode else self.down_weight_prefill,
            dtype=cfg.dtype,
            memory_config=self.decode_down_output_mem_config if is_decode else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_down_program_config
            if is_decode
            else _prefill_matmul_config(int(hidden_states.shape[-2]), LOCAL_INTERMEDIATE_SIZE, cfg.hidden_size),
            compute_kernel_config=self.decode_compute_kernel_config if is_decode else self.prefill_compute_kernel_config,
        )
        ttnn.deallocate(down_in)
        down = self._all_reduce(
            down_partial,
            memory_config=self.decode_hidden_mem_config if is_decode else ttnn.DRAM_MEMORY_CONFIG,
            ccl_dtype=None if is_decode else self.prefill_ccl_dtype,
        )
        ttnn.deallocate(down_partial)
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

    def _all_reduce(
        self,
        tensor: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig,
        ccl_dtype: ttnn.DataType | None = None,
    ) -> ttnn.Tensor:
        ccl_dtype = self.decode_ccl_dtype if ccl_dtype is None else ccl_dtype
        ccl_input = tensor
        if ccl_dtype != tensor.dtype:
            ccl_input = ttnn.typecast(tensor, dtype=ccl_dtype)
        if self.ccl_policy == "async_all_reduce":
            semaphores = self._ccl_semaphore_sets[self._ccl_semaphore_idx]
            self._ccl_semaphore_idx = (self._ccl_semaphore_idx + 1) % len(self._ccl_semaphore_sets)
            output = ttnn.experimental.all_reduce_async(
                ccl_input,
                cluster_axis=self.config.tp_axis,
                mesh_device=self.mesh_device,
                barrier_semaphores=semaphores["barrier"],
                rs_global_semaphores=semaphores["rs"],
                ag_global_semaphores=semaphores["ag"],
                math_op=ttnn.ReduceType.Sum,
                num_links=self.config.num_links,
                topology=self.config.topology,
                memory_config=memory_config,
            )
        else:
            output = ttnn.all_reduce(
                ccl_input,
                cluster_axis=self.config.tp_axis,
                num_links=self.config.num_links,
                topology=self.config.topology,
                memory_config=memory_config,
            )
        if ccl_input is not tensor:
            ttnn.deallocate(ccl_input)
        if output.dtype != self.config.dtype:
            typed_output = ttnn.typecast(output, dtype=self.config.dtype)
            ttnn.deallocate(output)
            output = typed_output
        return output


def mesh_strategy_summary() -> dict[str, object]:
    """Return the fixed mesh strategy in a structured form for tests/docs."""

    return {
        "model": MODEL_ID,
        "single_chip_baseline": "OptimizedDecoder",
        "mesh_shape": TARGET_MESH_SHAPE,
        "tp_factor": TP_FACTOR,
        "tp_axis": TP_AXIS,
        "topology": "Ring",
        "residual": {"mesh": "replicated", "shape": [1, 1, "T", HIDDEN_SIZE]},
        "attention": {
            "q_heads_per_device": LOCAL_NUM_HEADS,
            "kv_heads_per_device": LOCAL_NUM_KV_HEADS,
            "qkv_weight_per_device": [HIDDEN_SIZE, LOCAL_QKV_SIZE],
            "o_weight_per_device": [LOCAL_HIDDEN_SIZE, HIDDEN_SIZE],
            "collective": "all_reduce after o_proj",
        },
        "mlp": {
            "gate_up_weight_per_device": [HIDDEN_SIZE, LOCAL_GATE_UP_SIZE],
            "down_weight_per_device": [LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE],
            "intermediate_per_device": LOCAL_INTERMEDIATE_SIZE,
            "collective": "all_reduce after down_proj",
        },
        "kv_cache_per_device": ["num_blocks", LOCAL_NUM_KV_HEADS, DEFAULT_BLOCK_SIZE, HEAD_DIM],
        "moe": "not_applicable_dense_phi",
    }


def _validate_target_mesh(mesh_device: ttnn.MeshDevice) -> None:
    num_devices = mesh_device.get_num_devices()
    if num_devices != TP_FACTOR:
        raise ValueError(f"Phi multichip decoder requires exactly {TP_FACTOR} devices, got {num_devices}")


def _math_fidelity_from_name(name: str) -> ttnn.MathFidelity:
    normalized = name.lower()
    mapping = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported PHI35_MULTICHIP_MATMUL_FIDELITY={name!r}")
    return mapping[normalized]


def _ccl_dtype_from_name(name: str) -> ttnn.DataType:
    normalized = name.lower()
    mapping = {
        "bfloat16": ttnn.bfloat16,
        "bf16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
        "bf8": ttnn.bfloat8_b,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported PHI35_MULTICHIP_CCL_DTYPE={name!r}")
    return mapping[normalized]


def _create_async_all_reduce_semaphore_sets(mesh_device: ttnn.MeshDevice) -> list[dict[str, list[object]]]:
    grid = mesh_device.compute_with_storage_grid_size()
    core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    semaphore_sets = []
    for _ in range(4):
        semaphore_sets.append(
            {
                "rs": [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)],
                "ag": [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)],
                "barrier": [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)],
            }
        )
    ttnn.synchronize_device(mesh_device)
    return semaphore_sets


def _select_dram_matmul_num_cores(k: int, default_num_cores: int, min_in0_block_w: int) -> int:
    if min_in0_block_w <= 1:
        return default_num_cores
    k_tiles = k // TILE_SIZE
    for num_cores in range(default_num_cores, 0, -1):
        if k_tiles % num_cores != 0:
            continue
        if k_tiles // num_cores >= min_in0_block_w:
            return num_cores
    return default_num_cores


def _core_grid_for_num_cores(num_cores: int) -> ttnn.CoreGrid:
    for rows in range(1, 9):
        if num_cores % rows == 0:
            cols = num_cores // rows
            if cols <= 8:
                return ttnn.CoreGrid(x=cols, y=rows)
    raise ValueError(f"cannot represent {num_cores} cores as a <=8x8 CoreGrid")


def _width_sharded_decode_mem_config_for_num_cores(width: int, num_cores: int) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        (TILE_SIZE, width // num_cores),
        _core_grid_for_num_cores(num_cores),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _host_to_mesh(
    tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )


def _mesh_sharded_weight_to_device(
    weight: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    dtype: ttnn.DataType,
    mesh_shard_dim: int,
    local_k: int | None = None,
    local_n: int | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    if memory_config is None:
        if local_k is None or local_n is None:
            raise ValueError("local_k/local_n are required for DRAM-sharded decode weights")
        memory_config = _dram_sharded_weight_mem_config(mesh_device, local_k, local_n)
    return _host_to_mesh(
        weight.to(torch.bfloat16),
        mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=mesh_shard_dim),
    )


def _replicated_norm_weight_to_device(weight: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return _host_to_mesh(
        weight.reshape(1, 1, 1, -1).to(torch.bfloat16),
        mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_replicated_rope_tables(
    hf_config, config: Phi35MiniMultichipDecoderConfig, mesh_device: ttnn.MeshDevice
) -> dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]:
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
            _host_to_mesh(
                cos.to(torch.bfloat16),
                mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            _host_to_mesh(
                sin.to(torch.bfloat16),
                mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        )

    return {"short": make_tables(short_factor), "long": make_tables(long_factor)}


def _reorder_qkv_for_tp(weight_t: torch.Tensor) -> torch.Tensor:
    """Convert [Q-all, K-all, V-all] columns into per-device [Q_i, K_i, V_i] chunks."""

    q, k, v = torch.split(weight_t, HIDDEN_SIZE, dim=-1)
    local_head_width = LOCAL_NUM_HEADS * HEAD_DIM
    local_kv_width = LOCAL_NUM_KV_HEADS * HEAD_DIM
    chunks = []
    for device_idx in range(TP_FACTOR):
        q_start = device_idx * local_head_width
        kv_start = device_idx * local_kv_width
        chunks.extend(
            [
                q[:, q_start : q_start + local_head_width],
                k[:, kv_start : kv_start + local_kv_width],
                v[:, kv_start : kv_start + local_kv_width],
            ]
        )
    return torch.cat(chunks, dim=-1).contiguous()


def _reorder_gate_up_for_tp(weight_t: torch.Tensor) -> torch.Tensor:
    """Convert [gate-all, up-all] columns into per-device [gate_i, up_i] chunks."""

    gate, up = torch.split(weight_t, INTERMEDIATE_SIZE, dim=-1)
    chunks = []
    for device_idx in range(TP_FACTOR):
        start = device_idx * LOCAL_INTERMEDIATE_SIZE
        chunks.extend(
            [
                gate[:, start : start + LOCAL_INTERMEDIATE_SIZE],
                up[:, start : start + LOCAL_INTERMEDIATE_SIZE],
            ]
        )
    return torch.cat(chunks, dim=-1).contiguous()


def _is_decode_tensor(tensor: ttnn.Tensor) -> bool:
    return int(tensor.shape[-2]) == 1
