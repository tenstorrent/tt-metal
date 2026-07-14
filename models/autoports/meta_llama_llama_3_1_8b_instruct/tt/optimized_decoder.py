# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Optimized decoder candidate for meta-llama/Llama-3.1-8B-Instruct.

This stage preserves the forge-emitted single-token decode contract from the
functional decoder.  Prefill remains a stub because the source emit did not
contain a TTNN prefill graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from .functional_decoder import (
    EMITTED_BATCH_SIZE,
    EMITTED_CACHE_LEN,
    PREFILL_NOT_EMITTED_MESSAGE,
    Llama31DecoderConfig,
    _build_qvk_weight,
    _decode_height_sharded_memcfg,
    _device_tensor,
    _state_tensor,
)

TILE_SIZE = 32


@dataclass(frozen=True)
class OptimizedDecoderPolicy:
    """Runtime policy selected by the optimized decoder stage."""

    attention_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    output_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    norm_weight_dtype: ttnn.DataType = ttnn.bfloat16
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    qvk_in0_block_w: int | None = None
    o_in0_block_w: int | None = None
    gate_in0_block_w: int | None = None
    up_in0_block_w: int | None = None
    down_in0_block_w: int | None = None
    qvk_l1_core_count: int | None = None
    o_l1_core_count: int | None = None
    gate_l1_core_count: int | None = None
    up_l1_core_count: int | None = None
    down_l1_core_count: int | None = None
    use_dram_sharded_weights: bool = True
    use_decode_create_heads: bool = False
    use_explicit_sdpa_program_config: bool = True
    use_packed_gate_up_projection: bool = True


def _compute_kernel_config(
    math_fidelity: ttnn.MathFidelity,
    *,
    fp32_dest_acc_en: bool = False,
) -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )


def _compute_kernel_config_hifi4() -> ttnn.WormholeComputeKernelConfig:
    return _compute_kernel_config(ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)


def _find_largest_divisor(value: int, max_divisor: int = 16) -> int:
    for candidate in range(min(value, max_divisor), 0, -1):
        if value % candidate == 0:
            return candidate
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
    raise ValueError(f"cannot find core grid for {n_tiles} tiles")


def _find_grid_k_n(k_tiles: int, n_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    max_cores = max_rows * max_cols
    possible_cores = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
    possible_cores.sort(reverse=True)
    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise ValueError(f"cannot find core grid for K={k_tiles}, N={n_tiles} tiles")


def _core_grid_for_core_count(num_cores: int, max_rows: int = 8, max_cols: int = 8) -> ttnn.CoreGrid:
    if num_cores <= 0 or num_cores > max_rows * max_cols:
        raise ValueError(f"invalid core count {num_cores}")
    for rows in range(max_rows, 0, -1):
        if num_cores % rows == 0:
            cols = num_cores // rows
            if cols <= max_cols:
                return ttnn.CoreGrid(x=cols, y=rows)
    raise ValueError(f"cannot form core grid for {num_cores} cores")


def _dram_shard_core_grid(k: int, n: int | None = None) -> ttnn.CoreGrid:
    if n is None:
        rows, cols = _find_grid(k // TILE_SIZE)
    else:
        rows, cols = _find_grid_k_n(k // TILE_SIZE, n // TILE_SIZE)
    return ttnn.CoreGrid(x=cols, y=rows)


def _decode_width_sharded_memcfg(k: int, n: int | None = None, *, core_count: int | None = None) -> ttnn.MemoryConfig:
    grid = _core_grid_for_core_count(core_count) if core_count is not None else _dram_shard_core_grid(k, n)
    return ttnn.create_sharded_memory_config(
        (EMITTED_BATCH_SIZE, k // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_output_width_sharded_memcfg(k: int, n: int, *, core_count: int | None = None) -> ttnn.MemoryConfig:
    grid = _core_grid_for_core_count(core_count) if core_count is not None else _dram_shard_core_grid(k, n)
    return ttnn.create_sharded_memory_config(
        (EMITTED_BATCH_SIZE, n // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_sharded_weight_memcfg(mesh_device, k: int, n: int) -> ttnn.MemoryConfig:
    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
    )
    padded_n = math.ceil(n / (TILE_SIZE * dram_grid_size.x)) * (TILE_SIZE * dram_grid_size.x)
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (k, padded_n // dram_grid_size.x),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _dram_matmul_program_config(
    k: int,
    n: int,
    *,
    fused_activation=None,
    in0_block_w_override: int | None = None,
    input_l1_core_count: int | None = None,
):
    num_cores = _dram_shard_core_grid(k, n).num_cores
    input_num_cores = input_l1_core_count or num_cores
    in0_block_w = in0_block_w_override or _find_largest_divisor(k // (TILE_SIZE * input_num_cores))
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(n / (TILE_SIZE * num_cores)),
        fused_activation=fused_activation,
    )


def _linear_weight(
    tensor: torch.Tensor,
    mesh_device,
    *,
    dtype: ttnn.DataType,
    use_dram_sharded: bool,
) -> ttnn.Tensor:
    memory_config = (
        _dram_sharded_weight_memcfg(mesh_device, tensor.shape[-2], tensor.shape[-1])
        if use_dram_sharded
        else ttnn.DRAM_MEMORY_CONFIG
    )
    return _device_tensor(tensor, mesh_device, dtype=dtype, memory_config=memory_config)


def _build_gate_up_weight(state_dict: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").transpose(-2, -1)
    up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1)
    return torch.cat([gate, up], dim=-1).unsqueeze(0).unsqueeze(0)


class OptimizedDecoder(LightweightModule):
    """Single-layer optimized decode path; not a functional fallback wrapper."""

    def __init__(
        self,
        *,
        cfg: Llama31DecoderConfig,
        layer_idx: int,
        mesh_device,
        batch: int,
        cache_len: int,
        policy: OptimizedDecoderPolicy,
        qvk_proj_weight: ttnn.Tensor,
        o_proj_weight: ttnn.Tensor,
        gate_proj_weight: ttnn.Tensor,
        up_proj_weight: ttnn.Tensor,
        gate_up_proj_weight: ttnn.Tensor | None,
        down_proj_weight: ttnn.Tensor,
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.cache_len = cache_len
        self.policy = policy
        self.qvk_proj_weight = qvk_proj_weight
        self.o_proj_weight = o_proj_weight
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.gate_up_proj_weight = gate_up_proj_weight
        self.down_proj_weight = down_proj_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.norm_compute_kernel_config = _compute_kernel_config_hifi4()
        self.attention_compute_kernel_config = _compute_kernel_config(policy.attention_math_fidelity)
        self.mlp_compute_kernel_config = _compute_kernel_config(policy.mlp_math_fidelity)
        self.sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )
            if policy.use_explicit_sdpa_program_config
            else None
        )
        qvk_out = self.cfg.hidden_size + 2 * self.cfg.num_key_value_heads * self.cfg.head_dim
        self.qvk_input_memcfg = _decode_width_sharded_memcfg(
            self.cfg.hidden_size, qvk_out, core_count=policy.qvk_l1_core_count
        )
        self.o_input_memcfg = _decode_width_sharded_memcfg(
            self.cfg.hidden_size, self.cfg.hidden_size, core_count=policy.o_l1_core_count
        )
        self.mlp_gate_input_memcfg = _decode_width_sharded_memcfg(
            self.cfg.hidden_size, self.cfg.intermediate_size, core_count=policy.gate_l1_core_count
        )
        self.mlp_up_input_memcfg = _decode_width_sharded_memcfg(
            self.cfg.hidden_size, self.cfg.intermediate_size, core_count=policy.up_l1_core_count
        )
        self.mlp_down_input_memcfg = _decode_width_sharded_memcfg(
            self.cfg.intermediate_size, self.cfg.hidden_size, core_count=policy.down_l1_core_count
        )
        self.qvk_output_memcfg = _decode_output_width_sharded_memcfg(
            self.cfg.hidden_size, qvk_out, core_count=policy.qvk_l1_core_count
        )
        self.o_output_memcfg = _decode_output_width_sharded_memcfg(
            self.cfg.hidden_size, self.cfg.hidden_size, core_count=policy.o_l1_core_count
        )
        self.mlp_gate_output_memcfg = _decode_output_width_sharded_memcfg(
            self.cfg.hidden_size, self.cfg.intermediate_size, core_count=policy.gate_l1_core_count
        )
        self.mlp_up_output_memcfg = _decode_output_width_sharded_memcfg(
            self.cfg.hidden_size, self.cfg.intermediate_size, core_count=policy.up_l1_core_count
        )
        self.mlp_gate_up_packed_output_memcfg = _decode_output_width_sharded_memcfg(
            self.cfg.hidden_size, 2 * self.cfg.intermediate_size
        )
        self.mlp_down_output_memcfg = _decode_output_width_sharded_memcfg(
            self.cfg.intermediate_size, self.cfg.hidden_size, core_count=policy.down_l1_core_count
        )
        self.qvk_program_config = _dram_matmul_program_config(
            self.cfg.hidden_size,
            qvk_out,
            in0_block_w_override=policy.qvk_in0_block_w,
            input_l1_core_count=policy.qvk_l1_core_count,
        )
        self.o_program_config = _dram_matmul_program_config(
            self.cfg.hidden_size,
            self.cfg.hidden_size,
            in0_block_w_override=policy.o_in0_block_w,
            input_l1_core_count=policy.o_l1_core_count,
        )
        self.gate_program_config = _dram_matmul_program_config(
            self.cfg.hidden_size,
            self.cfg.intermediate_size,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            in0_block_w_override=policy.gate_in0_block_w,
            input_l1_core_count=policy.gate_l1_core_count,
        )
        self.up_program_config = _dram_matmul_program_config(
            self.cfg.hidden_size,
            self.cfg.intermediate_size,
            in0_block_w_override=policy.up_in0_block_w,
            input_l1_core_count=policy.up_l1_core_count,
        )
        self.gate_up_packed_program_config = _dram_matmul_program_config(
            self.cfg.hidden_size,
            2 * self.cfg.intermediate_size,
            in0_block_w_override=policy.gate_in0_block_w,
            input_l1_core_count=policy.gate_l1_core_count,
        )
        self.down_program_config = _dram_matmul_program_config(
            self.cfg.intermediate_size,
            self.cfg.hidden_size,
            in0_block_w_override=policy.down_in0_block_w,
            input_l1_core_count=policy.down_l1_core_count,
        )

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
        policy: OptimizedDecoderPolicy | None = None,
        **kwargs,
    ) -> "OptimizedDecoder":
        if kwargs:
            raise TypeError(f"unsupported OptimizedDecoder kwargs: {sorted(kwargs)}")
        policy = policy or OptimizedDecoderPolicy()
        cfg = Llama31DecoderConfig.from_hf_config(hf_config)
        if batch != EMITTED_BATCH_SIZE:
            raise ValueError(f"forge decode emit preserves batch={EMITTED_BATCH_SIZE}; got batch={batch}")
        if cache_len <= 0 or cache_len > cfg.max_position_embeddings:
            raise ValueError(f"cache_len must be in [1, {cfg.max_position_embeddings}], got {cache_len}")

        use_dram_sharded = policy.use_dram_sharded_weights
        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            cache_len=cache_len,
            policy=policy,
            qvk_proj_weight=_linear_weight(
                _build_qvk_weight(state_dict, layer_idx),
                mesh_device,
                dtype=policy.attention_weight_dtype,
                use_dram_sharded=use_dram_sharded,
            ),
            o_proj_weight=_linear_weight(
                _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
                .transpose(-2, -1)
                .unsqueeze(0)
                .unsqueeze(0),
                mesh_device,
                dtype=policy.output_weight_dtype,
                use_dram_sharded=use_dram_sharded,
            ),
            gate_proj_weight=_linear_weight(
                _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
                .transpose(-2, -1)
                .unsqueeze(0)
                .unsqueeze(0),
                mesh_device,
                dtype=policy.mlp_weight_dtype,
                use_dram_sharded=use_dram_sharded,
            ),
            up_proj_weight=_linear_weight(
                _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0),
                mesh_device,
                dtype=policy.mlp_weight_dtype,
                use_dram_sharded=use_dram_sharded,
            ),
            gate_up_proj_weight=(
                _linear_weight(
                    _build_gate_up_weight(state_dict, layer_idx),
                    mesh_device,
                    dtype=policy.mlp_weight_dtype,
                    use_dram_sharded=use_dram_sharded,
                )
                if policy.use_packed_gate_up_projection
                else None
            ),
            down_proj_weight=_linear_weight(
                _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
                .transpose(-2, -1)
                .unsqueeze(0)
                .unsqueeze(0),
                mesh_device,
                dtype=policy.mlp_weight_dtype,
                use_dram_sharded=use_dram_sharded,
            ),
            input_layernorm_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "input_layernorm.weight").reshape(1, 1, 1, -1),
                mesh_device,
                dtype=policy.norm_weight_dtype,
            ),
            post_attention_layernorm_weight=_device_tensor(
                _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").reshape(1, 1, 1, -1),
                mesh_device,
                dtype=policy.norm_weight_dtype,
            ),
        )

    def prefill_forward(self, *args, **kwargs) -> ttnn.Tensor:
        raise NotImplementedError(PREFILL_NOT_EMITTED_MESSAGE)

    def _matmul(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        *,
        input_memory_config,
        output_memory_config,
        program_config,
        compute_kernel_config,
        activation=None,
    ):
        if self.policy.use_dram_sharded_weights:
            x = ttnn.to_memory_config(x, input_memory_config)
        user_activation = None if self.policy.use_dram_sharded_weights else activation
        out = ttnn.matmul(
            x,
            weight,
            transpose_a=False,
            transpose_b=False,
            dtype=self.policy.activation_dtype,
            memory_config=output_memory_config if self.policy.use_dram_sharded_weights else ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config if self.policy.use_dram_sharded_weights else None,
            compute_kernel_config=compute_kernel_config,
            activation=user_activation,
        )
        if self.policy.use_dram_sharded_weights:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        return out

    def _create_qkv_heads(self, qvk: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        qvk = ttnn.reshape(qvk, [1, 1, self.batch, 6144], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.policy.use_decode_create_heads and self.layer_idx == 31:
            return ttnn.experimental.nlp_create_qkv_heads_decode(
                qvk,
                num_heads=self.cfg.num_attention_heads,
                num_kv_heads=self.cfg.num_key_value_heads,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        split_0, split_1, split_2 = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(qvk, [self.batch, 1, 6144], memory_config=ttnn.DRAM_MEMORY_CONFIG),
            None,
            num_heads=self.cfg.num_attention_heads,
            num_kv_heads=self.cfg.num_key_value_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.layer_idx == 31:
            return split_0, split_1, split_2
        return split_0, split_2, split_1

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
            memory_config=self.qvk_input_memcfg if self.policy.use_dram_sharded_weights else ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel_config,
        )
        qvk = self._matmul(
            normed,
            self.qvk_proj_weight,
            input_memory_config=self.qvk_input_memcfg,
            output_memory_config=self.qvk_output_memcfg,
            program_config=self.qvk_program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        query, key, value = self._create_qkv_heads(qvk)

        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.cfg.num_key_value_heads, 1, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.reshape(
            key, [1, self.batch, self.cfg.num_key_value_heads, self.cfg.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        key = ttnn.to_memory_config(key, _decode_height_sharded_memcfg(self.batch))
        key_idxs = (
            key_cache_update_idxs
            if key_cache_update_idxs is not None
            else ttnn.repeat(cache_position, ttnn.Shape([self.batch]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=key_idxs, share_cache=False, page_table=None
        )

        value = ttnn.slice(
            value,
            [0, 0, 0, 0],
            [self.batch, self.cfg.num_key_value_heads, 1, self.cfg.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            program_config=self.sdpa_program_config,
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
            attn_out, [1, 1, self.batch, self.cfg.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        attn_out = self._matmul(
            attn_out,
            self.o_proj_weight,
            input_memory_config=self.o_input_memcfg,
            output_memory_config=self.o_output_memcfg,
            program_config=self.o_program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        hidden_states = ttnn.add(
            attn_out, residual, dtype=self.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        mlp_in = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=self.mlp_gate_input_memcfg
            if self.policy.use_dram_sharded_weights
            else ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel_config,
        )
        if self.policy.use_packed_gate_up_projection:
            if self.gate_up_proj_weight is None:
                raise RuntimeError("packed gate/up policy requires a packed gate_up_proj_weight")
            gate_up = self._matmul(
                mlp_in,
                self.gate_up_proj_weight,
                input_memory_config=self.mlp_gate_input_memcfg,
                output_memory_config=self.mlp_gate_up_packed_output_memcfg,
                program_config=self.gate_up_packed_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            gate, up = ttnn.split(
                gate_up,
                self.cfg.intermediate_size,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            gate = self._matmul(
                mlp_in,
                self.gate_proj_weight,
                input_memory_config=self.mlp_gate_input_memcfg,
                output_memory_config=self.mlp_gate_output_memcfg,
                program_config=self.gate_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            )
            up = self._matmul(
                mlp_in,
                self.up_proj_weight,
                input_memory_config=self.mlp_up_input_memcfg,
                output_memory_config=self.mlp_up_output_memcfg,
                program_config=self.up_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
        mlp = ttnn.multiply(
            gate,
            up,
            dtype=self.policy.activation_dtype,
            memory_config=self.mlp_down_input_memcfg
            if self.policy.use_dram_sharded_weights
            else ttnn.DRAM_MEMORY_CONFIG,
        )
        mlp = self._matmul(
            mlp,
            self.down_proj_weight,
            input_memory_config=self.mlp_down_input_memcfg,
            output_memory_config=self.mlp_down_output_memcfg,
            program_config=self.down_program_config,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        return ttnn.add(mlp, hidden_states, dtype=self.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs) -> ttnn.Tensor:
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported OptimizedDecoder mode {mode!r}; expected 'decode' or 'prefill'")
