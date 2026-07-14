# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    EMITTED_BATCH_SIZE,
    EMITTED_DECODE_CACHE_LEN,
    FunctionalDecoder,
    Llama31DecoderConfig,
    _decode_height_sharded_memory_config,
    _fused_emit_weight,
    _get_layer_tensor,
    _optional_layer_tensor,
    _to_device_tensor,
    build_rope_tables,
)


@dataclass(frozen=True)
class OptimizedDecoderPolicy:
    """Precision and kernel choices for the single-chip optimized decoder."""

    projection_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    mlp_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    norm_dtype: ttnn.DataType = ttnn.bfloat16
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    enable_prefill_program_config: bool = True
    enable_decode_dram_sharded_matmul: bool = True
    decode_dram_sharded_grid: tuple[int, int] = (8, 1)
    enable_signposts: bool = False

    def compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def sdpa_decode_program_config(self) -> ttnn.SDPAProgramConfig:
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )


def _signpost(enabled: bool, name: str) -> None:
    if enabled:
        ttnn.tracy_message(name)


def _pad_to_dram_banks(width: int, num_banks: int) -> int:
    alignment = ttnn.TILE_SIZE * num_banks
    remainder = width % alignment
    return width if remainder == 0 else width + alignment - remainder


def _dram_sharded_weight_memory_config(mesh_device, k: int, n: int) -> ttnn.MemoryConfig:
    dram_grid = mesh_device.dram_grid_size()
    num_banks = dram_grid.x * dram_grid.y
    dram_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))}
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            dram_cores,
            [k, _pad_to_dram_banks(n, num_banks) // num_banks],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _l1_width_sharded_memory_config(batch: int, width: int) -> ttnn.MemoryConfig:
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [batch * ttnn.TILE_SIZE, width // 64], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _decode_dram_program_config(
    k: int, n: int, num_cores: int
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=max(1, (k // (ttnn.TILE_SIZE * num_cores)) // 4),
        per_core_M=1,
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
        fused_activation=None,
    )


def _prefill_program_config(k: int, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    grid = ttnn.CoreCoord(8, 8)
    per_core_n = math.ceil((n // ttnn.TILE_SIZE) / 64)
    in0_block_w = 8 if k == 4096 and n == 6144 else 16
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=min(per_core_n, 8),
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        mcast_in0=True,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Optimized single Llama-3.1-8B-Instruct decoder layer."""

    def __init__(self, *, policy: OptimizedDecoderPolicy | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy = policy or OptimizedDecoderPolicy()
        self.compute_kernel_config = self.policy.compute_kernel_config()
        self.decode_sdpa_program_config = self.policy.sdpa_decode_program_config()
        self.qkv_decode_weight = None
        self.o_decode_weight = None
        self.gate_decode_weight = None
        self.up_decode_weight = None
        self.down_decode_weight = None

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
        policy: OptimizedDecoderPolicy | None = None,
        **kwargs,
    ) -> "OptimizedDecoder":
        if kwargs:
            raise TypeError(f"unsupported OptimizedDecoder kwargs: {sorted(kwargs)}")
        cfg = Llama31DecoderConfig.from_hf_config(hf_config)
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        if max_seq_len <= 0 or max_seq_len > cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {cfg.max_position_embeddings}], got {max_seq_len}")

        selected_policy = policy or OptimizedDecoderPolicy()
        position_cos, position_sin = build_rope_tables(hf_config, max_seq_len, mesh_device)
        q_norm = _optional_layer_tensor(state_dict, layer_idx, "self_attn.q_norm.weight")
        k_norm = _optional_layer_tensor(state_dict, layer_idx, "self_attn.k_norm.weight")

        decoder = cls(
            policy=selected_policy,
            cfg=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            qkv_proj_weight=_to_device_tensor(
                _fused_emit_weight(state_dict, layer_idx),
                mesh_device,
                dtype=selected_policy.projection_weight_dtype,
            ),
            o_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight"),
                mesh_device,
                dtype=selected_policy.projection_weight_dtype,
            ),
            gate_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight"),
                mesh_device,
                dtype=selected_policy.mlp_weight_dtype,
            ),
            up_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight"),
                mesh_device,
                dtype=selected_policy.mlp_weight_dtype,
            ),
            down_proj_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight"),
                mesh_device,
                dtype=selected_policy.mlp_weight_dtype,
            ),
            input_layernorm_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "input_layernorm.weight"),
                mesh_device,
                dtype=selected_policy.norm_dtype,
            ),
            post_attention_layernorm_weight=_to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "post_attention_layernorm.weight"),
                mesh_device,
                dtype=selected_policy.norm_dtype,
            ),
            q_norm_weight=_to_device_tensor(q_norm, mesh_device, dtype=selected_policy.norm_dtype)
            if q_norm is not None
            else None,
            k_norm_weight=_to_device_tensor(k_norm, mesh_device, dtype=selected_policy.norm_dtype)
            if k_norm is not None
            else None,
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=None,
            max_seq_len=max_seq_len,
        )
        if selected_policy.enable_decode_dram_sharded_matmul:
            decoder.qkv_decode_weight = _to_device_tensor(
                _fused_emit_weight(state_dict, layer_idx),
                mesh_device,
                dtype=selected_policy.projection_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(
                    mesh_device, cfg.hidden_size, cfg.hidden_size + 2 * cfg.num_key_value_heads * cfg.head_dim
                ),
            )
            decoder.o_decode_weight = _to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").transpose(0, 1),
                mesh_device,
                dtype=selected_policy.projection_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, cfg.hidden_size, cfg.hidden_size),
            )
            decoder.gate_decode_weight = _to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").transpose(0, 1),
                mesh_device,
                dtype=selected_policy.mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, cfg.hidden_size, cfg.intermediate_size),
            )
            decoder.up_decode_weight = _to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(0, 1),
                mesh_device,
                dtype=selected_policy.mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, cfg.hidden_size, cfg.intermediate_size),
            )
            decoder.down_decode_weight = _to_device_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(0, 1),
                mesh_device,
                dtype=selected_policy.mlp_weight_dtype,
                memory_config=_dram_sharded_weight_memory_config(mesh_device, cfg.intermediate_size, cfg.hidden_size),
            )
        return decoder

    def _matmul(self, lhs: ttnn.Tensor, rhs: ttnn.Tensor, *, transpose_b: bool) -> ttnn.Tensor:
        return ttnn.matmul(
            lhs,
            rhs,
            transpose_a=False,
            transpose_b=transpose_b,
            dtype=self.policy.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _prefill_matmul(self, lhs: ttnn.Tensor, rhs: ttnn.Tensor, *, transpose_b: bool, k: int, n: int) -> ttnn.Tensor:
        program_config = None
        if self.policy.enable_prefill_program_config and lhs.shape[-2] <= ttnn.TILE_SIZE:
            program_config = _prefill_program_config(k, n)
        return ttnn.matmul(
            lhs,
            rhs,
            transpose_a=False,
            transpose_b=transpose_b,
            program_config=program_config,
            dtype=self.policy.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _decode_dram_sharded_matmul(self, lhs: ttnn.Tensor, rhs: ttnn.Tensor, *, k: int, n: int) -> ttnn.Tensor:
        grid = self.policy.decode_dram_sharded_grid
        num_cores = grid[0] * grid[1]
        if lhs.shape[-3] == self.batch and lhs.shape[-2] == 1:
            lhs = ttnn.reshape(lhs, [1, 1, self.batch, k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lhs = ttnn.interleaved_to_sharded(
            lhs,
            grid,
            [self.batch, k // num_cores],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        out = ttnn.matmul(
            lhs,
            rhs,
            transpose_a=False,
            transpose_b=False,
            program_config=_decode_dram_program_config(k, n, num_cores),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.policy.activation_dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, [1, self.batch, 1, n], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _to_l1_width_sharded(self, tensor: ttnn.Tensor, width: int) -> ttnn.Tensor:
        return ttnn.interleaved_to_sharded(
            tensor,
            (8, 8),
            [self.batch * ttnn.TILE_SIZE, width // 64],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    def _decode_attention_mlp(self, hidden_states: ttnn.Tensor, attn: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.cfg
        hidden_l1 = self._to_l1_width_sharded(hidden_states, cfg.hidden_size)
        attn = ttnn.reshape(
            attn,
            [1, self.batch, 1, cfg.num_attention_heads * cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = self._decode_dram_sharded_matmul(attn, self.o_decode_weight, k=cfg.hidden_size, n=cfg.hidden_size)
        hidden_l1_config = _l1_width_sharded_memory_config(self.batch, cfg.hidden_size)
        attn_residual = ttnn.add(
            self._to_l1_width_sharded(attn_out, cfg.hidden_size),
            hidden_l1,
            dtype=self.policy.activation_dtype,
            memory_config=hidden_l1_config,
        )
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=hidden_l1_config,
        )
        post_norm = ttnn.sharded_to_interleaved(post_norm, ttnn.DRAM_MEMORY_CONFIG)
        gate = self._decode_dram_sharded_matmul(
            post_norm, self.gate_decode_weight, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = self._decode_dram_sharded_matmul(
            post_norm, self.up_decode_weight, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        gated = ttnn.multiply(gate, up, dtype=self.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mlp_out = self._decode_dram_sharded_matmul(
            gated, self.down_decode_weight, k=cfg.intermediate_size, n=cfg.hidden_size
        )
        out = ttnn.add(
            self._to_l1_width_sharded(mlp_out, cfg.hidden_size),
            attn_residual,
            dtype=self.policy.activation_dtype,
            memory_config=hidden_l1_config,
        )
        return ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

    def _attention_mlp(self, hidden_states: ttnn.Tensor, attn: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        cfg = self.cfg
        attn = ttnn.reshape(
            attn,
            [1, self.batch, seq_len, cfg.num_attention_heads * cfg.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = self._prefill_matmul(
            attn, self.o_proj_weight, transpose_b=True, k=cfg.hidden_size, n=cfg.hidden_size
        )
        attn_residual = ttnn.add(
            attn_out, hidden_states, dtype=self.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = self._prefill_matmul(
            post_norm, self.gate_proj_weight, transpose_b=True, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = self._prefill_matmul(
            post_norm, self.up_proj_weight, transpose_b=True, k=cfg.hidden_size, n=cfg.intermediate_size
        )
        gated = ttnn.multiply(gate, up, dtype=self.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mlp_out = self._prefill_matmul(
            gated, self.down_proj_weight, transpose_b=True, k=cfg.intermediate_size, n=cfg.hidden_size
        )
        return ttnn.add(
            mlp_out, attn_residual, dtype=self.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor | None = None,
        position_sin: ttnn.Tensor | None = None,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        _signpost(self.policy.enable_signposts, "PERF_PREFILL")
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

        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.cfg.rms_norm_eps,
            weight=self.input_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv = self._prefill_matmul(
            normed,
            self.qkv_proj_weight,
            transpose_b=False,
            k=self.cfg.hidden_size,
            n=self.cfg.hidden_size + 2 * self.cfg.num_key_value_heads * self.cfg.head_dim,
        )
        q, k, v = self._prepare_qkv(
            qkv,
            seq_len,
            position_cos if position_cos is not None else self.position_cos,
            position_sin if position_sin is not None else self.position_sin,
        )
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = self._attention_mlp(hidden_states, attn, seq_len)
        _signpost(self.policy.enable_signposts, "PERF_PREFILL_END")
        return output

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
        _signpost(self.policy.enable_signposts, "PERF_DECODE")
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
        if self.policy.enable_decode_dram_sharded_matmul:
            qkv = self._decode_dram_sharded_matmul(
                normed,
                self.qkv_decode_weight,
                k=self.cfg.hidden_size,
                n=self.cfg.hidden_size + 2 * self.cfg.num_key_value_heads * self.cfg.head_dim,
            )
        else:
            qkv = self._matmul(normed, self.qkv_proj_weight, transpose_b=False)
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
            program_config=self.decode_sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn = ttnn.to_memory_config(attn, _decode_height_sharded_memory_config())
        attn = ttnn.experimental.nlp_concat_heads_decode(
            attn,
            sub_core_grids=attn.memory_config().shard_spec.grid,
            num_heads=self.cfg.num_attention_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.policy.enable_decode_dram_sharded_matmul:
            output = self._decode_attention_mlp(hidden_states, attn)
        else:
            output = self._attention_mlp(hidden_states, attn, 1)
        _signpost(self.policy.enable_signposts, "PERF_DECODE_END")
        return output

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str = "prefill", **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported OptimizedDecoder mode: {mode!r}")
