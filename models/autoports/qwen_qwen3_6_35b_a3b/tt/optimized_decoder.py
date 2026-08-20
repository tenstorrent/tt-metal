# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized TTNN decoder layer for ``Qwen/Qwen3.6-35B-A3B``.

The optimized decoder preserves the fused decoder public contract and starts
from the fused graph topology.  It adds an explicit per-role precision and
runtime policy so the measured path is not a functional or fused fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import ttnn

from .functional_decoder import (
    DEFAULT_MOE_CHUNK_SIZE,
    HIDDEN_SIZE,
    MODEL_ID,
    TILE_SIZE,
    FunctionalDecoderResult,
    QwenFullAttentionCache,
    QwenLinearAttentionState,
    _as_device_tensor,
    _layer_state,
    _require,
    _rms_weight,
    _shape,
    _slice,
    _slice_last,
    _text_config,
)
from .fused_decoder import _SIGMOID
from .fused_decoder import GRAPH_SUMMARY as FUSED_GRAPH_SUMMARY
from .fused_decoder import (
    FusedDecoder,
    FusedDecoderGraphSummary,
    _FusedFullAttention,
    _FusedLinearAttention,
    _FusedQwenMoe,
    _packed_gate_up_expert_weight,
    _silu_mul_fused,
)

AUTO_ROUTED_MOE_WEIGHT_DTYPE = "auto_by_layer"


@dataclass(frozen=True)
class OptimizedDecoderPolicy:
    """Runtime policy for optimized decoder material ops."""

    attention_weight_dtype: Any = ttnn.bfloat8_b
    linear_attention_weight_dtype: Any = ttnn.bfloat8_b
    shared_moe_weight_dtype: Any = ttnn.bfloat8_b
    routed_moe_weight_dtype: Any = AUTO_ROUTED_MOE_WEIGHT_DTYPE
    sparse_decode_output_dtype: Any = ttnn.bfloat16
    sparse_prefill_output_dtype: Any = ttnn.bfloat16
    sparse_decode_memory_config: Any = ttnn.DRAM_MEMORY_CONFIG
    sparse_prefill_memory_config: Any = ttnn.DRAM_MEMORY_CONFIG
    sparse_in0_block_w: int = 4
    sparse_core_count_cap: int | None = None
    sparse_out_subblock_h: int = 1
    sparse_out_subblock_w: int = 1
    use_decode_exact_nnz: bool = True
    use_decode_l1_sparse_inputs: bool = False
    use_prefill_l1_sparse_inputs: bool = False
    use_decode_sdpa_program_config: bool = True
    decode_sdpa_q_chunk_size: int = TILE_SIZE
    decode_sdpa_k_chunk_size: int = 2 * TILE_SIZE
    decode_sdpa_max_cores_per_head_batch: int = 16
    attention_compute_fidelity: str | None = "default"
    linear_attention_compute_fidelity: str | None = "default"
    router_compute_fidelity: str | None = "default"
    shared_moe_compute_fidelity: str | None = "default"
    routed_moe_compute_fidelity: str | None = "default"
    lm_head_compute_fidelity: str | None = "default"
    ccl_dtype: Literal["bf16", "bf8"] | None = None


DEFAULT_OPTIMIZED_POLICY = OptimizedDecoderPolicy()


@dataclass(frozen=True)
class OptimizedDecoderGraphSummary:
    """Small runtime-independent summary used by tests and docs."""

    fused_graph: FusedDecoderGraphSummary
    named_precision_policy: str
    bf16_norms_and_residuals: bool
    reduced_attention_weights: bool
    reduced_linear_attention_weights: bool
    reduced_moe_weights: bool


GRAPH_SUMMARY = OptimizedDecoderGraphSummary(
    fused_graph=FUSED_GRAPH_SUMMARY,
    named_precision_policy="qwen36_optimized_bfp8_dense_auto_routed_moe_bf16_state_sdpa_k64_sparsew4_exactnnz",
    bf16_norms_and_residuals=True,
    reduced_attention_weights=True,
    reduced_linear_attention_weights=True,
    reduced_moe_weights=True,
)


def _optimized_sparse_matmul_program_config(
    m: int, n: int, *, policy: OptimizedDecoderPolicy
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    n_tiles = int(math.ceil(n / TILE_SIZE))
    core_cap = policy.sparse_core_count_cap or 64
    best_cores = 1
    best_x = 1
    best_y = 1
    for num_cores in range(1, min(core_cap, 64, n_tiles) + 1):
        if n_tiles % num_cores != 0:
            continue
        for y in range(1, 9):
            if num_cores % y != 0:
                continue
            x = num_cores // y
            if x <= 8 and num_cores > best_cores:
                best_cores = num_cores
                best_x = x
                best_y = y
                break

    per_core_m = max(TILE_SIZE, m) // TILE_SIZE
    per_core_n = max(1, n_tiles // best_cores)
    out_subblock_h = max(1, min(policy.sparse_out_subblock_h, per_core_m))
    while per_core_m % out_subblock_h != 0:
        out_subblock_h -= 1
    out_subblock_w = max(1, min(policy.sparse_out_subblock_w, per_core_n))
    while per_core_n % out_subblock_w != 0:
        out_subblock_w -= 1

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_x, best_y),
        in0_block_w=policy.sparse_in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class _OptimizedQwenMoe(_FusedQwenMoe):
    def __init__(self, state: dict[str, Any], cfg, *, device, policy: OptimizedDecoderPolicy, chunk_size: int):
        super().__init__(state, cfg, device=device, dtype=policy.shared_moe_weight_dtype, chunk_size=chunk_size)
        self.policy = policy
        self.routed_gate_up = _packed_gate_up_expert_weight(
            state,
            "mlp.experts.gate_up_proj",
            device=device,
            dtype=policy.routed_moe_weight_dtype,
        )
        down = _require(state, "mlp.experts.down_proj")
        self.routed_down = _as_device_tensor(
            down.transpose(-1, -2).unsqueeze(0).contiguous(),
            device=device,
            dtype=policy.routed_moe_weight_dtype,
        )

    def _routed_decode(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        tokens = _shape(flat)[2]
        sparse_routing = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
        gate_up_config = _optimized_sparse_matmul_program_config(tokens, self.gate_up_width, policy=self.policy)
        down_config = _optimized_sparse_matmul_program_config(tokens, self.cfg.hidden_size, policy=self.policy)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        decode_nnz = tokens * self.cfg.num_experts_per_tok if self.policy.use_decode_exact_nnz else None
        gate_up_input = (
            ttnn.to_memory_config(flat, ttnn.L1_MEMORY_CONFIG) if self.policy.use_decode_l1_sparse_inputs else flat
        )

        gate_up = ttnn.sparse_matmul(
            gate_up_input,
            self.routed_gate_up,
            sparsity=sparse_routing,
            nnz=decode_nnz,
            memory_config=self.policy.sparse_decode_memory_config,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.policy.sparse_decode_output_dtype,
        )
        gate_up = ttnn.reshape(gate_up, (tokens, self.cfg.num_experts, 1, self.gate_up_width))
        gate_up = ttnn.transpose(gate_up, 1, 2)
        gate = _slice_last(gate_up, 0, self.cfg.moe_intermediate_size)
        up = _slice_last(gate_up, self.cfg.moe_intermediate_size, self.gate_up_width)

        expert_hidden = _silu_mul_fused(gate, up)
        expert_hidden = ttnn.reshape(expert_hidden, (tokens, self.cfg.num_experts, self.cfg.moe_intermediate_size))
        expert_hidden = ttnn.transpose(expert_hidden, 1, 0)
        expert_hidden = ttnn.reshape(
            expert_hidden,
            (1, self.cfg.num_experts, tokens, self.cfg.moe_intermediate_size),
        )
        down_input = (
            ttnn.to_memory_config(expert_hidden, ttnn.L1_MEMORY_CONFIG)
            if self.policy.use_decode_l1_sparse_inputs
            else expert_hidden
        )
        routed = ttnn.sparse_matmul(
            down_input,
            self.routed_down,
            sparsity=sparse_routing,
            nnz=decode_nnz,
            is_input_a_sparse=True,
            memory_config=self.policy.sparse_decode_memory_config,
            output_tile=output_tile,
            program_config=down_config,
            dtype=self.policy.sparse_decode_output_dtype,
        )
        routed = ttnn.permute(routed, (0, 2, 1, 3))
        routed = ttnn.reshape(routed, (tokens, self.cfg.num_experts, self.cfg.hidden_size))
        routing = ttnn.reshape(routing, (tokens, self.cfg.num_experts, 1))
        routed = ttnn.mul(routed, routing, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routed = ttnn.sum(routed, dim=1)
        return ttnn.unsqueeze_to_4D(routed)

    def _routed_prefill_chunk(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        logical_tokens = _shape(flat)[2]
        hidden = _shape(flat)[3]
        physical_tokens = int(math.ceil(logical_tokens / TILE_SIZE) * TILE_SIZE)
        if physical_tokens != logical_tokens:
            flat = ttnn.pad(flat, (1, 1, physical_tokens, hidden), (0, 0, 0, 0), 0.0)
            routing = ttnn.pad(routing, (1, 1, physical_tokens, self.cfg.num_experts), (0, 0, 0, 0), 0.0)

        group_size = physical_tokens // TILE_SIZE
        grouped = ttnn.reshape(flat, (1, group_size, TILE_SIZE, hidden))
        sparsity = ttnn.repeat(self.all_expert_sparsity, (1, 1, group_size, 1))
        nnz = self.cfg.num_experts * group_size
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _optimized_sparse_matmul_program_config(TILE_SIZE, self.gate_up_width, policy=self.policy)
        down_config = _optimized_sparse_matmul_program_config(TILE_SIZE, self.cfg.hidden_size, policy=self.policy)
        gate_up_input = (
            ttnn.to_memory_config(grouped, ttnn.L1_MEMORY_CONFIG)
            if self.policy.use_prefill_l1_sparse_inputs
            else grouped
        )

        gate_up = ttnn.sparse_matmul(
            gate_up_input,
            self.routed_gate_up,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=self.policy.sparse_prefill_memory_config,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.policy.sparse_prefill_output_dtype,
        )
        gate_up = ttnn.transpose(gate_up, 1, 3)
        gate_up = ttnn.reshape(gate_up, (1, self.cfg.num_experts, physical_tokens, self.gate_up_width))
        gate = _slice_last(gate_up, 0, self.cfg.moe_intermediate_size)
        up = _slice_last(gate_up, self.cfg.moe_intermediate_size, self.gate_up_width)

        expert_hidden = _silu_mul_fused(gate, up)
        expert_hidden = ttnn.reshape(
            expert_hidden, (1, self.cfg.num_experts, physical_tokens, self.cfg.moe_intermediate_size)
        )
        down_input = (
            ttnn.to_memory_config(expert_hidden, ttnn.L1_MEMORY_CONFIG)
            if self.policy.use_prefill_l1_sparse_inputs
            else expert_hidden
        )
        routed = ttnn.sparse_matmul(
            down_input,
            self.routed_down,
            sparsity=self.all_expert_sparsity,
            nnz=self.cfg.num_experts,
            is_input_a_sparse=True,
            memory_config=self.policy.sparse_prefill_memory_config,
            output_tile=output_tile,
            program_config=down_config,
            dtype=self.policy.sparse_prefill_output_dtype,
        )
        routed = ttnn.reshape(routed, (1, self.cfg.num_experts, physical_tokens, self.cfg.hidden_size))
        routing = ttnn.permute(routing, (0, 3, 2, 1))
        routed = ttnn.mul(routed, routing, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routed = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(routed, dims=[1]))
        routed = ttnn.reshape(routed, (1, 1, physical_tokens, self.cfg.hidden_size))
        if physical_tokens != logical_tokens:
            routed = _slice(routed, (0, 0, 0, 0), (1, 1, logical_tokens, self.cfg.hidden_size))
        return routed


class _OptimizedFullAttention(_FusedFullAttention):
    def __init__(self, state: dict[str, Any], cfg, *, device, dtype, policy: OptimizedDecoderPolicy):
        super().__init__(state, cfg, device=device, dtype=dtype)
        self.policy = policy

    def _decode_sdpa_program_config(self):
        if not self.policy.use_decode_sdpa_program_config:
            return None
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=self.policy.decode_sdpa_q_chunk_size,
            k_chunk_size=self.policy.decode_sdpa_k_chunk_size,
            exp_approx_mode=False,
            max_cores_per_head_batch=self.policy.decode_sdpa_max_cores_per_head_batch,
        )

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_embeddings: tuple[ttnn.Tensor, ttnn.Tensor],
        kv_cache: QwenFullAttentionCache,
        page_table: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        _, _, batch, _ = _shape(hidden_states)
        q_and_gate, k, v = self._project_qkgv(hidden_states)
        q, gate, k, v = self._reshape_decode_heads(q_and_gate, k, v, batch)
        q, k = self._norm_and_rope(q, k, position_embeddings, decode_layout=True)

        k_update = self._cache_update_tensor(k, batch=batch)
        v_update = self._cache_update_tensor(v, batch=batch, core_offset=batch)
        ttnn.experimental.paged_fused_update_cache(
            kv_cache.keys,
            k_update,
            kv_cache.values,
            v_update,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )

        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache.keys,
            kv_cache.values,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=self.cfg.head_dim**-0.5,
            program_config=self._decode_sdpa_program_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = ttnn.reshape(attn_out, (1, batch, 1, self.q_width))
        attn_out = ttnn.mul(
            attn_out,
            gate,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.linear(attn_out, self.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (1, 1, batch, self.cfg.hidden_size))


class OptimizedDecoder(FusedDecoder):
    """Qwen3.6-35B-A3B optimized decoder layer with the fused public contract."""

    graph_summary = GRAPH_SUMMARY
    default_policy = DEFAULT_OPTIMIZED_POLICY

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        cfg = _text_config(hf_config)
        if cfg.hidden_size != HIDDEN_SIZE:
            raise ValueError(f"{MODEL_ID} optimized decoder expects hidden_size={HIDDEN_SIZE}, got {cfg.hidden_size}")
        layer_type = cfg.layer_types[layer_idx]
        if layer_type not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported Qwen3.6 decoder layer_type: {layer_type}")

        policy = kwargs.get("policy", cls.default_policy)
        if not isinstance(policy, OptimizedDecoderPolicy):
            raise TypeError(f"policy must be OptimizedDecoderPolicy, got {type(policy)!r}")

        state = _layer_state(state_dict, layer_idx)
        input_norm = _rms_weight(state, "input_layernorm.weight", device=mesh_device, add_unit_offset=True)
        post_norm = _rms_weight(state, "post_attention_layernorm.weight", device=mesh_device, add_unit_offset=True)
        if layer_type == "linear_attention":
            token_mixer = _FusedLinearAttention(
                state,
                cfg,
                device=mesh_device,
                dtype=policy.linear_attention_weight_dtype,
            )
        else:
            token_mixer = _OptimizedFullAttention(
                state,
                cfg,
                device=mesh_device,
                dtype=policy.attention_weight_dtype,
                policy=policy,
            )
        moe_chunk_size = int(kwargs.get("moe_chunk_size", DEFAULT_MOE_CHUNK_SIZE))
        if moe_chunk_size <= 0:
            raise ValueError(f"moe_chunk_size must be positive, got {moe_chunk_size}")
        if policy.routed_moe_weight_dtype == AUTO_ROUTED_MOE_WEIGHT_DTYPE:
            routed_dtype = ttnn.bfloat4_b if layer_type == "full_attention" else ttnn.bfloat8_b
            policy = replace(policy, routed_moe_weight_dtype=routed_dtype)
        mlp = _OptimizedQwenMoe(state, cfg, device=mesh_device, policy=policy, chunk_size=moe_chunk_size)
        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            layer_type=layer_type,
            input_layernorm_weight=input_norm,
            post_attention_layernorm_weight=post_norm,
            token_mixer=token_mixer,
            mlp=mlp,
        )

    def forward(
        self, hidden_states: ttnn.Tensor, *, mode: Literal["prefill", "decode"], **kwargs
    ) -> FunctionalDecoderResult:
        return super().forward(hidden_states, mode=mode, **kwargs)


__all__ = [
    "OptimizedDecoder",
    "OptimizedDecoderGraphSummary",
    "OptimizedDecoderPolicy",
    "AUTO_ROUTED_MOE_WEIGHT_DTYPE",
    "DEFAULT_OPTIMIZED_POLICY",
    "GRAPH_SUMMARY",
    "FunctionalDecoderResult",
    "QwenFullAttentionCache",
    "QwenLinearAttentionState",
]
