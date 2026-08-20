# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""2x2 TTNN multichip decoder for ``Qwen/Qwen3.6-35B-A3B``.

This module starts from :mod:`optimized_decoder` as the single-chip baseline
and preserves its public layer contract.  The multichip runtime targets the
four-chip Blackhole p300c mesh available during bringup:

* mesh shape ``2x2``;
* tensor parallelism over mesh columns;
* expert parallel routing over mesh rows;
* replicated residual stream at decoder layer boundaries.

The implementation intentionally supports only that target mesh.  It keeps the
same prefill/decode logical shapes as the optimized decoder while making the
per-device weight, state, and KV-cache shapes explicit.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

import ttnn
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_tt_ccl,
)

from .functional_decoder import (
    DEFAULT_MOE_CHUNK_SIZE,
    HIDDEN_SIZE,
    LINEAR_ATTENTION_CHUNK_SIZE,
    MODEL_ID,
    TILE_SIZE,
    FunctionalDecoderResult,
    QwenFullAttentionCache,
    QwenLinearAttentionState,
    _concat_dim2_bounded,
    _l2_norm_last_dim,
    _layer_state,
    _require,
    _rms_norm,
    _rms_weight,
    _row_major_core_rangeset,
    _shape,
    _slice,
    _slice_last,
    _text_config,
)
from .fused_decoder import _SIGMOID, _SILU, _SOFTPLUS, _apply_partial_rope_fused, _silu_mul_fused
from .optimized_decoder import AUTO_ROUTED_MOE_WEIGHT_DTYPE, DEFAULT_OPTIMIZED_POLICY
from .optimized_decoder import GRAPH_SUMMARY as OPTIMIZED_GRAPH_SUMMARY
from .optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderGraphSummary,
    OptimizedDecoderPolicy,
    _optimized_sparse_matmul_program_config,
)
from .precision_config import compute_kernel_config_from_fidelity


@dataclass(frozen=True)
class MultichipMeshPlan:
    """Static mesh contract for this stage."""

    mesh_shape: tuple[int, int] = (2, 2)
    tensor_parallel_axis: int = 1
    expert_parallel_axis: int = 0
    tensor_parallel_size: int = 2
    expert_parallel_size: int = 2
    topology: Any = ttnn.Topology.Ring
    num_links: int = 2
    ccl_mode: Literal["all_reduce", "explicit_rs_ag"] = "all_reduce"
    ccl_dtype: Literal["bf16", "bf8"] = "bf16"
    residual_layout: Literal["replicated"] = "replicated"


def _target_mesh_plan_from_env() -> MultichipMeshPlan:
    plan = MultichipMeshPlan()
    raw_links = os.environ.get("QWEN36_MULTICHIP_NUM_LINKS")
    raw_mode = os.environ.get("QWEN36_MULTICHIP_CCL_MODE")
    raw_dtype = os.environ.get("QWEN36_MULTICHIP_CCL_DTYPE")
    if raw_links is None and raw_mode is None and raw_dtype is None:
        return plan
    num_links = plan.num_links if raw_links is None else int(raw_links)
    if num_links <= 0:
        raise ValueError(f"QWEN36_MULTICHIP_NUM_LINKS must be positive, got {raw_links!r}")
    ccl_mode = plan.ccl_mode if raw_mode is None else raw_mode
    if ccl_mode not in ("all_reduce", "explicit_rs_ag"):
        raise ValueError(f"unsupported QWEN36_MULTICHIP_CCL_MODE={raw_mode!r}")
    ccl_dtype = plan.ccl_dtype if raw_dtype is None else raw_dtype
    if ccl_dtype not in ("bf16", "bf8"):
        raise ValueError(f"unsupported QWEN36_MULTICHIP_CCL_DTYPE={raw_dtype!r}")
    return replace(plan, num_links=num_links, ccl_mode=ccl_mode, ccl_dtype=ccl_dtype)


TARGET_MESH_PLAN = _target_mesh_plan_from_env()


@dataclass(frozen=True)
class MultichipDecoderGraphSummary:
    """Runtime-independent summary used by tests and stage docs."""

    optimized_baseline: OptimizedDecoderGraphSummary
    target_mesh_shape: tuple[int, int]
    tensor_parallel_size: int
    expert_parallel_size: int
    ccl_num_links: int
    ccl_mode: str
    ccl_dtype: str
    replicated_residual_contract: bool
    full_attention_q_heads_per_device: int
    full_attention_kv_heads_per_device: int
    linear_attention_value_heads_per_device: int
    moe_active_decode_uses_routing_remap: bool
    moe_active_prefill_uses_token_sparse_path: bool
    moe_prefill_experts_per_ep_device: int


GRAPH_SUMMARY = MultichipDecoderGraphSummary(
    optimized_baseline=OPTIMIZED_GRAPH_SUMMARY,
    target_mesh_shape=TARGET_MESH_PLAN.mesh_shape,
    tensor_parallel_size=TARGET_MESH_PLAN.tensor_parallel_size,
    expert_parallel_size=TARGET_MESH_PLAN.expert_parallel_size,
    ccl_num_links=TARGET_MESH_PLAN.num_links,
    ccl_mode=TARGET_MESH_PLAN.ccl_mode,
    ccl_dtype=TARGET_MESH_PLAN.ccl_dtype,
    replicated_residual_contract=True,
    full_attention_q_heads_per_device=8,
    full_attention_kv_heads_per_device=1,
    linear_attention_value_heads_per_device=16,
    moe_active_decode_uses_routing_remap=True,
    moe_active_prefill_uses_token_sparse_path=True,
    moe_prefill_experts_per_ep_device=4,
)


def _validate_target_mesh(mesh_device, plan: MultichipMeshPlan = TARGET_MESH_PLAN) -> None:
    if not hasattr(mesh_device, "shape") or tuple(mesh_device.shape) != plan.mesh_shape:
        raise ValueError(f"{MODEL_ID} multichip decoder requires mesh shape {plan.mesh_shape}, got {mesh_device!r}")
    if int(mesh_device.get_num_devices()) != plan.tensor_parallel_size * plan.expert_parallel_size:
        raise ValueError("target mesh must expose exactly four devices")


def _replicate_mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _col_mapper(mesh_device):
    return ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=mesh_device.shape)


def _row_mapper(mesh_device):
    return ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_device.shape)


def _ep_row_mapper(mesh_device):
    return ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=mesh_device.shape)


def _mesh_tensor(
    tensor: torch.Tensor,
    *,
    device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=None,
) -> ttnn.Tensor:
    return ttnn.as_tensor(
        tensor.contiguous(),
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper if mesh_mapper is not None else _replicate_mapper(device),
    )


def _replicated_weight(state: dict[str, Any], key: str, *, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return _mesh_tensor(_require(state, key).reshape(1, 1, 1, -1), device=device, dtype=dtype)


def _packed_col_weight(chunks: list[torch.Tensor], *, device, dtype) -> ttnn.Tensor:
    packed = torch.cat(chunks, dim=0)
    return _mesh_tensor(
        packed.transpose(-1, -2).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=dtype,
        mesh_mapper=_col_mapper(device),
    )


def _single_col_weight(weight: torch.Tensor, *, device, dtype) -> ttnn.Tensor:
    return _mesh_tensor(
        weight.transpose(-1, -2).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=dtype,
        mesh_mapper=_col_mapper(device),
    )


def _row_parallel_weight(weight: torch.Tensor, *, device, dtype) -> ttnn.Tensor:
    return _mesh_tensor(
        weight.transpose(-1, -2).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=dtype,
        mesh_mapper=_row_mapper(device),
    )


def _expert_col_weight(weight: torch.Tensor, *, device, dtype) -> ttnn.Tensor:
    return _mesh_tensor(
        weight.transpose(-1, -2).unsqueeze(0),
        device=device,
        dtype=dtype,
        mesh_mapper=_col_mapper(device),
    )


def _expert_row_weight(weight: torch.Tensor, *, device, dtype) -> ttnn.Tensor:
    return _mesh_tensor(
        weight.transpose(-1, -2).unsqueeze(0),
        device=device,
        dtype=dtype,
        mesh_mapper=_row_mapper(device),
    )


def _all_reduce(tensor: ttnn.Tensor, plan: MultichipMeshPlan, *, cluster_axis: int) -> ttnn.Tensor:
    output_dtype = tensor.dtype
    if plan.ccl_dtype == "bf8":
        tensor = ttnn.typecast(tensor, dtype=ttnn.bfloat8_b)
    output_memory_config = tensor.memory_config()
    if plan.ccl_mode == "explicit_rs_ag":
        tt_ccl = get_tt_ccl(tensor.device())
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=plan.num_links,
            memory_config=output_memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=plan.topology,
            cluster_axis=cluster_axis,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        gathered = ttnn.experimental.all_gather_async(
            reduced,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=plan.num_links,
            memory_config=output_memory_config,
            topology=plan.topology,
            cluster_axis=cluster_axis,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        reduced.deallocate(True)
        output = gathered
    else:
        output = ttnn.all_reduce(
            tensor,
            num_links=plan.num_links,
            topology=plan.topology,
            cluster_axis=cluster_axis,
        )
    if plan.ccl_dtype == "bf8" and output_dtype != ttnn.bfloat8_b:
        output = ttnn.typecast(output, dtype=output_dtype)
    return output


def _all_reduce_tp(tensor: ttnn.Tensor, plan: MultichipMeshPlan) -> ttnn.Tensor:
    return _all_reduce(tensor, plan, cluster_axis=plan.tensor_parallel_axis)


def _all_reduce_ep(tensor: ttnn.Tensor, plan: MultichipMeshPlan) -> ttnn.Tensor:
    return _all_reduce(tensor, plan, cluster_axis=plan.expert_parallel_axis)


def _head_chunk(weight: torch.Tensor, start: int, count: int, width: int) -> torch.Tensor:
    return weight[start * width : (start + count) * width]


def _reordered_full_qkgv_weight(state: dict[str, Any], cfg, *, tp: int) -> list[torch.Tensor]:
    q_gate = _require(state, "self_attn.q_proj.weight")
    k = _require(state, "self_attn.k_proj.weight")
    v = _require(state, "self_attn.v_proj.weight")
    q_heads_per_device = cfg.num_attention_heads // tp
    kv_heads_per_device = cfg.num_key_value_heads // tp
    chunks = []
    for col in range(tp):
        q_chunk = _head_chunk(q_gate, col * q_heads_per_device, q_heads_per_device, 2 * cfg.head_dim)
        k_chunk = _head_chunk(k, col * kv_heads_per_device, kv_heads_per_device, cfg.head_dim)
        v_chunk = _head_chunk(v, col * kv_heads_per_device, kv_heads_per_device, cfg.head_dim)
        chunks.append(torch.cat([q_chunk, k_chunk, v_chunk], dim=0))
    return chunks


def _reordered_linear_qkvzba_weight(state: dict[str, Any], cfg, *, tp: int) -> list[torch.Tensor]:
    qkv = _require(state, "linear_attn.in_proj_qkv.weight")
    z = _require(state, "linear_attn.in_proj_z.weight")
    beta = _require(state, "linear_attn.in_proj_b.weight")
    alpha = _require(state, "linear_attn.in_proj_a.weight")
    key_heads_per_device = cfg.linear_num_key_heads // tp
    value_heads_per_device = cfg.linear_num_value_heads // tp
    key_width = cfg.linear_key_head_dim
    value_width = cfg.linear_value_head_dim
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    chunks = []
    for col in range(tp):
        key_start = col * key_heads_per_device * key_width
        key_end = key_start + key_heads_per_device * key_width
        value_start = col * value_heads_per_device * value_width
        value_end = value_start + value_heads_per_device * value_width
        chunks.append(
            torch.cat(
                [
                    qkv[key_start:key_end],
                    qkv[key_dim + key_start : key_dim + key_end],
                    qkv[2 * key_dim + value_start : 2 * key_dim + value_end],
                    z[value_start:value_end],
                    beta[col * value_heads_per_device : (col + 1) * value_heads_per_device],
                    alpha[col * value_heads_per_device : (col + 1) * value_heads_per_device],
                ],
                dim=0,
            )
        )
    return chunks


def _linear_conv_weight_chunks(state: dict[str, Any], cfg, *, tp: int) -> list[torch.Tensor]:
    conv = _require(state, "linear_attn.conv1d.weight")
    key_heads_per_device = cfg.linear_num_key_heads // tp
    value_heads_per_device = cfg.linear_num_value_heads // tp
    key_width = cfg.linear_key_head_dim
    value_width = cfg.linear_value_head_dim
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    chunks = []
    for col in range(tp):
        key_start = col * key_heads_per_device * key_width
        key_end = key_start + key_heads_per_device * key_width
        value_start = col * value_heads_per_device * value_width
        value_end = value_start + value_heads_per_device * value_width
        chunks.append(
            torch.cat(
                [
                    conv[key_start:key_end],
                    conv[key_dim + key_start : key_dim + key_end],
                    conv[2 * key_dim + value_start : 2 * key_dim + value_end],
                ],
                dim=0,
            )
        )
    return chunks


def _local_vector_chunks(vector: torch.Tensor, *, parts: int) -> torch.Tensor:
    return torch.cat([vector.chunk(parts, dim=0)[idx] for idx in range(parts)], dim=0)


def _packed_shared_gate_up_chunks(state: dict[str, Any], cfg, *, tp: int) -> list[torch.Tensor]:
    gate = _require(state, "mlp.shared_expert.gate_proj.weight")
    up = _require(state, "mlp.shared_expert.up_proj.weight")
    local = cfg.shared_expert_intermediate_size // tp
    chunks = []
    for col in range(tp):
        start = col * local
        end = start + local
        chunks.append(torch.cat([gate[start:end], up[start:end]], dim=0))
    return chunks


class _MultichipFullAttention:
    def __init__(self, state: dict[str, Any], cfg, *, device, dtype, policy: OptimizedDecoderPolicy):
        if cfg.attention_bias:
            raise NotImplementedError("Qwen3.6-35B-A3B text attention is expected to be bias-free")
        self.cfg = cfg
        self.device = device
        self.policy = policy
        ccl_dtype = policy.ccl_dtype or TARGET_MESH_PLAN.ccl_dtype
        self.plan = replace(TARGET_MESH_PLAN, ccl_dtype=ccl_dtype)
        self.compute_kernel_config = compute_kernel_config_from_fidelity(policy.attention_compute_fidelity)
        self.tp = self.plan.tensor_parallel_size
        self.local_q_heads = cfg.num_attention_heads // self.tp
        self.local_kv_heads = cfg.num_key_value_heads // self.tp
        self.local_q_width = self.local_q_heads * cfg.head_dim
        self.local_q_gate_width = 2 * self.local_q_width
        self.local_kv_width = self.local_kv_heads * cfg.head_dim
        self.local_qkgv_width = self.local_q_gate_width + 2 * self.local_kv_width
        self.qkgv_proj = _packed_col_weight(
            _reordered_full_qkgv_weight(state, cfg, tp=self.tp), device=device, dtype=dtype
        )
        self.o_proj = _row_parallel_weight(_require(state, "self_attn.o_proj.weight"), device=device, dtype=dtype)
        self.q_norm_weight = _rms_weight(state, "self_attn.q_norm.weight", device=device, add_unit_offset=True)
        self.k_norm_weight = _rms_weight(state, "self_attn.k_norm.weight", device=device, add_unit_offset=True)

    def _project_qkgv(self, x: ttnn.Tensor):
        packed = ttnn.linear(
            x,
            self.qkgv_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        q_and_gate = _slice_last(packed, 0, self.local_q_gate_width)
        k = _slice_last(packed, self.local_q_gate_width, self.local_q_gate_width + self.local_kv_width)
        v = _slice_last(packed, self.local_q_gate_width + self.local_kv_width, self.local_qkgv_width)
        return q_and_gate, k, v

    def _reshape_prefill_heads(self, q_and_gate: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, batch: int, seq_len: int):
        q_and_gate = ttnn.reshape(q_and_gate, (batch, seq_len, self.local_q_heads, 2 * self.cfg.head_dim))
        q = _slice_last(q_and_gate, 0, self.cfg.head_dim)
        gate = _slice_last(q_and_gate, self.cfg.head_dim, 2 * self.cfg.head_dim)
        gate = ttnn.reshape(gate, (1, batch, seq_len, self.local_q_width))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (batch, seq_len, self.local_kv_heads, self.cfg.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (batch, seq_len, self.local_kv_heads, self.cfg.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))
        return q, gate, k, v

    def _reshape_decode_heads(self, q_and_gate: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, batch: int):
        q_and_gate = ttnn.reshape(q_and_gate, (batch, 1, self.local_q_heads, 2 * self.cfg.head_dim))
        q = _slice_last(q_and_gate, 0, self.cfg.head_dim)
        gate = _slice_last(q_and_gate, self.cfg.head_dim, 2 * self.cfg.head_dim)
        q = ttnn.permute(q, (1, 0, 2, 3))
        gate = ttnn.reshape(gate, (1, batch, 1, self.local_q_width))
        q = ttnn.reshape(q, (1, batch, self.local_q_heads, self.cfg.head_dim))
        k = ttnn.reshape(k, (1, batch, self.local_kv_heads, self.cfg.head_dim))
        v = ttnn.reshape(v, (1, batch, self.local_kv_heads, self.cfg.head_dim))
        return q, gate, k, v

    def _norm_and_rope(self, q: ttnn.Tensor, k: ttnn.Tensor, position_embeddings, *, decode_layout: bool):
        q = _rms_norm(q, self.q_norm_weight, self.cfg.rms_norm_eps)
        k = _rms_norm(k, self.k_norm_weight, self.cfg.rms_norm_eps)
        cos, sin = position_embeddings
        q = _apply_partial_rope_fused(q, cos, sin, self.cfg.rotary_dim, decode_layout=decode_layout)
        k = _apply_partial_rope_fused(k, cos, sin, self.cfg.rotary_dim, decode_layout=decode_layout)
        return q, k

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

    def _decode_update_mem_config(self, batch: int, head_dim: int, *, core_offset: int = 0):
        grid = self.device.compute_with_storage_grid_size()
        shard_grid = _row_major_core_rangeset(grid, batch, start=core_offset)
        shard_spec = ttnn.ShardSpec(
            shard_grid,
            (TILE_SIZE, head_dim),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def _cache_update_tensor(self, tensor: ttnn.Tensor, *, batch: int, core_offset: int = 0) -> ttnn.Tensor:
        if _shape(tensor)[2] < TILE_SIZE:
            tensor = ttnn.pad(
                tensor,
                (1, batch, TILE_SIZE, self.cfg.head_dim),
                (0, 0, 0, 0),
                0.0,
            )
        return ttnn.to_memory_config(
            tensor, self._decode_update_mem_config(batch, self.cfg.head_dim, core_offset=core_offset)
        )

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_embeddings: tuple[ttnn.Tensor, ttnn.Tensor],
        kv_cache: QwenFullAttentionCache | None,
        page_table: ttnn.Tensor | None,
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        _, batch, seq_len, _ = _shape(hidden_states)
        q_and_gate, k, v = self._project_qkgv(hidden_states)
        q, gate, k, v = self._reshape_prefill_heads(q_and_gate, k, v, batch, seq_len)
        q, k = self._norm_and_rope(q, k, position_embeddings, decode_layout=False)

        keys = values = None
        if kv_cache is not None:
            if page_table is None:
                raise ValueError("full-attention paged prefill requires page_table when kv_cache is supplied")
            keys, values = kv_cache.keys, kv_cache.values
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
            k_fill = ttnn.typecast(k, dtype=keys.dtype) if k.dtype != keys.dtype else k
            v_fill = ttnn.typecast(v, dtype=values.dtype) if v.dtype != values.dtype else v
            if batch == 1:
                ttnn.experimental.paged_fill_cache(keys, k_fill, fill_page_table, batch_idx=user_id)
                ttnn.experimental.paged_fill_cache(values, v_fill, fill_page_table, batch_idx=user_id)
            else:
                for batch_idx in range(batch):
                    k_b = _slice(
                        k_fill,
                        (batch_idx, 0, 0, 0),
                        (batch_idx + 1, self.local_kv_heads, seq_len, self.cfg.head_dim),
                    )
                    v_b = _slice(
                        v_fill,
                        (batch_idx, 0, 0, 0),
                        (batch_idx + 1, self.local_kv_heads, seq_len, self.cfg.head_dim),
                    )
                    ttnn.experimental.paged_fill_cache(keys, k_b, fill_page_table, batch_idx=user_id + batch_idx)
                    ttnn.experimental.paged_fill_cache(values, v_b, fill_page_table, batch_idx=user_id + batch_idx)

        q_sdpa = ttnn.typecast(q, dtype=ttnn.bfloat16) if q.dtype != ttnn.bfloat16 else q
        k_sdpa = ttnn.typecast(k, dtype=ttnn.bfloat16) if k.dtype != ttnn.bfloat16 else k
        v_sdpa = ttnn.typecast(v, dtype=ttnn.bfloat16) if v.dtype != ttnn.bfloat16 else v

        if chunk_start_idx is not None:
            if kv_cache is None or page_table is None:
                raise ValueError("chunked full-attention prefill requires paged kv_cache and page_table")
            if isinstance(chunk_start_idx, ttnn.Tensor):
                attn_out = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_sdpa,
                    keys,
                    values,
                    page_table,
                    chunk_start_idx_tensor=chunk_start_idx,
                )
            else:
                attn_out = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_sdpa,
                    keys,
                    values,
                    page_table,
                    chunk_start_idx,
                )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=True,
                attn_mask=attention_mask,
                scale=self.cfg.head_dim**-0.5,
            )

        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (1, batch, seq_len, self.local_q_width))
        attn_out = ttnn.mul(
            attn_out,
            gate,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.linear(
            attn_out,
            self.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        return _all_reduce_tp(out, self.plan)

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
        attn_out = ttnn.reshape(attn_out, (1, batch, 1, self.local_q_width))
        attn_out = ttnn.mul(
            attn_out,
            gate,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.linear(
            attn_out,
            self.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        out = _all_reduce_tp(out, self.plan)
        return ttnn.reshape(out, (1, 1, batch, self.cfg.hidden_size))


class _MultichipLinearAttention:
    def __init__(self, state: dict[str, Any], cfg, *, device, dtype, policy: OptimizedDecoderPolicy):
        self.cfg = cfg
        self.device = device
        ccl_dtype = policy.ccl_dtype or TARGET_MESH_PLAN.ccl_dtype
        self.plan = replace(TARGET_MESH_PLAN, ccl_dtype=ccl_dtype)
        self.projection_compute_kernel_config = compute_kernel_config_from_fidelity(
            policy.linear_attention_compute_fidelity
        )
        self.state_compute_kernel_config = compute_kernel_config_from_fidelity(policy.linear_attention_compute_fidelity)
        self.tp = self.plan.tensor_parallel_size
        self.local_key_heads = cfg.linear_num_key_heads // self.tp
        self.local_value_heads = cfg.linear_num_value_heads // self.tp
        self.local_key_dim = cfg.linear_key_head_dim * self.local_key_heads
        self.local_value_dim = cfg.linear_value_head_dim * self.local_value_heads
        self.local_conv_dim = self.local_key_dim * 2 + self.local_value_dim
        self.local_packed_width = self.local_conv_dim + self.local_value_dim + 2 * self.local_value_heads
        self.repeat_factor = self.local_value_heads // self.local_key_heads

        self.in_proj_qkv_zba = _packed_col_weight(
            _reordered_linear_qkvzba_weight(state, cfg, tp=self.tp), device=device, dtype=dtype
        )
        self.out_proj = _row_parallel_weight(_require(state, "linear_attn.out_proj.weight"), device=device, dtype=dtype)
        self.norm_weight = _rms_weight(state, "linear_attn.norm.weight", device=device, add_unit_offset=False)

        dt_chunks = [
            chunk.reshape(1, 1, 1, -1) for chunk in _require(state, "linear_attn.dt_bias").chunk(self.tp, dim=0)
        ]
        self.dt_bias = _mesh_tensor(
            torch.cat(dt_chunks, dim=-1), device=device, dtype=ttnn.bfloat16, mesh_mapper=_col_mapper(device)
        )
        a = _require(state, "linear_attn.A_log").float().exp().neg()
        a_chunks = [chunk.reshape(1, 1, 1, -1) for chunk in a.chunk(self.tp, dim=0)]
        self.neg_exp_a_log = _mesh_tensor(
            torch.cat(a_chunks, dim=-1), device=device, dtype=ttnn.bfloat16, mesh_mapper=_col_mapper(device)
        )

        conv_chunks = _linear_conv_weight_chunks(state, cfg, tp=self.tp)
        conv_by_tap = []
        for tap in range(cfg.linear_conv_kernel_dim):
            per_col = [chunk[:, 0, tap].reshape(1, 1, 1, -1) for chunk in conv_chunks]
            conv_by_tap.append(
                _mesh_tensor(
                    torch.cat(per_col, dim=-1), device=device, dtype=ttnn.bfloat16, mesh_mapper=_col_mapper(device)
                )
            )
        self.conv_weights = tuple(conv_by_tap)

        chunk = LINEAR_ATTENTION_CHUNK_SIZE
        self.linear_chunk_size = chunk
        mask_shape = (1, 1, chunk, chunk)
        self.chunk_lower_mask = _mesh_tensor(
            torch.tril(torch.ones(mask_shape, dtype=torch.bfloat16)), device=device, dtype=ttnn.bfloat16
        )
        self.chunk_strict_lower_mask = _mesh_tensor(
            torch.tril(torch.ones(mask_shape, dtype=torch.bfloat16), diagonal=-1),
            device=device,
            dtype=ttnn.bfloat16,
        )
        self.chunk_eye = _mesh_tensor(
            torch.eye(chunk, dtype=torch.bfloat16).reshape(mask_shape), device=device, dtype=ttnn.bfloat16
        )
        self.chunk_ones_1x64 = _mesh_tensor(
            torch.ones((1, 1, 1, chunk), dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16
        )
        row_prefix_masks = []
        row_keep_masks = []
        for idx in range(chunk):
            prefix = torch.zeros(mask_shape, dtype=torch.bfloat16)
            prefix[..., idx, :idx] = 1
            keep = torch.ones(mask_shape, dtype=torch.bfloat16)
            keep[..., idx, :] = 0
            row_prefix_masks.append(_mesh_tensor(prefix, device=device, dtype=ttnn.bfloat16))
            row_keep_masks.append(_mesh_tensor(keep, device=device, dtype=ttnn.bfloat16))
        self.row_prefix_masks = tuple(row_prefix_masks)
        self.row_keep_masks = tuple(row_keep_masks)

    def _project_inputs(self, hidden_states: ttnn.Tensor):
        packed = ttnn.linear(
            hidden_states,
            self.in_proj_qkv_zba,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute_kernel_config,
        )
        mixed_qkv_raw = _slice_last(packed, 0, self.local_conv_dim)
        z = _slice_last(packed, self.local_conv_dim, self.local_conv_dim + self.local_value_dim)
        beta = _slice_last(
            packed,
            self.local_conv_dim + self.local_value_dim,
            self.local_conv_dim + self.local_value_dim + self.local_value_heads,
        )
        alpha = _slice_last(
            packed, self.local_conv_dim + self.local_value_dim + self.local_value_heads, self.local_packed_width
        )
        return mixed_qkv_raw, z, beta, alpha

    def _log_g(self, alpha: ttnn.Tensor) -> ttnn.Tensor:
        base = ttnn.add(alpha, self.dt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.mul(
            base,
            self.neg_exp_a_log,
            input_tensor_a_activations=_SOFTPLUS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _conv_step(self, mixed_qkv: ttnn.Tensor, state: QwenLinearAttentionState):
        next_conv_state = (state.conv_state[1], state.conv_state[2], state.conv_state[3], mixed_qkv)
        acc = ttnn.mul(next_conv_state[0], self.conv_weights[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for idx in range(1, self.cfg.linear_conv_kernel_dim):
            part = ttnn.mul(next_conv_state[idx], self.conv_weights[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            acc = ttnn.add(acc, part, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG), next_conv_state

    def _step(self, hidden_states: ttnn.Tensor, state: QwenLinearAttentionState):
        _, _, batch, _ = _shape(hidden_states)
        mixed_qkv_raw, z, beta, alpha = self._project_inputs(hidden_states)
        mixed_qkv, conv_state = self._conv_step(mixed_qkv_raw, state)
        g = self._log_g(alpha)

        query = _slice_last(mixed_qkv, 0, self.local_key_dim)
        key = _slice_last(mixed_qkv, self.local_key_dim, 2 * self.local_key_dim)
        value = _slice_last(mixed_qkv, 2 * self.local_key_dim, self.local_conv_dim)

        query = ttnn.reshape(query, (1, batch, self.local_key_heads, self.cfg.linear_key_head_dim))
        key = ttnn.reshape(key, (1, batch, self.local_key_heads, self.cfg.linear_key_head_dim))
        if self.repeat_factor != 1:
            query = ttnn.repeat_interleave(query, self.repeat_factor, dim=2)
            key = ttnn.repeat_interleave(key, self.repeat_factor, dim=2)
        value = ttnn.reshape(value, (1, batch, self.local_value_heads, self.cfg.linear_value_head_dim))

        heads = batch * self.local_value_heads
        query = ttnn.reshape(query, (1, heads, 1, self.cfg.linear_key_head_dim))
        key = ttnn.reshape(key, (1, heads, 1, self.cfg.linear_key_head_dim))
        value = ttnn.reshape(value, (1, heads, 1, self.cfg.linear_value_head_dim))
        query = _l2_norm_last_dim(query, self.cfg.linear_key_head_dim)
        key = _l2_norm_last_dim(key, self.cfg.linear_key_head_dim)
        query = ttnn.mul(query, self.cfg.linear_key_head_dim**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        beta = ttnn.reshape(beta, (1, heads, 1, 1))
        g = ttnn.reshape(g, (1, heads, 1, 1))
        g = ttnn.exp(g, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        decayed_state = ttnn.mul(state.recurrent_state, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kv_mem = ttnn.matmul(
            key,
            decayed_state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        delta = ttnn.mul(
            ttnn.subtract(value, kv_mem, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            beta,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key_t = ttnn.transpose(key, -2, -1)
        outer = ttnn.matmul(
            key_t,
            delta,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        recurrent_state = ttnn.add(decayed_state, outer, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        core = ttnn.matmul(
            query,
            recurrent_state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )

        core = ttnn.reshape(core, (1, 1, heads, self.cfg.linear_value_head_dim))
        z = ttnn.reshape(z, (1, 1, heads, self.cfg.linear_value_head_dim))
        core = _rms_norm(core, self.norm_weight, self.cfg.rms_norm_eps)
        core = ttnn.mul(core, z, input_tensor_b_activations=_SILU, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        core = ttnn.reshape(core, (1, 1, batch, self.local_value_dim))
        out = ttnn.linear(
            core,
            self.out_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute_kernel_config,
        )
        out = _all_reduce_tp(out, self.plan)
        return out, QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)

    def _conv_prefill(self, mixed_qkv_raw: ttnn.Tensor, state: QwenLinearAttentionState):
        _, batch, length, _ = _shape(mixed_qkv_raw)
        taps = tuple(ttnn.reshape(tap, (1, batch, 1, self.local_conv_dim)) for tap in state.conv_state)
        conv_input = ttnn.concat(
            [taps[1], taps[2], taps[3], mixed_qkv_raw], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        acc = None
        for idx in range(self.cfg.linear_conv_kernel_dim):
            window = _slice(conv_input, (0, 0, idx, 0), (1, batch, idx + length, self.local_conv_dim))
            part = ttnn.mul(window, self.conv_weights[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            acc = part if acc is None else ttnn.add(acc, part, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        history = ttnn.concat(
            [taps[0], taps[1], taps[2], taps[3], mixed_qkv_raw],
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hist_len = _shape(history)[2]
        last4 = _slice(
            history, (0, 0, hist_len - self.cfg.linear_conv_kernel_dim, 0), (1, batch, hist_len, self.local_conv_dim)
        )
        conv_state = tuple(
            ttnn.reshape(
                _slice(last4, (0, 0, idx, 0), (1, batch, idx + 1, self.local_conv_dim)),
                (1, 1, batch, self.local_conv_dim),
            )
            for idx in range(self.cfg.linear_conv_kernel_dim)
        )
        return ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG), conv_state

    def _reshape_prefill_heads(
        self,
        mixed_qkv: ttnn.Tensor,
        z: ttnn.Tensor,
        beta: ttnn.Tensor,
        log_g: ttnn.Tensor,
        batch: int,
        length: int,
    ):
        query = _slice_last(mixed_qkv, 0, self.local_key_dim)
        key = _slice_last(mixed_qkv, self.local_key_dim, 2 * self.local_key_dim)
        value = _slice_last(mixed_qkv, 2 * self.local_key_dim, self.local_conv_dim)

        query = ttnn.reshape(query, (batch, length, self.local_key_heads, self.cfg.linear_key_head_dim))
        key = ttnn.reshape(key, (batch, length, self.local_key_heads, self.cfg.linear_key_head_dim))
        if self.repeat_factor != 1:
            query = ttnn.repeat_interleave(query, self.repeat_factor, dim=2)
            key = ttnn.repeat_interleave(key, self.repeat_factor, dim=2)

        value = ttnn.reshape(value, (batch, length, self.local_value_heads, self.cfg.linear_value_head_dim))
        z = ttnn.reshape(z, (batch, length, self.local_value_heads, self.cfg.linear_value_head_dim))
        beta = ttnn.reshape(beta, (batch, length, self.local_value_heads, 1))
        log_g = ttnn.reshape(log_g, (batch, length, self.local_value_heads, 1))

        query = self._fold_prefill_heads(query, batch, length, self.cfg.linear_key_head_dim)
        key = self._fold_prefill_heads(key, batch, length, self.cfg.linear_key_head_dim)
        value = self._fold_prefill_heads(value, batch, length, self.cfg.linear_value_head_dim)
        z = self._fold_prefill_heads(z, batch, length, self.cfg.linear_value_head_dim)
        beta = self._fold_prefill_heads(beta, batch, length, 1)
        log_g = self._fold_prefill_heads(log_g, batch, length, 1)

        query = _l2_norm_last_dim(query, self.cfg.linear_key_head_dim)
        key = _l2_norm_last_dim(key, self.cfg.linear_key_head_dim)
        query = ttnn.mul(query, self.cfg.linear_key_head_dim**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return query, key, value, z, beta, log_g

    def _fold_prefill_heads(self, tensor: ttnn.Tensor, batch: int, length: int, head_dim: int) -> ttnn.Tensor:
        tensor = ttnn.permute(tensor, (0, 2, 1, 3))
        return ttnn.reshape(tensor, (1, batch * self.local_value_heads, length, head_dim))

    def _pad_linear_chunk(self, tensor: ttnn.Tensor, length: int, last_dim: int) -> ttnn.Tensor:
        if length == self.linear_chunk_size:
            return tensor
        shape = list(_shape(tensor))
        shape[2] = self.linear_chunk_size
        shape[3] = last_dim
        return ttnn.pad(tensor, tuple(shape), (0, 0, 0, 0), 0.0)

    def _solve_chunk_attn(self, attn0: ttnn.Tensor) -> ttnn.Tensor:
        solved = attn0
        for idx in range(1, self.linear_chunk_size):
            row = ttnn.mul(solved, self.row_prefix_masks[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            update = ttnn.matmul(
                row,
                solved,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.state_compute_kernel_config,
            )
            new_row = ttnn.add(row, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kept = ttnn.mul(solved, self.row_keep_masks[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            solved = ttnn.add(kept, new_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(solved, self.chunk_eye, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _chunk_gated_delta_rule(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        log_g: ttnn.Tensor,
        beta: ttnn.Tensor,
        recurrent_state: ttnn.Tensor,
    ):
        log_g = ttnn.cumsum(log_g, dim=2, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g_rows = ttnn.matmul(
            log_g,
            self.chunk_ones_1x64,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        g_cols = ttnn.transpose(g_rows, -2, -1)
        decay = ttnn.subtract(g_rows, g_cols, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        decay = ttnn.exp(decay, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        decay = ttnn.mul(decay, self.chunk_lower_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        k_beta = ttnn.mul(
            key,
            beta,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_beta = ttnn.mul(
            value,
            beta,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kk = ttnn.matmul(
            k_beta,
            ttnn.transpose(key, -2, -1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        attn0 = ttnn.mul(kk, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn0 = ttnn.mul(
            ttnn.mul(attn0, decay, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            self.chunk_strict_lower_mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_attn = self._solve_chunk_attn(attn0)

        local_value = ttnn.matmul(
            local_attn,
            v_beta,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        exp_g = ttnn.exp(log_g, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_cumdecay = ttnn.matmul(
            local_attn,
            ttnn.mul(k_beta, exp_g, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )

        v_prime = ttnn.matmul(
            k_cumdecay,
            recurrent_state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        v_new = ttnn.subtract(local_value, v_prime, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        content_attn = ttnn.matmul(
            query,
            ttnn.transpose(key, -2, -1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        content_attn = ttnn.mul(content_attn, decay, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_inter = ttnn.matmul(
            ttnn.mul(query, exp_g, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            recurrent_state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        core = ttnn.add(
            attn_inter,
            ttnn.matmul(
                content_attn,
                v_new,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.state_compute_kernel_config,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        last_idx = self.linear_chunk_size - 1
        g_last = _slice(log_g, (0, 0, last_idx, 0), (1, _shape(log_g)[1], last_idx + 1, 1))
        state_decay = ttnn.mul(
            recurrent_state,
            ttnn.exp(g_last, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state_scale = ttnn.exp(
            ttnn.subtract(g_last, log_g, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state_update_key = ttnn.mul(key, state_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        state_update = ttnn.matmul(
            ttnn.transpose(state_update_key, -2, -1),
            v_new,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.state_compute_kernel_config,
        )
        recurrent_state = ttnn.add(state_decay, state_update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return core, recurrent_state

    def _finish_prefill_chunk(self, core: ttnn.Tensor, z: ttnn.Tensor, batch: int, length: int) -> ttnn.Tensor:
        core = _rms_norm(core, self.norm_weight, self.cfg.rms_norm_eps)
        core = ttnn.mul(core, z, input_tensor_b_activations=_SILU, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        core = ttnn.reshape(core, (batch, self.local_value_heads, length, self.cfg.linear_value_head_dim))
        core = ttnn.permute(core, (0, 2, 1, 3))
        core = ttnn.reshape(core, (1, batch, length, self.local_value_dim))
        out = ttnn.linear(
            core,
            self.out_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute_kernel_config,
        )
        return _all_reduce_tp(out, self.plan)

    def decode_forward(self, hidden_states: ttnn.Tensor, *, linear_state: QwenLinearAttentionState):
        return self._step(hidden_states, linear_state)

    def prefill_forward(self, hidden_states: ttnn.Tensor, *, linear_state: QwenLinearAttentionState):
        _, batch, seq_len, _ = _shape(hidden_states)
        chunks = []
        next_state = linear_state
        for start in range(0, seq_len, self.linear_chunk_size):
            end = min(start + self.linear_chunk_size, seq_len)
            length = end - start
            hidden_chunk = _slice(hidden_states, (0, 0, start, 0), (1, batch, end, self.cfg.hidden_size))
            mixed_qkv_raw, z, beta, alpha = self._project_inputs(hidden_chunk)
            mixed_qkv, conv_state = self._conv_prefill(mixed_qkv_raw, next_state)
            log_g = self._log_g(alpha)

            query, key, value, z_heads, beta, log_g = self._reshape_prefill_heads(
                mixed_qkv, z, beta, log_g, batch, length
            )
            query = self._pad_linear_chunk(query, length, self.cfg.linear_key_head_dim)
            key = self._pad_linear_chunk(key, length, self.cfg.linear_key_head_dim)
            value = self._pad_linear_chunk(value, length, self.cfg.linear_value_head_dim)
            z_heads = self._pad_linear_chunk(z_heads, length, self.cfg.linear_value_head_dim)
            beta = self._pad_linear_chunk(beta, length, 1)
            log_g = self._pad_linear_chunk(log_g, length, 1)

            core, recurrent_state = self._chunk_gated_delta_rule(
                query, key, value, log_g, beta, next_state.recurrent_state
            )
            if length != self.linear_chunk_size:
                core = _slice(
                    core,
                    (0, 0, 0, 0),
                    (1, batch * self.local_value_heads, length, self.cfg.linear_value_head_dim),
                )
                z_heads = _slice(
                    z_heads,
                    (0, 0, 0, 0),
                    (1, batch * self.local_value_heads, length, self.cfg.linear_value_head_dim),
                )
            chunks.append(self._finish_prefill_chunk(core, z_heads, batch, length))
            next_state = QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)
        return _concat_dim2_bounded(chunks), next_state


class _MultichipQwenMoe:
    def __init__(self, state: dict[str, Any], cfg, *, device, policy: OptimizedDecoderPolicy, chunk_size: int):
        self.cfg = cfg
        self.device = device
        self.policy = policy
        ccl_dtype = policy.ccl_dtype or TARGET_MESH_PLAN.ccl_dtype
        self.plan = replace(TARGET_MESH_PLAN, ccl_dtype=ccl_dtype)
        self.router_compute_kernel_config = compute_kernel_config_from_fidelity(policy.router_compute_fidelity)
        self.shared_compute_kernel_config = compute_kernel_config_from_fidelity(policy.shared_moe_compute_fidelity)
        self.routed_compute_kernel_config = compute_kernel_config_from_fidelity(policy.routed_moe_compute_fidelity)
        self.tp = self.plan.tensor_parallel_size
        self.ep = self.plan.expert_parallel_size
        self.chunk_size = chunk_size
        self.local_moe_intermediate_size = cfg.moe_intermediate_size // self.tp
        self.local_shared_intermediate_size = cfg.shared_expert_intermediate_size // self.tp
        self.local_gate_up_width = 2 * self.local_moe_intermediate_size
        self.local_shared_gate_up_width = 2 * self.local_shared_intermediate_size

        self.router = _mesh_tensor(
            _require(state, "mlp.gate.weight").transpose(-1, -2).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
        )
        self.shared_gate_up = _packed_col_weight(
            _packed_shared_gate_up_chunks(state, cfg, tp=self.tp), device=device, dtype=policy.shared_moe_weight_dtype
        )
        self.shared_down = _row_parallel_weight(
            _require(state, "mlp.shared_expert.down_proj.weight"), device=device, dtype=policy.shared_moe_weight_dtype
        )
        self.shared_expert_gate = _mesh_tensor(
            _require(state, "mlp.shared_expert_gate.weight").transpose(-1, -2).unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
        )

        gate_up = _require(state, "mlp.experts.gate_up_proj")
        inter = cfg.moe_intermediate_size
        local_gate_up_chunks = []
        for col in range(self.tp):
            start = col * self.local_moe_intermediate_size
            end = start + self.local_moe_intermediate_size
            local_gate_up_chunks.append(
                torch.cat([gate_up[:, start:end, :], gate_up[:, inter + start : inter + end, :]], dim=1)
            )
        self.routed_gate_up = _expert_col_weight(
            torch.cat(local_gate_up_chunks, dim=1), device=device, dtype=policy.routed_moe_weight_dtype
        )
        self.routed_down = _expert_row_weight(
            _require(state, "mlp.experts.down_proj"), device=device, dtype=policy.routed_moe_weight_dtype
        )

    def _router_dense(self, flat: ttnn.Tensor) -> ttnn.Tensor:
        logits = ttnn.linear(
            flat,
            self.router,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.router_compute_kernel_config,
        )
        probs = ttnn.softmax(logits, dim=-1, numeric_stable=True)
        top_values, top_indices = ttnn.topk(probs, k=self.cfg.num_experts_per_tok, dim=-1, sorted=True)
        denom = ttnn.sum(top_values, dim=-1, keepdim=True)
        top_values = ttnn.div(top_values, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=top_indices, src=top_values)

    def _shared(self, flat: ttnn.Tensor) -> ttnn.Tensor:
        gate_up = ttnn.linear(
            flat,
            self.shared_gate_up,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.shared_compute_kernel_config,
        )
        gate = _slice_last(gate_up, 0, self.local_shared_intermediate_size)
        up = _slice_last(gate_up, self.local_shared_intermediate_size, self.local_shared_gate_up_width)
        hidden = _silu_mul_fused(gate, up)
        hidden = ttnn.linear(
            hidden,
            self.shared_down,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.shared_compute_kernel_config,
        )
        hidden = _all_reduce_tp(hidden, self.plan)
        gate_scalar = ttnn.linear(
            flat,
            self.shared_expert_gate,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.router_compute_kernel_config,
        )
        return ttnn.mul(
            hidden,
            gate_scalar,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _decode_sparsity(self, routing: ttnn.Tensor, tokens: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        sparse_routing = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
        sparse_routing = ttnn.reshape(sparse_routing, (tokens, self.cfg.num_experts))
        sparse_routing = ttnn.moe_routing_remap(
            sparse_routing,
            self.cfg.num_experts_per_tok,
            self.ep,
            self.plan.expert_parallel_axis,
        )
        routing_weights = ttnn.tilize_with_zero_padding(sparse_routing, use_multicore=True)
        return sparse_routing, routing_weights

    def _routed_decode(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        tokens = _shape(flat)[2]
        sparse_routing, routing_weights = self._decode_sparsity(routing, tokens)
        gate_up_config = _optimized_sparse_matmul_program_config(tokens, self.local_gate_up_width, policy=self.policy)
        down_config = _optimized_sparse_matmul_program_config(tokens, self.cfg.hidden_size, policy=self.policy)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        decode_nnz = tokens * (self.cfg.num_experts_per_tok // self.ep) if self.policy.use_decode_exact_nnz else None
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
            compute_kernel_config=self.routed_compute_kernel_config,
        )
        gate_up = ttnn.reshape(gate_up, (tokens, self.cfg.num_experts, 1, self.local_gate_up_width))
        gate_up = ttnn.transpose(gate_up, 1, 2)
        gate = _slice_last(gate_up, 0, self.local_moe_intermediate_size)
        up = _slice_last(gate_up, self.local_moe_intermediate_size, self.local_gate_up_width)

        expert_hidden = _silu_mul_fused(gate, up)
        expert_hidden = ttnn.reshape(expert_hidden, (tokens, self.cfg.num_experts, self.local_moe_intermediate_size))
        expert_hidden = ttnn.transpose(expert_hidden, 1, 0)
        expert_hidden = ttnn.reshape(
            expert_hidden,
            (1, self.cfg.num_experts, tokens, self.local_moe_intermediate_size),
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
            compute_kernel_config=self.routed_compute_kernel_config,
        )
        routed = ttnn.permute(routed, (0, 2, 1, 3))
        routed = ttnn.reshape(routed, (tokens, self.cfg.num_experts, self.cfg.hidden_size))
        routing_weights = ttnn.reshape(routing_weights, (tokens, self.cfg.num_experts, 1))
        routed = ttnn.mul(routed, routing_weights, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routed = ttnn.sum(routed, dim=1)
        routed = ttnn.unsqueeze_to_4D(routed)
        routed = _all_reduce_ep(routed, self.plan)
        return _all_reduce_tp(routed, self.plan)

    def _routed_prefill_chunk(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        tokens = _shape(flat)[2]
        hidden = _shape(flat)[3]
        outputs = []
        for start in range(tokens):
            flat_token = _slice(flat, (0, 0, start, 0), (1, 1, start + 1, hidden))
            routing_token = _slice(routing, (0, 0, start, 0), (1, 1, start + 1, self.cfg.num_experts))
            outputs.append(self._routed_decode(flat_token, routing_token))
        return _concat_dim2_bounded(outputs)

    def _routed_chunk(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        if _shape(flat)[2] == 1:
            return self._routed_decode(flat, routing)
        return self._routed_prefill_chunk(flat, routing)

    def _forward_chunk(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        routed = self._routed_chunk(flat, routing)
        shared = self._shared(flat)
        return ttnn.add(routed, shared, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        _, batch, seq_len, hidden = _shape(hidden_states)
        tokens = batch * seq_len
        flat = ttnn.reshape(hidden_states, (1, 1, tokens, hidden))
        routing = self._router_dense(flat)
        if tokens <= self.chunk_size:
            out = self._forward_chunk(flat, routing)
        else:
            outputs = []
            for start in range(0, tokens, self.chunk_size):
                end = min(start + self.chunk_size, tokens)
                flat_chunk = _slice(flat, (0, 0, start, 0), (1, 1, end, hidden))
                routing_chunk = _slice(routing, (0, 0, start, 0), (1, 1, end, self.cfg.num_experts))
                outputs.append(self._forward_chunk(flat_chunk, routing_chunk))
            out = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (1, batch, seq_len, hidden))

    __call__ = forward


class MultichipDecoder(OptimizedDecoder):
    """Qwen3.6-35B-A3B 2x2 multichip decoder layer."""

    graph_summary = GRAPH_SUMMARY
    mesh_plan = TARGET_MESH_PLAN
    single_chip_baseline_cls = OptimizedDecoder
    default_policy = DEFAULT_OPTIMIZED_POLICY

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        _validate_target_mesh(mesh_device, cls.mesh_plan)
        cfg = _text_config(hf_config)
        if cfg.hidden_size != HIDDEN_SIZE:
            raise ValueError(f"{MODEL_ID} multichip decoder expects hidden_size={HIDDEN_SIZE}, got {cfg.hidden_size}")
        layer_type = cfg.layer_types[layer_idx]
        if layer_type not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported Qwen3.6 decoder layer_type: {layer_type}")

        policy = kwargs.get("policy", cls.default_policy)
        if not isinstance(policy, OptimizedDecoderPolicy):
            raise TypeError(f"policy must be OptimizedDecoderPolicy, got {type(policy)!r}")
        if policy.routed_moe_weight_dtype == AUTO_ROUTED_MOE_WEIGHT_DTYPE:
            routed_dtype = ttnn.bfloat4_b if layer_type == "full_attention" else ttnn.bfloat8_b
            policy = replace(policy, routed_moe_weight_dtype=routed_dtype)

        state = _layer_state(state_dict, layer_idx)
        input_norm = _rms_weight(state, "input_layernorm.weight", device=mesh_device, add_unit_offset=True)
        post_norm = _rms_weight(state, "post_attention_layernorm.weight", device=mesh_device, add_unit_offset=True)
        if layer_type == "linear_attention":
            token_mixer = _MultichipLinearAttention(
                state,
                cfg,
                device=mesh_device,
                dtype=policy.linear_attention_weight_dtype,
                policy=policy,
            )
        else:
            token_mixer = _MultichipFullAttention(
                state,
                cfg,
                device=mesh_device,
                dtype=policy.attention_weight_dtype,
                policy=policy,
            )
        moe_chunk_size = int(kwargs.get("moe_chunk_size", DEFAULT_MOE_CHUNK_SIZE))
        if moe_chunk_size <= 0:
            raise ValueError(f"moe_chunk_size must be positive, got {moe_chunk_size}")
        mlp = _MultichipQwenMoe(state, cfg, device=mesh_device, policy=policy, chunk_size=moe_chunk_size)
        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            layer_type=layer_type,
            input_layernorm_weight=input_norm,
            post_attention_layernorm_weight=post_norm,
            token_mixer=token_mixer,
            mlp=mlp,
        )

    @classmethod
    def allocate_full_attention_cache(
        cls,
        *,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int | None = None,
        block_size: int = 32,
        dtype=ttnn.bfloat16,
    ) -> QwenFullAttentionCache:
        _validate_target_mesh(mesh_device, cls.mesh_plan)
        cfg = _text_config(hf_config)
        if max_seq_len is None:
            max_seq_len = cfg.max_position_embeddings
        max_blocks_per_seq = math.ceil(max_seq_len / block_size)
        max_num_blocks = max_batch_size * max_blocks_per_seq
        local_kv_heads = cfg.num_key_value_heads // cls.mesh_plan.tensor_parallel_size
        cache_shape = (max_num_blocks, cfg.num_key_value_heads, block_size, cfg.head_dim)
        keys = _mesh_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=mesh_device.shape),
        )
        values = _mesh_tensor(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=mesh_device.shape),
        )
        if local_kv_heads != 1:
            raise ValueError("Qwen3.6 target plan expects one KV head per TP column")
        return QwenFullAttentionCache(
            keys=keys,
            values=values,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def allocate_linear_attention_state(
        cls,
        *,
        hf_config,
        mesh_device,
        batch_size: int,
        dtype=ttnn.bfloat16,
    ) -> QwenLinearAttentionState:
        _validate_target_mesh(mesh_device, cls.mesh_plan)
        cfg = _text_config(hf_config)
        tp = cls.mesh_plan.tensor_parallel_size
        local_key_heads = cfg.linear_num_key_heads // tp
        local_value_heads = cfg.linear_num_value_heads // tp
        local_key_dim = local_key_heads * cfg.linear_key_head_dim
        local_value_dim = local_value_heads * cfg.linear_value_head_dim
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
        conv_dim += cfg.linear_value_head_dim * cfg.linear_num_value_heads
        heads = batch_size * local_value_heads
        conv_state = tuple(
            _mesh_tensor(
                torch.zeros((1, 1, batch_size, conv_dim), dtype=torch.bfloat16),
                device=mesh_device,
                dtype=dtype,
                mesh_mapper=_col_mapper(mesh_device),
            )
            for _ in range(cfg.linear_conv_kernel_dim)
        )
        recurrent_state = _mesh_tensor(
            torch.zeros((1, heads * tp, cfg.linear_key_head_dim, cfg.linear_value_head_dim), dtype=torch.bfloat16),
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=mesh_device.shape),
        )
        return QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)

    def forward(
        self, hidden_states: ttnn.Tensor, *, mode: Literal["prefill", "decode"], **kwargs
    ) -> FunctionalDecoderResult:
        return super().forward(hidden_states, mode=mode, **kwargs)


__all__ = [
    "MultichipDecoder",
    "MultichipDecoderGraphSummary",
    "MultichipMeshPlan",
    "TARGET_MESH_PLAN",
    "GRAPH_SUMMARY",
    "FunctionalDecoderResult",
    "QwenFullAttentionCache",
    "QwenLinearAttentionState",
]
