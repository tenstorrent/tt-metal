# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS 120B tensor-parallel decoder on a four-chip Blackhole mesh."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.attention.config import AttentionConfig as _AttentionConfig
from models.demos.gpt_oss.tt.attention.kv_cache import get_kv_memory_config, init_kv_cache
from models.demos.gpt_oss.tt.attention.operations import apply_rope
from models.demos.gpt_oss.tt.attention.weights import load_attention_weights
from models.demos.gpt_oss.tt.attention_configs import GPTOSSAttentionProgramConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts import ExpertConfig
from models.demos.gpt_oss.tt.experts.operations import (
    apply_routing_weights,
    apply_swiglu,
    apply_tensor_parallel_allreduce,
    reduce_experts,
)
from models.demos.gpt_oss.tt.mlp import MLP
from models.demos.gpt_oss.tt.topk import TopKRouter, topk_router
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links
from models.demos.gpt_oss.utils.substate import substate
from models.demos.gpt_oss_120b_qb2.tt.attention_prefill import prefill_forward
from models.demos.gpt_oss_120b_qb2.tt.decoder import DecoderLayer, DecodeRMSNorm, _local_layer_state_dict
from models.demos.gpt_oss_120b_qb2.tt.sliding_ring import ring_modulo_for_layer
from models.tt_transformers.tt.common import PagedAttentionConfig, rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotarySetup

SUPPORTED_MESH_SHAPES = ((1, 4),)
_SUPPORTED_LAYER_TYPES = {"sliding_attention", "full_attention"}
_DOWN_SUBBLOCK_WIDTH_BY_TP = {2: 1, 4: 3}
DECODE_K_CHUNK_SIZE = GPTOSSAttentionProgramConfig().decode_k_chunk_size


@dataclass(frozen=True)
class MultichipTensorPlan:
    """Calculated logical and padded per-rank dimensions for one mesh."""

    mesh_shape: tuple[int, int]
    tp: int
    hidden_size: int
    padded_local_hidden: int
    padded_hidden_size: int
    local_intermediate_size: int
    padded_local_intermediate_size: int
    local_q_heads: int
    local_kv_heads: int
    local_qkv_width: int


@dataclass(frozen=True)
class MultichipDecoderPolicy:
    """Static multichip dtype/topology policy selected at construction."""

    name: str = "p150_tp4_sparse_experts"
    attention_weight_dtype: object = ttnn.bfloat8_b
    expert_weight_dtype: object = ttnn.bfloat4_b
    kv_cache_dtype: object = ttnn.bfloat8_b
    residual_layout: str = "replicated"
    topology: object = ttnn.Topology.Ring
    decode_fused_router: bool = True
    prefill_token_group_sparsity: bool = True
    # Decode batches above one run as a single 32-row token group through the
    # gate-selected sparse experts (union of the batch's experts) instead of a
    # per-user loop over the batch-one graph.
    decode_grouped_batch: bool = True
    # Decode batches in (1, this] gather exactly top_k*batch expert slots by
    # index (duplicates allowed) so every dense stage scales with the batch
    # instead of with the 128-expert union mask.  Larger batches use the mask.
    decode_indexed_slots_max_batch: int = 8
    # Keep the grouped-decode expert intermediates in L1 instead of DRAM.
    decode_grouped_l1: bool = True
    # Prefill: gather each expert's routed tokens into per-expert slabs and run
    # the projections in the compact indexed sparse-matmul mode, instead of a
    # dense 128-expert expanded output per 32-token group. Static expert slabs
    # keep program shapes reusable across prompts and decoder layers.
    prefill_indexed_experts: bool = True
    # Below this many tokens the packed group-sparse path is used: the indexed
    # path syncs the host once per layer to size its expert slabs, which costs
    # more than it saves on short prompts.
    prefill_indexed_min_tokens: int = 512
    prefill_sliding_q_chunk_size_large: int = 128
    prefill_sliding_k_chunk_size_large: int = 128
    prefill_full_q_chunk_size_large: int = 256
    prefill_full_k_chunk_size_large: int = 512
    decode_separate_gate_up: bool = False
    router_weight_dtype: object = ttnn.bfloat16
    normalization_weight_dtype: object = ttnn.bfloat16
    router_prefill_input_l1: bool = False
    router_prefill_explicit_program_config: bool = False
    activation_ccl_dtype: object = ttnn.bfloat16
    attention_activation_ccl_dtype: object | None = ttnn.bfloat8_b
    expert_activation_ccl_dtype: object | None = None
    projection_math_fidelity: object = ttnn.MathFidelity.LoFi
    prefill_projection_math_fidelity: object = ttnn.MathFidelity.HiFi2
    attention_sdpa_math_fidelity: object = ttnn.MathFidelity.HiFi4
    expert_math_fidelity: object = ttnn.MathFidelity.LoFi
    router_math_fidelity: object = ttnn.MathFidelity.HiFi2
    residual_dtype: object = ttnn.bfloat16
    attention_projection_input_dtype: object = ttnn.bfloat8_b
    expert_intermediate_dtype: object = ttnn.bfloat16
    expert_gate_up_cores: tuple[int, int] = (6, 8)
    expert_gate_up_in0_block_w: int = 30
    expert_gate_up_subblock_w: int = 1
    expert_down_cores: tuple[int, int] = (5, 3)
    expert_down_in0_block_w: int = 12
    expert_down_subblock_w: int | None = 6
    expert_prefill_down_cores: tuple[int, int] = (5, 9)
    expert_prefill_down_in0_block_w: int = 12
    expert_prefill_down_subblock_w: int | None = 2


DEFAULT_MULTICHIP_POLICY = MultichipDecoderPolicy()
_SWEEP_POLICY_FIELDS = {
    "name",
    "attention_weight_dtype",
    "expert_weight_dtype",
    "kv_cache_dtype",
    "router_weight_dtype",
    "normalization_weight_dtype",
    "activation_ccl_dtype",
    "attention_activation_ccl_dtype",
    "expert_activation_ccl_dtype",
    "projection_math_fidelity",
    "prefill_projection_math_fidelity",
    "attention_sdpa_math_fidelity",
    "expert_math_fidelity",
    "router_math_fidelity",
    "residual_dtype",
    "attention_projection_input_dtype",
    "expert_intermediate_dtype",
}
_SWEEP_DTYPES = {ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b}
_SWEEP_MATH_FIDELITIES = {
    ttnn.MathFidelity.LoFi,
    ttnn.MathFidelity.HiFi2,
    ttnn.MathFidelity.HiFi3,
    ttnn.MathFidelity.HiFi4,
}


def _is_supported_multichip_policy(policy: MultichipDecoderPolicy) -> bool:
    """Validate precision overrides while keeping the selected geometry."""

    for field in fields(MultichipDecoderPolicy):
        if field.name not in _SWEEP_POLICY_FIELDS and getattr(policy, field.name) != getattr(
            DEFAULT_MULTICHIP_POLICY, field.name
        ):
            return False
    dtype_fields = (
        "attention_weight_dtype",
        "expert_weight_dtype",
        "kv_cache_dtype",
        "router_weight_dtype",
        "normalization_weight_dtype",
        "activation_ccl_dtype",
        "residual_dtype",
        "attention_projection_input_dtype",
        "expert_intermediate_dtype",
    )
    optional_dtype_fields = (
        "attention_activation_ccl_dtype",
        "expert_activation_ccl_dtype",
    )
    fidelity_fields = (
        "projection_math_fidelity",
        "prefill_projection_math_fidelity",
        "attention_sdpa_math_fidelity",
        "expert_math_fidelity",
        "router_math_fidelity",
    )
    return (
        all(getattr(policy, name) in _SWEEP_DTYPES for name in dtype_fields)
        and all(
            getattr(policy, name) is None or getattr(policy, name) in _SWEEP_DTYPES for name in optional_dtype_fields
        )
        and all(getattr(policy, name) in _SWEEP_MATH_FIDELITIES for name in fidelity_fields)
    )


def tensor_plan(mesh_shape, hf_config) -> MultichipTensorPlan:
    """Return the setup-time TP shape/padding plan and reject invalid meshes."""
    shape = tuple(int(value) for value in mesh_shape)
    if shape not in SUPPORTED_MESH_SHAPES:
        raise ValueError(f"multichip decoder supports mesh shapes {SUPPORTED_MESH_SHAPES}, got {shape}")
    tp = shape[1]
    if hf_config.hidden_size != 2880 or hf_config.intermediate_size != 2880 or hf_config.head_dim != 64:
        raise ValueError("Expected openai/gpt-oss-120b hidden/intermediate/head dimensions (2880, 2880, 64)")
    if hf_config.num_attention_heads % tp or hf_config.num_key_value_heads % tp:
        raise ValueError(
            f"TP={tp} must divide Q/KV heads ({hf_config.num_attention_heads}, {hf_config.num_key_value_heads})"
        )

    hidden_size = int(hf_config.hidden_size)
    local_hidden = hidden_size // tp
    padded_local_hidden = math.ceil(local_hidden / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    local_intermediate = int(hf_config.intermediate_size) // tp
    padded_local_intermediate = math.ceil(local_intermediate / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    qkv_width = int(hf_config.num_attention_heads) * int(hf_config.head_dim) + 2 * int(
        hf_config.num_key_value_heads
    ) * int(hf_config.head_dim)
    return MultichipTensorPlan(
        mesh_shape=shape,
        tp=tp,
        hidden_size=hidden_size,
        padded_local_hidden=padded_local_hidden,
        padded_hidden_size=padded_local_hidden * tp,
        local_intermediate_size=local_intermediate,
        padded_local_intermediate_size=padded_local_intermediate,
        local_q_heads=int(hf_config.num_attention_heads) // tp,
        local_kv_heads=int(hf_config.num_key_value_heads) // tp,
        local_qkv_width=qkv_width // tp,
    )


def _allreduce_physical_hidden(
    tensor,
    *,
    hidden_size,
    padded_hidden_size,
    mesh_config,
    ccl_manager,
    memory_config=None,
):
    """Reduce a tile-divisible physical width, then restore logical width.

    TP4's logical 2880-wide residual has 90 tiles, which cannot be evenly
    reduce-scattered over four ranks.  Its natural per-rank tile padding gives
    2944 = 4 * 736 columns (92 tiles), selecting the native ring RS+AG path.
    Padding is produced by the row-parallel projection weights rather than by
    the public activation contract.
    """
    if tensor.shape[-1] != padded_hidden_size:
        raise ValueError(f"physical-hidden collective expected width {padded_hidden_size}, got {tensor.shape[-1]}")
    kwargs = {}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    reduced = ttnn.all_reduce(
        tensor,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
        **kwargs,
    )
    tensor.deallocate(True)
    if padded_hidden_size == hidden_size:
        return reduced
    logical = ttnn.slice(
        reduced,
        starts=[0] * len(reduced.shape),
        ends=[
            *[int(reduced.shape[index]) for index in range(len(reduced.shape) - 1)],
            hidden_size,
        ],
        steps=[1] * len(reduced.shape),
    )
    reduced.deallocate(True)
    return logical


@dataclass
class AttentionConfig(_AttentionConfig):
    cache_position_modulo: int | None = None


class _PhysicalHiddenCollectiveAttention:
    """Chunked prefill and decode attention with TP4's 2944-column reduction."""

    def __init__(
        self,
        *,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        layer_idx,
        paged_attention_config,
        transformation_mats,
        weight_dtype,
        cache_dtype,
        prefill_projection_input_dtype,
        prefill_projection_compute_kernel_config,
        tensor_cache_path,
        create_kv_cache,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.paged_attention_config = paged_attention_config
        self.transformation_mats = transformation_mats
        self.cache_dtype = cache_dtype
        self.prefill_projection_input_dtype = prefill_projection_input_dtype
        self.prefill_projection_compute_kernel_config = prefill_projection_compute_kernel_config
        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )
        self.kv_cache = (
            init_kv_cache(
                mesh_device=mesh_device,
                config=config,
                mesh_config=mesh_config,
                paged_attention_config=paged_attention_config,
                cache_dtype=cache_dtype,
                tensor_cache_path=tensor_cache_path,
            )
            if create_kv_cache
            else None
        )
        self.layer_past = self.kv_cache
        self.kv_mem_cfg = get_kv_memory_config(
            mesh_device, config.max_local_batch_size, mesh_config.shard_size(config.num_kv_heads), config.head_dim
        )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.scaling

    def __call__(
        self,
        hidden_states,
        rope_mats,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
        batch_size=1,
        fill_seq_lens=None,
        chunk_start_idx=None,
        ring_tail_block=None,
        fill_start_idx=None,
    ):
        cache = kv_cache if kv_cache is not None else self.kv_cache
        if not is_decode:
            return prefill_forward(
                hidden_states=hidden_states,
                rope_mats=rope_mats,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                program_config=self.program_config,
                transformation_mat=self.transformation_mats["prefill"],
                position_idx=position_idx,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                user_id=user_id,
                batch_size=batch_size,
                projection_input_dtype=self.prefill_projection_input_dtype,
                projection_compute_kernel_config=self.prefill_projection_compute_kernel_config,
                fill_seq_lens=fill_seq_lens,
                chunk_start_idx=chunk_start_idx,
                ring_tail_block=ring_tail_block,
                fill_start_idx=fill_start_idx,
            )
        transformation_mat = self.transformation_mats["decode"] if self.transformation_mats else None
        return self._decode_forward(
            hidden_states,
            rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=cache,
            transformation_mat=transformation_mat,
        )

    def _decode_forward(
        self,
        hidden_states,
        rope_mats,
        *,
        position_idx,
        page_table,
        kv_cache,
        transformation_mat,
    ):
        """Canonical GPT-OSS decode with only the output-collective tail changed."""
        _, seq_len, batch_size, hidden_size = hidden_states.shape
        if seq_len != 1:
            raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

        xqkv_fused = ttnn.linear(
            hidden_states,
            self.weights.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.decode_projection_compute_kernel_config,
        )
        ttnn.add(xqkv_fused, self.weights.wqkv_bias, output_tensor=xqkv_fused)

        num_local_heads = self.mesh_config.shard_size(self.config.num_heads)
        num_local_kv_heads = self.mesh_config.shard_size(self.config.num_kv_heads)
        tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=num_local_heads,
            num_kv_heads=num_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv_fused.deallocate(True)

        # rotary_embedding_llama pairs user shards by physical core.  On
        # Blackhole, head creation follows the native 11-wide worker grid,
        # while RotarySetup deliberately uses its established 8x8 grid at
        # B32.  Reshard Q/K to the transformation matrix grid before RoPE so
        # logical user rows do not alias across the two physical orderings.
        rope_grid = transformation_mat.memory_config().shard_spec.grid
        rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.config.head_dim),
            core_grid=rope_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if tt_q.memory_config() != rope_memory_config:
            split_q = tt_q
            split_k = tt_k
            tt_q = ttnn.to_memory_config(split_q, rope_memory_config)
            tt_k = ttnn.to_memory_config(split_k, rope_memory_config)
            split_q.deallocate(True)
            split_k.deallocate(True)

        tt_q_orig = tt_q
        tt_k_orig = tt_k
        tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
        tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
        tt_q_orig.deallocate(True)
        tt_k_orig.deallocate(True)

        k_cache, v_cache = kv_cache
        tt_k = ttnn.to_memory_config(tt_k, self.kv_mem_cfg)
        tt_v = ttnn.to_memory_config(tt_v, self.kv_mem_cfg)
        ring_kwargs = (
            {"cache_position_modulo": self.config.cache_position_modulo}
            if self.config.cache_position_modulo is not None and page_table is not None
            else {}
        )
        ttnn.experimental.paged_update_cache(
            k_cache,
            tt_k,
            update_idxs_tensor=position_idx,
            page_table=page_table,
            **ring_kwargs,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            tt_v,
            update_idxs_tensor=position_idx,
            page_table=page_table,
            **ring_kwargs,
        )
        tt_k.deallocate(True)
        tt_v.deallocate(True)

        grid_size = ttnn.CoreCoord(8, 8)
        batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)
        padded_heads = math.ceil(num_local_heads / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        height_sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=(padded_heads, self.config.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if page_table is not None:
            tt_sdpa_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_q,
                k_cache,
                v_cache,
                cur_pos_tensor=position_idx,
                sliding_window_size=self.config.sliding_window,
                attention_sink=self.weights.decode_sinks,
                page_table_tensor=page_table,
                scale=self.config.scaling,
                program_config=self.program_config.get_decode_sdpa_config(self.mesh_device),
                compute_kernel_config=self.program_config.get_compute_kernel_config(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **ring_kwargs,
            )
        else:
            tt_sdpa_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_q,
                k_cache,
                v_cache,
                cur_pos_tensor=position_idx,
                sliding_window_size=self.config.sliding_window,
                attention_sink=self.weights.decode_sinks,
                scale=self.config.scaling,
                program_config=self.program_config.get_decode_sdpa_config(self.mesh_device),
                compute_kernel_config=self.program_config.get_compute_kernel_config(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
        tt_q.deallocate(True)

        tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
        tt_sdpa_tensor.deallocate(True)

        tt_out = ttnn.linear(
            tt_sdpa_out,
            self.weights.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.decode_projection_compute_kernel_config,
        )
        tt_sdpa_out.deallocate(True)
        tt_out = ttnn.add(tt_out, self.weights.o_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        if tt_out.dtype != self.activation_ccl_dtype:
            projection_output = tt_out
            tt_out = ttnn.typecast(projection_output, self.activation_ccl_dtype)
            projection_output.deallocate(True)

        tt_out = ttnn.reshape(
            tt_out,
            (1, 1, batch_size, self.decode_output_physical_hidden),
            (1, 1, ttnn.TILE_SIZE, self.decode_output_physical_hidden),
        )
        output = _allreduce_physical_hidden(
            tt_out,
            hidden_size=hidden_size,
            padded_hidden_size=self.decode_output_physical_hidden,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # BFP4 is legal for the attention projection and ring payload but is
        # not a legal input dtype for the following sharded RMSNorm.  Restore
        # only at that consumer boundary so the lower-movement CCL candidate
        # is measured with its intended payload instead of rejected at the
        # first API validation error.
        if output.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(output, self.residual_dtype)
            output.deallocate(True)
            output = converted
        return output


class _ReplicatedL1Router(TopKRouter):
    """Replicated BF16 router with setup-time L1-resident weights."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        tensor_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        *,
        prefill_input_l1=False,
        prefill_explicit_program_config=False,
        decode_fused_router=False,
    ):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.tensor_cache_path = tensor_cache_path
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight_l1_replicated"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            state_dict["bias"].unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias_l1_replicated"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.prefill_input_l1 = prefill_input_l1
        self.prefill_program_config = (
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(4, 4),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=1,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            if prefill_explicit_program_config
            else None
        )
        self.softmax_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # The fused router consumes one physical 32-row tile. Decode tensors
        # already own that physical tile even when their logical batch is
        # smaller; the operation preserves the sub-tile logical batch.
        self.use_fused_op = bool(decode_fused_router and self.num_experts == 128 and weight_dtype == ttnn.bfloat16)
        self._fused_bias = None
        self._fused_biases = {}
        self._dense_zero_templates = {}
        self._bias_torch = state_dict["bias"].unsqueeze(0).to(torch.bfloat16) if self.use_fused_op else None

    def _fused_call(self, hidden_states, use_throughput_experts):
        """Select the [batch, experts] bias tile that matches this decode width.

        The fused kernel validates ``bias.shape[0] == batch``.  One layer serves
        every decode bucket (1, 8, 32, ...), so keep one bias tensor per width;
        each is created on the first untraced call for that width.
        """
        batch_size = int(hidden_states.shape[0])
        self._fused_bias = self._fused_biases.get(batch_size)
        if self._fused_bias is None:
            self._init_fused_op(hidden_states.device(), batch_size)
            self._fused_biases[batch_size] = self._fused_bias
        if not use_throughput_experts:
            raise ValueError("fused routing requires sparse expert output")
        router_input = (
            ttnn.typecast(hidden_states, ttnn.bfloat16) if hidden_states.dtype != ttnn.bfloat16 else hidden_states
        )
        result = ttnn.experimental.topk_router_gpt(
            router_input,
            weight_tensor=self.weight,
            bias_tensor=self._fused_bias,
            k=self.top_k,
            num_experts=self.num_experts,
        )
        if router_input is not hidden_states:
            router_input.deallocate(True)
        return result

    def dense_decode_routing(self, hidden_states):
        """Return [batch, experts] routing weights, positive only at each row's top-k.

        Uses the same router path as batch-1 decode (fused when enabled) so the
        selected experts match the single-user graph exactly.
        """
        expert_indices, routing_scores = self(hidden_states, True)
        return self.scatter_dense_routing(expert_indices, routing_scores, deallocate=True)

    def scatter_dense_routing(self, expert_indices, routing_scores, *, deallocate):
        """Scatter [batch, top_k] indices/scores into a dense [batch, experts] row per user."""
        if expert_indices.layout != ttnn.TILE_LAYOUT:
            indices_tiled = ttnn.to_layout(expert_indices, ttnn.TILE_LAYOUT)
        else:
            indices_tiled = expert_indices
        if routing_scores.layout != ttnn.TILE_LAYOUT:
            scores_tiled = ttnn.to_layout(routing_scores, ttnn.TILE_LAYOUT)
        else:
            scores_tiled = routing_scores
        batch_size = int(indices_tiled.shape[0])
        template = self._dense_zero_templates.get(batch_size)
        if template is None:
            # Created once per width on the first untraced call; trace replays
            # only see the device-side clone below.
            template = ttnn.from_torch(
                torch.zeros((batch_size, self.num_experts), dtype=torch.bfloat16),
                device=indices_tiled.device(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(indices_tiled.device()),
            )
            self._dense_zero_templates[batch_size] = template
        dense = ttnn.clone(template, memory_config=ttnn.L1_MEMORY_CONFIG)
        routing_weights = ttnn.scatter(dense, dim=1, index=indices_tiled, src=scores_tiled)
        if routing_weights is not dense:
            dense.deallocate(True)
        if indices_tiled is not expert_indices:
            indices_tiled.deallocate(True)
        if scores_tiled is not routing_scores:
            scores_tiled.deallocate(True)
        if deallocate:
            expert_indices.deallocate(True)
            routing_scores.deallocate(True)
        return routing_weights

    def prefill_logits(self, hidden_states):
        """Return the [tokens, experts] router logits for a prefill sequence."""
        actual_tokens = hidden_states.volume() // self.hidden_dim
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
        router_input = hidden_states
        if self.prefill_input_l1:
            router_input = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        router_logits = ttnn.linear(
            router_input,
            self.weight,
            bias=self.bias,
            memory_config=(ttnn.L1_MEMORY_CONFIG if actual_tokens <= 128 else ttnn.DRAM_MEMORY_CONFIG),
            program_config=self.prefill_program_config,
            compute_kernel_config=self.compute_config,
        )
        if router_input is not hidden_states:
            router_input.deallocate(True)
        return router_logits

    def __call__(self, hidden_states, use_throughput_experts):
        """Apply the opt-in prefill placement/config while preserving decode."""
        actual_tokens = hidden_states.volume() // self.hidden_dim
        if actual_tokens <= ttnn.TILE_SIZE:
            if self.use_fused_op and use_throughput_experts:
                hidden_2d = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
                return self._fused_call(
                    hidden_2d,
                    use_throughput_experts=True,
                )
            return super().__call__(hidden_states, use_throughput_experts)

        router_logits = self.prefill_logits(hidden_states)
        expert_indices, expert_weights = topk_router(
            router_logits,
            self.top_k,
            use_throughput_experts,
            self.softmax_compute_config,
        )
        router_logits.deallocate(True)
        return expert_indices, expert_weights


@dataclass
class _PackedTPExpertsRuntime:
    """Weight-free state shared by the packed TP expert paths."""

    config: ExpertConfig
    program_config: GPTOSSProgramConfig
    prefill_sparsity: object


class _ActiveExpertTPMLP(MLP):
    """Packed TP sparse experts with batch-safe decode at the autoport boundary."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        *,
        tensor_cache_path,
        mesh_config,
        expert_weight_dtype,
        router_weight_dtype=ttnn.bfloat16,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        router_math_fidelity=ttnn.MathFidelity.HiFi2,
        expert_intermediate_dtype=ttnn.bfloat16,
        router_prefill_input_l1=False,
        router_prefill_explicit_program_config=False,
        decode_fused_router=False,
        prefill_token_group_sparsity=False,
        grouped_decode_batch=False,
        indexed_slots_max_batch=0,
        grouped_decode_l1=False,
        indexed_prefill=False,
        indexed_prefill_min_tokens=512,
        activation_ccl_dtype=ttnn.bfloat8_b,
        separate_gate_up=False,
        gate_up_cores=(3, 4),
        gate_up_in0_block_w=30,
        gate_up_subblock_w=1,
        down_cores=(5, 6),
        down_in0_block_w=12,
        down_subblock_w=None,
        prefill_down_cores=(5, 9),
        prefill_down_in0_block_w=12,
        prefill_down_subblock_w=2,
    ):
        # This implementation has its own packed TP weight layout for decode
        # and prefill.  Constructing MLP/Experts first would load the generic
        # six-tensor expert layout only to deallocate it immediately, doubling
        # setup-time host/device pressure for a 120B checkpoint.
        self.use_throughput_experts = False
        self.router = _ReplicatedL1Router(
            mesh_device,
            hf_config,
            substate(state_dict, "router"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
            weight_dtype=router_weight_dtype,
            math_fidelity=router_math_fidelity,
            prefill_input_l1=router_prefill_input_l1,
            prefill_explicit_program_config=router_prefill_explicit_program_config,
            decode_fused_router=decode_fused_router,
        )
        # Geometry is a policy because the legal subblock width depends on the
        # chosen core count.  Decode uses 45-core gate/up and 15-core down on
        # both meshes.  Prefill uses 45-core down on both meshes; measuring
        # 15-, 30-, and 45-core candidates through the full layer selected it.
        down_subblock_w = (
            _DOWN_SUBBLOCK_WIDTH_BY_TP[mesh_config.decode.tp] if down_subblock_w is None else down_subblock_w
        )
        prefill_down_subblock_w = (
            _DOWN_SUBBLOCK_WIDTH_BY_TP[mesh_config.prefill.tp]
            if prefill_down_subblock_w is None
            else prefill_down_subblock_w
        )
        program_config = GPTOSSProgramConfig(
            decode_gate_up_cores=gate_up_cores,
            decode_gate_up_in0_block_w=gate_up_in0_block_w,
            decode_gate_up_subblock_w=gate_up_subblock_w,
            decode_down_cores=down_cores,
            decode_down_in0_block_w=down_in0_block_w,
            decode_down_subblock_w=down_subblock_w,
            prefill_gate_up_cores=gate_up_cores,
            prefill_gate_up_in0_block_w=gate_up_in0_block_w,
            prefill_gate_up_subblock_w=gate_up_subblock_w,
            prefill_down_cores=prefill_down_cores,
            prefill_down_in0_block_w=prefill_down_in0_block_w,
            prefill_down_subblock_w=prefill_down_subblock_w,
        )
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = int(hf_config.hidden_size)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.local_intermediate_size = self.intermediate_size // mesh_config.decode.tp
        # Pad each rank's intermediate slice to a multiple of 64 so the packed
        # [gate | up] width is an even number of tiles that fills a rectangular
        # core grid exactly (the sparse matmul requires every grid core to have
        # work): 720 -> 768 gives 48 output tiles for a 6x8 grid.
        self.padded_local_intermediate_size = math.ceil(self.local_intermediate_size / (2 * ttnn.TILE_SIZE)) * (
            2 * ttnn.TILE_SIZE
        )
        self.num_experts = int(hf_config.num_local_experts)
        self.top_k = int(hf_config.num_experts_per_tok)
        expert_config = ExpertConfig(
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            num_experts_per_tok=self.top_k,
            swiglu_limit=hf_config.swiglu_limit,
        )
        prefill_ep = mesh_config.prefill.ep
        experts_per_ep = self.num_experts // prefill_ep
        prefill_sparsity_host = torch.zeros(1, 1, prefill_ep, self.num_experts)
        for ep_rank in range(prefill_ep):
            start = ep_rank * experts_per_ep
            prefill_sparsity_host[:, :, ep_rank, start : start + experts_per_ep] = 1
        prefill_sparsity = ttnn.from_torch(
            prefill_sparsity_host,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                dims=(-2, None) if prefill_ep > 1 else (None, None),
                mesh_shape=mesh_device.shape,
                mesh_device=mesh_device,
            ),
        )
        self.experts = _PackedTPExpertsRuntime(
            config=expert_config,
            program_config=program_config,
            prefill_sparsity=prefill_sparsity,
        )
        self.expert_weight_dtype = expert_weight_dtype
        self.activation_ccl_dtype = activation_ccl_dtype
        self.expert_intermediate_dtype = expert_intermediate_dtype
        self.expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=expert_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.separate_gate_up = separate_gate_up
        self.prefill_token_group_sparsity = prefill_token_group_sparsity
        self.grouped_decode_batch = grouped_decode_batch
        self.indexed_slots_max_batch = int(indexed_slots_max_batch)
        self.grouped_decode_l1 = bool(grouped_decode_l1)
        self.indexed_prefill = bool(indexed_prefill)
        self.indexed_prefill_min_tokens = int(indexed_prefill_min_tokens)
        # Longest prompt the indexed path runs as one piece.  Its expert-output
        # arena and gather-back table are each ~5.8 KB per routed slot, so the
        # 131072-token bucket (4.5 GB arena) does not fit beside the weights and
        # KV cache; longer prompts run in equal row chunks (65536 is the longest
        # sweep row measured in serving).
        self.indexed_prefill_chunk_tokens = int(os.environ.get("GPT_OSS_120B_INDEXED_PREFILL_CHUNK", "65536"))
        # Gate/up bias added inside the sparse matmul (ttnn.sparse_matmul(bias=...), per-group tile row)
        # instead of a per-group gather + add; needs a ttnn build whose sparse_matmul accepts `bias`.
        self.indexed_prefill_fused_bias = os.environ.get(
            "GPT_OSS_120B_INDEXED_FUSED_BIAS", "1"
        ) == "1" and "bias (ttnn.Tensor, optional)" in (ttnn.sparse_matmul.__doc__ or "")
        # Indexed prefill matmul blocking: (in0_block_w, out_block_h,
        # out_subblock_h, out_subblock_w) in tiles.  out_block_h > 1 makes the
        # kernel reuse each weight block across several slab tile rows instead
        # of re-reading it per row.  The in0 circular buffer is out_block_h *
        # in0_block_w tiles, double buffered (2 KB bf16 tiles): keep the whole
        # CB set near 300 KB, since the serving process holds ~600 KB of
        # resident L1 buffers per core and a (24, 8, 4, 2) down config
        # (~960 KB of CBs) clashed with them at decode warmup.
        self.indexed_prefill_gate_up_blocking = (15, 4, 2, 1)
        self.indexed_prefill_down_blocking = (12, 4, 2, 2)
        # Optional narrower dtype for the slab fed to the gate/up matmul (the
        # single in0 multicast sender is the matmul's bottleneck).
        self.indexed_prefill_slab_dtype = None
        self._prefill_constants = {}
        # Optional dict; when set, the indexed prefill stores host copies of its
        # routing intermediates for offline checking (untraced path only).
        self._prefill_debug = None
        # Optional dict; when set, the indexed prefill synchronizes at stage
        # boundaries and records elapsed seconds per stage (test/diagnostics).
        self._prefill_timing = None
        self._prefill_stats = None
        self._prefill_program_growth = None
        self._slot_selectors = {}
        self._load_indexed_decode_weights(
            substate(state_dict, "experts"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "indexed_decode"),
        )
        self.decode_uses_gate_selected_sparse_experts = True

    def _load_indexed_decode_weights(self, expert_state, *, tensor_cache_path):
        """Load a compact top-k decode representation with TP-sharded weights.

        Every rank's slice of the expert intermediate dimension is zero-padded
        from ``local_intermediate_size`` (720 at TP4) to the next tile multiple
        (768) on the host.  The packed gate/up halves and the down projection's
        contraction rows then sit on tile boundaries, so the per-rank gate/up
        split is a tile-aligned slice instead of an untilize/retilize fallback.
        The padding columns of gate and up are zero, SwiGLU maps them to zero,
        and the matching down rows are zero, so results are unchanged.
        """
        tp = self.mesh_config.decode.tp
        local = self.local_intermediate_size
        padded = self.padded_local_intermediate_size
        pad = padded - local
        gate = expert_state["gate_up_proj"][..., ::2].reshape(
            1, self.num_experts, self.hidden_size, self.intermediate_size
        )
        up = expert_state["gate_up_proj"][..., 1::2].reshape(
            1, self.num_experts, self.hidden_size, self.intermediate_size
        )
        gate_bias = expert_state["gate_up_proj_bias"][..., ::2].reshape(self.num_experts, self.intermediate_size)
        up_bias = expert_state["gate_up_proj_bias"][..., 1::2].reshape(self.num_experts, self.intermediate_size)

        def rank_slices(tensor):
            return [
                torch.nn.functional.pad(tensor[..., rank * local : (rank + 1) * local], (0, pad)) for rank in range(tp)
            ]

        gate_ranks = rank_slices(gate)
        up_ranks = rank_slices(up)
        gate_bias_ranks = rank_slices(gate_bias)
        up_bias_ranks = rank_slices(up_bias)
        # Arrange [gate_rank, up_rank] chunks consecutively.  Sharding the
        # resulting last dimension then gives every rank both operands for its
        # local SwiGLU instead of assigning whole gate/up halves to ranks.
        packed_gate_up = torch.cat(
            [torch.cat((gate_ranks[rank], up_ranks[rank]), dim=-1) for rank in range(tp)],
            dim=-1,
        )
        packed_gate_up_bias = torch.cat(
            [torch.cat((gate_bias_ranks[rank], up_bias_ranks[rank]), dim=-1) for rank in range(tp)],
            dim=-1,
        )
        column_mapper = self.mesh_config.column_parallel(self.mesh_device)
        row_mapper = self.mesh_config.row_parallel(self.mesh_device)
        suffix = f"_pad{padded}"
        if self.separate_gate_up:
            padded_gate = torch.cat(gate_ranks, dim=-1)
            padded_up = torch.cat(up_ranks, dim=-1)
            padded_gate_bias = torch.cat(gate_bias_ranks, dim=-1)
            padded_up_bias = torch.cat(up_bias_ranks, dim=-1)
            self.indexed_gate = ttnn.as_tensor(
                padded_gate,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.expert_weight_dtype,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_gate" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_up = ttnn.as_tensor(
                padded_up,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.expert_weight_dtype,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_up" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_gate_bias = ttnn.as_tensor(
                padded_gate_bias,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_gate_bias" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_up_bias = ttnn.as_tensor(
                padded_up_bias,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "separate_up_bias" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # [experts, 1, width] so the prefill bias add broadcasts over rows.
            self.prefill_gate_bias = ttnn.as_tensor(
                padded_gate_bias.unsqueeze(1),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "prefill_separate_gate_bias_t" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.prefill_up_bias = ttnn.as_tensor(
                padded_up_bias.unsqueeze(1),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "prefill_separate_up_bias_t" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_gate_up = None
            self.indexed_gate_up_bias = None
            self.indexed_gate_up_bias_tiled = None
            self.prefill_gate_up_bias = None
        else:
            self.indexed_gate_up = ttnn.as_tensor(
                packed_gate_up,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.expert_weight_dtype,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "packed_gate_up" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.indexed_gate_up_bias = ttnn.as_tensor(
                packed_gate_up_bias,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "packed_gate_up_bias" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Stored as [experts, 1, width] so the prefill/grouped-decode bias add
            # broadcasts over the token rows without a per-call transpose.
            self.prefill_gate_up_bias = ttnn.as_tensor(
                packed_gate_up_bias.unsqueeze(1),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "prefill_packed_gate_up_bias_t" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Per-group fused bias for the indexed sparse matmul: one 32-row tile block per expert with
            # the [1, width] bias in row 0 (the kernel adds tile row `expert` row-broadcast over the
            # slab rows), so ttnn.sparse_matmul(bias=...) replaces the per-group gather + add.
            fused_bias = torch.zeros(
                self.num_experts, ttnn.TILE_SIZE, packed_gate_up_bias.shape[-1], dtype=packed_gate_up_bias.dtype
            )
            fused_bias[:, 0, :] = packed_gate_up_bias
            self.indexed_gate_up_bias_tiled = ttnn.as_tensor(
                fused_bias,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=column_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "packed_gate_up_bias_tilerow" + suffix),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        down = expert_state["down_proj"].reshape(
            1,
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
        )
        padded_down = torch.cat(
            [
                torch.nn.functional.pad(down[:, :, rank * local : (rank + 1) * local, :], (0, 0, 0, pad))
                for rank in range(tp)
            ],
            dim=-2,
        )
        self.indexed_down = ttnn.as_tensor(
            padded_down,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.expert_weight_dtype,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down" + suffix),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_bias = expert_state["down_proj_bias"].reshape(self.num_experts, self.hidden_size)
        down_bias = torch.cat([down_bias] + [torch.zeros_like(down_bias)] * (tp - 1), dim=-1)
        self.indexed_down_bias = ttnn.as_tensor(
            down_bias,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=column_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # [experts, hidden] tile layout for folding the down bias through the
        # dense routing weights with one small matmul (prefill and batched decode).
        self.grouped_down_bias = ttnn.as_tensor(
            down_bias,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=column_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down_bias_tiled"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.indexed_unused_sparsity = ttnn.as_tensor(
            torch.zeros((1, 1, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "unused_sparsity"),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _run_indexed_decode(self, hidden_states):
        """Run gate-selected top-4 TP experts without materializing 128 outputs."""
        expert_indices, routing_scores = self.router(hidden_states, True)
        if expert_indices.layout == ttnn.ROW_MAJOR_LAYOUT:
            expert_indices_rm = expert_indices
        else:
            expert_indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
            expert_indices.deallocate(True)
        expert_indices_rm = ttnn.reshape(expert_indices_rm, (1, 1, 1, self.top_k))
        embedding_indices = ttnn.typecast(expert_indices_rm, ttnn.uint32)
        output_tile = ttnn.Tile([32, 32])

        if self.separate_gate_up:
            projections = []
            for weight, bias in (
                (self.indexed_gate, self.indexed_gate_bias),
                (self.indexed_up, self.indexed_up_bias),
            ):
                projected = ttnn.sparse_matmul(
                    hidden_states,
                    weight,
                    sparsity=self.indexed_unused_sparsity,
                    indices=expert_indices_rm,
                    is_input_b_sparse=True,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    output_tile=output_tile,
                    program_config=self.experts.program_config.get_decode_gate_up_config(
                        hidden_states.shape[2],
                        weight.shape[3],
                        k=hidden_states.shape[-1],
                    ),
                    compute_kernel_config=self.expert_compute_kernel_config,
                    dtype=self.expert_intermediate_dtype,
                )
                projected = ttnn.reshape(projected, (1, self.top_k, self.padded_local_intermediate_size))
                projected_bias = ttnn.embedding(
                    embedding_indices,
                    bias,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                projected = ttnn.add(projected, projected_bias, output_tensor=projected)
                projected_bias.deallocate(True)
                projections.append(projected)
            gate, up = projections
        else:
            gate_up = ttnn.sparse_matmul(
                hidden_states,
                self.indexed_gate_up,
                sparsity=self.indexed_unused_sparsity,
                indices=expert_indices_rm,
                is_input_b_sparse=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=self.experts.program_config.get_decode_gate_up_config(
                    hidden_states.shape[2],
                    self.indexed_gate_up.shape[3],
                    k=hidden_states.shape[-1],
                ),
                compute_kernel_config=self.expert_compute_kernel_config,
                dtype=self.expert_intermediate_dtype,
            )
            gate_up = ttnn.reshape(gate_up, (1, self.top_k, 2 * self.padded_local_intermediate_size))
            gate_up_bias = ttnn.embedding(
                embedding_indices,
                self.indexed_gate_up_bias,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
            gate_up_bias.deallocate(True)
            if gate_up.dtype == ttnn.bfloat4_b:
                converted = ttnn.typecast(gate_up, self.expert_intermediate_dtype)
                gate_up.deallocate(True)
                gate_up = converted
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0],
                [1, self.top_k, self.padded_local_intermediate_size],
                [1, 1, 1],
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, self.padded_local_intermediate_size],
                [1, self.top_k, 2 * self.padded_local_intermediate_size],
                [1, 1, 1],
            )
            gate_up.deallocate(True)
        down_input = apply_swiglu(gate, up, self.experts.config)
        down_input = ttnn.reshape(down_input, (1, self.top_k, 1, self.padded_local_intermediate_size))
        down = ttnn.sparse_matmul(
            down_input,
            self.indexed_down,
            sparsity=self.indexed_unused_sparsity,
            indices=expert_indices_rm,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self.experts.program_config.get_decode_down_config(
                down_input.shape[2],
                self.indexed_down.shape[-1],
                k=down_input.shape[-1],
            ),
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.expert_intermediate_dtype,
        )
        down_input.deallocate(True)
        expert_indices_rm.deallocate(True)
        output = ttnn.reshape(down, (1, self.top_k, self.hidden_size))
        down_bias = ttnn.embedding(
            embedding_indices,
            self.indexed_down_bias,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        embedding_indices.deallocate(True)
        output = ttnn.add(output, down_bias, output_tensor=output)
        down_bias.deallocate(True)
        if routing_scores.layout == ttnn.ROW_MAJOR_LAYOUT:
            routing_scores_rm = routing_scores
        else:
            routing_scores_rm = ttnn.to_layout(routing_scores, ttnn.ROW_MAJOR_LAYOUT)
            routing_scores.deallocate(True)
        routing_scores_rm = ttnn.reshape(routing_scores_rm, (1, self.top_k, 1))
        output = ttnn.mul(output, routing_scores_rm, output_tensor=output)
        routing_scores_rm.deallocate(True)
        if output.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(output, self.expert_intermediate_dtype)
            output.deallocate(True)
            output = converted
        output = ttnn.sum(output, dim=1)
        output = ttnn.unsqueeze_to_4D(output)
        output = ttnn.unsqueeze_to_4D(output)
        if output.dtype != self.activation_ccl_dtype:
            converted = ttnn.typecast(output, self.activation_ccl_dtype)
            output.deallocate(True)
            output = converted
        output = apply_tensor_parallel_allreduce(
            output,
            self.mesh_config,
            self.mesh_device,
            1,
            self.ccl_manager,
        )
        return ttnn.reshape(
            output,
            (1, 1, 1, self.hidden_size),
            (1, 1, ttnn.TILE_SIZE, self.hidden_size),
        )

    def _process_packed_prefill_chunk(self, hidden_states, routing_weights):
        """Run one tile-aligned prefill chunk through the shared packed weights."""
        _, batch_size, sequence_length, _ = hidden_states.shape
        if batch_size != 1 or sequence_length % ttnn.TILE_SIZE:
            raise ValueError("packed TP expert prefill requires batch 1 and tile-aligned internal chunks")
        groups = sequence_length // ttnn.TILE_SIZE
        hidden_4d = ttnn.reshape(hidden_states, (1, groups, ttnn.TILE_SIZE, self.hidden_size))
        if self.prefill_token_group_sparsity:
            # The sparse matmul consumes one expert mask per 32-token group.
            # Router weights are positive only at the selected top-k entries,
            # so a reduction over the group produces exactly the union of
            # experts needed by those tokens. This skips unused expert weight
            # matmuls without changing the dense downstream routing contract.
            grouped_routing = ttnn.reshape(
                routing_weights,
                (1, groups, ttnn.TILE_SIZE, self.num_experts),
            )
            group_sums = ttnn.sum(grouped_routing, dim=2, keepdim=True)
            group_sums_transposed = ttnn.permute(group_sums, (0, 2, 1, 3))
            sparsity = ttnn.to_layout(group_sums_transposed, ttnn.ROW_MAJOR_LAYOUT)
            group_sums.deallocate(True)
            gate_up_sparse_kwargs = {}
        else:
            sparsity = ttnn.repeat(self.experts.prefill_sparsity, (1, 1, groups, 1))
            gate_up_sparse_kwargs = {"nnz": self.num_experts * groups}
        output_tile = ttnn.Tile([32, 32])
        if self.separate_gate_up:
            projections = []
            for weight, bias in (
                (self.indexed_gate, self.prefill_gate_bias),
                (self.indexed_up, self.prefill_up_bias),
            ):
                projected = ttnn.sparse_matmul(
                    hidden_4d,
                    weight,
                    sparsity=sparsity,
                    **gate_up_sparse_kwargs,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=output_tile,
                    program_config=self.experts.program_config.get_prefill_gate_up_config(
                        hidden_4d.shape[2],
                        weight.shape[3],
                        k=hidden_4d.shape[-1],
                    ),
                    compute_kernel_config=self.expert_compute_kernel_config,
                    dtype=self.expert_intermediate_dtype,
                )
                if groups > 1:
                    projected = ttnn.transpose(projected, 1, 3)
                projected = ttnn.reshape(
                    projected,
                    (
                        batch_size,
                        self.num_experts,
                        sequence_length,
                        self.padded_local_intermediate_size,
                    ),
                )
                projected_bias = bias
                projected = ttnn.add(projected, projected_bias, output_tensor=projected)
                projections.append(projected)
            gate, up = projections
        else:
            gate_up = ttnn.sparse_matmul(
                hidden_4d,
                self.indexed_gate_up,
                sparsity=sparsity,
                **gate_up_sparse_kwargs,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=self.experts.program_config.get_prefill_gate_up_config(
                    hidden_4d.shape[2],
                    self.indexed_gate_up.shape[3],
                    k=hidden_4d.shape[-1],
                ),
                compute_kernel_config=self.expert_compute_kernel_config,
                dtype=self.expert_intermediate_dtype,
            )
            # The sparse output is [1, groups, 1, experts, rows, width]; with one
            # group the expert-major reshape is a view, so skip the transpose.
            if groups > 1:
                gate_up = ttnn.transpose(gate_up, 1, 3)
            gate_up = ttnn.reshape(
                gate_up,
                (
                    batch_size,
                    self.num_experts,
                    sequence_length,
                    2 * self.padded_local_intermediate_size,
                ),
            )
            gate_up_bias = self.prefill_gate_up_bias
            gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
            # TP4 owns 720 intermediate elements per rank.  That logical split is
            # intentionally not tile aligned, and slice's internal untilize cannot
            # produce a row-major BFP4 tensor.  Keep the public/logical shape and
            # pay the explicit conversion in the BFP4 experiment instead of
            # rejecting the lower-precision family at its first API boundary.
            if gate_up.dtype == ttnn.bfloat4_b:
                converted = ttnn.typecast(gate_up, self.expert_intermediate_dtype)
                gate_up.deallocate(True)
                gate_up = converted
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [
                    batch_size,
                    self.num_experts,
                    sequence_length,
                    self.padded_local_intermediate_size,
                ],
                [1, 1, 1, 1],
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.padded_local_intermediate_size],
                [
                    batch_size,
                    self.num_experts,
                    sequence_length,
                    2 * self.padded_local_intermediate_size,
                ],
                [1, 1, 1, 1],
            )
            gate_up.deallocate(True)
        down_input = apply_swiglu(gate, up, self.experts.config)
        down_input = ttnn.reshape(
            down_input,
            (1, self.num_experts, sequence_length, self.padded_local_intermediate_size),
        )

        # Down bias through the dense routing weights: one small [S, E] x [E, H] matmul.
        bias_term = ttnn.matmul(
            routing_weights,
            self.grouped_down_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.router.compute_config,
        )
        bias_term = ttnn.reshape(bias_term, (1, 1, sequence_length, self.hidden_size))
        prefill_sparsity_2d = ttnn.reshape(self.experts.prefill_sparsity, (1, self.num_experts))
        routing_weights = ttnn.mul(routing_weights, prefill_sparsity_2d, output_tensor=routing_weights)
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, sequence_length, 1))
        split_size = (
            ttnn.TILE_SIZE
            if self.prefill_token_group_sparsity
            else self.experts.program_config.get_down_split_size(sequence_length)
        )
        if sequence_length > split_size:
            down_inputs = ttnn.split(down_input, split_size, dim=2)
            down_input.deallocate(True)
            routing_splits = ttnn.split(routing_weights, split_size, dim=2)
            routing_weights.deallocate(True)
            if self.prefill_token_group_sparsity:
                down_sparsities = ttnn.split(sparsity, split_size // ttnn.TILE_SIZE, dim=2)
                sparsity.deallocate(True)
            else:
                down_sparsities = [self.experts.prefill_sparsity] * len(down_inputs)
        else:
            down_inputs = [down_input]
            routing_splits = [routing_weights]
            down_sparsities = [sparsity if self.prefill_token_group_sparsity else self.experts.prefill_sparsity]

        reduced_accumulator = None
        for down_input_split, routing_split, down_sparsity in zip(
            down_inputs,
            routing_splits,
            down_sparsities,
        ):
            split_sequence = down_input_split.shape[2]
            down = ttnn.sparse_matmul(
                down_input_split,
                self.indexed_down,
                sparsity=down_sparsity,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                program_config=self.experts.program_config.get_prefill_down_config(
                    down_input_split.shape[2],
                    self.indexed_down.shape[-1],
                    k=down_input_split.shape[-1],
                ),
                compute_kernel_config=self.expert_compute_kernel_config,
                dtype=self.expert_intermediate_dtype,
                **({} if self.prefill_token_group_sparsity else {"nnz": self.num_experts}),
            )
            if self.prefill_token_group_sparsity:
                down_sparsity.deallocate(True)
            down_input_split.deallocate(True)
            next_states = ttnn.reshape(down, (batch_size, self.num_experts, split_sequence, self.hidden_size))
            next_states = apply_routing_weights(next_states, routing_split)
            routing_split.deallocate(True)
            if next_states.dtype == ttnn.bfloat4_b:
                converted = ttnn.typecast(next_states, self.expert_intermediate_dtype)
                next_states.deallocate(True)
                next_states = converted
            reduced = reduce_experts(next_states)
            down.deallocate(True)
            if reduced_accumulator is None:
                reduced_accumulator = reduced
            else:
                concatenated = ttnn.concat((reduced_accumulator, reduced), dim=2)
                reduced_accumulator.deallocate(True)
                reduced.deallocate(True)
                reduced_accumulator = concatenated
        if reduced_accumulator.dtype != bias_term.dtype:
            converted = ttnn.typecast(bias_term, reduced_accumulator.dtype)
            bias_term.deallocate(True)
            bias_term = converted
        reduced_accumulator = ttnn.add(reduced_accumulator, bias_term, output_tensor=reduced_accumulator)
        bias_term.deallocate(True)
        return reduced_accumulator

    def _run_packed_prefill(self, hidden_states, routing_weights):
        """Chunked full-context prefill using the decode-shared packed TP weights."""
        sequence_length = hidden_states.shape[2]
        chunk_size = self.experts.program_config.sequence_chunk_size
        if sequence_length > chunk_size:
            hidden_chunks = ttnn.split(hidden_states, chunk_size, dim=2)
            routing_chunks = ttnn.split(routing_weights, chunk_size, dim=0)
        else:
            hidden_chunks = [hidden_states]
            routing_chunks = [routing_weights]

        output_accumulator = None
        for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
            output = self._process_packed_prefill_chunk(hidden_chunk, routing_chunk)
            if output_accumulator is None:
                output_accumulator = output
            else:
                concatenated = ttnn.concat((output_accumulator, output), dim=2)
                output_accumulator.deallocate(True)
                output.deallocate(True)
                output_accumulator = concatenated
        if output_accumulator.dtype != self.activation_ccl_dtype:
            converted = ttnn.typecast(output_accumulator, self.activation_ccl_dtype)
            output_accumulator.deallocate(True)
            output_accumulator = converted
        output = apply_tensor_parallel_allreduce(
            output_accumulator,
            self.mesh_config,
            self.mesh_device,
            sequence_length,
            self.ccl_manager,
        )
        return ttnn.reshape(
            output,
            (1, 1, sequence_length, self.hidden_size),
            (1, 1, max(ttnn.TILE_SIZE, sequence_length), self.hidden_size),
        )

    _PREFILL_ROUTING_GRANULARITY = 64

    def _prefill_constant(self, key, build):
        """Return a setup-only device constant for the untraced prefill path."""
        value = self._prefill_constants.get(key)
        if value is None:
            value = build()
            self._prefill_constants[key] = value
        return value

    def _prefill_u32(self, host):
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _stage_mark(self, name, started):
        """Record ``name`` stage time when timing is enabled; returns the new start."""
        if self._prefill_timing is None:
            return started
        import time

        ttnn.synchronize_device(self.mesh_device)
        now = time.perf_counter()
        self._prefill_timing[name] = self._prefill_timing.get(name, 0.0) + (now - started)
        # Program-cache growth attributed to this stage (new programs since the
        # previous mark); a steady-state prompt should add none.
        entries = self.mesh_device.num_program_cache_entries()
        growth = self._prefill_program_growth
        if growth is not None:
            growth[name] = growth.get(name, 0) + entries - growth.get("_last", entries)
        else:
            growth = self._prefill_program_growth = {}
        growth["_last"] = entries
        return now

    def _debug_capture(self, name, tensor):
        if self._prefill_debug is not None:
            self._prefill_debug[name] = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()

    # Tallest expert slab.  The sparse matmul stages per_core_M x in0_block_w
    # input tiles in L1, so slab height is bounded; experts with more tokens
    # take several consecutive slabs of this height (same expert id repeated)
    # plus one shorter power-of-two remainder slab.
    _PREFILL_MAX_SLAB_ROWS = 512

    @staticmethod
    def _reshape_via_tile(tensor, shape):
        """Reshape a ROW_MAJOR tensor through TILE layout.

        A ROW_MAJOR reshape that splits or merges sticks stages whole sticks in
        L1; at 16k+ tokens the flat slot vectors are 256 KB sticks and that
        staging no longer fits beside the resident L1 buffers.  The TILE-layout
        reshape moves tiles instead.
        """
        tiled = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
        reshaped = ttnn.reshape(tiled, shape)
        if reshaped is not tiled:
            tiled.deallocate(True)
        result = ttnn.to_layout(reshaped, ttnn.ROW_MAJOR_LAYOUT)
        reshaped.deallocate(True)
        return result

    @staticmethod
    def _indexed_prefill_matmul_config(cores, m, n, k, blocking):
        """1D-multicast sparse matmul config for an [m, k] x [k, n] slab matmul.

        ``blocking`` is (in0_block_w, out_block_h, out_subblock_h,
        out_subblock_w); every value is snapped to the kernel's divisibility
        rules (Kt % in0_block_w, per_core_M % out_block_h, out_block_h %
        out_subblock_h, per_core_N % out_subblock_w, subblock <= 8 tiles).
        """
        in0_block_w, out_block_h, out_subblock_h, out_subblock_w = blocking
        core_x, core_y = cores
        num_cores = core_x * core_y
        Kt = (k + 31) // 32
        Nt = (n + 31) // 32
        per_core_M = max(32, m) // 32
        per_core_N = (Nt + num_cores - 1) // num_cores

        def snap(value, total):
            value = max(1, min(value, total))
            while total % value:
                value -= 1
            return value

        in0_block_w = snap(in0_block_w, Kt)
        out_block_h = snap(out_block_h, per_core_M)
        out_subblock_w = snap(out_subblock_w, per_core_N)
        out_subblock_h = snap(min(out_subblock_h, 8 // out_subblock_w), out_block_h)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=out_block_h,
            out_block_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def _indexed_prefill_group(self, slab_tiled, member_ids, height):
        """Run one height group of expert slabs: gate/up -> SwiGLU -> down.

        ``slab_tiled`` is [1, count, height, hidden] TILE, ``member_ids`` is a
        device UINT32 [1, 1, 1, count] ROW_MAJOR expert-id list.  Returns
        [1, 1, count*height, hidden] TILE bf16 rows in slab order (unweighted;
        routing weights are applied when the rows are gathered back).  Both
        inputs are consumed.
        """
        count = int(member_ids.shape[-1])
        import time as _time

        _t = _time.perf_counter()
        hidden = self.hidden_size
        padded = self.padded_local_intermediate_size
        output_tile = ttnn.Tile([32, 32])
        member_ids_u16 = ttnn.typecast(member_ids, ttnn.uint16)
        _t = self._stage_mark("g.ids", _t)
        if self.indexed_prefill_slab_dtype is not None and slab_tiled.dtype != self.indexed_prefill_slab_dtype:
            narrowed = ttnn.typecast(slab_tiled, self.indexed_prefill_slab_dtype)
            slab_tiled.deallocate(True)
            slab_tiled = narrowed
            _t = self._stage_mark("g.slab_cast", _t)
        fused_bias = self.indexed_prefill_fused_bias and self.indexed_gate_up_bias_tiled is not None
        gate_up = ttnn.sparse_matmul(
            slab_tiled,
            self.indexed_gate_up,
            sparsity=self.indexed_unused_sparsity,
            indices=member_ids_u16,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self._indexed_prefill_matmul_config(
                self.experts.program_config.prefill_gate_up_cores,
                height,
                self.indexed_gate_up.shape[3],
                hidden,
                self.indexed_prefill_gate_up_blocking,
            ),
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.expert_intermediate_dtype,
            **({"bias": self.indexed_gate_up_bias_tiled} if fused_bias else {}),
        )
        slab_tiled.deallocate(True)
        _t = self._stage_mark("g.gate_up_mm", _t)
        gate_up = ttnn.reshape(gate_up, (1, count, height, 2 * padded))
        if not fused_bias:
            bias_rows = ttnn.embedding(
                member_ids,
                self.indexed_gate_up_bias,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            bias_rows = ttnn.to_layout(ttnn.reshape(bias_rows, (1, count, 1, 2 * padded)), ttnn.TILE_LAYOUT)
            _t = self._stage_mark("g.bias_gather", _t)
            gate_up = ttnn.add(gate_up, bias_rows, output_tensor=gate_up)
            bias_rows.deallocate(True)
            _t = self._stage_mark("g.bias_add", _t)
        if gate_up.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(gate_up, self.expert_intermediate_dtype)
            gate_up.deallocate(True)
            gate_up = converted
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, count, height, padded], [1, 1, 1, 1])
        up = ttnn.slice(gate_up, [0, 0, 0, padded], [1, count, height, 2 * padded], [1, 1, 1, 1])
        gate_up.deallocate(True)
        _t = self._stage_mark("g.slices", _t)
        down_input = self._fused_swiglu(gate, up)
        up.deallocate(True)
        _t = self._stage_mark("g.swiglu", _t)
        down = ttnn.sparse_matmul(
            down_input,
            self.indexed_down,
            sparsity=self.indexed_unused_sparsity,
            indices=member_ids_u16,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self._indexed_prefill_matmul_config(
                self.experts.program_config.prefill_down_cores,
                height,
                self.indexed_down.shape[-1],
                padded,
                self.indexed_prefill_down_blocking,
            ),
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.expert_intermediate_dtype,
        )
        down_input.deallocate(True)
        member_ids.deallocate(True)
        member_ids_u16.deallocate(True)
        _t = self._stage_mark("g.down_mm", _t)
        if down.dtype != ttnn.bfloat16:
            converted = ttnn.typecast(down, ttnn.bfloat16)
            down.deallocate(True)
            down = converted
        _t = self._stage_mark("g.down_cast", _t)
        return ttnn.reshape(down, (1, 1, count * height, hidden))

    def warmup_indexed_prefill_shapes(self):
        """Compile every (slab height, group size) matmul shape the indexed prefill can use.

        Program caches are keyed by shapes, not weights, so running each pair
        once on one layer covers every layer.  Group sizes are powers of two up
        to the expert count; heights are tile powers of two up to the cap.
        """
        if not self.indexed_prefill or self.separate_gate_up:
            return 0
        hidden = self.hidden_size
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        heights = []
        height = ttnn.TILE_SIZE
        while height <= self._PREFILL_MAX_SLAB_ROWS:
            heights.append(height)
            height *= 2
        sizes = []
        size = 1
        while size <= self.num_experts:
            sizes.append(size)
            size *= 2
        # One 32-row token table stands in for the prefill activations: each
        # group's slab is produced by the same tile-layout embedding gather the
        # production path uses, so those program shapes are compiled too.
        table = ttnn.from_torch(
            torch.zeros((ttnn.TILE_SIZE, hidden), dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        compiled = 0
        for height in heights:
            for count in sizes:
                group_rows = ttnn.from_torch(
                    torch.zeros((1, count * height), dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
                slab = ttnn.reshape(
                    ttnn.embedding(group_rows, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    (1, count, height, hidden),
                )
                group_rows.deallocate(True)
                member_ids = ttnn.from_torch(
                    torch.tensor([index % self.num_experts for index in range(count)], dtype=torch.int32).reshape(
                        1, 1, 1, count
                    ),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
                rows = self._indexed_prefill_group(slab, member_ids, height)
                rows.deallocate(True)
                compiled += 1
        table.deallocate(True)
        ttnn.synchronize_device(self.mesh_device)
        return compiled

    def _run_indexed_prefill(self, hidden_states):
        """Prefill MoE with per-expert token slabs and compact indexed matmuls.

        Each expert that received tokens gets ``count // max`` full-height slabs
        plus one slab whose height is the remainder rounded up to a power-of-two
        tile multiple.  Slabs of equal height form groups, split into
        power-of-two chunks, and each chunk runs as one indexed sparse matmul
        pair with compact outputs, so expert work scales with routed slots
        (plus at most one short slab of padding per expert) instead of with
        ``experts x tokens``, and every program shape comes from a small static
        set.  Routing weights are applied when the expert rows are gathered
        back; the down bias is folded in through the dense routing weights.
        Prefill is untraced and host-synchronised, so the router logits are read
        back and the top-k selection and slab bookkeeping run in torch.
        """
        experts = self.num_experts
        top_k = self.top_k
        hidden = self.hidden_size
        padded = self.padded_local_intermediate_size
        sequence_length = int(hidden_states.shape[2])
        granularity = self._PREFILL_ROUTING_GRANULARITY
        rows = ((sequence_length + granularity - 1) // granularity) * granularity
        routed_hidden = hidden_states
        if rows != sequence_length:
            routed_hidden = ttnn.pad(
                hidden_states,
                [(0, 0), (0, 0), (0, rows - sequence_length), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        slots = rows * top_k
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        import time as _time

        stage_started = _time.perf_counter()
        # The device top-k over [tokens, 128] costs ~10 ms at 16k tokens and its
        # result is read back anyway, so read the bf16 logits (same bytes as the
        # padded index/score tiles) and select the experts in torch: same bf16
        # logits, same top-4, fp32 softmax as the device HiFi3/fp32 path.
        router_logits = self.router.prefill_logits(routed_hidden)
        stage_started = self._stage_mark("router", stage_started)
        logits_host = ttnn.to_torch(ttnn.get_device_tensors(router_logits)[0])[:rows, :experts]
        router_logits.deallocate(True)
        top_logits, indices_host = torch.topk(logits_host.to(torch.float32), top_k, dim=-1, sorted=True)
        scores_host = torch.softmax(top_logits, dim=-1)
        indices_host = indices_host.to(torch.int64)
        if self._prefill_debug is not None:
            self._prefill_debug["expert_indices"] = indices_host.clone()
            self._prefill_debug["routing_scores"] = scores_host.clone()
        stage_started = self._stage_mark("routing_readback", stage_started)
        flat_experts = indices_host.reshape(-1)
        counts_tensor = torch.bincount(flat_experts, minlength=experts)
        counts_host = counts_tensor.tolist()
        max_rows = self._PREFILL_MAX_SLAB_ROWS
        # Each expert's tokens fill `full` consecutive slabs of the maximum
        # height plus at most one power-of-two remainder slab in a shorter
        # group, so padded rows stay below one short slab per expert.
        groups = {}  # height -> expert ids, one entry per slab
        full_slabs = [0] * experts
        for expert, count in enumerate(counts_host):
            if count == 0:
                continue
            full, remainder = divmod(count, max_rows)
            if remainder:
                height = max(ttnn.TILE_SIZE, 1 << (remainder - 1).bit_length())
                if height >= max_rows:
                    full += 1
                else:
                    groups.setdefault(height, []).append(expert)
            if full:
                full_slabs[expert] = full
                groups.setdefault(max_rows, []).extend([expert] * full)
        full_base_host = torch.zeros(experts, dtype=torch.int64)
        remainder_base_host = torch.zeros(experts, dtype=torch.int64)
        layout = []  # (height, expert ids, start row)
        capacity = 0
        for height in sorted(groups):
            members = groups[height]
            base_host = full_base_host if height == max_rows else remainder_base_host
            seen = set()
            for expert in members:
                if expert not in seen:
                    seen.add(expert)
                    base_host[expert] = capacity
                capacity += height
            # Split the group into power-of-two chunks (largest first, at most
            # num_experts ids per indexed call) so every matmul shape comes from
            # a small static set (heights x sizes) that stays in the program
            # cache across prompts, with no padding slabs.
            group_start = capacity - height * len(members)
            offset = 0
            while offset < len(members):
                remaining = len(members) - offset
                chunk = min(experts, 1 << (remaining.bit_length() - 1))
                chunk_members = members[offset : offset + chunk]
                layout.append((height, chunk_members, group_start + offset * height))
                offset += chunk
        # Round the slab buffer up to 1/8 of its power-of-two octave so the
        # capacity-sized ops (dispatch upload, token gather) repeat their shapes
        # across prompts while wasting at most 12.5% of the gather.
        if capacity > 256:
            unit = 1 << (capacity.bit_length() - 4)
            capacity = ((capacity + unit - 1) // unit) * unit
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts_host = flat_experts[order]
        offsets_host = torch.cumsum(counts_tensor, 0) - counts_tensor
        rank_host = torch.arange(slots, dtype=torch.int64) - offsets_host[sorted_experts_host]
        full_rows_host = (torch.tensor(full_slabs, dtype=torch.int64) * max_rows)[sorted_experts_host]
        destinations_sorted = torch.where(
            rank_host < full_rows_host,
            full_base_host[sorted_experts_host] + rank_host,
            remainder_base_host[sorted_experts_host] + rank_host - full_rows_host,
        )
        dispatch_rows_host = torch.zeros(capacity, dtype=torch.int32)
        dispatch_rows_host[destinations_sorted] = (order // top_k).to(torch.int32)
        if self._prefill_debug is not None:
            self._prefill_debug["layout"] = [(height, members, start, None) for height, members, start in layout]
            self._prefill_debug["capacity"] = capacity
            self._prefill_debug["dispatch_rows_host"] = dispatch_rows_host.clone()
        # The expert-output arena is sized to a half-octave class of the slab
        # capacity so the paged fill programs are keyed on (rows per group,
        # arena class) only, not on the per-prompt capacity.
        tile = ttnn.TILE_SIZE
        arena_rows = capacity
        if arena_rows > 256:
            unit = 1 << (arena_rows.bit_length() - 2)
            arena_rows = ((arena_rows + unit - 1) // unit) * unit
        if self._prefill_timing is not None:
            self._prefill_stats = {
                "slots": slots,
                "capacity": capacity,
                "groups": len(layout),
                "arena_rows": arena_rows,
                "heights": sorted(groups),
                "max_count": max(counts_host),
            }
        slot_to_destination_host = torch.empty(slots, dtype=torch.int64)
        slot_to_destination_host[order] = destinations_sorted
        # [top_k, rows]: row k holds every token's k-th slot destination / score.
        slot_columns_host = slot_to_destination_host.reshape(rows, top_k).transpose(0, 1).contiguous().to(torch.int32)
        slot_weights_host = scores_host.transpose(0, 1).contiguous().reshape(1, top_k, rows, 1).to(torch.bfloat16)
        if self._prefill_debug is not None:
            self._prefill_debug["slot_columns"] = slot_columns_host.clone()
        # Dense [rows, experts] routing weights for the down-bias fold.
        routing_dense_host = torch.zeros(rows, experts, dtype=torch.float32).scatter_(1, indices_host, scores_host)
        stage_started = self._stage_mark("host_layout", stage_started)

        def upload(host, dtype):
            return ttnn.from_torch(
                host,
                device=self.mesh_device,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        # Every device op in this path must see prompt-independent shapes and
        # attributes: each distinct program is cached for the life of the
        # process and holds a DRAM kernel-binary buffer, so per-prompt variants
        # (for example slices at routing-dependent offsets) grow the program
        # cache by hundreds of entries per prompt and, after enough prompts,
        # slow every later prefill.  Per-group index vectors are therefore
        # uploaded from the host layout instead of being sliced on device.
        dispatch_rows = upload(dispatch_rows_host.reshape(1, capacity), ttnn.uint32)
        slot_columns = upload(slot_columns_host, ttnn.uint32)
        # Expert outputs are scattered into one tile-layout arena with the paged
        # fill kernel (page table = the slab's 32-row block ids), instead of
        # untilizing every group and concatenating: the concat's program was keyed
        # on the per-prompt list of group shapes and leaked one program per layer
        # per prompt.  The arena is sized to a half-octave class so the fill
        # programs are keyed on (rows per group, arena class) only.
        group_inputs = []
        for height, members, start in layout:
            span = len(members) * height
            group_inputs.append(
                (
                    upload(dispatch_rows_host[start : start + span].reshape(1, span), ttnn.uint32),
                    upload(torch.tensor(members, dtype=torch.int32).reshape(1, 1, 1, len(members)), ttnn.uint32),
                    upload(
                        torch.arange(start // tile, (start + span) // tile, dtype=torch.int32).reshape(1, span // tile),
                        ttnn.int32,
                    ),
                )
            )
        slot_weights = ttnn.from_torch(
            slot_weights_host,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        routing_dense = ttnn.from_torch(
            routing_dense_host.to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._debug_capture("dispatch_rows", dispatch_rows)
        stage_started = self._stage_mark("upload", stage_started)

        # Token rows are gathered straight into each group's slab stack (one
        # tile-layout embedding per group) so no capacity-sized copy is needed.
        hidden_rm = ttnn.reshape(ttnn.to_layout(routed_hidden, ttnn.ROW_MAJOR_LAYOUT), (rows, hidden))
        if routed_hidden is not hidden_states:
            routed_hidden.deallocate(True)
        stage_started = self._stage_mark("token_rm", stage_started)

        if self.separate_gate_up:
            raise NotImplementedError("indexed prefill requires the packed gate/up expert layout")
        arena = ttnn.empty(
            (arena_rows // tile, 1, tile, hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for (height, members, _start), (group_rows, member_ids, group_pages) in zip(layout, group_inputs):
            count = len(members)
            slab_tiled = ttnn.reshape(
                ttnn.embedding(group_rows, hidden_rm, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                (1, count, height, hidden),
            )
            group_rows.deallocate(True)
            stage_started = self._stage_mark("g.token_gather", stage_started)
            down_tiled = self._indexed_prefill_group(slab_tiled, member_ids, height)
            stage_started = _time.perf_counter()
            ttnn.experimental.paged_fill_cache(arena, down_tiled, group_pages, batch_idx=0)
            down_tiled.deallocate(True)
            group_pages.deallocate(True)
            stage_started = self._stage_mark("g.arena_fill", stage_started)
        stage_started = self._stage_mark("expert_groups", stage_started)
        hidden_rm.deallocate(True)
        dispatch_rows.deallocate(True)
        out_rows = ttnn.reshape(
            ttnn.to_layout(ttnn.reshape(arena, (1, 1, arena_rows, hidden)), ttnn.ROW_MAJOR_LAYOUT),
            (arena_rows, hidden),
        )
        arena.deallocate(True)
        stage_started = self._stage_mark("arena_untilize", stage_started)

        # Gather each token's top_k weighted expert rows back and sum them:
        # one [top_k, rows] gather, one tilize, one reduction over top_k.
        gathered = ttnn.embedding(
            slot_columns, out_rows, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        slot_columns.deallocate(True)
        out_rows.deallocate(True)
        gathered = ttnn.reshape(gathered, (1, top_k, rows, hidden))
        gathered = ttnn.mul(gathered, slot_weights, output_tensor=gathered)
        slot_weights.deallocate(True)
        combined = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(gathered, dims=[1]))
        gathered.deallocate(True)
        stage_started = self._stage_mark("gather_back", stage_started)

        # Down bias through the dense routing weights.
        bias_term = ttnn.matmul(
            routing_dense,
            self.grouped_down_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.router.compute_config,
        )
        routing_dense.deallocate(True)
        bias_term = ttnn.reshape(bias_term, (1, 1, rows, hidden))
        combined = ttnn.add(combined, bias_term, output_tensor=combined)
        bias_term.deallocate(True)
        if rows != sequence_length:
            trimmed = ttnn.slice(combined, [0, 0, 0, 0], [1, 1, sequence_length, hidden], [1, 1, 1, 1])
            combined.deallocate(True)
            combined = trimmed
        if combined.dtype != self.activation_ccl_dtype:
            converted = ttnn.typecast(combined, self.activation_ccl_dtype)
            combined.deallocate(True)
            combined = converted
        output = apply_tensor_parallel_allreduce(
            combined,
            self.mesh_config,
            self.mesh_device,
            sequence_length,
            self.ccl_manager,
        )
        self._stage_mark("bias_allreduce", stage_started)
        return ttnn.reshape(
            output,
            (1, 1, sequence_length, hidden),
            (1, 1, max(ttnn.TILE_SIZE, sequence_length), hidden),
        )

    def _run_one(self, hidden_states, *, is_decode):
        if is_decode:
            return self._run_indexed_decode(hidden_states)
        rows = int(hidden_states.shape[-2])
        if self.indexed_prefill and rows >= self.indexed_prefill_min_tokens:
            chunk = self.indexed_prefill_chunk_tokens
            if rows <= chunk:
                return self._run_indexed_prefill(hidden_states)
            # Row chunks are independent (the router and the experts act per
            # token); prefill lengths above 1k are powers of two, so the
            # chunks are equal and the split/concat shapes form a bounded set.
            pieces = ttnn.split(hidden_states, chunk, dim=2)
            outputs = []
            for piece in pieces:
                outputs.append(self._run_indexed_prefill(piece))
                piece.deallocate(True)
            combined = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for output in outputs:
                output.deallocate(True)
            return combined
        expert_indices, expert_weights = self.router(hidden_states, False)
        output = self._run_packed_prefill(hidden_states, expert_weights)
        expert_indices.deallocate(True)
        return output

    def _slot_selector(self, batch_size):
        """Return the [1, top_k*batch, batch, 1] one-hot that maps slot (u, k) to row u.

        Created on the first (untraced) call for a batch width and reused by
        every later call, so trace capture never allocates it.
        """
        selector = self._slot_selectors.get(batch_size)
        if selector is None:
            host = torch.zeros((1, self.top_k * batch_size, batch_size, 1), dtype=torch.bfloat16)
            for user in range(batch_size):
                host[0, user * self.top_k : (user + 1) * self.top_k, user, 0] = 1.0
            # DRAM resident: persistent per-layer L1 tensors crowd out the
            # sampler's static circular buffers at the end of the decode step.
            selector = ttnn.from_torch(
                host,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._slot_selectors[batch_size] = selector
        return selector

    def _fused_swiglu(self, gate, up):
        """GPT-OSS SwiGLU (clamped gate/up, alpha-scaled sigmoid) in two fused passes.

        ``glu = min(gate, limit) * sigmoid(alpha * min(gate, limit))`` and
        ``out = glu * (clamp(up, -limit, limit) + 1)``; the unary chains run
        inside the two binary kernels instead of as seven separate passes.
        """
        config = self.experts.config
        limit = float(config.swiglu_limit)
        alpha = float(config.alpha)
        clamp_gate = ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, limit)
        glu = ttnn.mul(
            gate,
            gate,
            input_tensor_a_activations=[clamp_gate],
            input_tensor_b_activations=[
                clamp_gate,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, alpha),
                ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
            ],
            output_tensor=gate,
        )
        return ttnn.mul(
            glu,
            up,
            input_tensor_b_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -limit, limit),
                ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 1.0),
            ],
            output_tensor=glu,
        )

    def _routed_down_bias(self, routing_dense, rows):
        """Return sum_e w[row, e] * down_bias[e] as [1, 1, rows, hidden] via one small matmul."""
        bias_term = ttnn.matmul(
            routing_dense,
            self.grouped_down_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.router.compute_config,
        )
        return ttnn.reshape(bias_term, (1, 1, rows, self.hidden_size))

    def _run_indexed_slots_decode(self, hidden_states):
        """Run a small decode batch through its top_k*batch gathered expert slots.

        Slot (u, k) computes user u's k-th expert for every row of the tile;
        the routing score of (u, k) is applied to row u only, before the down
        projection, and one reduction over slots yields the [1, 1, batch,
        hidden] block output.  Duplicate experts across users are computed
        twice, which keeps the trace shape static.  The down bias is folded in
        through the dense routing weights with one small matmul.
        """
        batch_size = int(hidden_states.shape[-2])
        slots = self.top_k * batch_size
        rows = int(hidden_states.shape[2])
        expert_indices, routing_scores = self.router(hidden_states, True)
        if expert_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            expert_indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        else:
            expert_indices_rm = expert_indices
        # Flatten [batch, top_k] into one ROW_MAJOR stick of slot ids.  UINT16
        # reshape is not a device operation, so widen, flatten, and narrow.
        indices_u32 = ttnn.typecast(expert_indices_rm, ttnn.uint32)
        if expert_indices_rm is not expert_indices:
            expert_indices_rm.deallocate(True)
        embedding_indices = ttnn.reshape(indices_u32, (1, 1, 1, slots))
        slot_indices = ttnn.typecast(embedding_indices, ttnn.uint16)
        routing_dense = self.router.scatter_dense_routing(expert_indices, routing_scores, deallocate=False)
        output_tile = ttnn.Tile([32, 32])

        def gather_bias(bias, width):
            # Gather in ROW_MAJOR so the [1, slots, 1, width] view is free, then
            # tilize once; a TILE-layout reshape would untilize and retilize the
            # 32-row padded form of every slot.
            gathered = ttnn.embedding(
                embedding_indices,
                bias,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gathered = ttnn.reshape(gathered, (1, slots, 1, width))
            tiled = ttnn.to_layout(gathered, ttnn.TILE_LAYOUT)
            gathered.deallocate(True)
            return tiled

        if self.separate_gate_up:
            projections = []
            for weight, bias in (
                (self.indexed_gate, self.indexed_gate_bias),
                (self.indexed_up, self.indexed_up_bias),
            ):
                projected = ttnn.sparse_matmul(
                    hidden_states,
                    weight,
                    sparsity=self.indexed_unused_sparsity,
                    indices=slot_indices,
                    is_input_b_sparse=True,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    output_tile=output_tile,
                    program_config=self.experts.program_config.get_decode_gate_up_config(
                        rows,
                        weight.shape[3],
                        k=hidden_states.shape[-1],
                    ),
                    compute_kernel_config=self.expert_compute_kernel_config,
                    dtype=self.expert_intermediate_dtype,
                )
                projected = ttnn.reshape(projected, (1, slots, rows, self.padded_local_intermediate_size))
                projected_bias = gather_bias(bias, self.padded_local_intermediate_size)
                projected = ttnn.add(projected, projected_bias, output_tensor=projected)
                projected_bias.deallocate(True)
                projections.append(projected)
            gate, up = projections
        else:
            gate_up = ttnn.sparse_matmul(
                hidden_states,
                self.indexed_gate_up,
                sparsity=self.indexed_unused_sparsity,
                indices=slot_indices,
                is_input_b_sparse=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=self.experts.program_config.get_decode_gate_up_config(
                    rows,
                    self.indexed_gate_up.shape[3],
                    k=hidden_states.shape[-1],
                ),
                compute_kernel_config=self.expert_compute_kernel_config,
                dtype=self.expert_intermediate_dtype,
            )
            gate_up = ttnn.reshape(gate_up, (1, slots, rows, 2 * self.padded_local_intermediate_size))
            gate_up_bias = gather_bias(self.indexed_gate_up_bias, 2 * self.padded_local_intermediate_size)
            gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
            gate_up_bias.deallocate(True)
            if gate_up.dtype == ttnn.bfloat4_b:
                converted = ttnn.typecast(gate_up, self.expert_intermediate_dtype)
                gate_up.deallocate(True)
                gate_up = converted
            gate = ttnn.slice(
                gate_up,
                [0, 0, 0, 0],
                [1, slots, rows, self.padded_local_intermediate_size],
                [1, 1, 1, 1],
            )
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.padded_local_intermediate_size],
                [1, slots, rows, 2 * self.padded_local_intermediate_size],
                [1, 1, 1, 1],
            )
            gate_up.deallocate(True)
        down_input = self._fused_swiglu(gate, up)
        up.deallocate(True)

        # Per-slot, per-row weight: routing score of (u, k) on row u, zero elsewhere,
        # applied before the (linear) down projection on the narrower tensor.
        if routing_scores.layout != ttnn.TILE_LAYOUT:
            scores_tiled = ttnn.to_layout(routing_scores, ttnn.TILE_LAYOUT)
        else:
            scores_tiled = routing_scores
        scores_4d = ttnn.reshape(scores_tiled, (1, slots, 1, 1))
        slot_weights = ttnn.mul(self._slot_selector(batch_size), scores_4d)
        scores_4d.deallocate(True)
        if scores_tiled is not routing_scores:
            scores_tiled.deallocate(True)
        down_input = ttnn.mul(down_input, slot_weights, output_tensor=down_input)
        slot_weights.deallocate(True)

        down = ttnn.sparse_matmul(
            down_input,
            self.indexed_down,
            sparsity=self.indexed_unused_sparsity,
            indices=slot_indices,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=(
                self.experts.program_config.get_prefill_down_config(
                    rows,
                    self.indexed_down.shape[-1],
                    k=down_input.shape[-1],
                )
                if slots > self.top_k
                else self.experts.program_config.get_decode_down_config(
                    rows,
                    self.indexed_down.shape[-1],
                    k=down_input.shape[-1],
                )
            ),
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.expert_intermediate_dtype,
        )
        down_input.deallocate(True)
        slot_indices.deallocate(True)
        embedding_indices.deallocate(True)
        indices_u32.deallocate(True)
        expert_indices.deallocate(True)
        routing_scores.deallocate(True)
        output = ttnn.reshape(down, (1, slots, rows, self.hidden_size))
        if output.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(output, self.expert_intermediate_dtype)
            output.deallocate(True)
            output = converted
        reduced = reduce_experts(output)
        output.deallocate(True)
        # fast_reduce_nc reports the padded 32-row shape; restore the logical rows
        # so the bias add sees matching [1, 1, batch, hidden] operands.
        reduced = ttnn.reshape(
            reduced,
            (1, 1, rows, self.hidden_size),
            (1, 1, ttnn.TILE_SIZE, self.hidden_size),
        )
        bias_term = self._routed_down_bias(routing_dense, rows)
        routing_dense.deallocate(True)
        reduced = ttnn.add(reduced, bias_term, output_tensor=reduced)
        bias_term.deallocate(True)
        return self._finish_decode_block(reduced, batch_size)

    def _finish_decode_block(self, reduced, batch_size):
        """All-reduce the per-rank partial block output and restore the [1, 1, batch, hidden] view."""
        if reduced.dtype != self.activation_ccl_dtype:
            converted = ttnn.typecast(reduced, self.activation_ccl_dtype)
            reduced.deallocate(True)
            reduced = converted
        reduced = apply_tensor_parallel_allreduce(
            reduced,
            self.mesh_config,
            self.mesh_device,
            ttnn.TILE_SIZE,
            self.ccl_manager,
        )
        return ttnn.reshape(
            reduced,
            (1, 1, batch_size, self.hidden_size),
            (1, 1, ttnn.TILE_SIZE, self.hidden_size),
        )

    def _run_grouped_decode(self, hidden_states):
        """Run every decode user through one 32-row group of gate-selected experts.

        The decode activation already occupies one physical 32-row tile, so a
        batch of up to 32 users is one token group: the sparse expert matmuls
        read each expert in the batch's union once for all rows, and the dense
        per-row routing weights (applied before the linear down projection)
        keep each user's own top-k contributions.  Padded rows are zero: they
        add no experts to the union and contribute zero output.  The down bias
        is folded in through the dense routing weights with one small matmul.
        """
        batch_size = int(hidden_states.shape[-2])
        rows = ttnn.TILE_SIZE
        experts = self.num_experts
        local = self.padded_local_intermediate_size
        memory_config = ttnn.L1_MEMORY_CONFIG if self.grouped_decode_l1 else ttnn.DRAM_MEMORY_CONFIG
        output_tile = ttnn.Tile([32, 32])

        routing_dense = self.router.dense_decode_routing(hidden_states)
        grouped_hidden = hidden_states
        if ttnn.is_sharded(grouped_hidden):
            grouped_hidden = ttnn.to_memory_config(grouped_hidden, memory_config)
        if batch_size != rows:
            padded_hidden = ttnn.pad(
                grouped_hidden,
                [(0, 0), (0, 0), (0, rows - batch_size), (0, 0)],
                value=0.0,
                memory_config=memory_config,
            )
            if grouped_hidden is not hidden_states:
                grouped_hidden.deallocate(True)
            grouped_hidden = padded_hidden
            padded_routing = ttnn.pad(
                routing_dense,
                [(0, rows - batch_size), (0, 0)],
                value=0.0,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            routing_dense.deallocate(True)
            routing_dense = padded_routing

        # Union of the batch's experts: positive where any row routes to the expert.
        routing_4d = ttnn.reshape(routing_dense, (1, 1, rows, experts))
        group_sum = ttnn.sum(routing_4d, dim=2, keepdim=True)
        sparsity = ttnn.to_layout(group_sum, ttnn.ROW_MAJOR_LAYOUT)
        group_sum.deallocate(True)

        gate_up = ttnn.sparse_matmul(
            grouped_hidden,
            self.indexed_gate_up,
            sparsity=sparsity,
            memory_config=memory_config,
            output_tile=output_tile,
            program_config=self.experts.program_config.get_prefill_gate_up_config(
                rows,
                self.indexed_gate_up.shape[3],
                k=grouped_hidden.shape[-1],
            ),
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.expert_intermediate_dtype,
        )
        if grouped_hidden is not hidden_states:
            grouped_hidden.deallocate(True)
        # [1, 1, 1, experts, rows, width] -> expert-major view.
        gate_up = ttnn.reshape(gate_up, (1, experts, rows, 2 * local))
        gate_up = ttnn.add(gate_up, self.prefill_gate_up_bias, output_tensor=gate_up)
        if gate_up.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(gate_up, self.expert_intermediate_dtype)
            gate_up.deallocate(True)
            gate_up = converted
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, experts, rows, local], [1, 1, 1, 1])
        up = ttnn.slice(gate_up, [0, 0, 0, local], [1, experts, rows, 2 * local], [1, 1, 1, 1])
        gate_up.deallocate(True)
        down_input = self._fused_swiglu(gate, up)
        up.deallocate(True)

        # Routing weights per (expert, row), applied before the linear down projection.
        routing_t = ttnn.permute(routing_dense, (1, 0))
        routing_t = ttnn.reshape(routing_t, (1, experts, rows, 1))
        down_input = ttnn.mul(down_input, routing_t, output_tensor=down_input)
        routing_t.deallocate(True)

        down = ttnn.sparse_matmul(
            down_input,
            self.indexed_down,
            sparsity=sparsity,
            memory_config=memory_config,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=self.experts.program_config.get_prefill_down_config(
                rows,
                self.indexed_down.shape[-1],
                k=down_input.shape[-1],
            ),
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=self.expert_intermediate_dtype,
        )
        down_input.deallocate(True)
        sparsity.deallocate(True)
        down = ttnn.reshape(down, (1, experts, rows, self.hidden_size))
        if down.dtype == ttnn.bfloat4_b:
            converted = ttnn.typecast(down, self.expert_intermediate_dtype)
            down.deallocate(True)
            down = converted
        reduced = reduce_experts(down)
        down.deallocate(True)
        bias_term = self._routed_down_bias(routing_dense, rows)
        routing_dense.deallocate(True)
        reduced = ttnn.add(reduced, bias_term, output_tensor=reduced)
        bias_term.deallocate(True)
        return self._finish_decode_block(reduced, batch_size)

    def __call__(self, hidden_states, *, is_decode):
        if not is_decode or hidden_states.shape[-2] == 1:
            return self._run_one(hidden_states, is_decode=is_decode)
        if self.grouped_decode_batch:
            if hidden_states.shape[-2] <= self.indexed_slots_max_batch:
                return self._run_indexed_slots_decode(hidden_states)
            return self._run_grouped_decode(hidden_states)

        # The reusable sparse expert decode kernel represents users as its
        # batch dimension and currently accepts B=1.  Keep the autoport's
        # [1,1,B,H] public contract by executing the same active-expert graph
        # once per logical user, then restore the stack layout.  The loop is
        # static for the captured decode shape and remains device-only.
        # ``hidden_states`` arrives as a width-sharded [1, 1, B, H] tensor.
        # Splitting a sub-tile B (for example B=2) derives half-tile shard
        # heights and assigns different physical rows to otherwise identical
        # users.  Materialize each logical row explicitly in DRAM, which is
        # already the split op's fallback boundary and the sparse matmul's
        # supported batch-one input contract.
        def extract_user(user):
            sliced = ttnn.slice(
                hidden_states,
                [0, 0, user, 0],
                [1, 1, user + 1, self.hidden_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            row_major = ttnn.to_layout(sliced, ttnn.ROW_MAJOR_LAYOUT)
            sliced.deallocate(True)
            padded = ttnn.to_layout(row_major, ttnn.TILE_LAYOUT)
            row_major.deallocate(True)
            padded = ttnn.fill_implicit_tile_padding(padded, 0.0)
            return padded

        user_inputs = [extract_user(user) for user in range(hidden_states.shape[-2])]
        outputs = [self._run_one(user_input, is_decode=True) for user_input in user_inputs]
        output = ttnn.concat(outputs, dim=2)
        for user_output in outputs:
            user_output.deallocate(True)
        return output


class MultichipDecoder(LightweightModule):
    """Construct and execute the selected four-chip decoder."""

    optimization_manifest = (
        "p150_1d_tensor_parallel",
        "packed_qkv_column_parallel",
        "local_qkv_head_paged_cache",
        "row_parallel_output_ring_reduce",
        "tile_divisible_physical_hidden_attention_collective",
        "packed_rank_local_gate_up_and_row_parallel_down",
        "gate_selected_sparse_expert_tensor_parallel",
        "borrowed_l1_decode_residual",
        "decode_sharded_rmsnorm",
        "decode_attention_bfp8_prefill_attention_bf16_expert_bf16_collectives",
        "decode_lofi_prefill_qkv_hifi2_o_lofi_attention_projections",
        "sparse_expert_decode_45x15_prefill_45x45_geometry",
        "fused_decode_router",
        "route_derived_prefill_token_group_expert_sparsity",
        "layer_aware_long_prefill_sdpa_chunks",
        "replicated_decode_l1_prefill_dram_stack_residual_contract",
    )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=1,
        max_context_length=None,
        page_size=64,
        tensor_cache_path=None,
        calibrated_checkpoint_revision=None,
        policy: MultichipDecoderPolicy = DEFAULT_MULTICHIP_POLICY,
        create_kv_cache: bool = True,
    ):
        plan = tensor_plan(mesh_device.shape, hf_config)
        if not _is_supported_multichip_policy(policy):
            raise ValueError(f"unsupported multichip decoder policy: {policy!r}")
        if not 1 <= int(max_batch_size) <= 32:
            raise ValueError(f"max_batch_size must be within [1, 32], got {max_batch_size}")
        layer_type = hf_config.layer_types[layer_idx]
        if layer_type not in _SUPPORTED_LAYER_TYPES:
            raise ValueError(f"Unsupported GPT-OSS layer type {layer_type!r}")
        advertised_context = int(hf_config.max_position_embeddings)
        max_context_length = advertised_context if max_context_length is None else int(max_context_length)
        if not 1 <= max_context_length <= advertised_context:
            raise ValueError(f"max_context_length must be within [1, {advertised_context}], got {max_context_length}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        local_state = _local_layer_state_dict(state_dict, layer_idx)
        cache_root = str(Path(tensor_cache_path) / f"tp{plan.tp}") if tensor_cache_path is not None else None
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=plan.tp, ep=1, sp=1),
            prefill=ModeConfig(tp=plan.tp, ep=1, sp=1),
        )
        ccl_manager = CCLManager(
            mesh_device,
            num_links=get_default_num_links(mesh_device),
            topology=policy.topology,
        )
        if layer_type == "sliding_attention":
            prefill_q_chunk_size_large = policy.prefill_sliding_q_chunk_size_large
            prefill_k_chunk_size_large = policy.prefill_sliding_k_chunk_size_large
        else:
            prefill_q_chunk_size_large = policy.prefill_full_q_chunk_size_large
            prefill_k_chunk_size_large = policy.prefill_full_k_chunk_size_large
        program_config = GPTOSSAttentionProgramConfig(
            math_fidelity=policy.attention_sdpa_math_fidelity.name,
            prefill_q_chunk_size_large=prefill_q_chunk_size_large,
            prefill_k_chunk_size_large=prefill_k_chunk_size_large,
        )
        physical_context_length = (
            math.ceil(max_context_length / program_config.decode_k_chunk_size) * program_config.decode_k_chunk_size
        )
        paged_attention_config = PagedAttentionConfig(
            block_size=page_size,
            # SDPA reads whole K chunks.  A non-chunk-aligned logical
            # context therefore needs enough physical pages for the padded
            # final chunk even though positions beyond the logical limit are
            # never accepted by the public generator.
            max_num_blocks=max_batch_size * math.ceil(physical_context_length / page_size),
        )
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            sliding_window=(hf_config.sliding_window if layer_type == "sliding_attention" else None),
            max_seq_len=max_context_length,
            max_local_batch_size=max_batch_size,
            users_row_sharded=False,
            cache_position_modulo=ring_modulo_for_layer(layer_type),
        )
        rope_scaling_config = getattr(hf_config, "rope_scaling", None)
        rope_scaling = rope_scaling_model_factory(rope_scaling_config) if rope_scaling_config else None
        rope_theta = getattr(hf_config, "rope_theta", None) or getattr(hf_config, "default_theta", 150000.0)
        rope_setup = RotarySetup(
            device=mesh_device,
            batch_size=max_batch_size,
            head_dim=hf_config.head_dim,
            max_seq_len=max_context_length,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            datatype=ttnn.bfloat16,
        )
        attention_state = substate(local_state, "self_attn")
        attention = _PhysicalHiddenCollectiveAttention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=attention_state,
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=program_config,
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=rope_setup.get_both_trans_mats(),
            weight_dtype=policy.attention_weight_dtype,
            cache_dtype=policy.kv_cache_dtype,
            prefill_projection_input_dtype=policy.attention_projection_input_dtype,
            prefill_projection_compute_kernel_config=ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=policy.prefill_projection_math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            tensor_cache_path=get_cache_file_name(cache_root, "self_attn"),
            create_kv_cache=create_kv_cache,
        )
        attention.decode_output_physical_hidden = plan.padded_hidden_size
        attention.activation_ccl_dtype = policy.attention_activation_ccl_dtype or policy.activation_ccl_dtype
        attention.residual_dtype = policy.residual_dtype
        attention.decode_projection_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.projection_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        backend = DecoderLayer(
            mesh_device=mesh_device,
            hf_config=hf_config,
            layer_idx=layer_idx,
            layer_type=layer_type,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            page_size=page_size,
            input_layernorm=DecodeRMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "input_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "input_layernorm"),
                mesh_config=mesh_config,
                weight_dtype=policy.normalization_weight_dtype,
                enable_decode_sharding=DecodeRMSNorm.sharding_enabled_for(max_batch_size),
            ),
            post_attention_layernorm=DecodeRMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "post_attention_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "post_attention_layernorm"),
                mesh_config=mesh_config,
                weight_dtype=policy.normalization_weight_dtype,
                enable_decode_sharding=DecodeRMSNorm.sharding_enabled_for(max_batch_size),
            ),
            attention=attention,
            mlp=_ActiveExpertTPMLP(
                mesh_device,
                hf_config,
                substate(local_state, "mlp"),
                ccl_manager,
                tensor_cache_path=get_cache_file_name(cache_root, "mlp"),
                mesh_config=mesh_config,
                expert_weight_dtype=policy.expert_weight_dtype,
                router_weight_dtype=policy.router_weight_dtype,
                expert_math_fidelity=policy.expert_math_fidelity,
                router_math_fidelity=policy.router_math_fidelity,
                expert_intermediate_dtype=policy.expert_intermediate_dtype,
                router_prefill_input_l1=policy.router_prefill_input_l1,
                router_prefill_explicit_program_config=policy.router_prefill_explicit_program_config,
                decode_fused_router=policy.decode_fused_router,
                prefill_token_group_sparsity=policy.prefill_token_group_sparsity,
                grouped_decode_batch=policy.decode_grouped_batch,
                indexed_slots_max_batch=policy.decode_indexed_slots_max_batch,
                grouped_decode_l1=policy.decode_grouped_l1,
                indexed_prefill=policy.prefill_indexed_experts,
                indexed_prefill_min_tokens=policy.prefill_indexed_min_tokens,
                activation_ccl_dtype=policy.expert_activation_ccl_dtype or policy.activation_ccl_dtype,
                separate_gate_up=policy.decode_separate_gate_up,
                gate_up_cores=policy.expert_gate_up_cores,
                gate_up_in0_block_w=policy.expert_gate_up_in0_block_w,
                gate_up_subblock_w=policy.expert_gate_up_subblock_w,
                down_cores=policy.expert_down_cores,
                down_in0_block_w=policy.expert_down_in0_block_w,
                down_subblock_w=policy.expert_down_subblock_w,
                prefill_down_cores=policy.expert_prefill_down_cores,
                prefill_down_in0_block_w=policy.expert_prefill_down_in0_block_w,
                prefill_down_subblock_w=policy.expert_prefill_down_subblock_w,
            ),
            calibrated_checkpoint_revision=calibrated_checkpoint_revision,
        )
        backend.mesh_config = mesh_config
        backend.ccl_manager = ccl_manager
        return cls(backend=backend, tensor_plan=plan, policy=policy)

    def __init__(self, *, backend, tensor_plan, policy):
        self.backend = backend
        self.tensor_plan = tensor_plan
        self.policy = policy
        self.mesh_device = backend.mesh_device
        self.hf_config = backend.hf_config
        self.layer_idx = backend.layer_idx
        self.layer_type = backend.layer_type
        self.max_batch_size = backend.max_batch_size
        self.max_context_length = backend.max_context_length
        self.page_size = backend.page_size

    @property
    def kv_cache(self):
        return self.backend.kv_cache

    @property
    def input_layernorm(self):
        return self.backend.input_layernorm

    @property
    def post_attention_layernorm(self):
        return self.backend.post_attention_layernorm

    @property
    def self_attn(self):
        return self.backend.self_attn

    @property
    def mlp(self):
        return self.backend.mlp

    def prefill_forward(self, hidden_states, **kwargs):
        return self.backend.prefill_forward(hidden_states, **kwargs)

    def decode_forward(self, hidden_states, **kwargs):
        return self.backend.decode_forward(hidden_states, **kwargs)

    def forward(self, hidden_states, *, mode, **kwargs):
        return self.backend.forward(hidden_states, mode=mode, **kwargs)
