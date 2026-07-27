# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device GPT-OSS-20B decoder layer.

The runtime is independent of ``FunctionalDecoder``.  It retains the emitted
packed-QKV attention topology and replaces the functional dense all-expert MoE
with the GPT-OSS routed active-expert ``ttnn.sparse_matmul`` topology.
Candidate policies are construction-time only and exist to make the stage's
precision/topology A/B measurements reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
    SUPPORTED_CONTEXT,
    _dense_expert_weight,
    _require_tensor,
)
from models.common.lightweightmodule import LightweightModule
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object
    expert_weight_dtype: object
    cache_dtype: object
    attention_math_fidelity: object
    sparse_experts: bool = True
    static_sparse_nnz: int | None = None
    advisor_attention_seed: bool = False


POLICIES = {
    "default": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_experts": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bf16_experts": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bf16_cache": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat16,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_attention": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "attention_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "sparse_nnz4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        static_sparse_nnz=4,
    ),
    "advisor_seed": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        advisor_attention_seed=True,
    ),
    "dense_packed_bfp4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat4_b,
        cache_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.HiFi2,
        sparse_experts=False,
    ),
}


def _as_replicated_tensor(
    tensor,
    *,
    mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
    )


def _largest_divisor(value: int, ceiling: int) -> int:
    for candidate in range(min(value, ceiling), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _largest_power_of_two_divisor(value: int, ceiling: int) -> int:
    candidate = 1
    while candidate * 2 <= ceiling and value % (candidate * 2) == 0:
        candidate *= 2
    return candidate


class OptimizedDecoder(LightweightModule):
    """Single-device optimized GPT-OSS decoder with active-expert execution."""

    def __init__(
        self,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int,
        max_cache_len: int,
        weights: dict[str, ttnn.Tensor],
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        prefill_sparsity: ttnn.Tensor,
        policy: OptimizationPolicy,
        candidate: str,
    ):
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.weights = weights
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.prefill_sparsity = prefill_sparsity
        self.policy = policy
        self.candidate = candidate

        self.hidden_size = int(hf_config.hidden_size)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(hf_config.head_dim)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.num_experts = int(hf_config.num_local_experts)
        self.top_k = int(hf_config.num_experts_per_tok)
        self.eps = float(hf_config.rms_norm_eps)
        self.scale = self.head_dim**-0.5
        layer_types = getattr(hf_config, "layer_types", None)
        self.layer_type = layer_types[layer_idx] if layer_types is not None else None
        self.sliding_window = int(hf_config.sliding_window) if self.layer_type == "sliding_attention" else None

        self.norm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.attention_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.attention_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.router_softmax_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=_largest_power_of_two_divisor(max_cache_len, 128),
        )
        self.expert_program_config = GPTOSSProgramConfig()
        self._prefill_qkv_program_configs = {}
        self._prefill_o_program_configs = {}
        if policy.advisor_attention_seed:
            qkv_grid = ttnn.CoreCoord(11, 8)
            qkv_output_cores = ttnn.num_cores_to_corerangeset(80, qkv_grid, row_wise=True)
            self.advisor_qkv_output_memcfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE),
                core_grid=qkv_output_cores,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=qkv_grid,
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            o_grid = ttnn.CoreCoord(11, 9)
            o_output_cores = ttnn.num_cores_to_corerangeset(90, o_grid, row_wise=True)
            self.advisor_o_output_memcfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                core_grid=o_output_cores,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.advisor_o_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=o_grid,
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, object],
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=EMITTED_CACHE_LENGTH,
        candidate="default",
        **_kwargs,
    ):
        import torch
        import torch.nn.functional as F
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

        if candidate not in POLICIES:
            raise ValueError(f"unknown optimization candidate {candidate!r}; expected one of {sorted(POLICIES)}")
        policy = POLICIES[candidate]
        if batch != EMITTED_BATCH:
            raise ValueError(f"The emitted workload batch is {EMITTED_BATCH}, got {batch}")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
        if not 1 <= max_cache_len <= SUPPORTED_CONTEXT:
            raise ValueError(f"max_cache_len must be in [1, {SUPPORTED_CONTEXT}], got {max_cache_len}")
        if not isinstance(mesh_device, ttnn.MeshDevice):
            raise TypeError("OptimizedDecoder requires a ttnn.MeshDevice")
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(f"OptimizedDecoder requires a 1x1 mesh, got {tuple(mesh_device.shape)}")

        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        intermediate_size = int(hf_config.intermediate_size)
        num_experts = int(hf_config.num_local_experts)
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        if (hidden_size, q_dim, kv_dim, intermediate_size, num_experts) != (2880, 4096, 512, 2880, 32):
            raise ValueError(
                "OptimizedDecoder expects GPT-OSS-20B dimensions "
                "(hidden=2880, q=4096, kv=512, intermediate=2880, experts=32)"
            )

        q_weight = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_weight = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_weight = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        q_bias = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.bias")
        k_bias = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.bias")
        v_bias = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.bias")
        qkv_weight = torch.cat(
            [q_weight.transpose(-2, -1), k_weight.transpose(-2, -1), v_weight.transpose(-2, -1)],
            dim=-1,
        ).to(torch.bfloat16)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1).reshape(1, 1, -1).to(torch.bfloat16)

        o_weight = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        o_bias = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.bias")
        router_weight = _require_tensor(state_dict, layer_idx, "mlp.router.weight")
        router_bias = _require_tensor(state_dict, layer_idx, "mlp.router.bias")
        gate_up_weight = _dense_expert_weight(state_dict, layer_idx, "gate_up_proj")
        down_weight = _dense_expert_weight(state_dict, layer_idx, "down_proj")
        gate_up_bias = _require_tensor(state_dict, layer_idx, "mlp.experts.gate_up_proj_bias")
        down_bias = _require_tensor(state_dict, layer_idx, "mlp.experts.down_proj_bias")
        input_norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")
        post_attention_norm = _require_tensor(state_dict, layer_idx, "post_attention_layernorm.weight")
        sinks = _require_tensor(state_dict, layer_idx, "self_attn.sinks").to(torch.bfloat16)

        scale = head_dim**-0.5
        prefill_sinks = (sinks.reshape(1, num_heads, 1, 1) / scale).to(torch.bfloat16)
        decode_sinks = F.pad(sinks.reshape(num_heads, 1), (0, ttnn.TILE_SIZE - 1)) / scale

        rotary = GptOssRotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rotary_input = torch.empty(1, 1, max_cache_len, head_dim, dtype=torch.bfloat16)
        cos_half, sin_half = rotary(rotary_input, positions)
        cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(1)
        sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(1)

        norm_shape = (1, 1, hidden_size // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        weights = {
            "input_norm": _as_replicated_tensor(
                input_norm.reshape(norm_shape).to(torch.bfloat16),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "post_attention_norm": _as_replicated_tensor(
                post_attention_norm.reshape(norm_shape).to(torch.bfloat16),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "qkv_weight": _as_replicated_tensor(
                qkv_weight,
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
            ),
            "qkv_bias": _as_replicated_tensor(qkv_bias, mesh_device=mesh_device),
            "o_weight": _as_replicated_tensor(
                o_weight.transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
                dtype=policy.attention_weight_dtype,
            ),
            "o_bias": _as_replicated_tensor(
                o_bias.reshape(1, 1, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
            "prefill_sinks": _as_replicated_tensor(prefill_sinks, mesh_device=mesh_device),
            "decode_sinks": _as_replicated_tensor(decode_sinks.to(torch.bfloat16), mesh_device=mesh_device),
            "router_weight": _as_replicated_tensor(
                router_weight.transpose(-2, -1).to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
            "router_bias": _as_replicated_tensor(
                router_bias.reshape(1, -1).float(),
                mesh_device=mesh_device,
                dtype=ttnn.float32,
            ),
        }

        if policy.sparse_experts:
            gate_weight = gate_up_weight[..., ::2].reshape(
                1, num_experts, hidden_size, intermediate_size
            )
            up_weight = gate_up_weight[..., 1::2].reshape(
                1, num_experts, hidden_size, intermediate_size
            )
            weights.update(
                {
                    "gate_weight": _as_replicated_tensor(
                        gate_weight.to(torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=policy.expert_weight_dtype,
                    ),
                    "up_weight": _as_replicated_tensor(
                        up_weight.to(torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=policy.expert_weight_dtype,
                    ),
                    "gate_bias": _as_replicated_tensor(
                        gate_up_bias[..., ::2]
                        .reshape(1, num_experts, intermediate_size)
                        .to(torch.bfloat16),
                        mesh_device=mesh_device,
                    ),
                    "up_bias": _as_replicated_tensor(
                        gate_up_bias[..., 1::2]
                        .reshape(1, num_experts, intermediate_size)
                        .to(torch.bfloat16),
                        mesh_device=mesh_device,
                    ),
                    "down_weight": _as_replicated_tensor(
                        down_weight.reshape(1, num_experts, intermediate_size, hidden_size).to(torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=policy.expert_weight_dtype,
                    ),
                    "down_bias": _as_replicated_tensor(
                        down_bias.reshape(1, num_experts, hidden_size).to(torch.bfloat16),
                        mesh_device=mesh_device,
                    ),
                }
            )
        else:
            weights.update(
                {
                    "packed_gate_up_weight": _as_replicated_tensor(
                        gate_up_weight.to(torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=policy.expert_weight_dtype,
                    ),
                    "packed_gate_up_bias": _as_replicated_tensor(
                        gate_up_bias.reshape(num_experts, 1, 2 * intermediate_size).to(torch.bfloat16),
                        mesh_device=mesh_device,
                    ),
                    "down_weight": _as_replicated_tensor(
                        down_weight.to(torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=policy.expert_weight_dtype,
                    ),
                    "down_bias": _as_replicated_tensor(
                        down_bias.reshape(num_experts, 1, hidden_size).to(torch.bfloat16),
                        mesh_device=mesh_device,
                    ),
                }
            )

        prefill_sparsity = _as_replicated_tensor(
            torch.ones(1, 1, 1, num_experts, dtype=torch.bfloat16),
            mesh_device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            weights=weights,
            cos_cache=_as_replicated_tensor(cos, mesh_device=mesh_device),
            sin_cache=_as_replicated_tensor(sin, mesh_device=mesh_device),
            prefill_sparsity=prefill_sparsity,
            policy=policy,
            candidate=candidate,
        )

    def create_kv_cache(self):
        shape = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
        key_cache = ttnn.zeros(
            shape,
            dtype=self.policy.cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value_cache = ttnn.zeros(
            shape,
            dtype=self.policy.cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return key_cache, value_cache

    def _validate_hidden_states(self, hidden_states, *, expected_seq_len=None):
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != self.batch or shape[3] != self.hidden_size:
            raise ValueError(f"hidden_states must have shape [1, {self.batch}, seq, {self.hidden_size}], got {shape}")
        if expected_seq_len is not None and shape[2] != expected_seq_len:
            raise ValueError(f"expected sequence length {expected_seq_len}, got {shape[2]}")
        return shape[2]

    def _prefill_qkv_program_config(self, seq_len):
        padded_m_tiles = math.ceil(seq_len / ttnn.TILE_SIZE)
        grid_y = next(value for value in range(min(8, padded_m_tiles), 0, -1) if padded_m_tiles % value == 0)
        key = (grid_y, padded_m_tiles)
        if key not in self._prefill_qkv_program_configs:
            self._prefill_qkv_program_configs[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, grid_y),
                in0_block_w=10,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=padded_m_tiles // grid_y,
                per_core_N=20,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
        return self._prefill_qkv_program_configs[key]

    def _prefill_o_program_config(self, seq_len):
        padded_m_tiles = math.ceil(seq_len / ttnn.TILE_SIZE)
        grid_y = next(value for value in range(min(8, padded_m_tiles), 0, -1) if padded_m_tiles % value == 0)
        key = (grid_y, padded_m_tiles)
        if key not in self._prefill_o_program_configs:
            self._prefill_o_program_configs[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(10, grid_y),
                in0_block_w=16,
                out_subblock_h=1,
                out_subblock_w=3,
                per_core_M=padded_m_tiles // grid_y,
                per_core_N=9,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
        return self._prefill_o_program_configs[key]

    def _prefill_attention(self, hidden_states, key_cache, value_cache, seq_len):
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["input_norm"],
            compute_kernel_config=self.norm_compute_kernel_config,
        )
        fused = ttnn.matmul(
            normalized,
            self.weights["qkv_weight"],
            dtype=ttnn.bfloat16,
            program_config=self._prefill_qkv_program_config(seq_len),
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused = ttnn.add(fused, self.weights["qkv_bias"], output_tensor=fused)
        fused = ttnn.reshape(fused, [self.batch, seq_len, -1])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.cos_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        sin = ttnn.slice(self.sin_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(query, [0, 0, 0, 0], [1, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [1, self.num_kv_heads, seq_len, self.head_dim])
        cache_key = ttnn.typecast(key, self.policy.cache_dtype)
        cache_value = ttnn.typecast(value, self.policy.cache_dtype)
        ttnn.fill_cache(key_cache, cache_key, batch_idx=0)
        ttnn.fill_cache(value_cache, cache_value, batch_idx=0)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.weights["prefill_sinks"],
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                exp_approx_mode=False,
                q_chunk_size=32 if seq_len < 2048 else 256,
                k_chunk_size=32 if seq_len < 2048 else 256,
            ),
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.matmul(
            attended,
            self.weights["o_weight"],
            dtype=ttnn.bfloat16,
            program_config=self._prefill_o_program_config(seq_len),
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        projected = ttnn.add(projected, self.weights["o_bias"], output_tensor=projected)
        return ttnn.add(hidden_states, ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size]))

    def _decode_attention(
        self,
        hidden_states,
        key_cache,
        value_cache,
        cache_position,
        cache_position_tensor,
        attention_mask,
    ):
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["input_norm"],
            compute_kernel_config=self.norm_compute_kernel_config,
        )
        if self.policy.advisor_attention_seed:
            normalized = ttnn.to_memory_config(normalized, ttnn.L1_MEMORY_CONFIG)
            fused = ttnn.matmul(
                normalized,
                self.weights["qkv_weight"],
                dtype=ttnn.bfloat16,
                program_config=self.advisor_qkv_program_config,
                compute_kernel_config=self.attention_compute_kernel_config,
                memory_config=self.advisor_qkv_output_memcfg,
            )
        else:
            fused = ttnn.matmul(
                normalized,
                self.weights["qkv_weight"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.attention_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        fused = ttnn.add(fused, self.weights["qkv_bias"], output_tensor=fused)
        if self.policy.advisor_attention_seed:
            fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        fused = ttnn.reshape(fused, [1, 1, self.batch, -1])
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        cos = ttnn.slice(
            self.cos_cache,
            [0, 0, cache_position, 0],
            [1, 1, cache_position + 1, self.head_dim],
        )
        sin = ttnn.slice(
            self.sin_cache,
            [0, 0, cache_position, 0],
            [1, 1, cache_position + 1, self.head_dim],
        )
        query = ttnn.experimental.rotary_embedding(
            query, cos, sin, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        key = ttnn.experimental.rotary_embedding(
            key, cos, sin, 0, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=cache_position_tensor,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=cache_position_tensor,
            share_cache=False,
            page_table=None,
        )
        attended = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=False,
            attn_mask=attention_mask,
            attention_sink=self.weights["decode_sinks"],
            scale=self.scale,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.reshape(attended, [self.batch, self.num_heads * self.head_dim])
        if self.policy.advisor_attention_seed:
            attended = ttnn.to_memory_config(attended, ttnn.L1_MEMORY_CONFIG)
            projected = ttnn.matmul(
                attended,
                self.weights["o_weight"],
                dtype=ttnn.bfloat16,
                program_config=self.advisor_o_program_config,
                compute_kernel_config=self.attention_compute_kernel_config,
                memory_config=self.advisor_o_output_memcfg,
            )
        else:
            projected = ttnn.matmul(
                attended,
                self.weights["o_weight"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.attention_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        projected = ttnn.add(projected, self.weights["o_bias"], output_tensor=projected)
        if self.policy.advisor_attention_seed:
            projected = ttnn.to_memory_config(projected, ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.reshape(projected, [1, self.batch, 1, self.hidden_size])
        return ttnn.add(hidden_states, projected)

    def _routing(self, normalized, token_count):
        flat = ttnn.reshape(normalized, [token_count, self.hidden_size])
        router_input = ttnn.typecast(flat, ttnn.float32)
        router_logits = ttnn.linear(
            router_input,
            self.weights["router_weight"],
            bias=self.weights["router_bias"],
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(router_logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            compute_kernel_config=self.router_softmax_compute_kernel_config,
        )
        routing_weights = ttnn.scatter(
            ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_values,
        )
        return flat, routing_weights

    def _apply_swiglu(self, gate, up):
        gate = ttnn.clamp(gate, min=None, max=7.0, output_tensor=gate)
        up = ttnn.clamp(up, min=-7.0, max=7.0, output_tensor=up)
        gate_scaled = ttnn.multiply(gate, 1.703125)
        gate_sigmoid = ttnn.sigmoid(gate_scaled)
        gate = ttnn.multiply(gate, gate_sigmoid, output_tensor=gate)
        up = ttnn.add(up, 1.0, output_tensor=up)
        return ttnn.multiply(up, gate, output_tensor=up)

    def _sparse_decode_moe(self, normalized, routing_weights):
        sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])
        gate = ttnn.sparse_matmul(
            normalized,
            self.weights["gate_weight"],
            sparsity=sparsity,
            nnz=self.policy.static_sparse_nnz,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self.expert_program_config.get_decode_gate_up_config(
                normalized.shape[2], self.weights["gate_weight"].shape[3], k=normalized.shape[-1]
            ),
            dtype=ttnn.bfloat8_b,
        )
        gate = ttnn.reshape(gate, (self.batch, self.num_experts, 1, self.intermediate_size))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (self.batch, self.num_experts, self.intermediate_size))
        gate = ttnn.add(gate, self.weights["gate_bias"], output_tensor=gate)

        up = ttnn.sparse_matmul(
            normalized,
            self.weights["up_weight"],
            sparsity=sparsity,
            nnz=self.policy.static_sparse_nnz,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self.expert_program_config.get_decode_gate_up_config(
                normalized.shape[2], self.weights["up_weight"].shape[3], k=normalized.shape[-1]
            ),
            dtype=ttnn.bfloat8_b,
        )
        up = ttnn.reshape(up, (self.batch, self.num_experts, 1, self.intermediate_size))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (self.batch, self.num_experts, self.intermediate_size))
        up = ttnn.add(up, self.weights["up_bias"], output_tensor=up)
        down_input = self._apply_swiglu(gate, up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, self.num_experts, 1, self.intermediate_size))
        down = ttnn.sparse_matmul(
            down_input,
            self.weights["down_weight"],
            sparsity=sparsity,
            nnz=self.policy.static_sparse_nnz,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=self.expert_program_config.get_decode_down_config(
                down_input.shape[2], self.weights["down_weight"].shape[-1], k=down_input.shape[-1]
            ),
            dtype=ttnn.bfloat8_b,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (self.batch, self.num_experts, self.hidden_size))
        next_states = ttnn.add(next_states, self.weights["down_bias"], output_tensor=next_states)
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (self.batch, self.num_experts, 1))
        next_states = ttnn.multiply(next_states, routing_weights, output_tensor=next_states)
        # Keep the expert axis explicit until after routing.  fast_reduce_nc is
        # intended for the 4-D NC reduction used by prefill; on this 3-D decode
        # tensor it can reduce the wrong physical dimension after reshape.
        next_states = ttnn.sum(next_states, dim=1)
        return ttnn.reshape(next_states, (1, self.batch, 1, self.hidden_size))

    def _sparse_prefill_moe(self, normalized, routing_weights, seq_len):
        padded_seq_len = math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if padded_seq_len != seq_len:
            normalized = ttnn.pad(
                normalized,
                padding=((0, 0), (0, 0), (0, padded_seq_len - seq_len), (0, 0)),
                value=0.0,
            )
            _, routing_weights = self._routing(normalized, padded_seq_len)
        group_count = padded_seq_len // ttnn.TILE_SIZE
        hidden_groups = ttnn.reshape(
            normalized,
            (1, group_count, ttnn.TILE_SIZE, self.hidden_size),
        )
        sparsity = ttnn.repeat(self.prefill_sparsity, (1, 1, group_count, 1))
        output_tile = ttnn.Tile([32, 32])
        nnz = self.num_experts * group_count
        gate = ttnn.sparse_matmul(
            hidden_groups,
            self.weights["gate_weight"],
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self.expert_program_config.get_prefill_gate_up_config(
                ttnn.TILE_SIZE, self.weights["gate_weight"].shape[3], k=self.hidden_size
            ),
            dtype=ttnn.bfloat8_b,
        )
        gate = ttnn.transpose(gate, 1, 3)
        gate = ttnn.reshape(gate, (self.batch, self.num_experts, padded_seq_len, self.intermediate_size))
        gate_bias = ttnn.transpose(self.weights["gate_bias"], 1, 0)
        gate = ttnn.add(gate, gate_bias, output_tensor=gate)

        up = ttnn.sparse_matmul(
            hidden_groups,
            self.weights["up_weight"],
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=self.expert_program_config.get_prefill_gate_up_config(
                ttnn.TILE_SIZE, self.weights["up_weight"].shape[3], k=self.hidden_size
            ),
            dtype=ttnn.bfloat8_b,
        )
        up = ttnn.transpose(up, 1, 3)
        up = ttnn.reshape(up, (self.batch, self.num_experts, padded_seq_len, self.intermediate_size))
        up_bias = ttnn.transpose(self.weights["up_bias"], 1, 0)
        up = ttnn.add(up, up_bias, output_tensor=up)
        down_input = self._apply_swiglu(gate, up)
        down_input = ttnn.reshape(
            down_input,
            (1, self.num_experts, padded_seq_len, self.intermediate_size),
        )
        down = ttnn.sparse_matmul(
            down_input,
            self.weights["down_weight"],
            sparsity=self.prefill_sparsity,
            nnz=self.num_experts,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=self.expert_program_config.get_prefill_down_config(
                padded_seq_len, self.weights["down_weight"].shape[-1], k=self.intermediate_size
            ),
            dtype=ttnn.bfloat8_b,
        )
        next_states = ttnn.reshape(
            down,
            (self.batch, self.num_experts, padded_seq_len, self.hidden_size),
        )
        down_bias = ttnn.transpose(self.weights["down_bias"], 1, 0)
        next_states = ttnn.add(next_states, down_bias, output_tensor=next_states)
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(
            routing_weights,
            (self.batch, self.num_experts, padded_seq_len, 1),
        )
        next_states = ttnn.multiply(next_states, routing_weights, output_tensor=next_states)
        next_states = ttnn.experimental.fast_reduce_nc(next_states, dims=[1])
        next_states = ttnn.reshape(
            next_states,
            (1, self.batch, padded_seq_len, self.hidden_size),
        )
        if padded_seq_len != seq_len:
            next_states = ttnn.slice(
                next_states,
                [0, 0, 0, 0],
                [1, self.batch, seq_len, self.hidden_size],
            )
        return next_states

    def _dense_packed_moe(self, normalized, routing_weights, seq_len):
        token_count = self.batch * seq_len
        flat = ttnn.reshape(normalized, [token_count, self.hidden_size])
        expert_input = ttnn.reshape(flat, [1, token_count, self.hidden_size])
        expert_input = ttnn.repeat(expert_input, ttnn.Shape([self.num_experts, 1, 1]))
        gate_up = ttnn.matmul(
            expert_input,
            self.weights["packed_gate_up_weight"],
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up = ttnn.add(gate_up, self.weights["packed_gate_up_bias"])
        gate = ttnn.slice(
            gate_up,
            [0, 0, 0],
            [self.num_experts, token_count, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        up = ttnn.slice(
            gate_up,
            [0, 0, 1],
            [self.num_experts, token_count, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        activated = self._apply_swiglu(gate, up)
        expert_output = ttnn.matmul(
            activated,
            self.weights["down_weight"],
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        expert_output = ttnn.add(expert_output, self.weights["down_bias"])
        routing_weights = ttnn.permute(routing_weights, [1, 0])
        routing_weights = ttnn.reshape(routing_weights, [self.num_experts, token_count, 1])
        expert_output = ttnn.multiply(expert_output, routing_weights)
        expert_output = ttnn.sum(expert_output, dim=0)
        return ttnn.reshape(expert_output, [1, self.batch, seq_len, self.hidden_size])

    def _moe_forward(self, hidden_states, seq_len):
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["post_attention_norm"],
            compute_kernel_config=self.norm_compute_kernel_config,
        )
        token_count = self.batch * seq_len
        _, routing_weights = self._routing(normalized, token_count)
        if self.policy.sparse_experts:
            if seq_len == 1:
                expert_output = self._sparse_decode_moe(normalized, routing_weights)
            else:
                expert_output = self._sparse_prefill_moe(normalized, routing_weights, seq_len)
        else:
            expert_output = self._dense_packed_moe(normalized, routing_weights, seq_len)
        return ttnn.add(hidden_states, expert_output)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1")
        if seq_len > self.max_cache_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_cache_len={self.max_cache_len}")
        hidden_states = self._prefill_attention(hidden_states, key_cache, value_cache, seq_len)
        return self._moe_forward(hidden_states, seq_len)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        cache_position,
        cache_position_tensor,
        attention_mask,
    ):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        if not 0 <= cache_position < self.max_cache_len:
            raise ValueError(f"cache_position must be in [0, {self.max_cache_len}), got {cache_position}")
        expected_mask_shape = (1, 1, self.num_heads, self.max_cache_len)
        if tuple(attention_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"attention_mask must have logical shape {expected_mask_shape}, got {tuple(attention_mask.shape)}"
            )
        hidden_states = self._decode_attention(
            hidden_states,
            key_cache,
            value_cache,
            cache_position,
            cache_position_tensor,
            attention_mask,
        )
        return self._moe_forward(hidden_states, 1)

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
