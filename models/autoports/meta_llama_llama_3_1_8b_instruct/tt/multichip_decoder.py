# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-mesh tensor-parallel Llama 3.1 decoder layer.

The single-chip :class:`OptimizedDecoder` is the implementation baseline.  This
module keeps its optimized operation graph and precision policy while applying
1-D tensor parallelism to the four-chip Blackhole P300c ring on this host:
autoport:

* packed QKV and gate/up weights are column parallel;
* output and down weights are row parallel;
* Q/K/V, attention, and SwiGLU execute on device-local heads/features;
* the replicated residual stream is restored by two asynchronous all-reduces
  over the physical four-device ring; and
* each device owns two of the model's eight KV heads.

Prefill and decode expose the same logical ``[1, batch, seq, 4096]`` boundary,
so outputs can feed the next decoder directly.  Decode keeps that replicated
stream width-sharded in each device's L1.  Paged and contiguous KV-cache forms
are both supported; page tables are replicated while cache heads are sharded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    _config_value,
    _state_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    TILE_SIZE,
    OptimizationConfig,
    OptimizedDecoder,
    _advisor_matmul_program_config,
    _advisor_width_sharded_l1,
    _dram_matmul_program_config,
    _dram_weight_memory_config,
    _width_sharded_l1,
)
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl

TARGET_TP_DEGREE = 4
TARGET_MESH_SHAPE = (1, 4)
PAGED_BLOCK_SIZE = 64


@dataclass(frozen=True)
class MultiChipConfig:
    """Fixed full-P300 TP=4 policy plus bounded experiment controls."""

    optimized: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(
            decode_matmul_strategy="dram_sharded",
            qkv_cores=16,
            output_cores=16,
            gate_up_cores=16,
            down_cores=16,
            residual_cores=16,
        )
    )
    topology: object = ttnn.Topology.Ring
    num_links: int = 2
    collective_dtype: object = ttnn.bfloat16


def _mesh_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    mesh_mapper,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )


def _pack_tp_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp_degree: int) -> torch.Tensor:
    """Pack rank-local Q/K/V shards so a final mesh shard preserves Q-K-V order."""

    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    local_packs = []
    for rank in range(tp_degree):
        q_local = q_t.tensor_split(tp_degree, dim=-1)[rank]
        k_local = k_t.tensor_split(tp_degree, dim=-1)[rank]
        v_local = v_t.tensor_split(tp_degree, dim=-1)[rank]
        local_packs.append(torch.cat((q_local, k_local, v_local), dim=-1))
    return torch.cat(local_packs, dim=-1)


def _pack_tp_gate_up(gate_t: torch.Tensor, up_t: torch.Tensor, tp_degree: int) -> torch.Tensor:
    """Pack gate/up rank pairs before mesh sharding the fused projection."""

    local_packs = []
    gate_shards = gate_t.tensor_split(tp_degree, dim=-1)
    up_shards = up_t.tensor_split(tp_degree, dim=-1)
    for rank in range(tp_degree):
        local_packs.append(torch.cat((gate_shards[rank], up_shards[rank]), dim=-1))
    return torch.cat(local_packs, dim=-1)


class MultiChipDecoder(OptimizedDecoder):
    """Real TP=4 implementation derived from :class:`OptimizedDecoder`."""

    single_chip_baseline = OptimizedDecoder

    def __init__(
        self,
        *,
        multichip_config: MultiChipConfig,
        global_num_heads: int,
        global_num_kv_heads: int,
        global_intermediate_size: int,
        tt_ccl: TT_CCL | None = None,
        **kwargs,
    ):
        self.multichip_config = multichip_config
        self.tp_degree = TARGET_TP_DEGREE
        self.global_num_heads = global_num_heads
        self.global_num_kv_heads = global_num_kv_heads
        self.global_intermediate_size = global_intermediate_size
        self.local_hidden_size = kwargs["hidden_size"] // self.tp_degree
        super().__init__(optimization_config=multichip_config.optimized, **kwargs)

        if self.mesh_device.get_num_devices() != self.tp_degree:
            raise ValueError(
                f"MultiChipDecoder targets exactly {TARGET_MESH_SHAPE}, got "
                f"{self.mesh_device.get_num_devices()} devices"
            )
        if tuple(self.mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(f"MultiChipDecoder targets mesh {TARGET_MESH_SHAPE}, got {tuple(self.mesh_device.shape)}")
        if self.multichip_config.topology != ttnn.Topology.Ring:
            raise ValueError("The target P300 path requires ring topology")
        if self.global_num_heads != self.num_heads * self.tp_degree:
            raise ValueError("local/global query-head ownership is inconsistent")
        if self.global_num_kv_heads != self.num_kv_heads * self.tp_degree:
            raise ValueError("local/global KV-head ownership is inconsistent")
        if self.global_intermediate_size != self.intermediate_size * self.tp_degree:
            raise ValueError("local/global MLP ownership is inconsistent")

        padded_rows = TILE_SIZE * math.ceil(self.batch / TILE_SIZE)
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        if self.use_advisor_1d:
            # Shape-adapt the optimized baseline's interleaved BFP4 1-D
            # family to TP4-local projection widths.  These exact grids cover
            # the local N tiles without output padding.
            self.qkv_decode_program_config = _advisor_matmul_program_config(
                grid=(8, 3), in0_block_w=2, per_core_n=2, out_subblock_w=2
            )
            self.output_decode_program_config = _advisor_matmul_program_config(
                grid=(8, 8), in0_block_w=8, per_core_n=2, out_subblock_w=2
            )
            self.gate_up_decode_program_config = _advisor_matmul_program_config(
                grid=(7, 4), in0_block_w=2, per_core_n=4, out_subblock_w=4
            )
            self.down_decode_program_config = _advisor_matmul_program_config(
                grid=(8, 8), in0_block_w=8, per_core_n=2, out_subblock_w=2
            )
            self.advisor_qkv_output_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=1536, cores=24, grid=(8, 3)
            )
            self.advisor_gate_up_output_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=self.intermediate_size, cores=28, grid=(7, 4)
            )
            self.advisor_down_input_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=self.intermediate_size, cores=28, grid=(7, 4)
            )
            self.advisor_residual_output_mem_config = _advisor_width_sharded_l1(
                rows=padded_rows, width=self.hidden_size, cores=64, grid=(8, 8)
            )
        else:
            output_cores = self.optimization_config.output_cores
            self.output_input_mem_config = _width_sharded_l1(
                rows=padded_rows,
                width=self.local_hidden_size,
                cores=output_cores,
                device_grid=device_grid,
            )
            self.output_decode_program_config = _dram_matmul_program_config(
                role="tp_output",
                m=padded_rows,
                k=self.local_hidden_size,
                n=self.hidden_size,
                cores=output_cores,
                in0_block_w=self.optimization_config.output_in0_block_w,
            )

        # Persistent semaphores are setup-time, per-mesh state shared by every
        # decoder layer in a stack.  The vector-semaphore all-reduce API is the
        # generic P300 path and is trace-safe.
        self.tt_ccl = tt_ccl if tt_ccl is not None else get_tt_ccl(self.mesh_device)
        if self.tt_ccl.mesh_device is not self.mesh_device:
            raise ValueError("tt_ccl must belong to the decoder's mesh_device")
        self._barrier_semaphores = self.tt_ccl.barrier_semaphore_handles[2]

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        multichip_config: MultiChipConfig | None = None,
        tt_ccl: TT_CCL | None = None,
        **_kwargs,
    ) -> "MultiChipDecoder":
        if mesh_device.get_num_devices() != TARGET_TP_DEGREE or tuple(mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(
                f"MultiChipDecoder requires mesh {TARGET_MESH_SHAPE}; got shape={tuple(mesh_device.shape)}, "
                f"devices={mesh_device.get_num_devices()}"
            )
        if batch < 1 or batch > EMITTED_BATCH:
            raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

        policy = multichip_config or MultiChipConfig()
        if policy.optimized.decode_matmul_strategy not in {"dram_sharded", "advisor_1d"}:
            raise ValueError("The TP=4 path requires DRAM-sharded or interleaved 1-D decode matmuls")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        global_num_heads = int(_config_value(hf_config, "num_attention_heads"))
        global_num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // global_num_heads)
        global_intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))

        for name, value in (
            ("hidden_size", hidden_size),
            ("num_attention_heads", global_num_heads),
            ("num_key_value_heads", global_num_kv_heads),
            ("intermediate_size", global_intermediate_size),
        ):
            if value % TARGET_TP_DEGREE:
                raise ValueError(f"{name}={value} must divide TP={TARGET_TP_DEGREE}")
        if hidden_size != global_num_heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        output = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16)

        packed_qkv = _pack_tp_qkv(q, k, v, TARGET_TP_DEGREE)
        output_t = output.transpose(0, 1)
        gate_t = gate.transpose(0, 1)
        up_t = up.transpose(0, 1)
        down_t = down.transpose(0, 1)
        local_num_heads = global_num_heads // TARGET_TP_DEGREE
        local_num_kv_heads = global_num_kv_heads // TARGET_TP_DEGREE
        local_intermediate_size = global_intermediate_size // TARGET_TP_DEGREE
        local_qkv_width = (local_num_heads + 2 * local_num_kv_heads) * head_dim
        packed_gate_up = _pack_tp_gate_up(gate_t, up_t, TARGET_TP_DEGREE) if policy.optimized.packed_gate_up else None
        local_gate_up_width = local_intermediate_size * (2 if policy.optimized.packed_gate_up else 1)

        rotary = LlamaRotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
        cos, sin = rotary(rope_probe, positions)
        cos = cos.to(torch.bfloat16).unsqueeze(1)
        sin = sin.to(torch.bfloat16).unsqueeze(1)

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        shard_last = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        shard_first = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        attention_dtype = policy.optimized.attention_weight_dtype
        gate_up_dtype = policy.optimized.gate_up_weight_dtype
        down_dtype = policy.optimized.down_weight_dtype
        weight_memory_config = (
            (lambda _k, _n: ttnn.DRAM_MEMORY_CONFIG)
            if policy.optimized.decode_matmul_strategy == "advisor_1d"
            else (lambda k, n: _dram_weight_memory_config(mesh_device, k=k, n=n))
        )

        return cls(
            multichip_config=policy,
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            num_heads=local_num_heads,
            num_kv_heads=local_num_kv_heads,
            head_dim=head_dim,
            intermediate_size=local_intermediate_size,
            rms_norm_eps=rms_norm_eps,
            global_num_heads=global_num_heads,
            global_num_kv_heads=global_num_kv_heads,
            global_intermediate_size=global_intermediate_size,
            tt_ccl=tt_ccl,
            input_norm=_mesh_tensor(input_norm, mesh_device, mesh_mapper=replicate),
            post_attention_norm=_mesh_tensor(post_attention_norm, mesh_device, mesh_mapper=replicate),
            qkv_weight=_mesh_tensor(
                packed_qkv,
                mesh_device,
                mesh_mapper=shard_last,
                dtype=attention_dtype,
                memory_config=weight_memory_config(hidden_size, local_qkv_width),
            ),
            output_weight=_mesh_tensor(
                output_t,
                mesh_device,
                mesh_mapper=shard_first,
                dtype=attention_dtype,
                memory_config=weight_memory_config(hidden_size // TARGET_TP_DEGREE, hidden_size),
            ),
            gate_weight=_mesh_tensor(
                packed_gate_up if packed_gate_up is not None else gate_t,
                mesh_device,
                mesh_mapper=shard_last,
                dtype=gate_up_dtype,
                memory_config=weight_memory_config(hidden_size, local_gate_up_width),
            ),
            up_weight=(
                None
                if packed_gate_up is not None
                else _mesh_tensor(
                    up_t,
                    mesh_device,
                    mesh_mapper=shard_last,
                    dtype=gate_up_dtype,
                    memory_config=weight_memory_config(hidden_size, local_intermediate_size),
                )
            ),
            down_weight=_mesh_tensor(
                down_t,
                mesh_device,
                mesh_mapper=shard_first,
                dtype=down_dtype,
                memory_config=weight_memory_config(local_intermediate_size, hidden_size),
            ),
            rotary_cos=_mesh_tensor(cos, mesh_device, mesh_mapper=replicate),
            rotary_sin=_mesh_tensor(sin, mesh_device, mesh_mapper=replicate),
            position_indices=_mesh_tensor(
                torch.arange(max_cache_len, dtype=torch.int32).unsqueeze(1).expand(-1, batch),
                mesh_device,
                mesh_mapper=replicate,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            ),
        )

    @staticmethod
    def mesh_mapper_for_input(mesh_device):
        """The stacked decoder boundary is a replicated logical residual."""

        return ttnn.ReplicateTensorToMesh(mesh_device)

    @staticmethod
    def mesh_mapper_for_cache(mesh_device):
        """Shard the global KV-head dimension across the TP mesh."""

        return ttnn.ShardTensorToMesh(mesh_device, dim=1)

    def _validate_caches(self, key_cache, value_cache, page_table=None) -> None:
        key_shape = tuple(key_cache.shape)
        value_shape = tuple(value_cache.shape)
        if key_shape != value_shape:
            raise ValueError(f"key/value cache shapes differ: {key_shape} versus {value_shape}")
        if page_table is None:
            expected = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
            if key_shape != expected:
                raise ValueError(f"contiguous key/value caches must have local shape {expected}, got {key_shape}")
            return

        if len(key_shape) != 4 or key_shape[1] != self.num_kv_heads or key_shape[3] != self.head_dim:
            raise ValueError(
                "paged key/value caches must have local shape "
                f"[blocks, {self.num_kv_heads}, block_size, {self.head_dim}], got {key_shape}"
            )
        if key_shape[2] != PAGED_BLOCK_SIZE:
            raise ValueError(f"paged block size must be {PAGED_BLOCK_SIZE}, got {key_shape[2]}")
        page_shape = tuple(page_table.shape)
        if len(page_shape) != 2 or page_shape[0] != self.batch:
            raise ValueError(f"page_table must have local shape [{self.batch}, pages], got {page_shape}")

    def _all_reduce_partial(self, partial, *, memory_config=None):
        output_memory_config = memory_config or self.residual_mem_config
        payload = partial
        if self.multichip_config.collective_dtype != ttnn.bfloat16:
            payload = ttnn.typecast(
                partial,
                self.multichip_config.collective_dtype,
                memory_config=output_memory_config,
            )
        reduced = ttnn.experimental.all_reduce_async(
            payload,
            num_devices=self.tp_degree,
            barrier_semaphores=self._barrier_semaphores,
            rs_global_semaphores=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            ag_global_semaphores=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            math_op=ttnn.ReduceType.Sum,
            num_links=self.multichip_config.num_links,
            memory_config=output_memory_config,
            topology=self.multichip_config.topology,
        )
        if self.multichip_config.collective_dtype != ttnn.bfloat16:
            reduced = ttnn.typecast(reduced, ttnn.bfloat16, memory_config=output_memory_config)
        return reduced

    def _fill_cache(self, cache, values, *, page_table, seq_len: int) -> None:
        cache_values = values if values.dtype == cache.dtype else ttnn.typecast(values, cache.dtype)
        for user_id in range(self.batch):
            user = ttnn.slice(
                cache_values,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            if page_table is None:
                ttnn.fill_cache(cache, user, user_id)
            else:
                ttnn.experimental.paged_fill_cache(cache, user, page_table, batch_idx=user_id)

    def _mlp_prefill_tp(self, hidden_states, seq_len: int):
        return self._all_reduce_partial(
            super()._mlp_prefill(hidden_states, seq_len), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _mlp_decode_tp(self, hidden_states):
        return self._all_reduce_partial(super()._mlp_decode(hidden_states))

    def prefill_forward(self, hidden_states, key_cache, value_cache, *, page_table=None):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache, page_table)
        if page_table is not None and math.ceil(seq_len / PAGED_BLOCK_SIZE) > page_table.shape[1]:
            raise ValueError("page_table does not cover the logical prefill length")

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel,
        )
        fused_qkv = self._prefill_linear(
            normed,
            self.qkv_weight,
            role="tp_qkv",
            seq_len=seq_len,
            k=self.hidden_size,
            n=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            compute_kernel=self.attention_compute_kernel,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._fill_cache(key_cache, key, page_table=page_table, seq_len=seq_len)
        self._fill_cache(value_cache, value, page_table=page_table, seq_len=seq_len)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.prefill_sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.local_hidden_size])
        attention = self._prefill_linear(
            attention,
            self.output_weight,
            role="tp_output",
            seq_len=seq_len,
            k=self.local_hidden_size,
            n=self.hidden_size,
            compute_kernel=self.attention_compute_kernel,
        )
        attention = self._all_reduce_partial(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel,
        )
        hidden_states = self._mlp_prefill_tp(hidden_states, seq_len)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int, page_table=None):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache, page_table)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")
        if page_table is not None and current_pos // PAGED_BLOCK_SIZE >= page_table.shape[1]:
            raise ValueError("page_table does not cover current_pos")

        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.batch, self.hidden_size])
        hidden_states = ttnn.to_memory_config(hidden_states, self.residual_mem_config)
        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            program_config=self.norm_program_config,
            memory_config=self.residual_mem_config,
            compute_kernel_config=self.norm_compute_kernel,
        )
        normed = ttnn.to_memory_config(
            normed,
            ttnn.L1_MEMORY_CONFIG if self.use_advisor_1d else self.qkv_input_mem_config,
        )
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.advisor_qkv_output_mem_config if self.use_advisor_1d else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            ),
            program_config=self.qkv_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        fused_qkv = ttnn.sharded_to_interleaved(fused_qkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_heads_mem_config,
        )

        cos = ttnn.slice(
            self.rotary_cos,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        sin = ttnn.slice(
            self.rotary_sin,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        query = ttnn.experimental.rotary_embedding(query, cos, sin, 0, memory_config=self.decode_heads_mem_config)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, 0, memory_config=self.decode_heads_mem_config)

        update_indices = ttnn.slice(
            self.position_indices,
            [current_pos, 0],
            [current_pos + 1, self.batch],
            [1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        update_indices = ttnn.reshape(update_indices, [self.batch])
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=page_table,
        )

        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        if page_table is None:
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                is_causal=True,
                cur_pos_tensor=update_indices,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel,
            )
        else:
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query,
                key_cache,
                value_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=update_indices,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel,
            )
        attention = ttnn.to_memory_config(attention, self.decode_heads_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.local_hidden_size],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(
            attention,
            ttnn.L1_MEMORY_CONFIG if self.use_advisor_1d else self.output_input_mem_config,
        )
        attention = ttnn.linear(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.advisor_residual_output_mem_config if self.use_advisor_1d else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            ),
            program_config=self.output_decode_program_config,
            compute_kernel_config=self.attention_compute_kernel,
        )
        attention = ttnn.to_memory_config(attention, self.residual_mem_config)
        attention = self._all_reduce_partial(attention)
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=self.residual_mem_config,
        )

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            program_config=self.norm_program_config,
            memory_config=self.residual_mem_config,
            compute_kernel_config=self.norm_compute_kernel,
        )
        hidden_states = self._mlp_decode_tp(hidden_states)
        hidden_states = ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=self.residual_mem_config,
        )
        return ttnn.reshape(hidden_states, [1, self.batch, 1, self.hidden_size])

    def forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        mode: str,
        current_pos: int | None = None,
        page_table=None,
    ):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, key_cache, value_cache, page_table=page_table)
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.decode_forward(
                hidden_states,
                key_cache,
                value_cache,
                current_pos=current_pos,
                page_table=page_table,
            )
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


__all__ = [
    "MultiChipConfig",
    "MultiChipDecoder",
    "PAGED_BLOCK_SIZE",
    "TARGET_MESH_SHAPE",
    "TARGET_TP_DEGREE",
]
