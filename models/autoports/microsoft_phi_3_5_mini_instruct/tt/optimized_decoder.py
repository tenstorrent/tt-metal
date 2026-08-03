# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized Phi-3.5 decoder layer.

The functional decoder remains the semantic reference.  This implementation
keeps the same public prefill/decode and paged-cache contracts while replacing
the material dense path with explicit phase-specific TTNN configurations:

* decode residuals and matmul intermediates are width-sharded in L1;
* decode weights are width-sharded across Blackhole DRAM banks;
* QKV stays packed while decode gate/up uses the faster measured split form;
* prefill uses DRAM-interleaved activations; explicit 2-D programs are sweepable;
* every material matmul has an explicit precision and compute-fidelity policy.

Torch conversion is confined to ``from_state_dict``.  Runtime forwards contain
TTNN operations only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import (
    DEFAULT_PAGE_SIZE,
    HF_ADVERTISED_CONTEXT,
    PCC_ACCEPTANCE,
    PREFILL_SDPA_MAX_SEQ,
    FunctionalDecoder,
    _require,
)


@dataclass(frozen=True)
class OptimizationPolicy:
    """Named precision/config candidate used by tests and the final default."""

    attention_weight_dtype: object = ttnn.bfloat4_b
    gate_up_weight_dtype: object = ttnn.bfloat4_b
    down_weight_dtype: object = ttnn.bfloat4_b
    kv_cache_dtype: object = ttnn.bfloat8_b
    attention_math_fidelity: object = ttnn.MathFidelity.LoFi
    gate_up_math_fidelity: object = ttnn.MathFidelity.LoFi
    down_math_fidelity: object = ttnn.MathFidelity.LoFi
    decode_core_count: int = 16
    in0_block_w_qkv: int = 6
    in0_block_w_o: int = 6
    in0_block_w_gate_up: int = 6
    in0_block_w_down: int = 16
    use_explicit_prefill_programs: bool = False
    use_explicit_decode_sdpa: bool = True
    split_decode_qkv: bool = False
    split_decode_gate_up: bool = True
    # Advisor-challenger winner: keep Phi's explicit rotate-half chain in L1.
    use_advisor_decode_rope_l1: bool = True


DEFAULT_OPTIMIZATION_POLICY = OptimizationPolicy()


def _blackhole_compute_config(fidelity):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _rectangular_grid(num_cores: int, device_grid) -> ttnn.CoreGrid:
    for y in range(min(device_grid.y, num_cores), 0, -1):
        if num_cores % y == 0 and num_cores // y <= device_grid.x:
            return ttnn.CoreGrid(x=num_cores // y, y=y)
    raise ValueError(f"cannot place {num_cores} rectangular cores on {device_grid}")


def _dram_weight_memory_config(mesh_device, *, k: int, n: int):
    dram = mesh_device.dram_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    dram_cores = dram.x * dram.y
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * ttnn.TILE_SIZE * dram_cores
    shard = ttnn.ShardSpec(cores, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)


def _to_device_weight(tensor, mesh_device, *, dtype, memory_config):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=memory_config,
    )


def _prefill_program(m: int, k: int, n: int, grid=(8, 8)):
    grid_x, grid_y = grid
    per_core_m = math.ceil(math.ceil(m / ttnn.TILE_SIZE) / grid_y)
    per_core_n = math.ceil(math.ceil(n / ttnn.TILE_SIZE) / grid_x)
    in0_tiles = math.ceil(k / ttnn.TILE_SIZE)
    in0_block_w = min(8, in0_tiles)
    while in0_tiles % in0_block_w:
        in0_block_w -= 1
    out_subblock_w = min(4, per_core_n)
    while per_core_n % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


class OptimizedDecoder(FunctionalDecoder):
    """Phi decoder with explicit optimized prefill and decode paths."""

    def __init__(self, *, optimization_policy: OptimizationPolicy, **kwargs):
        super().__init__(**kwargs)
        self.optimization_policy = optimization_policy
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.decode_grid = _rectangular_grid(optimization_policy.decode_core_count, device_grid)
        self.decode_residual_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.hidden_size // optimization_policy.decode_core_count),
            core_grid=self.decode_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_mlp_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.intermediate_size // optimization_policy.decode_core_count),
            core_grid=self.decode_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        block_w = self.hidden_size // ttnn.TILE_SIZE // optimization_policy.decode_core_count
        subblock_w = min(8, block_w)
        while block_w % subblock_w:
            subblock_w -= 1
        self.decode_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.decode_grid.x, self.decode_grid.y],
            subblock_w=subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        self.decode_programs = {
            "qkv": self._decode_matmul_program(
                self.hidden_size, 3 * self.hidden_size, optimization_policy.in0_block_w_qkv
            ),
            "o_proj": self._decode_matmul_program(
                self.hidden_size, self.hidden_size, optimization_policy.in0_block_w_o
            ),
            "gate_up": self._decode_matmul_program(
                self.hidden_size, 2 * self.intermediate_size, optimization_policy.in0_block_w_gate_up
            ),
            "down": self._decode_matmul_program(
                self.intermediate_size, self.hidden_size, optimization_policy.in0_block_w_down
            ),
        }
        if optimization_policy.split_decode_qkv:
            for role in ("q_proj", "k_proj", "v_proj"):
                self.decode_programs[role] = self._decode_matmul_program(
                    self.hidden_size, self.hidden_size, optimization_policy.in0_block_w_qkv
                )
        if optimization_policy.split_decode_gate_up:
            for role in ("gate_proj", "up_proj"):
                self.decode_programs[role] = self._decode_matmul_program(
                    self.hidden_size, self.intermediate_size, optimization_policy.in0_block_w_gate_up
                )
        self.compute_configs = {
            "attention": _blackhole_compute_config(optimization_policy.attention_math_fidelity),
            "gate_up": _blackhole_compute_config(optimization_policy.gate_up_math_fidelity),
            "down": _blackhole_compute_config(optimization_policy.down_math_fidelity),
        }
        self.decode_sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )
            if optimization_policy.use_explicit_decode_sdpa
            else None
        )

    def _decode_matmul_program(self, k: int, n: int, in0_block_w: int):
        policy = self.optimization_policy
        k_tiles_per_core = k // ttnn.TILE_SIZE // policy.decode_core_count
        if k_tiles_per_core % in0_block_w:
            raise ValueError(f"in0_block_w={in0_block_w} does not divide {k_tiles_per_core} K tiles/core")
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * policy.decode_core_count)),
            fused_activation=None,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, object],
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_context=HF_ADVERTISED_CONTEXT,
        page_size=DEFAULT_PAGE_SIZE,
        optimization_policy=DEFAULT_OPTIMIZATION_POLICY,
        **_kwargs,
    ):
        import torch

        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(f"OptimizedDecoder requires a 1x1 mesh, got {tuple(mesh_device.shape)}")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx {layer_idx} is outside the configured layer range")
        if (
            int(hf_config.hidden_size),
            int(hf_config.intermediate_size),
            int(hf_config.num_attention_heads),
            int(hf_config.num_key_value_heads),
        ) != (3072, 8192, 32, 32):
            raise ValueError("OptimizedDecoder targets Phi-3.5 Mini's real dense layer shape")
        if not 1 <= max_context <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_context must be in [1, {hf_config.max_position_embeddings}], got {max_context}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        hidden = int(hf_config.hidden_size)
        inter = int(hf_config.intermediate_size)
        qkv = _require(state_dict, layer_idx, "self_attn.qkv_proj.weight")
        o_proj = _require(state_dict, layer_idx, "self_attn.o_proj.weight")
        gate_up = _require(state_dict, layer_idx, "mlp.gate_up_proj.weight")
        down = _require(state_dict, layer_idx, "mlp.down_proj.weight")
        input_norm = _require(state_dict, layer_idx, "input_layernorm.weight")
        post_norm = _require(state_dict, layer_idx, "post_attention_layernorm.weight")

        head_dim = hidden // int(hf_config.num_attention_heads)
        rope = hf_config.rope_scaling
        positions = torch.arange(max_context, dtype=torch.float32).unsqueeze(1)
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        amplitude = math.sqrt(
            1
            + math.log(int(hf_config.max_position_embeddings) / int(hf_config.original_max_position_embeddings))
            / math.log(int(hf_config.original_max_position_embeddings))
        )

        def rope_table(factors):
            inv_freq = 1.0 / (torch.tensor(factors, dtype=torch.float32) * float(hf_config.rope_theta) ** exponent)
            freqs = positions * inv_freq.unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            return (emb.cos() * amplitude).to(torch.bfloat16), (emb.sin() * amplitude).to(torch.bfloat16)

        short_cos, short_sin = rope_table(rope["short_factor"])
        long_cos, long_sin = rope_table(rope["long_factor"])
        norm_shape = (1, 1, hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        weights = {
            "input_norm": ttnn.from_torch(
                input_norm.reshape(norm_shape).to(torch.bfloat16),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "post_norm": ttnn.from_torch(
                post_norm.reshape(norm_shape).to(torch.bfloat16),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "qkv": _to_device_weight(
                qkv.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.attention_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, k=hidden, n=3 * hidden),
            ),
            "o_proj": _to_device_weight(
                o_proj.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.attention_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, k=hidden, n=hidden),
            ),
            "gate_up": _to_device_weight(
                gate_up.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.gate_up_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, k=hidden, n=2 * inter),
            ),
            "down": _to_device_weight(
                down.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.down_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, k=inter, n=hidden),
            ),
            # The large 2-D prefill candidate consumes DRAM-sharded weights.
            # Keep phase-specific interleaved copies as well: the TTNN default
            # prefill program requires interleaved B and is a measured candidate.
            "qkv_prefill": _to_device_weight(
                qkv.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.attention_weight_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "o_proj_prefill": _to_device_weight(
                o_proj.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.attention_weight_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "gate_up_prefill": _to_device_weight(
                gate_up.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.gate_up_weight_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "down_prefill": _to_device_weight(
                down.transpose(-2, -1),
                mesh_device,
                dtype=optimization_policy.down_weight_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        }
        if optimization_policy.split_decode_qkv:
            for role, value in zip(("q_proj", "k_proj", "v_proj"), qkv.chunk(3, dim=0)):
                weights[role] = _to_device_weight(
                    value.transpose(-2, -1),
                    mesh_device,
                    dtype=optimization_policy.attention_weight_dtype,
                    memory_config=_dram_weight_memory_config(mesh_device, k=hidden, n=hidden),
                )
        if optimization_policy.split_decode_gate_up:
            for role, value in zip(("gate_proj", "up_proj"), gate_up.chunk(2, dim=0)):
                weights[role] = _to_device_weight(
                    value.transpose(-2, -1),
                    mesh_device,
                    dtype=optimization_policy.gate_up_weight_dtype,
                    memory_config=_dram_weight_memory_config(mesh_device, k=hidden, n=inter),
                )

        def table_to_device(value):
            return ttnn.from_torch(
                value,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_context=max_context,
            page_size=page_size,
            weights=weights,
            short_cos=table_to_device(short_cos),
            short_sin=table_to_device(short_sin),
            long_cos=table_to_device(long_cos),
            long_sin=table_to_device(long_sin),
            optimization_policy=optimization_policy,
        )

    def create_paged_kv_cache(self, *, num_physical_blocks=None):
        blocks_per_user = math.ceil(self.max_context / self.page_size)
        num_physical_blocks = num_physical_blocks or self.batch * blocks_per_user
        shape = (num_physical_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        kwargs = dict(
            dtype=self.optimization_policy.kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.zeros(shape, **kwargs), ttnn.zeros(shape, **kwargs)

    def _decode_norm(self, hidden_states, weight):
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memory_config)
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=weight,
            program_config=self.decode_norm_program_config,
            memory_config=self.decode_residual_memory_config,
        )

    def _decode_linear(self, hidden_states, role, output_memory_config):
        compute_role = (
            "attention"
            if role in ("qkv", "q_proj", "k_proj", "v_proj", "o_proj")
            else "gate_up"
            if role in ("gate_proj", "up_proj")
            else role
        )
        return ttnn.linear(
            hidden_states,
            self.weights[role],
            dtype=ttnn.bfloat16,
            program_config=self.decode_programs[role],
            compute_kernel_config=self.compute_configs[compute_role],
            memory_config=output_memory_config,
        )

    def _decode_mlp(self, hidden_states):
        normalized = self._decode_norm(hidden_states, self.weights["post_norm"])
        if self.optimization_policy.split_decode_gate_up:
            gate = self._decode_linear(normalized, "gate_proj", self.decode_mlp_memory_config)
            up = self._decode_linear(normalized, "up_proj", self.decode_mlp_memory_config)
        else:
            gate_up = self._decode_linear(normalized, "gate_up", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
            gate_up_shape = tuple(gate_up.shape)
            gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
            up = ttnn.slice(
                gate_up, [0, 0, 0, self.intermediate_size], [*gate_up_shape[:-1], 2 * self.intermediate_size]
            )
        activated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=self.decode_mlp_memory_config,
        )
        down = self._decode_linear(activated, "down", self.decode_residual_memory_config)
        return ttnn.add(hidden_states, down, memory_config=self.decode_residual_memory_config)

    def _decode_rope(self, query, key, current_positions, *, use_long_rope):
        if not self.optimization_policy.use_advisor_decode_rope_l1:
            return super()._decode_rope(query, key, current_positions, use_long_rope=use_long_rope)
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.reshape(
            ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG),
            [1, 1, self.batch, self.head_dim],
        )
        sin = ttnn.reshape(
            ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG),
            [1, 1, self.batch, self.head_dim],
        )
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG), cos, sin)
        height_memory_config = self._decode_concat_memory_config()
        return (
            ttnn.to_memory_config(query, height_memory_config),
            ttnn.to_memory_config(key, height_memory_config),
        )

    def _prefill_linear(self, hidden_states, role, *, seq_len, k, n, compute_role):
        program_config = (
            _prefill_program(seq_len, k, n) if self.optimization_policy.use_explicit_prefill_programs else None
        )
        weight = self.weights[role] if program_config is not None else self.weights[f"{role}_prefill"]
        return ttnn.linear(
            hidden_states,
            weight,
            dtype=ttnn.bfloat16,
            program_config=program_config,
            compute_kernel_config=self.compute_configs[compute_role],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_mlp(self, hidden_states, seq_len):
        normalized = self._norm(hidden_states, self.weights["post_norm"])
        gate_up = self._prefill_linear(
            normalized,
            "gate_up",
            seq_len=seq_len,
            k=self.hidden_size,
            n=2 * self.intermediate_size,
            compute_role="gate_up",
        )
        shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*shape[:-1], self.intermediate_size])
        up = ttnn.slice(gate_up, [0, 0, 0, self.intermediate_size], [*shape[:-1], 2 * self.intermediate_size])
        activated = ttnn.multiply(ttnn.silu(gate), up)
        down = self._prefill_linear(
            activated,
            "down",
            seq_len=seq_len,
            k=self.intermediate_size,
            n=self.hidden_size,
            compute_role="down",
        )
        return ttnn.add(hidden_states, down)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache, page_table, user_id=0):
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[:2] != (1, self.batch) or shape[3] != self.hidden_size:
            raise ValueError(f"prefill hidden_states must be [1,{self.batch},S,{self.hidden_size}], got {shape}")
        seq_len = shape[2]
        if not 1 < seq_len <= self.max_context:
            raise ValueError(f"prefill sequence must be in [2,{self.max_context}], got {seq_len}")
        residual = hidden_states
        normalized = self._norm(hidden_states, self.weights["input_norm"])
        fused = self._prefill_linear(
            normalized,
            "qkv",
            seq_len=seq_len,
            k=self.hidden_size,
            n=3 * self.hidden_size,
            compute_role="attention",
        )
        fused = ttnn.reshape(fused, [self.batch, seq_len, 3 * self.hidden_size])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query, key = self._prefill_rope(query, key, seq_len)
        query = ttnn.slice(query, [0, 0, 0, 0], [self.batch, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        value = ttnn.slice(value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        key_fill = ttnn.typecast(key, self.optimization_policy.kv_cache_dtype)
        value_fill = ttnn.typecast(value, self.optimization_policy.kv_cache_dtype)
        for batch_idx in range(self.batch):
            user_key = ttnn.slice(
                key_fill, [batch_idx, 0, 0, 0], [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim]
            )
            user_value = ttnn.slice(
                value_fill, [batch_idx, 0, 0, 0], [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim]
            )
            ttnn.experimental.paged_fill_cache(
                key_cache, user_key, page_table, batch_idx=user_id + batch_idx, block_size=self.page_size
            )
            ttnn.experimental.paged_fill_cache(
                value_cache, user_value, page_table, batch_idx=user_id + batch_idx, block_size=self.page_size
            )
        if seq_len <= PREFILL_SDPA_MAX_SEQ:
            attended = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attended_chunks = []
            chunk_start = 0
            while chunk_start < seq_len:
                chunk_capacity = PREFILL_SDPA_MAX_SEQ if chunk_start == 0 else 4 * ttnn.TILE_SIZE
                chunk_len = min(chunk_capacity, seq_len - chunk_start)
                padded_len = math.ceil(chunk_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                query_chunk = ttnn.slice(
                    query,
                    [0, 0, chunk_start, 0],
                    [self.batch, self.num_heads, chunk_start + chunk_len, self.head_dim],
                )
                if padded_len != chunk_len:
                    query_chunk = ttnn.pad(
                        query_chunk, [(0, 0), (0, 0), (0, padded_len - chunk_len), (0, 0)], value=0.0
                    )
                if chunk_start == 0 and chunk_len == PREFILL_SDPA_MAX_SEQ:
                    prefix_key = ttnn.slice(
                        key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, chunk_len, self.head_dim]
                    )
                    prefix_value = ttnn.slice(
                        value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, chunk_len, self.head_dim]
                    )
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        prefix_key,
                        prefix_value,
                        is_causal=True,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    mask = self._offset_causal_mask(chunk_start=chunk_start, query_len=padded_len, key_len=seq_len)
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        key,
                        value,
                        attn_mask=mask,
                        is_causal=False,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                            packer_l1_acc=False,
                        ),
                    )
                if padded_len != chunk_len:
                    output_chunk = ttnn.slice(
                        output_chunk, [0, 0, 0, 0], [self.batch, self.num_heads, chunk_len, self.head_dim]
                    )
                attended_chunks.append(output_chunk)
                chunk_start += chunk_len
            attended = attended_chunks[0] if len(attended_chunks) == 1 else ttnn.concat(attended_chunks, dim=2)
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = self._prefill_linear(
            attended,
            "o_proj",
            seq_len=seq_len,
            k=self.hidden_size,
            n=self.hidden_size,
            compute_role="attention",
        )
        projected = ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size])
        return self._prefill_mlp(ttnn.add(residual, projected), seq_len)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        use_long_rope,
    ):
        shape = tuple(hidden_states.shape)
        if shape != (1, 1, self.batch, self.hidden_size):
            raise ValueError(f"decode hidden_states must be [1,1,{self.batch},{self.hidden_size}], got {shape}")
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape [{self.batch}], got {tuple(current_positions.shape)}")
        residual = ttnn.to_memory_config(hidden_states, self.decode_residual_memory_config)
        normalized = self._decode_norm(residual, self.weights["input_norm"])
        if self.optimization_policy.split_decode_qkv:
            qkv_parts = [
                ttnn.to_memory_config(
                    self._decode_linear(normalized, role, ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for role in ("q_proj", "k_proj", "v_proj")
            ]
            fused = ttnn.concat(qkv_parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            fused = self._decode_linear(normalized, "qkv", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        query, key = self._decode_rope(query, key, current_positions, use_long_rope=use_long_rope)
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
        )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            cur_pos_tensor=current_positions,
            page_table_tensor=page_table,
            scale=self.scale,
            program_config=self.decode_sdpa_program_config,
            compute_kernel_config=self.compute_configs["attention"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self._decode_concat_memory_config())
        attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
        if self.batch < ttnn.TILE_SIZE:
            attended = ttnn.slice(attended, [0, 0, 0, 0], [1, 1, self.batch, self.hidden_size])
        attended = ttnn.to_memory_config(attended, self.decode_residual_memory_config)
        projected = self._decode_linear(attended, "o_proj", self.decode_residual_memory_config)
        return self._decode_mlp(ttnn.add(residual, projected, memory_config=self.decode_residual_memory_config))


__all__ = [
    "DEFAULT_OPTIMIZATION_POLICY",
    "DEFAULT_PAGE_SIZE",
    "HF_ADVERTISED_CONTEXT",
    "OptimizationPolicy",
    "OptimizedDecoder",
    "PCC_ACCEPTANCE",
]
