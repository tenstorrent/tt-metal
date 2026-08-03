# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-device optimized Qwen3.6-27B decoder layer.

This path starts from :class:`FusedDecoder` and keeps its packed projection,
cache, recurrent-state, non-aligned sequence, and public tensor contracts.  The
optimization policy is explicit so profiler evidence can be tied to the dtype
and fidelity actually used by each material projection group.
"""

from __future__ import annotations

import math
import os
import types
from dataclasses import dataclass

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder


@dataclass(frozen=True)
class OptimizedPrecisionPolicy:
    attention_weight_dtype: ttnn.DataType
    mlp_gate_up_weight_dtype: ttnn.DataType
    mlp_down_weight_dtype: ttnn.DataType
    math_fidelity: ttnn.MathFidelity
    dram_sharded: bool = False
    max_in0_block_w: int = 2
    large_prefill_config: bool = False


class _ScopedTtnnProxy:
    """Delegate TTNN calls while overriding selected ops for one invocation."""

    def __init__(self, *, linear, concat=None):
        self.linear = linear
        if concat is not None:
            self.concat = concat

    def __getattr__(self, name):
        return getattr(ttnn, name)


POLICIES = {
    "bfp8_hifi2": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat8_b,
        mlp_down_weight_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp8_lofi": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat8_b,
        mlp_down_weight_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bfp4_mlp_lofi": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bfp4_all_lofi": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bfp4_all_dram_w4": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded=True,
        max_in0_block_w=4,
    ),
    "bfp4_all_dram_w8": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded=True,
        max_in0_block_w=8,
        large_prefill_config=True,
    ),
    "bfp4_all_dram_w8_default_prefill": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded=True,
        max_in0_block_w=8,
    ),
    "bfp4_all_dram_w10": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded=True,
        max_in0_block_w=10,
    ),
    "bfp4_all_dram_w20": OptimizedPrecisionPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        dram_sharded=True,
        max_in0_block_w=20,
    ),
}


class OptimizedDecoder(FusedDecoder):
    """Fused decoder with explicit reduced-weight and compute policies."""

    @classmethod
    def from_state_dict(cls, state_dict, *, optimization_policy="bfp4_all_dram_w8", **kwargs):
        if optimization_policy not in POLICIES:
            raise ValueError(f"unknown optimization_policy={optimization_policy!r}; expected one of {tuple(POLICIES)}")
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.optimization_policy_name = optimization_policy
        decoder.optimization_policy = POLICIES[optimization_policy]
        decoder.compute_kernel_config = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=decoder.optimization_policy.math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.decode_program_configs = {}
        decoder.decode_input_memory_configs = {}
        decoder.decode_output_memory_configs = {}
        decoder.prefill_weights = {}

        attention_names = (
            ("packed_qkv", "o_proj") if decoder.layer_kind == "full_attention" else ("packed_linear_inputs", "out_proj")
        )
        # FusedDecoder retains the unpacked BF16 source projections after
        # building the packed tensor.  No fused/optimized runtime method reads
        # them; release those references before materializing the two compact
        # BFP4 representations.  The resulting persistent projection storage
        # is smaller than the fused baseline, despite keeping phase-specific
        # prefill and decode layouts.
        packed_source_names = (
            ("q_proj", "k_proj", "v_proj")
            if decoder.layer_kind == "full_attention"
            else ("in_qkv", "in_z", "in_b", "in_a")
        )
        for name in packed_source_names:
            decoder.weights.pop(name, None)
        for name in attention_names:
            decoder.weights[name] = ttnn.typecast(
                decoder.weights[name],
                decoder.optimization_policy.attention_weight_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        for name in ("mlp_gate", "mlp_up"):
            decoder.weights[name] = ttnn.typecast(
                decoder.weights[name],
                decoder.optimization_policy.mlp_gate_up_weight_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        decoder.weights["mlp_down"] = ttnn.typecast(
            decoder.weights["mlp_down"],
            decoder.optimization_policy.mlp_down_weight_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if decoder.optimization_policy.dram_sharded:
            # Prefill uses the large-M interleaved matmul family; decode uses
            # the one-tile-M DRAM-sharded family.  Keep each representation
            # materialized so neither measured path reshards weights at run
            # time.
            decoder.prefill_weights = {
                name: decoder.weights[name] for name in (["mlp_gate", "mlp_up", "mlp_down"] + list(attention_names))
            }
            decoder._materialize_dram_sharded_weights()
        return decoder

    @staticmethod
    def _largest_divisor_at_most(value, maximum):
        return next(divisor for divisor in range(min(value, maximum), 0, -1) if value % divisor == 0)

    def _materialize_dram_sharded_weights(self):
        dram_size = self.mesh_device.dram_grid_size()
        dram_cores = dram_size.x * dram_size.y
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1),
                )
            }
        )
        # The current DRAM-sharded matmul factory maps Blackhole's eight DRAM
        # banks onto the first eight logical workers.  Keep the activation and
        # output shard contract on that exact contiguous grid.
        worker_grid = ttnn.CoreGrid(x=dram_cores, y=1)
        material_names = ["mlp_gate", "mlp_up", "mlp_down"]
        material_names += (
            ["packed_qkv", "o_proj"] if self.layer_kind == "full_attention" else ["packed_linear_inputs", "out_proj"]
        )
        for name in material_names:
            weight = self.weights[name]
            k, n = int(weight.shape[-2]), int(weight.shape[-1])
            padded_n = math.ceil(n / (32 * dram_cores)) * (32 * dram_cores)
            weight_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (k, padded_n // dram_cores),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            self.weights[name] = ttnn.to_memory_config(weight, weight_memory_config)
            input_tiles_per_core = k // (32 * dram_cores)
            block_w = self._largest_divisor_at_most(
                input_tiles_per_core,
                self.optimization_policy.max_in0_block_w,
            )
            self.decode_program_configs[name] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block_w,
                per_core_M=1,
                per_core_N=math.ceil(n / (32 * dram_cores)),
                fused_activation=None,
            )
            self.decode_input_memory_configs[name] = ttnn.create_sharded_memory_config(
                shape=(32, k // dram_cores),
                core_grid=worker_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.decode_output_memory_configs[name] = ttnn.create_sharded_memory_config(
                shape=(32, math.ceil(n / (32 * dram_cores)) * 32),
                core_grid=worker_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

    def _linear(self, activation, weight_name, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        program_config = self.decode_program_configs.get(weight_name)
        if program_config is not None:
            activation = ttnn.to_memory_config(
                activation,
                self.decode_input_memory_configs[weight_name],
            )
            memory_config = self.decode_output_memory_configs[weight_name]
        return ttnn.linear(
            activation,
            self.weights[weight_name],
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            program_config=program_config,
        )

    def prefill_forward(self, *, hidden_states, page_table, current_positions):
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "input_norm")
        hidden_states = self._token_mixer_prefill(hidden_states, page_table, current_positions)
        hidden_states = ttnn.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "post_attention_norm")
        hidden_states = self._mlp_prefill(hidden_states)
        return ttnn.add(residual, hidden_states)

    def _decode_residual_memory_config(self):
        return self.decode_input_memory_configs["mlp_gate"]

    def _rms_norm_decode_sharded(self, hidden_states, name):
        memory_config = self._decode_residual_memory_config()
        if not hidden_states.is_sharded():
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config)
        block_w = self.hidden_size // 8 // 32
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights[name],
            memory_config=memory_config,
            program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[8, 1],
                subblock_w=4,
                block_h=1,
                block_w=block_w,
                inplace=False,
            ),
        )

    def decode_forward(self, *, hidden_states, page_table, current_positions):
        if not self.optimization_policy.dram_sharded:
            return super().decode_forward(
                hidden_states=hidden_states,
                page_table=page_table,
                current_positions=current_positions,
            )
        residual_memory_config = self._decode_residual_memory_config()
        residual = ttnn.to_memory_config(hidden_states, residual_memory_config)
        hidden_states = self._rms_norm_decode_sharded(residual, "input_norm")
        hidden_states = self._token_mixer_decode(hidden_states, page_table, current_positions)
        if not hidden_states.is_sharded():
            hidden_states = ttnn.to_memory_config(hidden_states, residual_memory_config)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=residual_memory_config)
        residual = hidden_states
        hidden_states = self._rms_norm_decode_sharded(hidden_states, "post_attention_norm")
        hidden_states = self._mlp(hidden_states)
        return ttnn.add(residual, hidden_states, memory_config=residual_memory_config)

    def _mlp(self, hidden_states):
        gate = self._linear(hidden_states, "mlp_gate")
        up = self._linear(hidden_states, "mlp_up")
        hidden_states = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._linear(hidden_states, "mlp_down")

    def _mlp_prefill(self, hidden_states):
        def project(activation, name):
            weight = self.prefill_weights.get(name, self.weights[name])
            return ttnn.linear(
                activation,
                weight,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
                program_config=self._prefill_program_config(activation, weight),
            )

        gate = project(hidden_states, "mlp_gate")
        up = project(hidden_states, "mlp_up")
        hidden_states = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return project(hidden_states, "mlp_down")

    def _prefill_program_config(self, activation, weight):
        if not self.optimization_policy.large_prefill_config:
            return None
        activation_shape = [int(activation.shape[index]) for index in range(len(activation.shape))]
        # TTNN pads the sequence axis independently for each leading batch
        # slice; do not flatten logical rows before accounting for that tile
        # padding (batch32 x seq33 is 64 M tiles, not 33).
        m_tiles = math.prod(activation_shape[:-2]) * math.ceil(activation_shape[-2] / 32)
        # The explicit multicast setup cost loses at the two-tile batch-1
        # point; retain TTNN's default small-M factory there. It wins
        # decisively once enough M tiles exist to occupy the grid.
        if m_tiles < 10:
            return None
        k_tiles = int(weight.shape[-2]) // 32
        n_tiles = math.ceil(int(weight.shape[-1]) / 32)
        grid_y = min(m_tiles, 10)
        # Eight columns keeps the 16,480-wide packed linear projection from
        # collapsing to a five-column divisor and over-allocating per-core
        # output CBs. The factory pads the final column when N is not divisible.
        grid_x = min(n_tiles, 8)
        per_core_n = math.ceil(n_tiles / grid_x)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=self._largest_divisor_at_most(k_tiles, 4),
            out_subblock_h=1,
            out_subblock_w=self._largest_divisor_at_most(per_core_n, 4),
            per_core_M=math.ceil(m_tiles / grid_y),
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )

    def _call_fused_with_scoped_ttnn(self, method, *args, linear, concat=None):
        """Call a fused method with invocation-local TTNN op overrides.

        ``FusedDecoder`` spells out the already-validated attention and
        recurrent topology.  Rebinding only its function-global ``ttnn`` name
        keeps that implementation while avoiding any mutation of the
        process-global TTNN module (and therefore remains thread-safe).
        """
        method_globals = method.__globals__.copy()
        method_globals["ttnn"] = _ScopedTtnnProxy(linear=linear, concat=concat)
        scoped_method = types.FunctionType(
            method.__code__,
            method_globals,
            name=method.__name__,
            argdefs=method.__defaults__,
            closure=method.__closure__,
        )
        return scoped_method(self, *args)

    def _optimized_prefill_linear(self, activation, weight, **kwargs):
        weight_name = next((name for name, value in self.weights.items() if value is weight), None)
        if weight_name in self.prefill_weights:
            weight = self.prefill_weights[weight_name]
        kwargs["compute_kernel_config"] = self.compute_kernel_config
        kwargs["dtype"] = ttnn.bfloat16
        kwargs["program_config"] = self._prefill_program_config(activation, weight)
        return ttnn.linear(activation, weight, **kwargs)

    def _optimized_decode_linear(self, activation, weight, **kwargs):
        weight_name = next(name for name, value in self.weights.items() if value is weight)
        program_config = self.decode_program_configs.get(weight_name)
        if program_config is not None:
            activation = ttnn.to_memory_config(
                activation,
                self.decode_input_memory_configs[weight_name],
            )
            kwargs["memory_config"] = self.decode_output_memory_configs[weight_name]
            kwargs["program_config"] = program_config
        kwargs["compute_kernel_config"] = self.compute_kernel_config
        kwargs["dtype"] = ttnn.bfloat16
        output = ttnn.linear(activation, weight, **kwargs)
        if weight_name == "packed_qkv":
            output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
        if weight_name == "packed_linear_inputs" and output.is_sharded():
            # Beta/decay are much narrower than one projection shard.
            # Cross once before exact slicing; recurrent math remains in its
            # original dedicated layouts.
            output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
        return output

    def _partial_rope_decode(self, tensor, current_positions):
        """Default-off reproduction hook for the rejected advisor repeat placements."""
        scope = os.environ.get("QWEN_ADVISOR_ROPE_REPEAT_SCOPE", "")
        cores = int(os.environ.get("QWEN_ADVISOR_ROPE_REPEAT_CORES", "0"))
        heads = int(tensor.shape[2])
        selected = scope == "both" or (scope == "query" and heads == self.num_heads) or (
            scope == "key" and heads == self.num_kv_heads
        )
        if not selected or cores <= 0:
            return super()._partial_rope_decode(tensor, current_positions)

        rotary_dim = int(self.head_dim * float(self.hf_config.partial_rotary_factor))
        rotary, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
        cos = ttnn.embedding(current_positions, self.rope["cos"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(current_positions, self.rope["sin"], layout=ttnn.TILE_LAYOUT)
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)[:, : self.batch, :, :]
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)[:, : self.batch, :, :]
        grid_width, grid_height = min(cores, 11), math.ceil(cores / 11)
        ranges = {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_width - 1, grid_height - 2))}
        ranges.add(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, grid_height - 1),
                ttnn.CoreCoord((cores - 1) % grid_width, grid_height - 1),
            )
        )
        repeat_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ttnn.CoreRangeSet(ranges), (32, rotary_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )
        cos = ttnn.repeat(cos, [1, 1, heads, 1], memory_config=repeat_memory)
        sin = ttnn.repeat(sin, [1, 1, heads, 1], memory_config=repeat_memory)
        rotary = ttnn.add(ttnn.multiply(rotary, cos), ttnn.multiply(self._rotate_half(rotary), sin))
        rotary = ttnn.to_memory_config(rotary, ttnn.L1_MEMORY_CONFIG)
        return ttnn.to_memory_config(
            ttnn.concat([rotary, passthrough], dim=-1), self.decode_attention_memory_config
        )

    @staticmethod
    def _optimized_decode_concat(tensors, *args, **kwargs):
        if tensors and tensors[0].is_sharded():
            # Slices preserve the packed projection's physical shard width,
            # so Q/K/V no longer have compatible width-sharded logical shapes
            # for concat. Cross the narrow create-heads boundary once in L1.
            tensors = [ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG) for tensor in tensors]
            kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
        return ttnn.concat(tensors, *args, **kwargs)

    # Route material projection calls through the explicit optimized policy.
    def _full_attention_prefill(self, hidden_states, page_table, current_positions):
        return self._call_fused_with_scoped_ttnn(
            FusedDecoder._full_attention_prefill,
            hidden_states,
            page_table,
            current_positions,
            linear=self._optimized_prefill_linear,
        )

    def _full_attention_decode(self, hidden_states, page_table, current_positions):
        return self._call_fused_with_scoped_ttnn(
            FusedDecoder._full_attention_decode,
            hidden_states,
            page_table,
            current_positions,
            linear=self._optimized_decode_linear,
            concat=self._optimized_decode_concat,
        )

    def _linear_attention_prefill_chunk(self, hidden_states):
        return self._call_fused_with_scoped_ttnn(
            FusedDecoder._linear_attention_prefill_chunk,
            hidden_states,
            linear=self._optimized_prefill_linear,
        )

    def _linear_attention_decode(self, hidden_states):
        return self._call_fused_with_scoped_ttnn(
            FusedDecoder._linear_attention_decode,
            hidden_states,
            linear=self._optimized_decode_linear,
        )
