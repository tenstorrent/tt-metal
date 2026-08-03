# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device decoder for ``google/gemma-4-26B-A4B-it``.

The optimized class deliberately owns both public forward paths.  It reuses
the functional decoder's correctness helpers and paged-cache implementation,
but never dispatches through ``FunctionalDecoder.prefill_forward`` or
``FunctionalDecoder.decode_forward``.  Material operations are overridden
below as optimization candidates are selected.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MLP_INTERMEDIATE_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
)
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config
from models.demos.gemma4.tt.experts.operations import apply_geglu


def optimization_candidate_matrix() -> dict[str, Any]:
    """Return the exact machine-readable decoder sweep contract."""

    dense_roles = {
        "packed_gate_up": {"k": HIDDEN_SIZE, "n": 2 * MLP_INTERMEDIATE_SIZE},
        "dense_down": {"k": MLP_INTERMEDIATE_SIZE, "n": HIDDEN_SIZE},
    }
    sparse_roles = {
        "expert_gate_up": {"k": HIDDEN_SIZE, "n": MOE_INTERMEDIATE_SIZE},
        "expert_down": {"k": MOE_INTERMEDIATE_SIZE, "n": HIDDEN_SIZE},
    }

    def widths(k: int, core_counts: tuple[int, ...]) -> dict[str, list[int]]:
        k_tiles = k // TILE_SIZE
        return {
            str(cores): [width for width in range(1, k_tiles // cores + 1) if (k_tiles // cores) % width == 0]
            for cores in core_counts
            if k_tiles % cores == 0
        }

    for role in dense_roles.values():
        # Blackhole exposes eight DRAM banks to this single-device path.
        # Include every bank count that divides K; in particular dense-down's
        # 66 K tiles admits 3 and 6 cores, which a power-of-two-only matrix
        # silently omitted.
        core_counts = tuple(cores for cores in range(1, 9) if (role["k"] // TILE_SIZE) % cores == 0)
        role["dram_core_counts"] = list(core_counts)
        role["in0_block_w"] = widths(role["k"], core_counts)
    for role in sparse_roles.values():
        role["in0_block_w"] = widths(role["k"], (1,))

    return {
        "decode_batches": {
            "1": {"logical_m": 1, "physical_m": TILE_SIZE, "per_core_M": 1},
            "32": {"logical_m": 32, "physical_m": TILE_SIZE, "per_core_M": 1},
        },
        "dense_decode": dense_roles,
        "sparse_decode": sparse_roles,
        "weight_compute_pairs": [
            {"weight": "bfloat16", "fidelity": "HiFi4"},
            {"weight": "bfloat8_b", "fidelity": "HiFi2"},
            {"weight": "bfloat8_b", "fidelity": "LoFi"},
            {"weight": "bfloat4_b", "fidelity": "LoFi"},
        ],
        "kv_cache": [{"dtype": "bfloat16", "control": True}, {"dtype": "bfloat8_b", "control": False}],
        "prefill_sequence_lengths": [1024, 1023],
        "prefill_program_families": ["framework_default", "large_multicore_reuse"],
        "movement_families": [
            "dram_interleaved",
            "dram_sharded_weight_l1_width_sharded_output",
            "l1_width_sharded_chain",
        ],
    }


class OptimizedDecoder(FunctionalDecoder):
    """Gemma-4 decoder whose measured runtime is the optimized implementation."""

    implementation = "optimized"
    optimization_candidate = "selected_default"
    sparse_in0_block_w = 11
    dram_dense_gate_cores = 8
    dram_dense_down_cores = 6
    dram_dense_gate_in0_block_w = 11
    dram_dense_down_in0_block_w = 11
    dram_dense_weight_dtype = ttnn.bfloat8_b
    dense_compute_fidelity: ttnn.MathFidelity | None = None
    expert_compute_fidelity: ttnn.MathFidelity | None = ttnn.MathFidelity.LoFi

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: Any,
        attention_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        dense_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        expert_weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        **kwargs: Any,
    ) -> "OptimizedDecoder":
        """Load sensitive tensors in BF16, then reduce projection groups.

        Keeping the group conversion explicit prevents a blanket low-precision
        loader policy from reducing RMSNorm, router, or layer-scalar tensors.
        """
        decoder = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weight_dtype=ttnn.bfloat16,
            expert_weight_dtype=ttnn.bfloat16,
            **kwargs,
        )

        def material_compute_config(
            fidelity: ttnn.MathFidelity | None,
        ) -> ttnn.DeviceComputeKernelConfig | None:
            if fidelity is None:
                return None
            arch = mesh_device.arch() if hasattr(mesh_device, "arch") else ttnn.device.GetDefaultDevice().arch()
            return ttnn.init_device_compute_kernel_config(
                arch,
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        decoder.dense_compute_kernel_config = material_compute_config(cls.dense_compute_fidelity)
        decoder.expert_compute_kernel_config = material_compute_config(cls.expert_compute_fidelity)

        def cast(weight: ttnn.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
            if weight.dtype == dtype:
                return weight
            return ttnn.typecast(weight, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        decoder.weights = replace(
            decoder.weights,
            qkv=cast(decoder.weights.qkv, attention_weight_dtype),
            o_proj=cast(decoder.weights.o_proj, attention_weight_dtype),
            mlp_gate=cast(decoder.weights.mlp_gate, dense_weight_dtype),
            mlp_up=cast(decoder.weights.mlp_up, dense_weight_dtype),
            mlp_down=cast(decoder.weights.mlp_down, dense_weight_dtype),
            expert_gate=cast(decoder.weights.expert_gate, expert_weight_dtype),
            expert_up=cast(decoder.weights.expert_up, expert_weight_dtype),
            expert_down=cast(decoder.weights.expert_down, expert_weight_dtype),
        )
        decoder.expert_weights = replace(
            decoder.expert_weights,
            gate_proj=decoder.weights.expert_gate,
            up_proj=decoder.weights.expert_up,
            down_proj=decoder.weights.expert_down,
        )
        decoder.precision_policy = {
            "attention_weights": attention_weight_dtype,
            "dense_weights": dense_weight_dtype,
            "expert_weights": expert_weight_dtype,
            "activations": decoder.activation_dtype,
            "norms": ttnn.bfloat16,
            "router": ttnn.float32,
            "dense_compute_fidelity": (
                str(cls.dense_compute_fidelity) if cls.dense_compute_fidelity is not None else "framework_default"
            ),
            "expert_compute_fidelity": (
                str(cls.expert_compute_fidelity)
                if cls.expert_compute_fidelity is not None
                else "gate_HiFi4_up_down_framework_default"
            ),
        }
        expected_dtypes = {
            "qkv": attention_weight_dtype,
            "o_proj": attention_weight_dtype,
            "mlp_gate": dense_weight_dtype,
            "mlp_up": dense_weight_dtype,
            "mlp_down": dense_weight_dtype,
            "expert_gate": expert_weight_dtype,
            "expert_up": expert_weight_dtype,
            "expert_down": expert_weight_dtype,
        }
        actual_dtypes = {name: getattr(decoder.weights, name).dtype for name in expected_dtypes}
        if actual_dtypes != expected_dtypes:
            raise RuntimeError(
                f"optimized weight policy did not materialize: expected={expected_dtypes}, actual={actual_dtypes}"
            )
        decoder.router_combined_scale = ttnn.mul(
            decoder.weights.router_scale,
            decoder.router_hidden_scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        decoder.mlp_gate_up = ttnn.concat(
            [decoder.weights.mlp_gate, decoder.weights.mlp_up],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # The packed tensor has exactly the combined source volume. Release
        # both source buffers so packing does not reduce the context contract.
        ttnn.deallocate(decoder.weights.mlp_gate)
        ttnn.deallocate(decoder.weights.mlp_up)
        decoder.candidate_provenance = {
            "name": cls.optimization_candidate,
            "dense_weight_dtype": str(dense_weight_dtype),
            "expert_weight_dtype": str(expert_weight_dtype),
            "dense_compute_fidelity": decoder.precision_policy["dense_compute_fidelity"],
            "expert_compute_fidelity": decoder.precision_policy["expert_compute_fidelity"],
        }
        if cls.optimization_candidate.startswith("dram_sharded_dense_"):
            decoder._materialize_dram_sharded_dense_weights()
        return decoder

    @staticmethod
    def _core_range_set(core_count: int) -> ttnn.CoreRangeSet:
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_count - 1, 0))})

    @classmethod
    def _width_sharded_memory_config(
        cls,
        *,
        core_count: int,
        shard_height: int,
        shard_width: int,
        buffer_type: ttnn.BufferType,
    ) -> ttnn.MemoryConfig:
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type,
            ttnn.ShardSpec(
                cls._core_range_set(core_count),
                (shard_height, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _materialize_dram_sharded_dense_weights(self) -> None:
        """Prepare the two dense decode projections in DRAM-bank shards."""
        gate_cores, down_cores = self.dram_dense_gate_cores, self.dram_dense_down_cores
        gate_padded_width = 4352
        down_padded_width = 3072
        gate_weight_mem = self._width_sharded_memory_config(
            core_count=gate_cores,
            shard_height=HIDDEN_SIZE,
            shard_width=gate_padded_width // gate_cores,
            buffer_type=ttnn.BufferType.DRAM,
        )
        down_weight_mem = self._width_sharded_memory_config(
            core_count=down_cores,
            shard_height=MLP_INTERMEDIATE_SIZE,
            shard_width=down_padded_width // down_cores,
            buffer_type=ttnn.BufferType.DRAM,
        )
        # Retain the interleaved tensors for prefill; the DRAM-sharded
        # program family is decode-only and cannot serve the prefill planner.
        packed_padded = ttnn.pad(
            self.mlp_gate_up,
            [(0, 0), (0, 0), (0, 0), (0, gate_padded_width - 2 * MLP_INTERMEDIATE_SIZE)],
            0.0,
        )
        down_padded = ttnn.pad(
            self.weights.mlp_down,
            [(0, 0), (0, 0), (0, 0), (0, down_padded_width - HIDDEN_SIZE)],
            0.0,
        )
        self.mlp_gate_up_decode = ttnn.to_memory_config(
            ttnn.typecast(packed_padded, self.dram_dense_weight_dtype), gate_weight_mem
        )
        self.mlp_down_decode = ttnn.to_memory_config(
            ttnn.typecast(down_padded, self.dram_dense_weight_dtype), down_weight_mem
        )
        self.candidate_provenance.update(
            {
                "family": "dram_sharded_weight_l1_width_sharded_activation_output",
                "packed_gate_up": {
                    "core_count": gate_cores,
                    "input_shard": [TILE_SIZE, HIDDEN_SIZE // gate_cores],
                    "logical_n": 2 * MLP_INTERMEDIATE_SIZE,
                    "padded_n": gate_padded_width,
                    "weight_shard": [HIDDEN_SIZE, gate_padded_width // gate_cores],
                    "output_shard": [TILE_SIZE, gate_padded_width // gate_cores],
                    "in0_block_w": self.dram_dense_gate_in0_block_w,
                    "per_core_M": 1,
                    "per_core_N": (gate_padded_width // TILE_SIZE) // gate_cores,
                },
                "dense_down": {
                    "core_count": down_cores,
                    "input_shard": [TILE_SIZE, MLP_INTERMEDIATE_SIZE // down_cores],
                    "logical_n": HIDDEN_SIZE,
                    "padded_n": down_padded_width,
                    "weight_shard": [MLP_INTERMEDIATE_SIZE, down_padded_width // down_cores],
                    "output_shard": [TILE_SIZE, down_padded_width // down_cores],
                    "in0_block_w": self.dram_dense_down_in0_block_w,
                    "per_core_M": 1,
                    "per_core_N": (down_padded_width // TILE_SIZE) // down_cores,
                },
                "weight_dtype": str(self.dram_dense_weight_dtype),
                "compute": "HiFi4 fp32_dest_acc_en",
            }
        )

    def _dram_sharded_dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate_cores, down_cores = self.dram_dense_gate_cores, self.dram_dense_down_cores
        gate_padded_width = 4352
        down_padded_width = 3072
        compute = self.dense_compute_kernel_config or self.correctness_compute_config
        gate_input_mem = self._width_sharded_memory_config(
            core_count=gate_cores,
            shard_height=TILE_SIZE,
            shard_width=HIDDEN_SIZE // gate_cores,
            buffer_type=ttnn.BufferType.L1,
        )
        gate_output_mem = self._width_sharded_memory_config(
            core_count=gate_cores,
            shard_height=TILE_SIZE,
            shard_width=gate_padded_width // gate_cores,
            buffer_type=ttnn.BufferType.L1,
        )
        x = ttnn.to_memory_config(x, gate_input_mem)
        gate_up = ttnn.linear(
            x,
            self.mlp_gate_up_decode,
            dtype=self.activation_dtype,
            memory_config=gate_output_mem,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=self.dram_dense_gate_in0_block_w,
                per_core_M=1,
                per_core_N=(gate_padded_width // TILE_SIZE) // gate_cores,
                fused_activation=None,
            ),
            compute_kernel_config=compute,
        )
        gate_up = ttnn.sharded_to_interleaved(gate_up, ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(
            gate_up,
            [0, 0, 0, 0],
            [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], MLP_INTERMEDIATE_SIZE],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, MLP_INTERMEDIATE_SIZE],
            [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], 2 * MLP_INTERMEDIATE_SIZE],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG)
        down_input_mem = self._width_sharded_memory_config(
            core_count=down_cores,
            shard_height=TILE_SIZE,
            shard_width=MLP_INTERMEDIATE_SIZE // down_cores,
            buffer_type=ttnn.BufferType.L1,
        )
        down_output_mem = self._width_sharded_memory_config(
            core_count=down_cores,
            shard_height=TILE_SIZE,
            shard_width=down_padded_width // down_cores,
            buffer_type=ttnn.BufferType.L1,
        )
        hidden = ttnn.to_memory_config(hidden, down_input_mem)
        output = ttnn.linear(
            hidden,
            self.mlp_down_decode,
            dtype=self.activation_dtype,
            memory_config=down_output_mem,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=self.dram_dense_down_in0_block_w,
                per_core_M=1,
                per_core_N=(down_padded_width // TILE_SIZE) // down_cores,
                fused_activation=None,
            ),
            compute_kernel_config=compute,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.slice(
            output,
            [0, 0, 0, 0],
            [output.shape[0], output.shape[1], output.shape[2], HIDDEN_SIZE],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        self._executing_decode = False
        batch = hidden_states.shape[0]
        if batch < 1:
            raise ValueError("prefill requires at least one batch row")
        if batch == 1:
            return self._prefill_forward_single_user_optimized(
                hidden_states,
                position_cos=position_cos,
                position_sin=position_sin,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                chunk_page_table=chunk_page_table,
                cache_position_modulo=cache_position_modulo,
            )
        if user_id + batch > page_table.shape[0]:
            raise ValueError(
                f"prefill batch rows [{user_id}, {user_id + batch}) exceed page table batch {page_table.shape[0]}"
            )
        outputs = []
        for batch_index in range(batch):
            table_index = user_id + batch_index
            outputs.append(
                self._prefill_forward_single_user_optimized(
                    ttnn.slice(
                        hidden_states,
                        [batch_index, 0, 0, 0],
                        [batch_index + 1, 1, hidden_states.shape[2], hidden_states.shape[3]],
                    ),
                    position_cos=ttnn.slice(
                        position_cos,
                        [batch_index, 0, 0, 0],
                        [batch_index + 1, 1, position_cos.shape[2], position_cos.shape[3]],
                    ),
                    position_sin=ttnn.slice(
                        position_sin,
                        [batch_index, 0, 0, 0],
                        [batch_index + 1, 1, position_sin.shape[2], position_sin.shape[3]],
                    ),
                    page_table=ttnn.slice(
                        page_table,
                        [table_index, 0],
                        [table_index + 1, page_table.shape[1]],
                    ),
                    kv_cache=kv_cache,
                    user_id=0,
                    chunk_page_table=(
                        None
                        if chunk_page_table is None
                        else ttnn.slice(
                            chunk_page_table,
                            [table_index, 0],
                            [table_index + 1, chunk_page_table.shape[1]],
                        )
                    ),
                    cache_position_modulo=cache_position_modulo,
                )
            )
        return ttnn.concat(outputs, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _prefill_forward_single_user_optimized(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        chunk_page_table: ttnn.Tensor | None,
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        logical_seq_len = hidden_states.shape[-2]
        if logical_seq_len < 1:
            raise ValueError("prefill requires at least one logical token")
        padded_seq_len = ((logical_seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        if padded_seq_len != logical_seq_len:
            pad = [(0, 0), (0, 0), (0, padded_seq_len - logical_seq_len), (0, 0)]
            hidden_states = ttnn.pad(hidden_states, pad, 0.0)
            position_cos = ttnn.pad(position_cos, pad, 0.0)
            position_sin = ttnn.pad(position_sin, pad, 0.0)

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_prefill(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            cache_position_modulo=cache_position_modulo,
            logical_seq_len=logical_seq_len,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_1 = self._rms_norm(
            self._dense_mlp(self._rms_norm(hidden_states, self.weights.pre_ff_ln)),
            self.weights.post_ff_ln_1,
        )
        router_weights = self._router_weights(residual)
        hidden_2 = self._rms_norm(
            self._moe_prefill(self._rms_norm(residual, self.weights.pre_ff_ln_2), router_weights),
            self.weights.post_ff_ln_2,
        )
        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._apply_layer_scalar(hidden_states)
        if padded_seq_len != logical_seq_len:
            hidden_states = ttnn.slice(
                hidden_states,
                starts=[0, 0, 0, 0],
                ends=[1, 1, logical_seq_len, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return hidden_states

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        self._executing_decode = True
        if hidden_states.shape[-2] < 1:
            raise ValueError("decode requires at least one batch row")
        residual = hidden_states
        attn_out = self._attention_decode(
            self._rms_norm(hidden_states, self.weights.input_ln),
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        hidden_states = ttnn.add(
            residual,
            self._rms_norm(attn_out, self.weights.post_attn_ln),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        residual = hidden_states
        hidden_1 = self._rms_norm(
            self._dense_mlp(self._rms_norm(hidden_states, self.weights.pre_ff_ln)),
            self.weights.post_ff_ln_1,
        )
        router_weights = self._router_weights(residual)
        hidden_2 = self._rms_norm(
            self._moe_decode(self._rms_norm(residual, self.weights.pre_ff_ln_2), router_weights),
            self.weights.post_ff_ln_2,
        )
        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._apply_layer_scalar(hidden_states)

    def _router_weights(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        """Router with the two immutable scale factors folded at setup."""
        tokens = residual.shape[-2]
        router_in = self._rms_norm(residual, None)
        router_in = ttnn.mul(router_in, self.router_combined_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.weights.router_proj,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        routing = ttnn.mul(routing, self.weights.router_per_expert_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.typecast(routing, ttnn.bfloat16)
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])

    def _moe_decode_single_user(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Top-8 sparse experts using the selected widest legal K block."""
        batch = hidden_states.shape[2]
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _build_sparse_matmul_config(batch, MOE_INTERMEDIATE_SIZE, in0_block_w=self.sparse_in0_block_w)
        down_config = _build_sparse_matmul_config(batch, HIDDEN_SIZE, in0_block_w=self.sparse_in0_block_w)
        common = {
            "sparsity": sparsity,
            "nnz": TOP_K_EXPERTS,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_tile": output_tile,
            "program_config": gate_up_config,
            "dtype": self.activation_dtype,
        }
        gate = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_gate,
            compute_kernel_config=self.expert_compute_kernel_config or self.correctness_compute_config,
            **common,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, sparse_intermediate))
        up = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_up,
            compute_kernel_config=self.expert_compute_kernel_config,
            **common,
        )
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, sparse_intermediate))
        down_input = ttnn.transpose(apply_geglu(gate, up), 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, sparse_intermediate))
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
            compute_kernel_config=self.expert_compute_kernel_config,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.sum(ttnn.mul(next_states, routing_3d), dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Packed same-input gate/up projection, including on-device split."""
        if self.optimization_candidate.startswith("dram_sharded_dense_") and getattr(self, "_executing_decode", False):
            return self._dram_sharded_dense_mlp(x)
        prefill_program_config = None
        if self.optimization_candidate == "large_prefill_multicore" and x.shape[-2] >= 1024:
            prefill_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(6, 8),
                in0_block_w=11,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=4,
                per_core_N=22,
                transpose_mcast=False,
                fused_activation=None,
            )
            self.candidate_provenance.update(
                {
                    "family": "large_prefill_multicore_reuse",
                    "packed_gate_up": {
                        "grid": [6, 8],
                        "in0_block_w": 11,
                        "out_subblock": [1, 2],
                        "per_core_M": 4,
                        "per_core_N": 22,
                    },
                }
            )
        gate_up = ttnn.linear(
            x,
            self.mlp_gate_up,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=prefill_program_config,
            compute_kernel_config=self.dense_compute_kernel_config,
        )
        gate = ttnn.slice(
            gate_up,
            starts=[0, 0, 0, 0],
            ends=[gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], MLP_INTERMEDIATE_SIZE],
            steps=[1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.slice(
            gate_up,
            starts=[0, 0, 0, MLP_INTERMEDIATE_SIZE],
            ends=[gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], 2 * MLP_INTERMEDIATE_SIZE],
            steps=[1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.dense_compute_kernel_config,
        )

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs: Any) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
