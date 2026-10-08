# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDA_SOFTPLUS_BETA, KDA_SOFTPLUS_THRESHOLD, KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import _layout as _selection_layout
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_CHUNK_SIZE,
    KDA_NORM_MEMORY_CONFIG,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_RECURRENT_STATE_DTYPE,
    KDAProgramConfig,
    decay_projection_config,
    tuned_projection_matmul_configs,
)
from models.demos.deepseek_v3_d_p.tt.kda.convolution import exchange_convolution_carry
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import KDARecurrence
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for divisor in range(limit, 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _effective_qkv_channel_chunk_size(channels: int, configured_chunk_size: int) -> int:
    """Resolve a configured ceiling to an exact TP-local channel divisor."""
    return ttnn.TILE_SIZE * _largest_divisor_at_most(
        channels // ttnn.TILE_SIZE, configured_chunk_size // ttnn.TILE_SIZE
    )


@dataclass(frozen=True)
class _ProjectedInputs:
    # The fused projection; its leading Q+K+V columns are the convolution channels.
    qkv: ttnn.Tensor
    # The fused projection again; the decay's low-rank activations are its columns from decay_rank_offset.
    decay_rank: ttnn.Tensor
    decay_rank_offset: int
    output_gate: ttnn.Tensor
    output_gate_offset: int
    # The fused projection again; beta's pre-sigmoid logits are its columns from beta_offset.
    beta: ttnn.Tensor
    beta_offset: int


@dataclass(frozen=True)
class KdaState:
    """Caller-owned KDA carries.

    ``recurrent`` is TP-local and must be replicated across the SP axis.
    ``convolution`` is the BF16 row-major DRAM stream tail with shape
    ``[B, kernel_size - 1, Q_local + K_local + V_local]``. Its channels are
    sharded across TP and the complete tail is replicated across SP. This history
    seeds the logical sequence start; the halo exchange supplies predecessor
    histories from projected tokens for the other segments. Construct state with
    :meth:`ttKDA.allocate_state` or reuse a state returned by :meth:`ttKDA.forward`.
    """

    recurrent: ttnn.Tensor
    convolution: ttnn.Tensor


class ttKDA:
    """Prefill KDA for one fixed physical geometry and caller-owned logical state.

    ``active_seq_len`` is the global physical token count, matching MLA's
    construction contract. Each call supplies exactly ``active_seq_len / SP``
    local rows. Construct another instance for a different physical length;
    weights may be shared. Runtime ``actual_start`` changes chronology within
    the constructed graph without changing grouping or reading device values.
    """

    def __init__(
        self,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        layer_idx: int = 0,
        weight_cache_path: Path | None = None,
        tt_ccl: TT_CCL | None = None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        program_config: KDAProgramConfig | None = None,
        weights: KDAWeights | None = None,
        *,
        active_seq_len: int,
    ) -> None:
        if tp_axis not in (0, 1) or sp_axis not in (0, 1) or sp_axis == tp_axis:
            raise ValueError(f"KDA requires distinct 2D SP/TP axes, got SP={sp_axis}, TP={tp_axis}")
        program_config = program_config or KDAProgramConfig()
        self.device = mesh_device
        self.tensor_parallel_axis = tp_axis
        self.sequence_parallel_axis = sp_axis
        self.sequence_parallel_size = (
            tuple(mesh_device.shape)[self.sequence_parallel_axis] if isinstance(mesh_device, ttnn.MeshDevice) else 1
        )
        if active_seq_len <= 0 or active_seq_len % (self.sequence_parallel_size * KDA_CHUNK_SIZE):
            raise ValueError("active_seq_len must give a positive tile-aligned local physical length")
        self.active_seq_len_local = active_seq_len // self.sequence_parallel_size
        self._is_sequence_parallel = self.sequence_parallel_size > 1
        self._tp_cluster_axis = None if not self._is_sequence_parallel else self.tensor_parallel_axis
        self._activate_decay = self._softplus_decay if config.gate_lower_bound is None else self._bounded_decay
        uses_grouped_scan = (
            self.sequence_parallel_size > 1 or program_config.recurrence.local_scan_strategy == "grouped"
        )
        if uses_grouped_scan and config.head_k_dim != config.head_v_dim:
            raise ValueError("grouped KDA affine prefix currently requires K == V")
        self.tp_ccl_topology = program_config.tp_ccl_topology
        self.gated_rms_output_dtype = program_config.gated_rms_output_dtype
        if weights is not None and state_dict is not None:
            raise ValueError("pass either constructed KDAWeights or host state_dict, not both")
        if weights is None:
            weights = load_kda_weights(
                mesh_device,
                config,
                state_dict,
                weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.kda",
                tensor_parallel_axis=tp_axis,
            )
        expected_tp_size = tuple(mesh_device.shape)[tp_axis] if isinstance(mesh_device, ttnn.MeshDevice) else 1
        if weights.tensor_parallel_size != expected_tp_size or weights.tensor_parallel_axis != tp_axis:
            raise ValueError(
                "KDAWeights placement does not match the layer mesh: "
                f"weights TP={weights.tensor_parallel_size} axis={weights.tensor_parallel_axis}, "
                f"layer TP={expected_tp_size} axis={tp_axis}"
            )
        self.weights = weights
        self.tensor_parallel_size = self.weights.tensor_parallel_size
        self.config = replace(config, num_heads=config.num_heads // self.tensor_parallel_size)
        qkv_channel_chunk_size = _effective_qkv_channel_chunk_size(
            self._convolution_width, program_config.qkv_channel_chunk_size
        )
        self.qkv_convolution_program_config = ttnn.QkvCausalConv1dSiluProgramConfig(
            channel_chunk_size=qkv_channel_chunk_size
        )
        if self.tensor_parallel_size > 1 and tt_ccl is None:
            raise ValueError("tt_ccl is required for tensor-parallel KDA")
        self.tt_ccl = tt_ccl
        # Ordinary matmuls (input and decay projections) keep packer L1 accumulation; the input
        # projection takes its fidelity from the program config.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.input_projection_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=program_config.input_projection_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.input_projection_minimal_matmul_config, self.output_projection_program_config = (
            tuned_projection_matmul_configs(
                mesh_device.compute_with_storage_grid_size(),
                self.active_seq_len_local,
                *tuple(self.weights.output_projection.shape)[-2:],
                program_config.input_projection_math_fidelity,
            )
            if program_config.tuned_projection_matmuls
            else (None, None)
        )
        # The bounded gate's per-head scale is folded into the decay projection, which then applies the sigmoid.
        self.decay_activation = (
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if config.gate_lower_bound is not None else None
        )
        self.decay_projection_config = decay_projection_config(
            mesh_device.compute_with_storage_grid_size(), self.active_seq_len_local
        )
        # Experimental KDA operations reject packer_l1_acc=True because their kernels do not
        # accumulate through L1. Keep this separate from projection matmuls, which accept the flag.
        self.kda_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.recurrence = KDARecurrence(
            mesh_device,
            program_config.recurrence,
            sequence_parallel_axis=self.sequence_parallel_axis,
            local_rows=self.active_seq_len_local,
            heads=self.config.num_heads,
            key_dim=self.config.head_k_dim,
            value_dim=self.config.head_v_dim,
            # The bounded gate's lower-bound scale is applied by chunk preparation.
            gate_scale=1.0 if config.gate_lower_bound is None else config.gate_lower_bound,
        )
        self.output_projection_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=program_config.output_projection_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @property
    def _convolution_width(self) -> int:
        return self.config.q_dim + self.config.k_dim + self.config.v_dim

    def allocate_state(self, batch_size: int = 1) -> KdaState:
        """Allocate the canonical device-resident KDA carries for one prefill stream."""
        if batch_size != 1:
            raise ValueError(f"KDA prefill currently requires batch_size=1, got {batch_size}")
        return KdaState(
            recurrent=ttnn.zeros(
                (batch_size, self.config.num_heads, self.config.head_k_dim, self.config.head_v_dim),
                dtype=KDA_RECURRENT_STATE_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            ),
            convolution=ttnn.zeros(
                (batch_size, self.config.conv_kernel_size - 1, self._convolution_width),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    @staticmethod
    def _validate_runtime_bound(bound: ttnn.Tensor, name: str) -> None:
        if not isinstance(bound, ttnn.Tensor):
            raise TypeError(f"{name} must be a device UINT32 scalar")
        if (
            bound.dtype != ttnn.uint32
            or bound.layout != ttnn.ROW_MAJOR_LAYOUT
            or any(dimension != 1 for dimension in bound.shape)
        ):
            raise ValueError(f"device {name} must be a UINT32 row-major scalar")

    def _validate_forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
        actual_start: ttnn.Tensor,
    ) -> None:
        """Validate shape/type plus the documented SP state-distribution contract."""
        self._validate_runtime_bound(actual_start, "actual_start")
        if len(hidden_states.shape) != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states shape {tuple(hidden_states.shape)} must be [B,T,{self.config.hidden_size}]"
            )
        batch = hidden_states.shape[0]
        sequence = hidden_states.shape[1]
        if batch != 1:
            raise ValueError(f"KDA prefill currently requires batch size 1, got B={batch}")
        if sequence <= 0 or sequence % KDA_CHUNK_SIZE != 0:
            raise ValueError(
                f"KDA prefill requires local T to be positive and divisible by {KDA_CHUNK_SIZE}, got T={sequence}"
            )
        if sequence != self.active_seq_len_local:
            raise ValueError(
                f"hidden_states local T={sequence} does not match constructed T={self.active_seq_len_local}"
            )
        expected_recurrent = (batch, self.config.num_heads, self.config.head_k_dim, self.config.head_v_dim)
        expected_convolution = (batch, self.config.conv_kernel_size - 1, self._convolution_width)
        if tuple(state.recurrent.shape) != expected_recurrent:
            raise ValueError(f"recurrent state shape {tuple(state.recurrent.shape)} != {expected_recurrent}")
        if tuple(state.convolution.shape) != expected_convolution:
            raise ValueError(f"convolution state shape {tuple(state.convolution.shape)} != {expected_convolution}")
        if state.recurrent.dtype != KDA_RECURRENT_STATE_DTYPE:
            raise ValueError(f"recurrent state dtype {state.recurrent.dtype} != {KDA_RECURRENT_STATE_DTYPE}")
        if state.convolution.dtype != ttnn.bfloat16 or state.convolution.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("convolution state must be BF16 row-major")

    def _convolve_qkv(
        self,
        qkv: ttnn.Tensor,
        incoming_layer_carry: ttnn.Tensor,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        config = self.config
        if not self._is_sequence_parallel:
            # The last three valid rows, selected on device from the chronology.
            (new_state,) = ttnn.experimental.kda.select_history_rows(
                qkv,
                _selection_layout.LOCAL_FINAL_HISTORY,
                actual_start,
                self.sequence_parallel_axis,
                self.active_seq_len_local,
                width=self._convolution_width,
                actual_end=actual_end,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            predecessor = incoming_layer_carry
        else:
            predecessor, new_state = exchange_convolution_carry(
                qkv,
                sequence_parallel_axis=self.sequence_parallel_axis,
                actual_start=actual_start,
                actual_end=actual_end,
                local_rows=self.active_seq_len_local,
                width=self._convolution_width,
            )
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv,
            incoming_layer_carry,
            *self.weights.convolution_taps,
            config.q_dim,
            config.k_dim,
            config.v_dim,
            program_config=self.qkv_convolution_program_config,
            actual_start=actual_start,
            sequence_parallel_axis=self.sequence_parallel_axis,
            predecessor_carry=predecessor,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return q, k, v, new_state

    def _project_inputs(
        self,
        hidden_states: ttnn.Tensor,
    ) -> _ProjectedInputs:
        """Run the fused input projection and split its semantic outputs."""
        config = self.config
        weights = self.weights
        if self.input_projection_minimal_matmul_config is not None:
            projected = ttnn.experimental.minimal_matmul(
                hidden_states,
                weights.input_projection,
                config=self.input_projection_minimal_matmul_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.input_projection_compute_config,
            )
        else:
            projected = ttnn.linear(
                hidden_states,
                weights.input_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.input_projection_compute_config,
            )
        auxiliary_start = self._convolution_width
        return _ProjectedInputs(
            # The convolution reads its channels in place from the tiled projection.
            qkv=projected,
            # The decay projection reads its low-rank columns in place.
            decay_rank=projected,
            decay_rank_offset=auxiliary_start,
            # The gated norm reads its gate columns straight from the fused projection, which therefore
            # stays allocated until the norm instead of only its gate slice.
            output_gate=projected,
            output_gate_offset=auxiliary_start + config.head_k_dim,
            # Chunk preparation reads beta's logits in place and applies the sigmoid.
            beta=projected,
            beta_offset=auxiliary_start + config.head_k_dim + config.v_dim,
        )

    def _compute_decay(self, projected: ttnn.Tensor, decay_rank_offset: int) -> ttnn.Tensor:
        """Evaluate the decay gate consumed by the recurrence; chunk preparation activates beta itself.

        The low-rank activations are read in place from the fused projection's columns at ``decay_rank_offset``.
        """
        weights = self.weights
        gate = ttnn.experimental.minimal_matmul(
            projected,
            weights.decay_output_projection,
            bias_tensor=weights.decay_bias_flat,
            fused_activation=self.decay_activation,
            config=self.decay_projection_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            in0_column_offset=decay_rank_offset,
        )
        return self._activate_decay(gate)

    def _softplus_decay(self, gate: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.multiply(
            self.weights.decay_scale_flat,
            gate,
            input_tensor_b_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, KDA_SOFTPLUS_BETA, KDA_SOFTPLUS_THRESHOLD)
            ],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _bounded_decay(gate: ttnn.Tensor) -> ttnn.Tensor:
        # The projection already applied sigmoid(scale * (x @ W + bias)); chunk preparation applies the lower bound.
        return gate

    def _kda_rms_norm(
        self,
        output: ttnn.Tensor,
        output_gate: ttnn.Tensor,
        output_gate_offset: int,
    ) -> ttnn.Tensor:
        """Apply the KDA gated RMSNorm epilogue."""
        config, weights = self.config, self.weights
        return ttnn.experimental.kda.sigmoid_gated_rms_norm(
            output,
            output_gate,
            weights.norm,
            config.num_heads,
            epsilon=config.norm_eps,
            memory_config=KDA_NORM_MEMORY_CONFIG,
            compute_kernel_config=self.kda_compute_config,
            output_dtype=self.gated_rms_output_dtype,
            gate_column_offset=output_gate_offset,
        )

    def _project_output(
        self,
        output: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Project normalized heads and perform the required TP reduction."""
        weights = self.weights
        output = ttnn.linear(
            output,
            weights.output_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.output_projection_program_config,
            compute_kernel_config=self.output_projection_compute_config,
        )
        if self.tensor_parallel_size > 1:
            cluster_axis = self._tp_cluster_axis
            output = ttnn.experimental.reduce_scatter_minimal_async(
                output,
                dim=-1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                num_links=self.tt_ccl.get_num_links(cluster_axis),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=cluster_axis,
            )
        return output

    def selections(self, actual_start: ttnn.Tensor, actual_end: ttnn.Tensor | None = None) -> ChronologicalSelections:
        """The chronological selection table for one call's bounds.

        It depends only on the bounds and on the SP geometry, so layers that share both can share one table; see
        ``forward``'s ``selections``.
        """
        return ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(
                actual_start,
                self.sequence_parallel_axis,
                self.active_seq_len_local,
                self.config.num_heads,
                self.config.head_k_dim,
                self.config.head_v_dim,
                actual_end=actual_end,
            ),
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None = None,
        selections: ChronologicalSelections | None = None,
    ) -> tuple[ttnn.Tensor, KdaState]:
        """Run prefill KDA and return replacement logical carries.

        ``actual_start`` is the absolute global position of this chunk's first
        token. It selects the chronological SP segment order for MLA's
        block-cyclic layout. The caller owns a replicated UINT32 row-major scalar
        and must keep it alive at the captured address throughout trace use.
        Update its contents before replay; every SP rank must observe the same
        nonnegative, 32-aligned position. No device-to-host value validation is
        performed. Pass an explicit zero-valued tensor for a zero-start call.

        Optional ``actual_end`` is a replicated device scalar defining the
        exclusive global valid end. The interval is nonempty, 32-aligned, and
        no larger than the constructed capacity. Omission means full capacity.
        Bounds may change during trace replay; their addresses must stay alive.
        Padded output rows are unspecified; returned carries stop at the valid end.

        The input state is only read. No tensor reachable from it is used as a
        ``ttnn.copy`` destination or retained on this layer. The returned output
        is sequence-partitioned along SP and, when TP > 1, reduce-scattered on
        the hidden dimension; TP == 1 returns the full hidden dimension.

        Optional ``selections`` is a table from ``selections()`` for these same
        bounds. The layer derives its selections on device from the bounds, so
        it is accepted for compatibility and not read.
        """
        self._validate_forward(hidden_states, state, actual_start)
        if actual_end is not None:
            self._validate_runtime_bound(actual_end, "actual_end")
        del selections  # Selections are derived on device from the bounds.
        projected = self._project_inputs(hidden_states)
        convolution_state = ttnn.to_layout(
            state.convolution, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q, k, v, new_convolution = self._convolve_qkv(projected.qkv, convolution_state, actual_start, actual_end)
        gate = self._compute_decay(projected.decay_rank, projected.decay_rank_offset)
        result = self.recurrence(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=projected.beta,
            beta_logits_column_offset=projected.beta_offset,
            initial_state=state.recurrent,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        output = self._kda_rms_norm(result.output, projected.output_gate, projected.output_gate_offset)
        output = self._project_output(output)
        return output, KdaState(recurrent=result.final_state, convolution=new_convolution)
