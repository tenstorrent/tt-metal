# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_BETA_DTYPE,
    KDA_CHUNK_SIZE,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_RECURRENT_STATE_DTYPE,
    KDAProgramConfig,
)
from models.demos.deepseek_v3_d_p.tt.kda.convolution import exchange_convolution_carry
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import KDARecurrence
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL


def _slice_width(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    stop = list(tensor.shape)
    begin = [0] * len(stop)
    begin[-1] = start
    stop[-1] = end
    return ttnn.slice(tensor, tuple(begin), tuple(stop), memory_config=ttnn.DRAM_MEMORY_CONFIG)


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
    qkv: ttnn.Tensor
    decay_rank: ttnn.Tensor
    output_gate: ttnn.Tensor
    beta: ttnn.Tensor


@dataclass(frozen=True)
class _KDAInputs:
    q: ttnn.Tensor
    k: ttnn.Tensor
    v: ttnn.Tensor
    decay: ttnn.Tensor
    beta: ttnn.Tensor
    output_gate: ttnn.Tensor


@dataclass(frozen=True)
class KdaState:
    """Caller-owned KDA carries.

    ``recurrent`` is TP-local and must be replicated across the SP axis.
    ``convolution`` is the BF16 row-major DRAM stream tail with shape
    ``[B, kernel_size - 1, Q_local + K_local + V_local]``. Its channels are
    sharded across TP and the complete tail is replicated across SP. The halo
    exchange derives each partition entry carry from it. Construct state with
    :meth:`ttKDA.allocate_state` or reuse a state returned by :meth:`ttKDA.forward`.
    """

    recurrent: ttnn.Tensor
    convolution: ttnn.Tensor


class ttKDA:
    """Prefill-only KDA layer with immutable, caller-owned logical state."""

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
    ) -> None:
        if tp_axis not in (0, 1) or sp_axis not in (0, 1) or sp_axis == tp_axis:
            raise ValueError(f"KDA requires distinct 2D SP/TP axes, got SP={sp_axis}, TP={tp_axis}")
        program_config = program_config or KDAProgramConfig()
        self.device = mesh_device
        self.layer_idx = layer_idx
        self.tensor_parallel_axis = tp_axis
        self.sequence_parallel_axis = sp_axis
        self.sequence_parallel_size = (
            tuple(mesh_device.shape)[self.sequence_parallel_axis] if isinstance(mesh_device, ttnn.MeshDevice) else 1
        )
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
        # Ordinary matmuls (input and decay projections) keep packer L1 accumulation.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
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
            sequence_parallel_axis=(self.sequence_parallel_axis if self.sequence_parallel_size > 1 else None),
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

    def _validate_forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
    ) -> None:
        """Validate shape/type plus the documented SP state-distribution contract."""
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
        convolution_state: ttnn.Tensor,
        sequence: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Run depthwise convolution and emit Q/K/V without post-convolution slices."""
        config = self.config
        channels = self._convolution_width
        qkv_row_major = ttnn.to_layout(
            qkv,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state_row_major = ttnn.to_layout(
            convolution_state,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.sequence_parallel_size > 1:
            state_row_major, new_state = exchange_convolution_carry(
                qkv_row_major,
                state_row_major,
                sequence_parallel_axis=self.sequence_parallel_axis,
            )
        else:
            new_state = ttnn.slice(
                qkv_row_major,
                (0, sequence - (config.conv_kernel_size - 1), 0),
                (qkv_row_major.shape[0], sequence, channels),
            )
        # The replacement state is BF16 row-major DRAM [B, K - 1, Q_local + K_local + V_local],
        # channel-sharded across TP and replicated across SP.
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv_row_major,
            state_row_major,
            *self.weights.convolution_taps,
            config.q_dim,
            config.k_dim,
            config.v_dim,
            program_config=self.qkv_convolution_program_config,
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
        projected = ttnn.linear(
            hidden_states,
            weights.input_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        auxiliary_start = self._convolution_width
        return _ProjectedInputs(
            qkv=_slice_width(projected, 0, auxiliary_start),
            decay_rank=_slice_width(projected, auxiliary_start, auxiliary_start + config.head_k_dim),
            output_gate=_slice_width(
                projected,
                auxiliary_start + config.head_k_dim,
                auxiliary_start + config.head_k_dim + config.v_dim,
            ),
            beta=_slice_width(
                projected,
                auxiliary_start + config.head_k_dim + config.v_dim,
                auxiliary_start + config.head_k_dim + config.v_dim + config.num_heads,
            ),
        )

    def _compute_gates(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        beta: ttnn.Tensor,
        decay_rank: ttnn.Tensor,
        output_gate: ttnn.Tensor,
    ) -> _KDAInputs:
        """Evaluate decay and write gates while preserving the output gate for the epilogue."""
        config, weights = self.config, self.weights
        # Preserve the sigmoid result at the FP32 precision required by chunk preparation.
        beta_for_recurrence = ttnn.sigmoid(
            ttnn.typecast(
                beta,
                KDA_BETA_DTYPE,
                memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            ),
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        )
        gate = ttnn.linear(
            decay_rank,
            weights.decay_output_projection,
            bias=weights.decay_bias_flat,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        if config.gate_lower_bound is None:
            gate = ttnn.multiply(
                weights.decay_scale_flat,
                gate,
                input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = ttnn.multiply(
                weights.decay_scale_flat,
                gate,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate = ttnn.multiply(gate, config.gate_lower_bound, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return _KDAInputs(
            q=q,
            k=k,
            v=v,
            decay=gate,
            beta=beta_for_recurrence,
            output_gate=output_gate,
        )

    def _kda_prefill(
        self,
        inputs: _KDAInputs,
        recurrent_state: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the KDA recurrence and return its raw output and updated state."""
        new_state, output = self.recurrence(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            gate=inputs.decay,
            beta=inputs.beta,
            initial_state=recurrent_state,
        )
        return output, new_state

    def _kda_rms_norm(
        self,
        output: ttnn.Tensor,
        output_gate: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply the KDA gated RMSNorm epilogue."""
        config, weights = self.config, self.weights
        return ttnn.experimental.kda.sigmoid_gated_rms_norm(
            output,
            output_gate,
            weights.norm,
            config.num_heads,
            epsilon=config.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.kda_compute_config,
            output_dtype=self.gated_rms_output_dtype,
        )

    def _project_output(
        self,
        output: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Project normalized heads and perform the required TP reduction."""
        weights = self.weights
        assert output.dtype == self.gated_rms_output_dtype
        output = ttnn.linear(
            output,
            weights.output_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.output_projection_compute_config,
        )
        if self.tensor_parallel_size > 1:
            assert self.tt_ccl is not None
            cluster_axis = None if self.sequence_parallel_size == 1 else self.tensor_parallel_axis
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

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
    ) -> tuple[ttnn.Tensor, KdaState]:
        """Run prefill KDA and return replacement logical carries.

        The input state is only read. No tensor reachable from it is used as a
        ``ttnn.copy`` destination or retained on this layer. The returned output
        is sequence-partitioned along SP and, when TP > 1, reduce-scattered on
        the hidden dimension; TP == 1 returns the full hidden dimension.
        """
        self._validate_forward(hidden_states, state)
        sequence = hidden_states.shape[1]
        projected = self._project_inputs(hidden_states)
        q, k, v, new_convolution = self._convolve_qkv(projected.qkv, state.convolution, sequence=sequence)
        inputs = self._compute_gates(
            q,
            k,
            v,
            beta=projected.beta,
            decay_rank=projected.decay_rank,
            output_gate=projected.output_gate,
        )
        output, new_recurrent = self._kda_prefill(inputs, state.recurrent)
        output = self._kda_rms_norm(output, inputs.output_gate)
        output = self._project_output(output)
        return output, KdaState(recurrent=new_recurrent, convolution=new_convolution)
