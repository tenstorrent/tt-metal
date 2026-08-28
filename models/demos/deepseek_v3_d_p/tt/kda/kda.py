# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda import ops
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_BETA_DTYPE,
    KDA_CHUNK_SIZE,
    KDA_GATE_DTYPE,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_QKV_DTYPE,
    KDA_RECURRENT_STATE_DTYPE,
    KDAProgramConfig,
)
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


def _output_projection_program_config(
    sequence: int,
    input_width: int,
    output_width: int,
    out_block_w_cap: int | None,
    grid: tuple[int, int],
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    per_core_n = max(1, math.ceil(output_width / ttnn.TILE_SIZE / grid[0]))
    # Keep two output tiles per block when possible to bound accumulation footprint.
    out_block_w_limit = max(1, per_core_n // 2)
    if out_block_w_cap is not None:
        out_block_w_limit = min(out_block_w_limit, out_block_w_cap)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=_largest_divisor_at_most(
            input_width // ttnn.TILE_SIZE,
            # Four input tiles balances reuse with the per-core L1 budget on Blackhole.
            min(4, max(1, input_width // ttnn.TILE_SIZE // grid[0])),
        ),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(sequence / ttnn.TILE_SIZE / grid[1])),
        per_core_N=per_core_n,
        out_block_w=_largest_divisor_at_most(per_core_n, out_block_w_limit),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))}
        ),
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
    ``convolution`` is an SP-replicated stream tail; the halo exchange derives
    each partition entry carry from it. Construct state with
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
        self.recurrence_config = program_config.recurrence
        self.output_projection_out_block_w = program_config.output_projection_out_block_w
        output_grid = mesh_device.compute_with_storage_grid_size()
        self.output_projection_grid = (output_grid.x, output_grid.y)
        self.tp_ccl_topology = program_config.tp_ccl_topology
        self.gated_rms_output_dtype = program_config.gated_rms_output_dtype
        if weights is not None and state_dict:
            raise ValueError("pass either constructed KDAWeights or host state_dict, not both")
        if weights is None:
            loaded = load_kda_weights(
                mesh_device,
                config,
                state_dict,
                weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.kda",
                tensor_parallel_axis=tp_axis,
            )
            assert loaded is not None
            weights = loaded
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
        # Every experimental KDA operation rejects packer_l1_acc=true: none of their compute
        # kernels accumulate through L1, so the flag would be a silent no-op and the operations
        # fail the configuration early rather than accept an untruthful contract. Keep this
        # separate from self.compute_config, which configures the projection matmuls where the
        # flag is both accepted and worth roughly 70% of layer wall time.
        self.kda_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.affine_prefix_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.recurrence_config.affine_prefix_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.grouped_scan_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.recurrence_config.grouped_scan_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.recurrence_compute_config = ops._RecurrenceComputeConfig(
            preparation=self.kda_compute_config,
            affine_prefix=self.affine_prefix_compute_config,
            grouped_scan=self.grouped_scan_compute_config,
        )
        # Real-K3 component A/B: output-projection HiFi2 retained PCC >=0.999987 for every LoudBox
        # layout and improved median component latency by 3.580%-5.165%.
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
    ) -> tuple[int, int]:
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
        local_chunks = sequence // KDA_CHUNK_SIZE
        summary_group_chunks = self.recurrence_config.summary_group_chunks
        sequence_parallel_axis = self.sequence_parallel_axis if self.sequence_parallel_size > 1 else None
        if ops._uses_grouped_scan(
            num_chunks=local_chunks,
            program_config=self.recurrence_config,
            sequence_parallel_axis=sequence_parallel_axis,
        ):
            if self.config.head_k_dim != self.config.head_v_dim:
                raise ValueError(
                    f"grouped KDA currently requires K == V, got {self.config.head_k_dim} and {self.config.head_v_dim}"
                )
            ops._validate_grouped_scan_capacity(
                batch_heads=batch * self.config.num_heads,
                num_chunks=local_chunks,
                summary_group_chunks=summary_group_chunks,
                device=hidden_states.device(),
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
        return batch, sequence

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
            state_row_major, new_state = ops.convolution_halo(
                qkv_row_major,
                state_row_major,
                sequence_parallel_axis=self.sequence_parallel_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            new_state = ttnn.slice(
                qkv_row_major,
                (0, sequence - (config.conv_kernel_size - 1), 0),
                (qkv_row_major.shape[0], sequence, channels),
            )
            # Retained by real-K3 T=5120 component A/B: 74.36-75.76% faster at direct Q/K/V PCC >=0.999989.
        # Cap blocks at the operation's measured production size while keeping the
        # chunk an exact divisor for smaller synthetic and TP-local channel counts.
        channel_chunk_size = ttnn.TILE_SIZE * _largest_divisor_at_most(
            channels // ttnn.TILE_SIZE, 768 // ttnn.TILE_SIZE
        )
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv_row_major,
            state_row_major,
            *self.weights.convolution_taps,
            config.q_dim,
            config.k_dim,
            config.v_dim,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=channel_chunk_size),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.kda_compute_config,
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

    def _assert_flat_recurrence_contract(
        self,
        inputs: _KDAInputs,
        recurrent_state: ttnn.Tensor,
    ) -> None:
        """Assert postconditions of the internal recurrence producers."""
        batch, sequence, heads = inputs.beta.shape
        key_width = heads * self.config.head_k_dim
        value_width = heads * self.config.head_v_dim
        assert tuple(inputs.q.shape) == (batch, sequence, key_width)
        assert tuple(inputs.k.shape) == (batch, sequence, key_width)
        assert tuple(inputs.v.shape) == (batch, sequence, value_width)
        assert tuple(inputs.decay.shape) == (batch, sequence, key_width)
        assert tuple(recurrent_state.shape) == (
            batch,
            heads,
            self.config.head_k_dim,
            self.config.head_v_dim,
        )
        for tensor in (inputs.q, inputs.k, inputs.v):
            assert tensor.dtype == KDA_QKV_DTYPE
            assert tensor.layout == ttnn.TILE_LAYOUT
        assert inputs.decay.dtype == KDA_GATE_DTYPE
        assert inputs.decay.layout == ttnn.TILE_LAYOUT
        assert inputs.beta.dtype == KDA_BETA_DTYPE
        assert inputs.beta.layout == ttnn.TILE_LAYOUT
        assert recurrent_state.dtype == KDA_RECURRENT_STATE_DTYPE
        assert recurrent_state.layout == ttnn.TILE_LAYOUT
        assert recurrent_state.memory_config() == KDA_OUTPUT_MEMORY_CONFIG

    def _kda_prefill(
        self,
        inputs: _KDAInputs,
        recurrent_state: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the KDA recurrence and return its raw output and updated state."""
        self._assert_flat_recurrence_contract(inputs, recurrent_state)
        recurrence_inputs = ops._FlatRecurrenceInputs(
            q=inputs.q,
            k=inputs.k,
            v=inputs.v,
            gate=inputs.decay,
            beta=inputs.beta,
        )
        result = ops._chunk_recurrence(
            recurrence_inputs,
            recurrent_state,
            program_config=self.recurrence_config,
            compute_config=self.recurrence_compute_config,
            sequence_parallel_axis=(self.sequence_parallel_axis if self.sequence_parallel_size > 1 else None),
        )
        return result.output, result.final_state

    def _kda_rms_norm(
        self,
        output: ttnn.Tensor,
        output_gate: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply the KDA gated RMSNorm epilogue."""
        config, weights = self.config, self.weights
        # Retained by real-K3 T=5120 component A/B: 92.72-93.92% faster at output PCC >=0.999990.
        return ttnn.experimental.kda.sigmoid_gated_rms_norm(
            output,
            output_gate,
            weights.norm,
            config.num_heads,
            epsilon=config.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.kda_compute_config,
            # Real-K3 component A/B: direct BF16 output retained PCC 1.0 for every LoudBox layout
            # and improved median component latency by 0.655%-1.339%.
            output_dtype=self.gated_rms_output_dtype,
        )

    def _project_output(
        self,
        output: ttnn.Tensor,
        *,
        batch: int,
        sequence: int,
    ) -> ttnn.Tensor:
        """Project normalized heads and perform the required TP reduction."""
        config, weights = self.config, self.weights
        assert output.dtype == self.gated_rms_output_dtype
        if self.tensor_parallel_size > 1:
            assert self.tt_ccl is not None
            output = ttnn.reshape(output, (1, batch, sequence, config.v_dim))
            output = ttnn.linear(
                output,
                weights.output_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=_output_projection_program_config(
                    sequence,
                    config.v_dim,
                    weights.output_projection.shape[-1],
                    self.output_projection_out_block_w,
                    self.output_projection_grid,
                ),
                compute_kernel_config=self.output_projection_compute_config,
            )
            # The calibrated 1D SP1xTP8 path uses the axis-less CCL pool; 2D meshes target the TP axis.
            cluster_axis = None if self.sequence_parallel_size == 1 else self.tensor_parallel_axis
            output = ttnn.experimental.reduce_scatter_minimal_async(
                output,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                num_links=self.tt_ccl.get_num_links(cluster_axis),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=cluster_axis,
            )
            output = ttnn.reshape(output, (batch, sequence, output.shape[-1]))
        else:
            # Single-device tests retain TTNN's default matmul selection; the tuned program config
            # belongs to the distributed projection followed by reduce-scatter above.
            output = ttnn.linear(
                output,
                weights.output_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.output_projection_compute_config,
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
        batch, sequence = self._validate_forward(hidden_states, state)
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
        output = self._project_output(output, batch=batch, sequence=sequence)
        return output, KdaState(recurrent=new_recurrent, convolution=new_convolution)
