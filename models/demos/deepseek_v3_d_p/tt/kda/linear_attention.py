# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill skeleton shared by the linear-attention layers on the KDA path (``ttKDA``, ``ttGDN``).

Both layers run the same graph around their own projection split and gates: caller-owned state, the
chronological selections, a fused input projection, the q/k/v causal convolution with its SP carry exchange,
the chunked delta-rule recurrence (``KDARecurrence``), the gated RMSNorm epilogue and the output projection with
its TP reduce-scatter. This module owns that graph and its fixed execution policy; the layers own their weights,
the layout of their fused projection, their gates and their output-gate activation.

All dimensions here are TP-local. q and k carry ``key_heads`` heads, v, the recurrent state, the gates and the
output norm carry ``value_heads`` heads (``value_heads % key_heads == 0``; value head ``h`` reads key head
``h // (value_heads / key_heads)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_CHUNK_SIZE,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_RECURRENT_STATE_DTYPE,
    KDAProgramConfig,
    tuned_projection_matmul_configs,
)
from models.demos.deepseek_v3_d_p.tt.kda.convolution import exchange_convolution_carry
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import KDARecurrence, RecurrenceResult
from models.tt_transformers.tt.ccl import TT_CCL


@dataclass(frozen=True)
class KdaState:
    """Caller-owned carries of a linear-attention layer on the KDA path.

    ``recurrent`` is TP-local, ``[B, value_heads, K, V]`` FP32, and must be replicated across the SP axis.
    ``convolution`` is the BF16 row-major DRAM stream tail with shape
    ``[B, kernel_size - 1, Q_local + K_local + V_local]``. Its channels are
    sharded across TP and the complete tail is replicated across SP. This history
    seeds the logical sequence start; the halo exchange supplies predecessor
    histories from projected tokens for the other segments. Construct state with
    the layer's ``allocate_state`` or reuse a state returned by its ``forward``.
    """

    recurrent: ttnn.Tensor
    convolution: ttnn.Tensor


@dataclass(frozen=True)
class LinearAttentionGeometry:
    """TP-local dimensions of one linear-attention layer."""

    hidden_size: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_kernel_size: int
    norm_eps: float

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.key_heads, self.value_heads, self.key_dim, self.value_dim) <= 0:
            raise ValueError(f"linear-attention dimensions must be positive, got {self}")
        if self.value_heads % self.key_heads:
            raise ValueError(f"value_heads {self.value_heads} must be a multiple of key_heads {self.key_heads}")

    @property
    def q_width(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def k_width(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def v_width(self) -> int:
        return self.value_heads * self.value_dim

    @property
    def convolution_width(self) -> int:
        return self.q_width + self.k_width + self.v_width


def slice_width(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    """Columns ``[start, end)`` of the last dimension, in DRAM."""
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


def mesh_axis_size(mesh_device: ttnn.Device | ttnn.MeshDevice, axis: int) -> int:
    return tuple(mesh_device.shape)[axis] if isinstance(mesh_device, ttnn.MeshDevice) else 1


class LinearAttentionSkeleton:
    """Constructor-fixed shared graph of one linear-attention layer for one physical geometry.

    ``active_seq_len`` is the global physical token count; each call supplies ``active_seq_len / SP`` local rows.
    ``scalar_decay`` selects the recurrence's decay layout: one log decay per (value head, token) ``[B, T, H]``
    (GDN) instead of one per (value head, key channel, token) ``[B, T, H * K]`` (KDA).
    ``layer_name`` only prefixes error messages.
    """

    def __init__(
        self,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        geometry: LinearAttentionGeometry,
        program_config: KDAProgramConfig,
        *,
        layer_name: str,
        sequence_parallel_axis: int,
        tensor_parallel_axis: int,
        active_seq_len: int,
        tt_ccl: TT_CCL | None,
        scalar_decay: bool,
    ) -> None:
        if (
            tensor_parallel_axis not in (0, 1)
            or sequence_parallel_axis not in (0, 1)
            or sequence_parallel_axis == tensor_parallel_axis
        ):
            raise ValueError(
                f"{layer_name} requires distinct 2D SP/TP axes, got SP={sequence_parallel_axis}, "
                f"TP={tensor_parallel_axis}"
            )
        self.layer_name = layer_name
        self.device = mesh_device
        self.geometry = geometry
        self.sequence_parallel_axis = sequence_parallel_axis
        self.tensor_parallel_axis = tensor_parallel_axis
        self.sequence_parallel_size = mesh_axis_size(mesh_device, sequence_parallel_axis)
        self.tensor_parallel_size = mesh_axis_size(mesh_device, tensor_parallel_axis)
        if active_seq_len <= 0 or active_seq_len % (self.sequence_parallel_size * KDA_CHUNK_SIZE):
            raise ValueError("active_seq_len must give a positive tile-aligned local physical length")
        self.active_seq_len_local = active_seq_len // self.sequence_parallel_size
        self.is_sequence_parallel = self.sequence_parallel_size > 1
        self._tp_cluster_axis = None if not self.is_sequence_parallel else self.tensor_parallel_axis
        uses_grouped_scan = self.is_sequence_parallel or program_config.recurrence.local_scan_strategy == "grouped"
        if uses_grouped_scan and geometry.key_dim != geometry.value_dim:
            raise ValueError(f"grouped {layer_name} affine prefix currently requires K == V")
        if self.tensor_parallel_size > 1 and tt_ccl is None:
            raise ValueError(f"tt_ccl is required for tensor-parallel {layer_name}")
        self.tt_ccl = tt_ccl
        self.tp_ccl_topology = program_config.tp_ccl_topology
        self.gated_rms_output_dtype = program_config.gated_rms_output_dtype
        self.qkv_convolution_program_config = ttnn.QkvCausalConv1dSiluProgramConfig(
            channel_chunk_size=_effective_qkv_channel_chunk_size(
                geometry.convolution_width, program_config.qkv_channel_chunk_size
            )
        )
        # Ordinary matmuls (input and decay projections) keep packer L1 accumulation.
        self.matmul_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # The output projection weight is [V_local, hidden] on every TP rank.
        self.input_projection_minimal_matmul_config, self.output_projection_program_config = (
            tuned_projection_matmul_configs(
                mesh_device.compute_with_storage_grid_size(),
                self.active_seq_len_local,
                geometry.v_width,
                geometry.hidden_size,
            )
            if program_config.tuned_projection_matmuls
            else (None, None)
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
            heads=geometry.value_heads,
            key_heads=geometry.key_heads,
            key_dim=geometry.key_dim,
            value_dim=geometry.value_dim,
            scalar_decay=scalar_decay,
        )
        self.output_projection_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=program_config.output_projection_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def allocate_state(self, batch_size: int = 1) -> KdaState:
        """Allocate the canonical device-resident carries for one prefill stream."""
        if batch_size != 1:
            raise ValueError(f"{self.layer_name} prefill currently requires batch_size=1, got {batch_size}")
        geometry = self.geometry
        return KdaState(
            recurrent=ttnn.zeros(
                (batch_size, geometry.value_heads, geometry.key_dim, geometry.value_dim),
                dtype=KDA_RECURRENT_STATE_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            ),
            convolution=ttnn.zeros(
                (batch_size, geometry.conv_kernel_size - 1, geometry.convolution_width),
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

    def validate_forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
    ) -> None:
        """Validate shape/type plus the documented SP state-distribution contract."""
        geometry, name = self.geometry, self.layer_name
        self._validate_runtime_bound(actual_start, "actual_start")
        if len(hidden_states.shape) != 3 or hidden_states.shape[-1] != geometry.hidden_size:
            raise ValueError(f"hidden_states shape {tuple(hidden_states.shape)} must be [B,T,{geometry.hidden_size}]")
        batch = hidden_states.shape[0]
        sequence = hidden_states.shape[1]
        if batch != 1:
            raise ValueError(f"{name} prefill currently requires batch size 1, got B={batch}")
        if sequence <= 0 or sequence % KDA_CHUNK_SIZE != 0:
            raise ValueError(
                f"{name} prefill requires local T to be positive and divisible by {KDA_CHUNK_SIZE}, got T={sequence}"
            )
        if sequence != self.active_seq_len_local:
            raise ValueError(
                f"hidden_states local T={sequence} does not match constructed T={self.active_seq_len_local}"
            )
        expected_recurrent = (batch, geometry.value_heads, geometry.key_dim, geometry.value_dim)
        expected_convolution = (batch, geometry.conv_kernel_size - 1, geometry.convolution_width)
        if tuple(state.recurrent.shape) != expected_recurrent:
            raise ValueError(f"recurrent state shape {tuple(state.recurrent.shape)} != {expected_recurrent}")
        if tuple(state.convolution.shape) != expected_convolution:
            raise ValueError(f"convolution state shape {tuple(state.convolution.shape)} != {expected_convolution}")
        if state.recurrent.dtype != KDA_RECURRENT_STATE_DTYPE:
            raise ValueError(f"recurrent state dtype {state.recurrent.dtype} != {KDA_RECURRENT_STATE_DTYPE}")
        if state.convolution.dtype != ttnn.bfloat16 or state.convolution.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("convolution state must be BF16 row-major")
        if actual_end is not None:
            self._validate_runtime_bound(actual_end, "actual_end")

    def selections(self, actual_start: ttnn.Tensor, actual_end: ttnn.Tensor | None) -> ChronologicalSelections:
        """Device chronology of this call; all geometries use the same selection graph for full and padded calls."""
        geometry = self.geometry
        return ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(
                actual_start,
                self.sequence_parallel_axis,
                self.active_seq_len_local,
                geometry.value_heads,
                geometry.key_dim,
                geometry.value_dim,
                actual_end=actual_end,
            ),
        )

    def project_input(self, hidden_states: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """Run the fused input projection; the layer splits its columns."""
        if self.input_projection_minimal_matmul_config is not None:
            return ttnn.experimental.minimal_matmul(
                hidden_states,
                weight,
                config=self.input_projection_minimal_matmul_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.matmul_compute_config,
            )
        return ttnn.linear(
            hidden_states,
            weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.matmul_compute_config,
        )

    def convolve(
        self,
        projected_qkv: ttnn.Tensor,
        convolution_state: ttnn.Tensor,
        selections: ChronologicalSelections,
        actual_start: ttnn.Tensor,
        taps: tuple[ttnn.Tensor, ...],
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Causal q/k/v convolution with SiLU; returns q, k, v and the replacement convolution carry."""
        geometry = self.geometry
        qkv = ttnn.to_layout(projected_qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        incoming_layer_carry = ttnn.to_layout(
            convolution_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if not self.is_sequence_parallel:
            new_state = selections.select_local_final_history(qkv, 1)
            predecessor = incoming_layer_carry
        else:
            predecessor, new_state = exchange_convolution_carry(
                qkv, sequence_parallel_axis=self.sequence_parallel_axis, selections=selections
            )
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv,
            incoming_layer_carry,
            *taps,
            geometry.q_width,
            geometry.k_width,
            geometry.v_width,
            program_config=self.qkv_convolution_program_config,
            actual_start=actual_start,
            sequence_parallel_axis=self.sequence_parallel_axis,
            predecessor_carry=predecessor,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # FP32 DST carries the tap sum in FP32. The op default (BF16 DST) rounds every partial sum to BF16, which
            # biased q/k/v by up to ~1 ulp: a 0.70 output error on Qwen3.8-2.4T text (tt_metal_tracker-g1b.5.19),
            # and for KDA a positive state bias that only masked the prep's k_dec_t contraction (g1b.4.18).
            compute_kernel_config=self.kda_compute_config,
        )
        return q, k, v, new_state

    def recur(
        self,
        *,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        gate: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: ttnn.Tensor,
        selections: ChronologicalSelections,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
    ) -> RecurrenceResult:
        """Chunked delta-rule recurrence over this call's rows from the caller's recurrent state."""
        return self.recurrence(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            initial_state=initial_state,
            selections=selections if self.is_sequence_parallel else None,
            actual_start=actual_start,
            actual_end=actual_end,
        )

    def gated_rms_norm(self, output: ttnn.Tensor, gate: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """``weight * rmsnorm(output) * sigmoid(gate)`` per value head, head-first in, time-first out."""
        return ttnn.experimental.kda.sigmoid_gated_rms_norm(
            output,
            gate,
            weight,
            self.geometry.value_heads,
            epsilon=self.geometry.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.kda_compute_config,
            output_dtype=self.gated_rms_output_dtype,
        )

    def project_output(self, output: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """Project normalized heads and perform the required TP reduction."""
        output = ttnn.linear(
            output,
            weight,
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
