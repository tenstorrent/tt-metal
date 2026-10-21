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
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_CHUNK_SIZE,
    KDA_DECAY_L1_BYTES_PER_CORE,
    KDA_HISTORY_EXCHANGE_ROWS,
    KDA_HISTORY_EXCHANGE_SEMAPHORES,
    KDA_NORM_OUTPUT_L1_BYTES_PER_CORE,
    KDA_OUTPUT_MEMORY_CONFIG,
    KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE,
    KDA_QKV_L1_BYTES_PER_CORE,
    KDA_RECURRENT_STATE_DTYPE,
    KDAProgramConfig,
    block_projection_program_config,
    decay_projection_program_config,
    l1_when_it_fits,
    matmul_grid,
    tuned_projection_matmul_configs,
)
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import CarryProjection, KDARecurrence
from models.demos.deepseek_v3_d_p.tt.kda.relay_affine_prefix import RelayOutputGate
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


def reduce_scatter_tp_partials(
    partials: ttnn.Tensor, tt_ccl: TT_CCL, cluster_axis: int | None, topology: ttnn.Topology
) -> ttnn.Tensor:
    """Reduce-scatter the output projection's TP partial sums over the hidden dimension into DRAM."""
    return ttnn.experimental.reduce_scatter_minimal_async(
        partials,
        dim=-1,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=tt_ccl.get_num_links(cluster_axis),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=cluster_axis,
    )


@dataclass(frozen=True)
class _ProjectedInputs:
    # The fused input projection. The QKV columns, the decay's low-rank activations, the output gate and beta's
    # pre-sigmoid logits are read in place from their column offsets, so it stays allocated until the gated norm.
    projected: ttnn.Tensor
    decay_rank_offset: int
    output_gate_offset: int
    beta_offset: int


@dataclass(frozen=True)
class _HistoryExchange:
    """Persistent state of the convolution's fabric history exchange.

    Peers write this rank's buffer and bump its semaphores, possibly before this rank's call starts, so both outlive
    every call. The semaphores live on the exchange core (``qkv_causal_conv1d_silu_exchange_core``), and in L1_SMALL when
    the device has one: their counts carry across calls, and a fabric mux may overwrite raw L1 below it.

    Every KDA layer on a mesh shares one exchange. The entry barrier orders calls, not layers, so layers taking turns
    are no different from one layer called repeatedly, and per-layer copies would exhaust a small L1_SMALL at depth.
    """

    buffer: ttnn.Tensor
    semaphores: list[ttnn.GlobalSemaphore]

    @classmethod
    def shared(cls, mesh_device: ttnn.MeshDevice, width: int) -> _HistoryExchange:
        """The mesh's exchange for this width, allocated on first use (mesh ids are never reused)."""
        key = (mesh_device.id(), width)
        if key not in _history_exchanges:
            _history_exchanges[key] = cls.allocate(mesh_device, width)
        return _history_exchanges[key]

    @classmethod
    def allocate(cls, mesh_device: ttnn.MeshDevice, width: int) -> _HistoryExchange:
        exchange_core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
            mesh_device.compute_with_storage_grid_size()
        )
        exchange_cores = ttnn.CoreRangeSet({ttnn.CoreRange(exchange_core, exchange_core)})
        l1_small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
        semaphore_buffer_type = ttnn.BufferType.L1_SMALL if l1_small.total_bytes_per_bank > 0 else ttnn.BufferType.L1
        return cls(
            buffer=ttnn.zeros(
                (1, KDA_HISTORY_EXCHANGE_ROWS, width),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            semaphores=[
                ttnn.create_global_semaphore(mesh_device, exchange_cores, 0, semaphore_buffer_type)
                for _ in range(KDA_HISTORY_EXCHANGE_SEMAPHORES)
            ],
        )


_history_exchanges: dict[tuple[int, int], _HistoryExchange] = {}


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

    On geometries whose chunk terms and decay gate fit their L1 budgets (see config.py), those tensors live in L1
    while the recurrence runs. Open the device with an L1_SMALL region, as the model's prefill tests do, so the CCL
    semaphores allocated on the first call do not fragment L1 between warm-up and trace capture.
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
        # A bounded gate's projection already applies sigmoid(scale * (x @ W + bias)); chunk preparation applies the
        # lower bound, so only the softplus gate needs a separate activation.
        self._bounded_gate = config.gate_lower_bound is not None
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
                projection_dtype=program_config.projection_dtype,
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
        self.qkv_memory_config = l1_when_it_fits(
            mesh_device, self.active_seq_len_local * self._convolution_width * 2, KDA_QKV_L1_BYTES_PER_CORE
        )
        # The output projection takes no output dtype, so its partial sums keep the norm output's.
        output_element_bytes = 4 if self.gated_rms_output_dtype == ttnn.float32 else 2
        self.norm_output_memory_config = l1_when_it_fits(
            mesh_device,
            self.active_seq_len_local * self.config.v_dim * output_element_bytes,
            KDA_NORM_OUTPUT_L1_BYTES_PER_CORE,
        )
        # The partial sums only feed the reduce-scatter; a single TP rank returns the projection itself.
        self.partial_output_memory_config = (
            l1_when_it_fits(
                mesh_device,
                self.active_seq_len_local * config.hidden_size * output_element_bytes,
                KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE,
            )
            if self.tensor_parallel_size > 1
            else ttnn.DRAM_MEMORY_CONFIG
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
        grid = mesh_device.compute_with_storage_grid_size()
        # The projection matmuls run on at most max_matmul_cores cores (Blackhole di/dt).
        mm_grid = matmul_grid(grid, program_config.max_matmul_cores)
        self._matmul_grid = mm_grid
        self._relay_column_cores = [
            ttnn.CoreCoord(x, y) for x in range(grid.x - 1, grid.x - 4, -1) for y in range(grid.y)
        ]
        self.recurrence = KDARecurrence(
            mesh_device,
            program_config.recurrence,
            sequence_parallel_axis=self.sequence_parallel_axis,
            local_rows=self.active_seq_len_local,
            heads=self.config.num_heads,
            key_dim=self.config.head_k_dim,
            value_dim=self.config.head_v_dim,
            # The bounded gate's lower-bound scale is applied by chunk preparation.
            gate_scale=config.gate_lower_bound if self._bounded_gate else 1.0,
            # The relay takes the right-hand worker columns, leaving the output gate's matmul a full-height block.
            affine_relay_cores=self._relay_column_cores if program_config.overlap_output_gate else None,
        )
        # The output gate's columns become their own matmul, run beside the SP prefix relay on the worker columns left
        # of the relay's; the input projection keeps Q, K, V, the decay rank, beta and the first output-gate heads that
        # fit in its per-core width anyway. Split once on device from the fused weight. The relay's final carry (its own
        # program on the last worker column, mostly waiting on the fabric) then runs beside the output projection, on
        # the other columns, after the gated norm. All three are ttnn.linear's 2D multicast matmul on worker blocks.
        self.overlap_output_gate = program_config.overlap_output_gate and self.recurrence.uses_affine_relay
        self.carry_with_output_projection = self.overlap_output_gate
        self.input_projection_gate_columns = 0
        self.output_gate_weight = None
        self._output_gate_program_config = None
        self._output_projection_block_config = None
        input_projection_block_config = None
        if self.overlap_output_gate:
            fused = self.weights.input_projection
            hidden, width = tuple(fused.shape)[-2:]
            gate_start = self._convolution_width + self.config.head_k_dim
            gate_end = gate_start + self.config.v_dim
            self.input_projection_gate_columns = self._input_projection_gate_tiles(mm_grid, program_config, width) * (
                ttnn.TILE_SIZE
            )
            kept_end = gate_start + self.input_projection_gate_columns
            self.output_gate_weight = ttnn.slice(fused, [0, kept_end], [hidden, gate_end])
            leading = ttnn.slice(fused, [0, 0], [hidden, kept_end])
            trailing = ttnn.slice(fused, [0, gate_end], [hidden, width])
            # Both pieces are whole tiles of the fused weight; concat widens to BF16, and casting back requantizes
            # each tile with its original block exponents, so the projection weight is unchanged.
            joined = ttnn.concat([leading, trailing], dim=-1)
            object.__setattr__(self.weights, "input_projection", ttnn.typecast(joined, fused.dtype))
            for tensor in (joined, leading, trailing, fused):
                ttnn.deallocate(tensor)
            relay_cores = self.recurrence.affine_relay_cores
            # The kept heads' gate columns are copied into the gate's output by the relay columns' spare cores.
            self._gate_copy_cores = [c for c in self._relay_column_cores if c not in relay_cores]
            # The relay's head cores multiply while the gate runs, so they share the gate's max_matmul_cores.
            gate_columns = min(
                min(c.x for c in self._relay_column_cores),
                (program_config.max_matmul_cores - len(self.recurrence.affine_relay_head_cores)) // grid.y,
            )
            if gate_columns < 1:
                raise ValueError(
                    f"no worker column left for the output gate beside the SP relay ({grid.x}x{grid.y} grid, "
                    f"max_matmul_cores {program_config.max_matmul_cores}); disable overlap_output_gate"
                )
            rows = self.active_seq_len_local // ttnn.TILE_SIZE
            self._output_gate_program_config = block_projection_program_config(
                gate_columns,
                grid.y,
                rows,
                self.output_gate_weight.shape[-1] // ttnn.TILE_SIZE,
                program_config.output_gate_k_block,
            )
            # The output projection on the worker columns left of the final carry's, within max_matmul_cores.
            carry_x = min(c.x for c in self.recurrence.final_carry_cores)
            self._output_projection_block_config = block_projection_program_config(
                min(carry_x, mm_grid.x),
                grid.y,
                rows,
                self.weights.output_projection.shape[-1] // ttnn.TILE_SIZE,
                program_config.output_projection_k_block,
            )
            # The input projection on the matmul grid, 2 rows of tiles per core.
            input_projection_block_config = block_projection_program_config(
                mm_grid.x,
                mm_grid.y,
                rows,
                self.weights.input_projection.shape[-1] // ttnn.TILE_SIZE,
                program_config.input_projection_k_block,
            )
        self.input_projection_program_config, self.output_projection_program_config = (
            tuned_projection_matmul_configs(
                mm_grid,
                self.active_seq_len_local,
                self.weights.input_projection.shape[-1],
                *tuple(self.weights.output_projection.shape)[-2:],
            )
            if program_config.tuned_projection_matmuls
            else (None, None)
        )
        if input_projection_block_config is not None:
            self.input_projection_program_config = input_projection_block_config
        # The bounded gate's per-head scale is folded into the decay projection, which then applies the sigmoid.
        decay_width = self.config.num_heads * self.config.head_k_dim
        self.decay_program_config = decay_projection_program_config(
            mm_grid,
            self.active_seq_len_local,
            self.config.head_k_dim,
            decay_width,
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if self._bounded_gate else None,
        )
        # The prepared chunk terms read the decay gate; L1 spares its DRAM round trip.
        self.decay_memory_config = l1_when_it_fits(
            mesh_device, self.active_seq_len_local * decay_width * 2, KDA_DECAY_L1_BYTES_PER_CORE
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
        self.output_projection_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=program_config.output_projection_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._history_exchange = _HistoryExchange.shared(mesh_device, self._convolution_width)

    def _input_projection_gate_tiles(self, grid: ttnn.CoreCoord, program_config, width: int) -> int:
        """Output-gate tile columns the input projection keeps: its per-core width without the gate's columns plus
        input_projection_extra_gate_tiles (one off the tuned schedule). K3 at 640 rows on the 11-column matmul grid:
        293 tiles take 27 per core column; at 32, 59 gate tiles fit. The gate's matmul beside the relay sets the
        relay's window."""
        base_tiles = -(-(width - self.config.v_dim) // ttnn.TILE_SIZE)
        if program_config.tuned_projection_matmuls:
            out_k, out_n = tuple(self.weights.output_projection.shape)[-2:]
            base = tuned_projection_matmul_configs(
                grid, self.active_seq_len_local, width - self.config.v_dim, out_k, out_n
            )[0]
            if base is None:
                return 0
            extra = program_config.input_projection_extra_gate_tiles
        else:
            # The matmul program's own blocking (T=2048 and T=2560: 25 tiles per core) and one more column.
            base_per_core = -(-base_tiles // grid.x)
            extra = 1
        per_core = (base.per_core_N if program_config.tuned_projection_matmuls else base_per_core) + extra
        return max(0, min(self.config.v_dim // ttnn.TILE_SIZE, per_core * grid.x - base_tiles))

    def _matmul_core_grid(self, program_config) -> ttnn.CoreGrid | None:
        """The core grid of an auto-configured projection linear: the matmul grid (a program config sets its own)."""
        return None if program_config is not None else ttnn.CoreGrid(y=self._matmul_grid.y, x=self._matmul_grid.x)

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

    def _project_inputs(
        self,
        hidden_states: ttnn.Tensor,
    ) -> _ProjectedInputs:
        """Run the fused input projection and locate its semantic columns."""
        config = self.config
        weights = self.weights
        # The tuned schedule writes the projection to L1: it is only read back by this layer (the QKV, decay, beta
        # and output-gate columns, all in place), and its DRAM write would otherwise sit after the last K block,
        # outside the compute.
        projected = ttnn.linear(
            hidden_states,
            weights.input_projection,
            program_config=self.input_projection_program_config,
            core_grid=self._matmul_core_grid(self.input_projection_program_config),
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG if self.input_projection_program_config is None else ttnn.L1_MEMORY_CONFIG
            ),
            compute_kernel_config=self.input_projection_compute_config,
        )
        auxiliary_start = self._convolution_width
        # With overlap_output_gate the projection keeps only the first heads' output-gate columns.
        gate_width = self.input_projection_gate_columns if self.overlap_output_gate else config.v_dim
        return _ProjectedInputs(
            projected=projected,
            decay_rank_offset=auxiliary_start,
            output_gate_offset=auxiliary_start + config.head_k_dim,
            beta_offset=auxiliary_start + config.head_k_dim + gate_width,
        )

    def _compute_decay(self, projected: ttnn.Tensor, decay_rank_offset: int) -> ttnn.Tensor:
        """Evaluate the decay gate consumed by the recurrence.

        The low-rank activations are read in place from the fused projection's columns at ``decay_rank_offset``.
        """
        weights = self.weights
        gate = ttnn.linear(
            projected,
            weights.decay_output_projection,
            bias=weights.decay_bias_flat,
            program_config=self.decay_program_config,
            memory_config=self.decay_memory_config,
            compute_kernel_config=self.compute_config,
            in0_column_offset=decay_rank_offset,
        )
        return gate if self._bounded_gate else self._softplus_decay(gate)

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
            memory_config=self.norm_output_memory_config,
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
            memory_config=self.partial_output_memory_config,
            program_config=self.output_projection_program_config,
            core_grid=self._matmul_core_grid(self.output_projection_program_config),
            compute_kernel_config=self.output_projection_compute_config,
        )
        return self._reduce_scatter(output)

    def _reduce_scatter(self, output: ttnn.Tensor) -> ttnn.Tensor:
        """The TP reduction of the output projection's partial sums."""
        if self.tensor_parallel_size > 1:
            output = reduce_scatter_tp_partials(output, self.tt_ccl, self._tp_cluster_axis, self.tp_ccl_topology)
        return output

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None = None,
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
        exclusive global valid end. The interval is nonempty and no larger than
        the constructed capacity; the end need not be 32-aligned. Omission means
        full capacity. Bounds may change during trace replay; their addresses must
        stay alive. Padded rows may hold any values, including NaN: padded output
        rows are unspecified, and returned carries stop at the valid end.

        The input state is only read. No tensor reachable from it is used as a
        ``ttnn.copy`` destination or retained on this layer. The returned output
        is sequence-partitioned along SP and, when TP > 1, reduce-scattered on
        the hidden dimension; TP == 1 returns the full hidden dimension.
        """
        self._validate_forward(hidden_states, state, actual_start)
        if actual_end is not None:
            self._validate_runtime_bound(actual_end, "actual_end")
        inputs = self._project_inputs(hidden_states)
        q, k, v, new_convolution = ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
            inputs.projected,
            state.convolution,
            *self.weights.convolution_taps,
            self.config.q_dim,
            self.config.k_dim,
            self.config.v_dim,
            program_config=self.qkv_convolution_program_config,
            actual_start=actual_start,
            exchange_buffer=self._history_exchange.buffer,
            exchange_semaphores=self._history_exchange.semaphores,
            actual_end=actual_end,
            sequence_parallel_axis=self.sequence_parallel_axis,
            memory_config=self.qkv_memory_config,
        )
        gate = self._compute_decay(inputs.projected, inputs.decay_rank_offset)
        output_gate = None
        if self.overlap_output_gate:
            # Its L1 output is allocated when the relay dispatches, so it is not live during chunk preparation; the
            # heads the input projection kept are copied in by the relay columns' spare cores.
            output_gate = RelayOutputGate(
                input=hidden_states,
                weight=self.output_gate_weight,
                program_config=self._output_gate_program_config,
                compute_config=self.input_projection_compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                copy_source=inputs.projected,
                copy_source_column=inputs.output_gate_offset // ttnn.TILE_SIZE,
                copied_columns=self.input_projection_gate_columns // ttnn.TILE_SIZE,
                copy_cores=self._gate_copy_cores,
            )

        result = self.recurrence(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=inputs.projected,
            beta_logits_column_offset=inputs.beta_offset,
            initial_state=state.recurrent,
            actual_start=actual_start,
            actual_end=actual_end,
            release_gate=True,
            output_gate=output_gate,
            defer_final_carry=self.carry_with_output_projection,
        )
        if self.overlap_output_gate:
            output = self._kda_rms_norm(result.output, result.output_gate, 0)
            ttnn.deallocate(result.output_gate)
        else:
            output = self._kda_rms_norm(result.output, inputs.projected, inputs.output_gate_offset)
        # The norm read the output gate, the projection's last columns in use.
        ttnn.deallocate(inputs.projected)
        if self.carry_with_output_projection:
            normed = output
            # The output projection on the cores the final carry leaves (and max_matmul_cores allows).
            final_state, output = result.final_state.run(
                CarryProjection(
                    input=normed,
                    weight=self.weights.output_projection,
                    program_config=self._output_projection_block_config,
                    compute_config=self.output_projection_compute_config,
                    memory_config=self.partial_output_memory_config,
                )
            )
            ttnn.deallocate(normed)
            output = self._reduce_scatter(output)
        else:
            output, final_state = self._project_output(output), result.final_state
        return output, KdaState(recurrent=final_state, convolution=new_convolution)
