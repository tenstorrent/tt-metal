# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn
from models.experimental.kimi_delta_attention.config import KDAConfig, KDAProgramConfig
from models.experimental.kimi_delta_attention.tt.const_tiles import build_kda_const_tiles
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights, load_kda_weights
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
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    grid = (8, 8)
    per_core_n = max(1, math.ceil(output_width / ttnn.TILE_SIZE / grid[0]))
    out_block_w_limit = max(1, per_core_n // 2)
    if out_block_w_cap is not None:
        out_block_w_limit = min(out_block_w_limit, out_block_w_cap)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=_largest_divisor_at_most(
            input_width // ttnn.TILE_SIZE,
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


class KimiDeltaAttention:
    """Stateful, fully device-resident KDA correctness implementation."""

    def __init__(
        self,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        tensor_cache_path: Path | None = None,
        tt_ccl: TT_CCL | None = None,
        tensor_parallel_axis: int = 1,
        program_config: KDAProgramConfig | None = None,
        summary_group_chunks: int | None = None,
        cache_name_prefix: str = "kda",
        weights: KDAWeights | None = None,
    ) -> None:
        if tensor_parallel_axis not in (0, 1):
            raise ValueError(f"tensor_parallel_axis must be 0 or 1, got {tensor_parallel_axis}")
        program_config = program_config or KDAProgramConfig()
        if summary_group_chunks is None:
            summary_group_chunks = program_config.summary_group_chunks
        if summary_group_chunks <= 0:
            raise ValueError(f"summary_group_chunks must be positive, got {summary_group_chunks}")
        self.device = mesh_device
        self.tensor_parallel_axis = tensor_parallel_axis
        self.sequence_parallel_axis = 1 - tensor_parallel_axis
        self.sequence_parallel_size = (
            tuple(mesh_device.shape)[self.sequence_parallel_axis] if isinstance(mesh_device, ttnn.MeshDevice) else 1
        )
        self.summary_group_chunks = summary_group_chunks
        self.output_projection_out_block_w = program_config.output_projection_out_block_w
        self.recurrent_state_dtype = program_config.recurrent_state_dtype
        self.tp_ccl_topology = program_config.tp_ccl_topology
        self.affine_summary_dtype = program_config.affine_summary_dtype
        self.grouped_scan_output_dtype = program_config.grouped_scan_output_dtype
        self.use_bf16_prep_intermediates = program_config.use_bf16_prep_intermediates
        self.gated_rms_output_dtype = program_config.gated_rms_output_dtype
        if weights is not None and state_dict:
            raise ValueError("pass either constructed KDAWeights or host state_dict, not both")
        if weights is None:
            loaded = load_kda_weights(
                mesh_device,
                config,
                state_dict,
                tensor_cache_path,
                cache_name_prefix=cache_name_prefix,
                tensor_parallel_axis=tensor_parallel_axis,
            )
            assert loaded is not None
            weights = loaded
        expected_tp_size = (
            tuple(mesh_device.shape)[tensor_parallel_axis] if isinstance(mesh_device, ttnn.MeshDevice) else 1
        )
        if weights.tensor_parallel_size != expected_tp_size or weights.tensor_parallel_axis != tensor_parallel_axis:
            raise ValueError(
                "KDAWeights placement does not match the layer mesh: "
                f"weights TP={weights.tensor_parallel_size} axis={weights.tensor_parallel_axis}, "
                f"layer TP={expected_tp_size} axis={tensor_parallel_axis}"
            )
        self.weights = weights
        self.tensor_parallel_size = self.weights.tensor_parallel_size
        self.global_config = config
        if self.sequence_parallel_size > 1 and config.head_k_dim != config.head_v_dim:
            raise ValueError(
                f"sequence-parallel KDA currently requires K == V, got {config.head_k_dim} and {config.head_v_dim}"
            )
        self.config = replace(config, num_heads=config.num_heads // self.tensor_parallel_size)
        if self.tensor_parallel_size > 1 and tt_ccl is None:
            raise ValueError("tt_ccl is required for tensor-parallel KDA")
        self.tt_ccl = tt_ccl
        self.chunk_const_tiles = build_kda_const_tiles(mesh_device)
        self.recurrent_state: ttnn.Tensor | None = None
        self.convolution_state: ttnn.Tensor | None = None
        self.use_inplace_state = False
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.affine_prefix_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=program_config.affine_prefix_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.grouped_scan_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=program_config.grouped_scan_math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
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

    def reset_state(self, batch_size: int | None = None) -> None:
        """Allocate zero cache for a batch, or release logical ownership."""
        if batch_size is None:
            self.recurrent_state = None
            self.convolution_state = None
            self.use_inplace_state = False
            return
        if batch_size != 1:
            raise ValueError(f"KDA prefill currently requires batch_size=1, got {batch_size}")
        self.recurrent_state = ttnn.zeros(
            (
                batch_size,
                self.config.num_heads,
                self.config.head_k_dim,
                self.config.head_v_dim,
            ),
            dtype=self.recurrent_state_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.convolution_state = ttnn.zeros(
            (
                batch_size,
                self.config.conv_kernel_size - 1,
                self._convolution_width,
            ),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.use_inplace_state = False

    def set_external_state(
        self,
        recurrent_state: ttnn.Tensor,
        convolution_state: ttnn.Tensor,
    ) -> None:
        """Adopt trace-stable external buffers and update them in place."""
        batch = recurrent_state.shape[0]
        if batch != 1:
            raise ValueError(f"KDA prefill currently requires state batch 1, got {batch}")
        expected_recurrent = (
            batch,
            self.config.num_heads,
            self.config.head_k_dim,
            self.config.head_v_dim,
        )
        expected_convolution = (
            batch,
            self.config.conv_kernel_size - 1,
            self._convolution_width,
        )
        if tuple(recurrent_state.shape) != expected_recurrent:
            raise ValueError(f"recurrent_state shape {tuple(recurrent_state.shape)} != {expected_recurrent}")
        if tuple(convolution_state.shape) != expected_convolution:
            raise ValueError(f"convolution_state shape {tuple(convolution_state.shape)} != {expected_convolution}")
        if recurrent_state.dtype != self.recurrent_state_dtype:
            raise ValueError(f"recurrent_state dtype {recurrent_state.dtype} != {self.recurrent_state_dtype}")
        if convolution_state.dtype != ttnn.bfloat16:
            raise ValueError(f"convolution_state dtype {convolution_state.dtype} != {ttnn.bfloat16}")
        self.recurrent_state = recurrent_state
        self.convolution_state = convolution_state
        self.use_inplace_state = True

    def _validate_forward(
        self,
        hidden_states: ttnn.Tensor,
    ) -> tuple[int, int]:
        if len(hidden_states.shape) != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states shape {tuple(hidden_states.shape)} must be [B,T,{self.config.hidden_size}]"
            )
        batch = hidden_states.shape[0]
        sequence = hidden_states.shape[1]
        if batch != 1:
            raise ValueError(f"KDA prefill currently requires batch size 1, got B={batch}")
        if sequence <= 0 or sequence % ttnn.TILE_SIZE != 0:
            raise ValueError(f"KDA prefill requires local T to be positive and divisible by 32, got T={sequence}")
        if self.sequence_parallel_size > 1:
            local_chunks = sequence // ttnn.TILE_SIZE
            if local_chunks % self.summary_group_chunks != 0:
                raise ValueError(
                    f"local chunk count {local_chunks} must be divisible by "
                    f"summary_group_chunks {self.summary_group_chunks}"
                )
        if self.recurrent_state is None or self.convolution_state is None:
            raise RuntimeError("KDA state is uninitialized; call reset_state(batch_size) first")
        if self.recurrent_state.shape[0] != batch:
            raise ValueError(f"state batch {self.recurrent_state.shape[0]} != input batch {batch}")
        return batch, sequence

    def _convolve_qkv(
        self,
        qkv: ttnn.Tensor,
        sequence: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Run depthwise convolution and emit Q/K/V without post-convolution slices."""
        assert self.convolution_state is not None
        config = self.config
        channels = self._convolution_width
        qkv_row_major = ttnn.to_layout(
            qkv,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state_row_major = ttnn.to_layout(
            self.convolution_state,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.sequence_parallel_size > 1:
            state_row_major, new_state = ttnn.transformer.kda_convolution_halo(
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
        if self.convolution_state.layout != ttnn.ROW_MAJOR_LAYOUT:
            new_state = ttnn.to_layout(
                new_state,
                self.convolution_state.layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        # Retained by real-K3 T=5120 component A/B: 74.36-75.76% faster at direct Q/K/V PCC >=0.999989.
        # Reproduce with tests/perf/test_fusion_ab.py; exact results are in perf_targets/bh_loudbox_fusion_ab.json.
        q, k, v = ttnn.transformer.kda_causal_conv1d_split(
            qkv_row_major,
            state_row_major,
            *self.weights.convolution_taps,
            config.q_dim,
            config.k_dim,
            config.v_dim,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
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
        beta = ttnn.sigmoid(beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            beta=beta,
            output_gate=output_gate,
        )

    def _kda_prefill(
        self,
        inputs: _KDAInputs,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the KDA recurrence and return its raw output and updated state."""
        assert self.recurrent_state is not None
        q, k = inputs.q, inputs.k
        key_dim = q.shape[-1] // inputs.beta.shape[-1]
        output, new_recurrent_state = ttnn.transformer.chunk_kda(
            q,
            k,
            inputs.v,
            inputs.decay,
            inputs.beta,
            scale=key_dim**-0.5,
            initial_state=self.recurrent_state,
            output_final_state=True,
            output_head_major=True,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            eye=self.chunk_const_tiles[0],
            tril=self.chunk_const_tiles[1],
            ones=self.chunk_const_tiles[2],
            masks=self.chunk_const_tiles[3],
            summary_group_chunks=self.summary_group_chunks,
            sequence_parallel_axis=(self.sequence_parallel_axis if self.sequence_parallel_size > 1 else None),
            # Real-K3 component A/B: BF16 affine-summary storage retained PCC 0.999429/0.999545/0.999702
            # for SP1xTP8/SP2xTP4/SP4xTP2, with <=0.687% median and <=0.858% 95% UCB latency cost.
            affine_summary_dtype=self.affine_summary_dtype,
            # Real-K3 component A/B: affine-prefix HiFi2 retained PCC 1.0 for every LoudBox layout;
            # median latency changed -0.228%/+0.299%/-0.276% for SP1xTP8/SP2xTP4/SP4xTP2.
            affine_prefix_compute_kernel_config=self.affine_prefix_compute_config,
            # Real-K3 component A/B: BF16 grouped-scan output retained PCC >=0.999984 for every
            # LoudBox layout and improved median component latency by 0.199%-0.347%.
            grouped_scan_output_dtype=self.grouped_scan_output_dtype,
            # Real-K3 component A/B: grouped final-scan HiFi2 retained PCC >=0.995789 for every
            # LoudBox layout; median latency changed -0.365%/-0.473%/+0.094% for SP1xTP8/SP2xTP4/SP4xTP2.
            grouped_scan_compute_kernel_config=self.grouped_scan_compute_config,
            use_bf16_prep_intermediates=self.use_bf16_prep_intermediates,
        )
        assert new_recurrent_state is not None
        if new_recurrent_state.dtype != self.recurrent_state_dtype:
            new_recurrent_state = ttnn.typecast(new_recurrent_state, self.recurrent_state_dtype)
        return output, new_recurrent_state

    def _kda_rms_norm(
        self,
        output: ttnn.Tensor,
        output_gate: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply the KDA gated RMSNorm epilogue."""
        config, weights = self.config, self.weights
        # Retained by real-K3 T=5120 component A/B: 92.72-93.92% faster at output PCC >=0.999990.
        # Reproduce with tests/perf/test_fusion_ab.py; exact results are in perf_targets/bh_loudbox_fusion_ab.json.
        return ttnn.transformer.kda_gated_rms_norm(
            output,
            output_gate,
            weights.norm,
            config.num_heads,
            epsilon=config.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
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
        output = ttnn.reshape(output, (batch, sequence, config.v_dim))
        if self.tensor_parallel_size > 1:
            assert self.tt_ccl is not None
            # Real-K3 T=5120 full-layer A/B rejected fused MMRS: -3.84% SP2 and -1.56% SP4.
            # Standalone matmul and RS already overlap; see perf_targets/bh_loudbox_fusion_ab.json.
            if output.dtype != ttnn.bfloat16:
                output = ttnn.typecast(output, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
                ),
                compute_kernel_config=self.output_projection_compute_config,
            )
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
        else:
            output = ttnn.linear(
                output,
                weights.output_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.output_projection_compute_config,
            )
        return output

    def _commit_state(
        self,
        new_recurrent_state: ttnn.Tensor,
        new_convolution_state: ttnn.Tensor,
    ) -> None:
        """Commit eager or trace-stable cache updates without exposing the mutation mode."""
        assert self.recurrent_state is not None
        assert self.convolution_state is not None
        if self.use_inplace_state:
            ttnn.copy(new_recurrent_state, self.recurrent_state)
            ttnn.copy(new_convolution_state, self.convolution_state)
        else:
            self.recurrent_state = new_recurrent_state
            self.convolution_state = new_convolution_state

    def forward(
        self,
        hidden_states: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run prefill KDA without any host tensor operation or implicit fallback."""
        batch, sequence = self._validate_forward(hidden_states)
        projected = self._project_inputs(hidden_states)
        q, k, v, new_convolution_state = self._convolve_qkv(
            projected.qkv,
            sequence=sequence,
        )
        inputs = self._compute_gates(
            q,
            k,
            v,
            beta=projected.beta,
            decay_rank=projected.decay_rank,
            output_gate=projected.output_gate,
        )
        output, new_recurrent_state = self._kda_prefill(inputs)
        output = self._kda_rms_norm(output, inputs.output_gate)
        output = self._project_output(output, batch=batch, sequence=sequence)
        self._commit_state(new_recurrent_state, new_convolution_state)
        return output
