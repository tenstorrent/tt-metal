# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles
from models.demos.blackhole.qwen36.tt.tp_common import matmul_reduce_scatter_prefill
from models.experimental.kimi_delta_attention.config import KDAConfig, KDAProgramConfig
from models.experimental.kimi_delta_attention.tt.recurrence import kda_prefill
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights, load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL


def _slice_width(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    stop = list(tensor.shape)
    begin = [0] * len(stop)
    begin[-1] = start
    stop[-1] = end
    return ttnn.slice(tensor, tuple(begin), tuple(stop), memory_config=ttnn.DRAM_MEMORY_CONFIG)


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
        state_dict: Mapping[str, torch.Tensor],
        tensor_cache_path: Path | None = None,
        tt_ccl: TT_CCL | None = None,
        tensor_parallel_axis: int = 1,
        program_config: KDAProgramConfig | None = None,
        summary_group_chunks: int | None = None,
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
        self.weights: KDAWeights = load_kda_weights(
            mesh_device,
            config,
            state_dict,
            tensor_cache_path,
            tensor_parallel_axis=tensor_parallel_axis,
        )
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
        self.chunk_const_tiles = build_fused_const_tiles(mesh_device, _FUSED_CHUNK_SIZE)
        self.recurrent_state: ttnn.Tensor | None = None
        self.convolution_state: ttnn.Tensor | None = None
        self.affine_identity: ttnn.Tensor | None = None
        self.affine_zero: ttnn.Tensor | None = None
        self.use_inplace_state = False
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @property
    def _convolution_width(self) -> int:
        return self.config.q_dim + self.config.k_dim + self.config.v_dim

    def _allocate_affine_constants(self, batch_size: int) -> None:
        if self.sequence_parallel_size == 1:
            self.affine_identity = None
            self.affine_zero = None
            return
        state_shape = (
            batch_size,
            self.config.num_heads,
            self.config.head_k_dim,
            self.config.head_v_dim,
        )
        identity = torch.eye(self.config.head_k_dim, dtype=torch.float32).reshape(1, 1, *state_shape[-2:])
        identity = identity.expand(state_shape).contiguous()
        self.affine_identity = ttnn.from_torch(
            identity,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.affine_zero = ttnn.zeros(
            state_shape,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def reset_state(self, batch_size: int | None = None) -> None:
        """Allocate zero cache for a batch, or release logical ownership."""
        if batch_size is None:
            self.recurrent_state = None
            self.convolution_state = None
            self.affine_identity = None
            self.affine_zero = None
            self.use_inplace_state = False
            return
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.recurrent_state = ttnn.zeros(
            (
                batch_size,
                self.config.num_heads,
                self.config.head_k_dim,
                self.config.head_v_dim,
            ),
            dtype=self.config.recurrent_state_dtype,
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
        self._allocate_affine_constants(batch_size)
        self.use_inplace_state = False

    def set_external_state(
        self,
        recurrent_state: ttnn.Tensor,
        convolution_state: ttnn.Tensor,
    ) -> None:
        """Adopt trace-stable external buffers and update them in place."""
        batch = recurrent_state.shape[0]
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
        if recurrent_state.dtype != self.config.recurrent_state_dtype:
            raise ValueError(f"recurrent_state dtype {recurrent_state.dtype} != {self.config.recurrent_state_dtype}")
        if convolution_state.dtype != ttnn.bfloat16:
            raise ValueError(f"convolution_state dtype {convolution_state.dtype} != {ttnn.bfloat16}")
        self.recurrent_state = recurrent_state
        self.convolution_state = convolution_state
        self._allocate_affine_constants(batch)
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
                (1, sequence, channels),
            )
        if self.convolution_state.layout != ttnn.ROW_MAJOR_LAYOUT:
            new_state = ttnn.to_layout(
                new_state,
                self.convolution_state.layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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
        config = self.config
        assert self.recurrent_state is not None
        output, new_recurrent_state = kda_prefill(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.decay,
            inputs.beta,
            self.recurrent_state,
            self.chunk_const_tiles,
            summary_group_chunks=self.summary_group_chunks,
            sequence_parallel_axis=(self.sequence_parallel_axis if self.sequence_parallel_size > 1 else None),
            affine_identity=self.affine_identity,
            affine_zero=self.affine_zero,
        )
        if new_recurrent_state.dtype != config.recurrent_state_dtype:
            new_recurrent_state = ttnn.typecast(new_recurrent_state, config.recurrent_state_dtype)
        return output, new_recurrent_state

    def _kda_rms_norm(
        self,
        output: ttnn.Tensor,
        output_gate: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply the KDA gated RMSNorm epilogue."""
        config, weights = self.config, self.weights
        return ttnn.transformer.kda_gated_rms_norm(
            output,
            output_gate,
            weights.norm,
            config.num_heads,
            epsilon=config.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
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
            # Keep the MMRS input and output dtypes equal: mixed page sizes corrupt the fused collective.
            # BF16 halves the projection intermediate and reduce-scatter traffic; accumulation remains FP32.
            output = ttnn.typecast(output, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            output = matmul_reduce_scatter_prefill(
                output,
                weights.output_projection,
                self.tt_ccl,
                self.compute_config,
                ttnn.Topology.Ring,
                self.tensor_parallel_size,
                ttnn.bfloat16,
                cluster_axis=None if self.sequence_parallel_size == 1 else self.tensor_parallel_axis,
                out_block_w_cap=self.output_projection_out_block_w,
            )
        else:
            output = ttnn.linear(
                output,
                weights.output_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
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
            if new_convolution_state.layout != self.convolution_state.layout:
                new_convolution_state = ttnn.to_layout(
                    new_convolution_state,
                    self.convolution_state.layout,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
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
