# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles
from models.demos.blackhole.qwen36.tt.tp_common import matmul_reduce_scatter_prefill
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.experimental.kimi_delta_attention.config import KDAConfig, KDAProgramConfig
from models.experimental.kimi_delta_attention.tt.recurrence import kda_prefill
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights, load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce


def _slice_width(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    stop = list(tensor.shape)
    begin = [0] * len(stop)
    begin[-1] = start
    stop[-1] = end
    return ttnn.slice(tensor, tuple(begin), tuple(stop), memory_config=ttnn.DRAM_MEMORY_CONFIG)


@dataclass(frozen=True)
class _ProjectedInputs:
    combined: ttnn.Tensor
    qkv: ttnn.Tensor


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
        self._prepared_convolution_weights: dict[int, ttnn.Tensor] = {}
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
        if self.sequence_parallel_size > 1:
            if sequence % ttnn.TILE_SIZE != 0:
                raise ValueError(f"sequence-parallel KDA prefill requires local T divisible by 32, got T={sequence}")
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

    def _causal_conv1d_prefill(
        self,
        qkv: ttnn.Tensor,
        sequence: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Run depthwise convolution and emit Q/K/V without post-convolution slices."""
        assert self.convolution_state is not None
        config = self.config
        channels = self._convolution_width
        input_length = sequence + config.conv_kernel_size - 1
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
        # The generic depthwise conv exceeds Blackhole L1 at K3's 4608 local channels.
        # Keep the large-width exception local until conv1d gains a suitable sliced config.
        use_split_convolution = sequence > 640 or channels >= 4608
        if use_split_convolution:
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

        conv_input = ttnn.concat(
            [state_row_major, qkv_row_major],
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conv_input = ttnn.reshape(conv_input, (1, input_length, 1, channels))
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        if input_length not in self._prepared_convolution_weights:
            self._prepared_convolution_weights[input_length] = ttnn.prepare_conv_weights(
                weight_tensor=self.weights.convolution_weight,
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=channels,
                out_channels=channels,
                batch_size=1,
                input_height=1,
                input_width=input_length,
                kernel_size=(1, config.conv_kernel_size),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=False,
                groups=channels,
                device=self.device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
                compute_config=self.compute_config,
            )
        output = ttnn.conv1d(
            input_tensor=conv_input,
            weight_tensor=self._prepared_convolution_weights[input_length],
            device=self.device,
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_length=input_length,
            kernel_size=config.conv_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=channels,
            dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=self.compute_config,
            slice_config=(
                ttnn.Conv2dL1FullSliceConfig
                if sequence <= 640
                else ttnn.Conv2dSliceConfig(
                    slice_type=ttnn.Conv2dDRAMSliceWidth,
                    num_slices=0,
                )
            ),
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (1, sequence, channels))
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.silu(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = _slice_width(output, 0, config.q_dim)
        k = _slice_width(output, config.q_dim, config.q_dim + config.k_dim)
        v = _slice_width(output, config.q_dim + config.k_dim, channels)
        return q, k, v, new_state

    def _project_inputs(
        self,
        hidden_states: ttnn.Tensor,
        *,
        head_major: bool,
        memory_config: ttnn.MemoryConfig,
    ) -> _ProjectedInputs:
        """Run the fused input projection and expose its QKV branch."""
        weights = self.weights
        projected = ttnn.linear(
            hidden_states,
            weights.input_projection_prefill if head_major else weights.input_projection,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
        )
        qkv = _slice_width(projected, 0, self._convolution_width)
        return _ProjectedInputs(
            combined=projected,
            qkv=qkv,
        )

    def _convolve_qkv(
        self,
        qkv: ttnn.Tensor,
        *,
        batch: int,
        sequence: int,
        head_major: bool,
        memory_config: ttnn.MemoryConfig,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Apply causal convolution and return Q/K/V plus the next convolution state."""
        config, weights = self.config, self.weights
        if batch == 1 and sequence >= ttnn.TILE_SIZE:
            q, k, v, new_convolution_state = self._causal_conv1d_prefill(qkv, sequence)
        else:
            assert self.convolution_state is not None
            convolution_state = self.convolution_state
            if convolution_state.layout != ttnn.TILE_LAYOUT:
                convolution_state = ttnn.to_layout(
                    convolution_state,
                    ttnn.TILE_LAYOUT,
                    memory_config=memory_config,
                )
            qkv, new_convolution_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                config.conv_kernel_size,
                self.device,
                memory_config=memory_config,
                conv_state=convolution_state,
                weight_taps=weights.convolution_taps,
            )
            q = _slice_width(qkv, 0, config.q_dim)
            k = _slice_width(qkv, config.q_dim, config.q_dim + config.k_dim)
            v = _slice_width(qkv, config.q_dim + config.k_dim, self._convolution_width)
        if not head_major:
            q = ttnn.reshape(q, (batch, sequence, config.num_heads, config.head_k_dim))
            k = ttnn.reshape(k, (batch, sequence, config.num_heads, config.head_k_dim))
            v = ttnn.reshape(v, (batch, sequence, config.num_heads, config.head_v_dim))
        return q, k, v, new_convolution_state

    def _compute_gates(
        self,
        projected: _ProjectedInputs,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        batch: int,
        sequence: int,
        head_major: bool,
    ) -> _KDAInputs:
        """Evaluate decay and write gates while preserving the output gate for the epilogue."""
        config, weights = self.config, self.weights
        output_gate_width = config.v_dim if head_major or weights.output_gate_is_direct else config.head_v_dim
        auxiliary_start = self._convolution_width
        decay_rank = _slice_width(projected.combined, auxiliary_start, auxiliary_start + config.head_k_dim)
        output_gate = _slice_width(
            projected.combined,
            auxiliary_start + config.head_k_dim,
            auxiliary_start + config.head_k_dim + output_gate_width,
        )
        beta = _slice_width(
            projected.combined,
            auxiliary_start + config.head_k_dim + output_gate_width,
            auxiliary_start + config.head_k_dim + output_gate_width + config.num_heads,
        )
        beta = ttnn.sigmoid(beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if head_major:
            decay_bias = weights.decay_bias_flat
            decay_scale = weights.decay_scale_flat
            gate = ttnn.linear(
                decay_rank,
                weights.decay_output_projection,
                bias=decay_bias,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
        else:
            raw_gate = ttnn.linear(
                decay_rank,
                weights.decay_output_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
            raw_gate = ttnn.reshape(raw_gate, (batch, sequence, config.num_heads, config.head_k_dim))
            decay_bias = weights.decay_bias
            decay_scale = weights.decay_scale
            gate = ttnn.add(raw_gate, decay_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if config.gate_lower_bound is None:
            gate = ttnn.multiply(
                decay_scale,
                gate,
                input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
                dtype=ttnn.bfloat16 if head_major else ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = ttnn.multiply(
                decay_scale,
                gate,
                dtype=ttnn.bfloat16 if head_major else ttnn.float32,
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

    def _run_kda_and_norm(
        self,
        inputs: _KDAInputs,
        *,
        batch: int,
        sequence: int,
        head_major: bool,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the recurrence and return normalized heads regardless of kernel fusion."""
        config, weights = self.config, self.weights
        assert self.recurrent_state is not None
        # The long-context grouped-prefix path returns raw scan output; normalize it
        # after regrouping instead of asking the serial scan for a fused RMS tensor.
        use_group_prefix = (
            head_major
            and (self.sequence_parallel_size > 1 or (sequence >= 5120 and sequence % 256 == 0))
            and os.getenv("QWEN_KDA_SERIAL_SCAN") is None
        )
        fuse_scan_rms = (
            head_major and sequence > 640 and os.getenv("QWEN_KDA_GROUP_PREFIX") is None and not use_group_prefix
        )
        output, new_recurrent_state = kda_prefill(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.decay,
            inputs.beta,
            self.recurrent_state,
            self.chunk_const_tiles,
            rms_gate=inputs.output_gate if fuse_scan_rms else None,
            rms_weight=weights.norm if fuse_scan_rms else None,
            rms_epsilon=config.norm_eps,
            summary_group_chunks=self.summary_group_chunks,
            sequence_parallel_axis=(self.sequence_parallel_axis if self.sequence_parallel_size > 1 else None),
            affine_identity=self.affine_identity,
            affine_zero=self.affine_zero,
        )
        if new_recurrent_state.dtype != config.recurrent_state_dtype:
            new_recurrent_state = ttnn.typecast(new_recurrent_state, config.recurrent_state_dtype)
        if head_major:
            output_gate = inputs.output_gate
        elif weights.output_gate_is_direct:
            output_gate = ttnn.reshape(inputs.output_gate, (batch, sequence, config.num_heads, config.head_v_dim))
        else:
            assert weights.output_gate_projection is not None
            output_gate = ttnn.linear(
                inputs.output_gate,
                weights.output_gate_projection,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
            output_gate = ttnn.reshape(output_gate, (batch, sequence, config.num_heads, config.head_v_dim))
        if head_major and not fuse_scan_rms:
            output = ttnn.transformer.kda_gated_rms_norm(
                output,
                output_gate,
                weights.norm,
                config.num_heads,
                epsilon=config.norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
        elif not head_major:
            output = ttnn.rms_norm(
                output,
                weight=weights.norm,
                epsilon=config.norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            output = ttnn.multiply(
                output_gate,
                output,
                input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return output, new_recurrent_state

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
        fused_output_collective = self.tensor_parallel_size > 1 and (
            self.sequence_parallel_size > 1 or config.v_dim >= 8 * ttnn.TILE_SIZE
        )
        if fused_output_collective:
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
        if self.tensor_parallel_size > 1 and not fused_output_collective:
            assert self.tt_ccl is not None
            output = ttnn.reshape(output, (batch, 1, sequence, self.global_config.hidden_size))
            output = tt_all_reduce(
                output,
                self.device,
                self.tt_ccl,
                cluster_axis=None if self.sequence_parallel_size == 1 else self.tensor_parallel_axis,
                dim=3,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            output = ttnn.reshape(
                output,
                (batch, sequence, self.global_config.hidden_size // self.tensor_parallel_size),
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
        head_major = sequence % ttnn.TILE_SIZE == 0
        memory_config = (
            ttnn.L1_MEMORY_CONFIG if batch * sequence * self._convolution_width <= 65536 else ttnn.DRAM_MEMORY_CONFIG
        )

        projected = self._project_inputs(hidden_states, head_major=head_major, memory_config=memory_config)
        q, k, v, new_convolution_state = self._convolve_qkv(
            projected.qkv,
            batch=batch,
            sequence=sequence,
            head_major=head_major,
            memory_config=memory_config,
        )
        inputs = self._compute_gates(
            projected,
            q,
            k,
            v,
            batch=batch,
            sequence=sequence,
            head_major=head_major,
        )
        output, new_recurrent_state = self._run_kda_and_norm(
            inputs,
            batch=batch,
            sequence=sequence,
            head_major=head_major,
        )
        output = self._project_output(output, batch=batch, sequence=sequence)
        self._commit_state(new_recurrent_state, new_convolution_state)
        return output
