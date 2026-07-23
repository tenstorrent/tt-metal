# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles
from models.demos.blackhole.qwen36.tt.tp_common import matmul_reduce_scatter_prefill
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tt.recurrence import chunk_kda_recurrence, fused_kda_recurrence
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights, load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce


def _slice_width(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    stop = list(tensor.shape)
    begin = [0] * len(stop)
    begin[-1] = start
    stop[-1] = end
    return ttnn.slice(tensor, tuple(begin), tuple(stop), memory_config=ttnn.DRAM_MEMORY_CONFIG)


class KimiDeltaAttention:
    """Stateful, fully device-resident KDA correctness implementation."""

    def __init__(
        self,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor],
        tensor_cache_path: Path | None = None,
        tt_ccl: TT_CCL | None = None,
    ) -> None:
        self.device = mesh_device
        self.weights: KDAWeights = load_kda_weights(
            mesh_device,
            config,
            state_dict,
            tensor_cache_path,
        )
        self.tensor_parallel_size = self.weights.tensor_parallel_size
        self.global_config = config
        self.config = replace(config, num_heads=config.num_heads // self.tensor_parallel_size)
        if self.tensor_parallel_size > 1 and tt_ccl is None:
            raise ValueError("tt_ccl is required for tensor-parallel KDA")
        self.tt_ccl = tt_ccl
        self.chunk_const_tiles = build_fused_const_tiles(mesh_device, _FUSED_CHUNK_SIZE)
        self._prepared_convolution_weights: dict[int, ttnn.Tensor] = {}
        self.recurrent_state: ttnn.Tensor | None = None
        self.convolution_state: ttnn.Tensor | None = None
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

    def reset_state(self, batch_size: int | None = None) -> None:
        """Allocate zero cache for a batch, or release logical ownership."""
        if batch_size is None:
            self.recurrent_state = None
            self.convolution_state = None
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
            layout=ttnn.TILE_LAYOUT,
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
        self.use_inplace_state = True

    def _validate_forward(
        self,
        hidden_states: ttnn.Tensor,
        mode: Literal["recurrent", "chunk"],
        chunk_size: int | None,
        valid_len: int | None,
    ) -> tuple[int, int]:
        if mode not in ("recurrent", "chunk"):
            raise ValueError(f"unsupported KDA mode: {mode}")
        if len(hidden_states.shape) != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states shape {tuple(hidden_states.shape)} must be [B,T,{self.config.hidden_size}]"
            )
        batch = hidden_states.shape[0]
        sequence = hidden_states.shape[1]
        if mode == "recurrent" and sequence != 1:
            raise ValueError(f"recurrent mode requires T=1, got T={sequence}")
        if mode == "chunk" and chunk_size not in (None, 32):
            raise ValueError(f"chunk KDA currently requires chunk_size=32, got {chunk_size}")
        if valid_len is not None and valid_len != sequence:
            raise NotImplementedError(f"composed KDA currently requires valid_len == T, got {valid_len} != {sequence}")
        if self.recurrent_state is None or self.convolution_state is None:
            raise RuntimeError("KDA state is uninitialized; call reset_state(batch_size) first")
        if self.recurrent_state.shape[0] != batch:
            raise ValueError(f"state batch {self.recurrent_state.shape[0]} != input batch {batch}")
        return batch, sequence

    def _causal_conv1d_prefill(
        self,
        qkv: ttnn.Tensor,
        sequence: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run trace-safe native depthwise convolution over QKV and its carry."""
        assert self.convolution_state is not None
        config = self.config
        channels = self._convolution_width
        input_length = sequence + config.conv_kernel_size - 1
        new_state = ttnn.slice(
            qkv,
            (0, sequence - (config.conv_kernel_size - 1), 0),
            (1, sequence, channels),
        )
        new_state = ttnn.to_memory_config(
            ttnn.to_layout(new_state, ttnn.TILE_LAYOUT),
            ttnn.DRAM_MEMORY_CONFIG,
        )
        conv_input = ttnn.concat(
            [self.convolution_state, qkv],
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conv_input = ttnn.to_layout(
            conv_input,
            ttnn.ROW_MAJOR_LAYOUT,
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
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (1, sequence, channels))
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.silu(output, memory_config=ttnn.DRAM_MEMORY_CONFIG), new_state

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        mode: Literal["recurrent", "chunk"] = "recurrent",
        chunk_size: int | None = None,
        valid_len: int | None = None,
    ) -> ttnn.Tensor:
        """Run KDA without any host tensor operation or implicit fallback."""
        batch, sequence = self._validate_forward(hidden_states, mode, chunk_size, valid_len)
        config, weights = self.config, self.weights
        memory_config = (
            ttnn.L1_MEMORY_CONFIG if batch * sequence * self._convolution_width <= 65536 else ttnn.DRAM_MEMORY_CONFIG
        )

        qkv = ttnn.linear(
            hidden_states,
            weights.qkv_projection,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
        )
        if mode == "chunk" and batch == 1 and sequence >= ttnn.TILE_SIZE:
            qkv, new_convolution_state = self._causal_conv1d_prefill(qkv, sequence)
        else:
            qkv, new_convolution_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                config.conv_kernel_size,
                self.device,
                memory_config=memory_config,
                conv_state=self.convolution_state,
                weight_taps=weights.convolution_taps,
            )
        q = _slice_width(qkv, 0, config.q_dim)
        k = _slice_width(qkv, config.q_dim, config.q_dim + config.k_dim)
        if mode == "recurrent" or sequence % ttnn.TILE_SIZE != 0:
            q = ttnn.reshape(q, (batch, sequence, config.num_heads, config.head_k_dim))
            k = ttnn.reshape(k, (batch, sequence, config.num_heads, config.head_k_dim))
        v = _slice_width(qkv, config.q_dim + config.k_dim, self._convolution_width)
        if mode == "recurrent" or sequence % ttnn.TILE_SIZE != 0:
            v = ttnn.reshape(v, (batch, sequence, config.num_heads, config.head_v_dim))

        auxiliary = ttnn.linear(
            hidden_states,
            weights.auxiliary_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        decay_rank = _slice_width(auxiliary, 0, config.head_k_dim)
        output_gate_rank = _slice_width(
            auxiliary,
            config.head_k_dim,
            config.head_k_dim + config.head_v_dim,
        )
        beta = _slice_width(
            auxiliary,
            config.head_k_dim + config.head_v_dim,
            config.head_k_dim + config.head_v_dim + config.num_heads,
        )
        beta = ttnn.sigmoid(beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        raw_gate = ttnn.linear(
            decay_rank,
            weights.decay_output_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        head_major = mode == "chunk" and sequence % ttnn.TILE_SIZE == 0
        if head_major:
            decay_bias = weights.decay_bias_flat
            decay_scale = weights.decay_scale_flat
        else:
            raw_gate = ttnn.reshape(raw_gate, (batch, sequence, config.num_heads, config.head_k_dim))
            decay_bias = weights.decay_bias
            decay_scale = weights.decay_scale
        gate = ttnn.add(raw_gate, decay_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.softplus(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.multiply(decay_scale, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        assert self.recurrent_state is not None
        if mode == "recurrent":
            output, new_recurrent_state = fused_kda_recurrence(q, k, v, gate, beta, self.recurrent_state)
        else:
            output, new_recurrent_state = chunk_kda_recurrence(
                q, k, v, gate, beta, self.recurrent_state, self.chunk_const_tiles
            )
        if new_recurrent_state.dtype != config.recurrent_state_dtype:
            new_recurrent_state = ttnn.typecast(new_recurrent_state, config.recurrent_state_dtype)
        output_gate = ttnn.linear(
            output_gate_rank,
            weights.output_gate_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        output_gate = ttnn.reshape(output_gate, (batch, sequence, config.num_heads, config.head_v_dim))
        if head_major:
            output_gate = ttnn.reshape(
                ttnn.permute(output_gate, (0, 2, 1, 3)),
                (batch * config.num_heads, sequence, config.head_v_dim),
            )
        output = ttnn.rms_norm(
            output,
            weight=weights.norm,
            epsilon=config.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_gate = ttnn.typecast(ttnn.sigmoid(output_gate), ttnn.float32)
        output = ttnn.multiply(output, output_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if head_major:
            output = ttnn.reshape(output, (batch, config.num_heads, sequence, config.head_v_dim))
            output = ttnn.experimental.nlp_concat_heads(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (batch, sequence, config.v_dim))
        fused_output_collective = (
            self.tensor_parallel_size > 1 and mode == "chunk" and config.v_dim >= 8 * ttnn.TILE_SIZE
        )
        if fused_output_collective:
            assert self.tt_ccl is not None
            output = matmul_reduce_scatter_prefill(
                output,
                weights.output_projection,
                self.tt_ccl,
                self.compute_config,
                ttnn.Topology.Ring,
                self.tensor_parallel_size,
                output.dtype,
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
                cluster_axis=0,
                dim=3,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            output = ttnn.reshape(
                output,
                (batch, sequence, self.global_config.hidden_size // self.tensor_parallel_size),
            )

        if self.use_inplace_state:
            ttnn.copy(new_recurrent_state, self.recurrent_state)
            ttnn.copy(new_convolution_state, self.convolution_state)
        else:
            self.recurrent_state = new_recurrent_state
            self.convolution_state = new_convolution_state
        return output
