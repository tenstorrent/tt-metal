# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Normalization layer implementations for TTNN."""

from torch import nn
import torch
import ttnn
from models.experimental.tt_symbiote.core.module import (
    TTNNModule,
    run_on_devices,
    DeviceArch,
    set_distributed_tensor_config,
)
from models.experimental.tt_symbiote.core.utils import tree_map


class TTNNLayerNorm(TTNNModule):
    """TTNN-accelerated LayerNorm."""

    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm):
        """Create TTNNLayerNorm from PyTorch LayerNorm."""
        if layer_norm.weight is None:
            print(f"Warning: LayerNorm layer {layer_norm} has no weight. Using standard LayerNorm.")
            return layer_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = layer_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        """Preprocess LayerNorm weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = nn.LayerNorm(normalized_shape=1)
        self.tt_weight = ttnn.from_torch(self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias = ttnn.from_torch(self.torch_layer.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through LayerNorm."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.layer_norm(
            input_tensor,
            weight=self.tt_weight,
            bias=self.tt_bias,
        )
        return tt_output


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TTNNRMSNorm(TTNNModule):
    @classmethod
    def from_torch(cls, rms_norm: DeepseekV2RMSNorm):
        """Create from PyTorch RMSNorm."""
        if rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        """Preprocess RMSNorm weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = DeepseekV2RMSNorm(hidden_size=1)
        self.tt_weight = ttnn.from_torch(
            self.torch_layer.weight.unsqueeze(0).expand(32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.torch_layer.variance_epsilon)
        return x


class TTNNDistributedRMSNorm(TTNNModule):
    """
    Distributed RMSNorm implementation that performs the reduction across devices in the forward pass.

    """

    @classmethod
    def from_torch(cls, rms_norm: "RMSNorm"):
        """Create from PyTorch RMSNorm."""
        if rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device.

        Weight is replicated (not sharded) because rms_norm_post_all_gather receives
        full input after embedding/attention; gamma must match input padded shape.
        """
        dim = self.torch_layer.weight.shape[0]
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        self.weight_distributed = ttnn.as_tensor(
            self.torch_layer.weight.unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // 32, 32]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)

    @run_on_devices(DeviceArch.T3K, DeviceArch.P150x8, DeviceArch.N150)
    def forward(self, inp):
        original_shape = inp.shape
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        needs_squeeze = len(original_shape) == 3
        if needs_squeeze:
            inp = ttnn.unsqueeze(inp, 1)
        if self.device.get_num_devices() == 1:
            tt_out = ttnn.rms_norm(inp, weight=self.weight_distributed, epsilon=self.torch_layer.variance_epsilon)
        else:
            tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)
            tt_stats = ttnn.experimental.all_gather_async(
                tt_stats,
                dim=-1,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
            tt_out = ttnn.rms_norm_post_all_gather(
                inp,
                tt_stats,
                epsilon=self.torch_layer.variance_epsilon,
                weight=self.weight_distributed,
            )
            tt_stats.deallocate(True)
        if needs_squeeze:
            tt_out = ttnn.squeeze(tt_out, 1)
        return tt_out

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = self.device_state.get_replicated_tensor_config((1, 1, self.torch_layer.weight.shape[0]))
        return tree_map(set_distributed_tensor_config(cfg), output_tensors)
