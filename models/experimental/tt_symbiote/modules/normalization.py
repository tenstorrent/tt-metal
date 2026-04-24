# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Normalization layer implementations for TTNN."""

from torch import nn
import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.run_config import trace_enabled


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
        if not hasattr(rms_norm, "weight") or rms_norm.weight is None:
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


@trace_enabled
class TTNNLocalRMSNorm(TTNNModule):
    """
    Local (per-device) RMSNorm for per-head norms (Q-norm, K-norm, V-norm) in Gemma4 attention.

    Unlike TTNNDistributedRMSNorm, this operates on local tensors per device and does NOT
    perform any cross-device reduction. Supports Gemma4RMSNorm which uses `eps` instead of
    `variance_epsilon` and may have `with_scale=False` (no learnable weight).
    """

    @classmethod
    def from_torch(cls, rms_norm):
        """Create from PyTorch RMSNorm (supports Gemma4RMSNorm with optional scale)."""
        # Gemma4RMSNorm may have with_scale=False, meaning weight is None
        has_scale = getattr(rms_norm, "with_scale", True)
        if has_scale and rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        """Preprocess RMSNorm weights for TTNN."""
        # Gemma4RMSNorm uses `eps` instead of `variance_epsilon`
        self.eps = getattr(self.torch_layer, "eps", getattr(self.torch_layer, "variance_epsilon", 1e-6))

        has_scale = getattr(self.torch_layer, "with_scale", True)
        if has_scale and hasattr(self.torch_layer, "weight") and self.torch_layer.weight is not None:
            weight = self.torch_layer.weight
        else:
            # Create dummy all-ones weight for Gemma4RMSNorm with with_scale=False
            dim = self._infer_dim()
            weight = torch.ones(dim)

        # Shape [1, dim] for broadcasting over 4D inputs [batch, heads, seq, head_dim]
        self.tt_weight = ttnn.from_torch(weight.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _infer_dim(self):
        """Infer the normalization dimension from the torch layer."""
        if hasattr(self.torch_layer, "weight") and self.torch_layer.weight is not None:
            return self.torch_layer.weight.shape[0]
        # Fallback: check for normalized_shape or similar attributes
        if hasattr(self.torch_layer, "normalized_shape"):
            return self.torch_layer.normalized_shape
        if hasattr(self.torch_layer, "dim"):
            return self.torch_layer.dim
        # Stashed by attention module for Gemma4RMSNorm(with_scale=False)
        if hasattr(self.torch_layer, "_norm_dim"):
            return self.torch_layer._norm_dim
        raise ValueError("Cannot infer normalization dimension from torch layer with no weight")

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device with replication across mesh."""
        self.tt_weight = ttnn.to_device(
            self.tt_weight,
            self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through local RMSNorm. Input is 4D [batch, heads, seq, head_dim]."""
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)
        return x


@trace_enabled
class TTNNDistributedRMSNorm(TTNNModule):
    """
    Distributed RMSNorm implementation that performs the reduction across devices in the forward pass.

    """

    @classmethod
    def from_torch(cls, rms_norm: "RMSNorm"):
        """Create from PyTorch RMSNorm."""
        if not hasattr(rms_norm, "weight") or rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        dim = self.torch_layer.weight.shape[0]
        # Pad to nearest multiple of 32 for TILE compatibility
        padded_dim = ((dim + 31) // 32) * 32
        weight = self.torch_layer.weight
        if padded_dim != dim:
            weight = torch.nn.functional.pad(weight, (0, padded_dim - dim), value=1.0)
        self.weight_distributed = ttnn.as_tensor(
            weight.unsqueeze(0).view(1, 1, padded_dim).reshape([1, 1, padded_dim // 32, 32]).to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=(ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=list(self.device.shape))),
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @run_on_devices(DeviceArch.N300, DeviceArch.T3K)
    def forward(self, inp):
        original_shape = inp.shape
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)  # Add batch dimension for RMSNorm
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
        )
        # AllGather stats — use Ring topology for trace compatibility.
        # Linear topology may allocate dynamic intermediates not pinned by trace.
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Ring,
        )
        # Run distributed rmsnorm part 2
        eps = getattr(self.torch_layer, "variance_epsilon", getattr(self.torch_layer, "eps", 1e-6))
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=eps,
            weight=self.weight_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_stats.deallocate(True)

        # Squeeze back to original shape if we added a batch dimension
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])

        return tt_out
