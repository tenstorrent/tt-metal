# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Normalization layer implementations for TTNN."""

from torch import nn
import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, tree_map, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig, trace_enabled


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


@trace_enabled
class TTNNDistributedRMSNorm(TTNNModule):
    """
    Distributed RMSNorm implementation that performs the reduction across devices in the forward pass.

    """

    @property
    def _is_distributed(self):
        """True when running on a multi-device mesh (col-sharded activations; CCL not required for output metadata)."""
        return self.device is not None and self.device.get_num_devices() > 1

    def set_output_tensors_config_impl(self, output_tensors):
        """Col-sharded activations: same mesh composer / logical shape as Qwen decoder attention (dim=-1)."""

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    e.set_distributed_tensor_config(
                        DistributedTensorConfig(
                            mesh_mapper=mesh_mapper,
                            mesh_composer=mesh_composer,
                            logical_shape_fn=logical_shape_for_col_sharded,
                        )
                    )
            return e

        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)
        return tree_map(set_col_sharded_config, output_tensors)

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
        """Move weights to TTNN device."""
        dim = int(self.torch_layer.weight.shape[0])
        assert dim % 32 == 0, f"TTNNDistributedRMSNorm gamma length {dim} must be divisible by 32"
        w_bf16 = self.torch_layer.weight.to(torch.bfloat16)

        if self.device is None or self.device.get_num_devices() <= 1:
            relayout = w_bf16.view(1, 1, dim // 32, 32)
            self.weight_distributed = ttnn.as_tensor(relayout, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
            return

        mesh_shape = list(self.device.shape)
        ncol = int(mesh_shape[-1])
        n_dev = int(self.device.get_num_devices())
        ntiles = dim // 32

        # ShardTensor2dMesh(..., dims=(None, 2), mesh_shape=(1, ncol)) requires the sharded axis
        # (tile rows) to align with the mesh — e.g. code_predictor q_norm (dim=128 → 4 tiles) on T3K
        # (ncol=8) does not shard as 8 chunks. Width-sharding [1,1,1,dim] breaks
        # rms_norm_post_all_gather (gamma last dim must pad to TILE_WIDTH=32). Use PyTorch RMSNorm
        # for those subgraphs instead (see test_qwen_omni ``_restore_torch_rmsnorm_in_code_predictor``).
        if ntiles % ncol == 0:
            relayout = w_bf16.view(1, 1, ntiles, 32)
            mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=mesh_shape)
        else:
            raise RuntimeError(
                f"TTNNDistributedRMSNorm: gamma (dim={dim}, ntiles={ntiles}) is incompatible with mesh {mesh_shape}: "
                f"need (dim//32) % mesh_width == 0. For small norms (e.g. talker code_predictor), keep HF RMSNorm on CPU."
            )

        self.weight_distributed = ttnn.as_tensor(
            relayout,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, inp):
        original_shape = inp.shape
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)  # Add batch dimension for RMSNorm
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)
        # AllGather stats
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=self.torch_layer.variance_epsilon,
            weight=self.weight_distributed,
        )
        tt_stats.deallocate(True)

        # Squeeze back to original shape if we added a batch dimension
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])

        return tt_out
