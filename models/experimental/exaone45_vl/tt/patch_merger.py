# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen25_vl.tt.vision_rmsnorm import RMSNorm


class PatchMerger(LightweightModule):
    def __init__(self, mesh_device, args, state_dict, weight_cache_path, dtype):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args

        self.mlp_size = args.hf_config.vision_config.hidden_size * (
            args.hf_config.vision_config.spatial_merge_size**2
        )
        state_dict_prefix = args.get_state_dict_prefix(self.__class__.__name__)

        # Create the RMSNorm layer
        self.norm = RMSNorm(
            device=mesh_device,
            dim=args.hf_config.vision_config.hidden_size,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix + ".",
            weight_key="ln_q",
            weight_dtype=ttnn.bfloat16,
            eps=1e-6,  # Exaone4_5_PatchMerger hard-codes this
        )

        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        torch_bias = lambda name: self.state_dict[f"{state_dict_prefix}.{name}.bias"]

        # Create the MLP weights - note different dimensions for each layer
        as_weight_tensor = lambda name, type, in_dim, out_dim: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )
        as_bias_tensor = lambda name, type: ttnn.as_tensor(
            torch_bias(name).reshape(1, 1, 1, -1),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(f"{name}.bias"),
        )

        # First layer: hidden_size -> hidden_size
        self.w1 = as_weight_tensor("feed_forward.0", dtype, self.mlp_size, self.mlp_size)
        # Second layer: hidden_size -> out_dim
        self.w2 = as_weight_tensor("feed_forward.2", dtype, self.mlp_size, args.hf_config.vision_config.out_hidden_size)
        # The HF merger MLP linears carry biases (the qwen25_vl PatchMerger drops them;
        # a real accuracy bug for EXAONE's 5120-wide output).
        self.b1 = as_bias_tensor("feed_forward.0", dtype)
        self.b2 = as_bias_tensor("feed_forward.2", dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Apply RMSNorm
        x = self.norm(x)

        # Reshape to merge spatial dimensions
        # bug in ttnn.reshape tilized causing hangs https://github.com/tenstorrent/tt-metal/issues/29932
        # using workaround of converting to row major first
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], x.shape[1], -1, self.mlp_size))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # First linear + GELU
        x = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            activation="gelu",
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Second linear
        x = ttnn.linear(
            x,
            self.w2,
            bias=self.b2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return x
