# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_vl.tt.vision_layernorm import LayerNorm


class PatchMerger(LightweightModule):
    def __init__(
        self, mesh_device, args, state_dict, state_dict_prefix, weight_cache_path, dtype, postshuffle_norm=False
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.postshuffle_norm = postshuffle_norm
        self.mlp_size = args.hf_config.vision_config.hidden_size * (
            args.hf_config.vision_config.spatial_merge_size**2
        )
        state_dict_prefix = (
            args.get_state_dict_prefix(self.__class__.__name__) if state_dict_prefix is None else state_dict_prefix
        )

        # Create the LayerNorm layer
        self.norm = LayerNorm(
            device=mesh_device,
            dim=args.hf_config.vision_config.hidden_size if not postshuffle_norm else self.mlp_size,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix + ".norm",
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            eps=1e-6,  # Qwen3_VLPatchMerger hard-codes this
        )

        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        torch_bias = lambda name: self.state_dict[f"{state_dict_prefix}.{name}.bias"]

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Create the MLP weights - note different dimensions for each layer
        as_weight_tensor = lambda name, type, in_dim, out_dim: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(f"{name}.weight"),
        )
        as_bias_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_bias(name),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(f"{name}.bias"),
        )

        # First layer: hidden_size -> hidden_size
        self.w1 = as_weight_tensor("linear_fc1", dtype, self.mlp_size, self.mlp_size)
        # Second layer: hidden_size -> out_dim
        self.w2 = as_weight_tensor("linear_fc2", dtype, self.mlp_size, args.hf_config.vision_config.out_hidden_size)
        self.b1 = as_bias_tensor("linear_fc1", dtype, self.mlp_size)
        self.b2 = as_bias_tensor("linear_fc2", dtype, args.hf_config.vision_config.out_hidden_size)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Apply RMSNorm

        if self.postshuffle_norm:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (x.shape[0], x.shape[1], -1, self.mlp_size))
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
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
