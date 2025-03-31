# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.common.rmsnorm import RMSNorm


class PatchMerger(LightweightModule):
    def __init__(self, mesh_device, args, state_dict, weight_cache_path, dtype):
        super().__init__()

        # state_dict = torch.load("ref_merger_state_dict.pt")

        # state_dict = convert_hf_to_meta(state_dict, args.head_dim)
        # state_dict_prefix = args.get_state_dict_prefix("PatchMerger")
        # state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args

        self.mlp_size = args.hf_config.vision_config.hidden_size * (
            args.hf_config.vision_config.spatial_merge_size**2
        )
        state_dict_prefix = args.get_state_dict_prefix(self.__class__.__name__)

        # Create the RMSNorm layer
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.hf_config.vision_config.hidden_size,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix + ".",
                weight_key="ln_q",
                weight_dtype=ttnn.bfloat16,
                eps=1e-6,  # Qwen2_5_VLPatchMerger hard-codes this
                is_distributed=False,
            ),
            args,
            TG=args.is_galaxy,
        )

        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Create the MLP weights - note different dimensions for each layer
        as_sharded_tensor = lambda name, type, in_dim, out_dim: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-2, -1), mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        # First layer: hidden_size -> hidden_size
        self.w1 = as_sharded_tensor("feed_forward.0", dtype, self.mlp_size, self.mlp_size)
        # Second layer: hidden_size -> out_dim
        self.w2 = as_sharded_tensor(
            "feed_forward.2", dtype, self.mlp_size, args.hf_config.vision_config.out_hidden_size
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        def save_tensor(x, name):
            from os import getenv

            x_torch = ttnn.to_torch(
                x,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(1, 3), mesh_shape=self.args.cluster_shape
                ),
            )
            x_torch = torch.Tensor(x_torch[:, 0:1, :, : x.shape[-1]].squeeze())
            prefix = getenv("MPREFIX") or ""
            torch.save(x_torch, prefix + name)

        # Apply RMSNorm
        x = self.norm(x, mode="prefill")
        save_tensor(x, "1_post_norm.pt")

        # Reshape to merge spatial dimensions
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        save_tensor(x, "1a_untilized.pt")
        x = ttnn.reshape(x, (x.shape[0], x.shape[1], -1, self.mlp_size))
        save_tensor(x, "1b_reshaped.pt")
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        save_tensor(x, "2_tilized.pt")

        # Reshape to merge spatial dimensions
        # x = ttnn.to_torch(x)
        # x = torch.reshape(x, (x.shape[0], x.shape[1], -1, self.mlp_size))
        # x = ttnn.from_torch(
        #     x,
        #     device=self.mesh_device,
        #     mesh_mapper=ttnn.ShardTensor2dMesh(
        #         self.mesh_device,
        #         dims=(None, 3) if self.args.is_galaxy else (None, None),
        #         mesh_shape=self.args.cluster_shape,
        #     ),
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     layout=ttnn.TILE_LAYOUT,
        # )

        # First linear + GELU
        x = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        save_tensor(x, "3_post_linear.pt")
        x = ttnn.gelu(x)
        save_tensor(x, "4_post_gelu.pt")

        # Second linear
        x = ttnn.linear(
            x,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        save_tensor(x, "5_post_linear.pt")
        return x
