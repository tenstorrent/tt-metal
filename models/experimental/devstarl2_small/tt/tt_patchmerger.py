# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import vision_slice_memcfg
import ttnn
import torch


class TTMistral3PatchMerger(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path=None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.spatial_merge_size = 2
        self.patch_size = args.vision_patch_size
        self.args = args

        def get_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)

        def as_tensor_data(tensor_data, dtype, inner_h, inner_w):
            cache_name = None
            if weight_cache_path is not None:
                cache_name = weight_cache_path / f"{state_dict_prefix}merging_layer.weight.{inner_h}_{inner_w}.tile"
            return ttnn.as_tensor(
                tensor_data,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name,
            )

        merging_weights = get_weight("merging_layer")
        input_dim, output_dim = merging_weights.shape
        hidden_dim = input_dim // self.spatial_merge_size**2
        merging_weights = merging_weights.reshape(
            hidden_dim,
            self.spatial_merge_size,
            self.spatial_merge_size,
            output_dim,
        )
        self.merging_weights = [
            as_tensor_data(merging_weights[:, inner_h, inner_w, :].contiguous(), dtype, inner_h, inner_w)
            for inner_h in range(self.spatial_merge_size)
            for inner_w in range(self.spatial_merge_size)
        ]
        # mcast_in0 only valid when M fits one core block; else use auto matmul config.
        self._small_m_per_core_tiles = 8
        self.merge_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=32,
            out_subblock_h=8,
            out_subblock_w=1,
            per_core_M=self._small_m_per_core_tiles,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def _ensure_tile(self, tensor: ttnn.Tensor, mem_cfg: ttnn.MemoryConfig) -> ttnn.Tensor:
        if tensor.get_layout() != ttnn.TILE_LAYOUT:
            return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, memory_config=mem_cfg)
        if mem_cfg.buffer_type == ttnn.BufferType.L1 and tensor.memory_config().buffer_type != ttnn.BufferType.L1:
            return ttnn.to_memory_config(tensor, mem_cfg)
        return tensor

    def forward(self, image_features: ttnn.Tensor, image_sizes) -> ttnn.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(ttnn.split(image_features, tokens_per_image, dim=0)):
            h, w = image_sizes[image_index]
            merged_h = h // self.spatial_merge_size
            merged_w = w // self.spatial_merge_size

            slice_mem_cfg = vision_slice_memcfg(h * w)
            image_tokens = self._ensure_tile(image_tokens, slice_mem_cfg)

            grid = ttnn.reshape(
                image_tokens,
                (merged_h, self.spatial_merge_size, merged_w, self.spatial_merge_size, d),
            )
            if (
                slice_mem_cfg.buffer_type == ttnn.BufferType.L1
                and grid.memory_config().buffer_type != ttnn.BufferType.L1
            ):
                grid = ttnn.to_memory_config(grid, slice_mem_cfg)
            merged_patches = merged_h * merged_w
            merged_patch_tiles = (merged_patches + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
            use_small_config = merged_patch_tiles <= self._small_m_per_core_tiles
            merge_program_config = self.merge_program_config if use_small_config else None

            merged = None
            weight_index = 0
            for inner_h in range(self.spatial_merge_size):
                for inner_w in range(self.spatial_merge_size):
                    patch = ttnn.slice(
                        grid,
                        (0, inner_h, 0, inner_w, 0),
                        (merged_h, inner_h + 1, merged_w, inner_w + 1, d),
                        memory_config=slice_mem_cfg,
                    )
                    patch = ttnn.reshape(patch, (merged_h * merged_w, d))
                    linear_kwargs = dict(
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    if merge_program_config is not None:
                        linear_kwargs["program_config"] = merge_program_config
                    projected = ttnn.linear(
                        patch,
                        self.merging_weights[weight_index],
                        **linear_kwargs,
                    )
                    ttnn.deallocate(patch)
                    if merged is None:
                        merged = projected
                    else:
                        prev_merged = merged
                        merged = ttnn.add(merged, projected)
                        ttnn.deallocate(prev_merged)
                        ttnn.deallocate(projected)
                    weight_index += 1

            ttnn.deallocate(grid)
            permuted_tensor.append(merged)

        image_features = ttnn.concat(permuted_tensor, dim=0)

        return image_features


__all__ = ["TTMistral3PatchMerger"]
