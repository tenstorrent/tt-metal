from models.common.lightweightmodule import LightweightModule
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

        def as_tensor_data(tensor_data, dtype):
            return ttnn.as_tensor(
                tensor_data,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            as_tensor_data(merging_weights[:, inner_h, inner_w, :].contiguous(), dtype)
            for inner_h in range(self.spatial_merge_size)
            for inner_w in range(self.spatial_merge_size)
        ]
        self.merge_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=32,
            out_subblock_h=8,
            out_subblock_w=1,
            per_core_M=8,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        self.merge_large_m_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=8,
            per_core_N=4,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def forward(self, image_features: ttnn.Tensor, image_sizes) -> ttnn.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(ttnn.split(image_features, tokens_per_image, dim=0)):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            merged_h = h // self.spatial_merge_size
            merged_w = w // self.spatial_merge_size

            image_tokens = ttnn.to_layout(image_tokens, ttnn.ROW_MAJOR_LAYOUT)

            image_grid = ttnn.view(image_tokens, (h, w, d))
            grid = ttnn.reshape(
                image_grid,
                (merged_h, self.spatial_merge_size, merged_w, self.spatial_merge_size, d),
            )
            merged_patches = merged_h * merged_w
            merged_patch_tiles = (merged_patches + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
            merge_program_config = (
                self.merge_large_m_program_config
                if ((merged_patch_tiles + 7) // 8) * 32 > 110
                else self.merge_program_config
            )

            merged = None
            weight_index = 0
            for inner_h in range(self.spatial_merge_size):
                for inner_w in range(self.spatial_merge_size):
                    patch = ttnn.slice(
                        grid,
                        (0, inner_h, 0, inner_w, 0),
                        (merged_h, inner_h + 1, merged_w, inner_w + 1, d),
                    )
                    patch = ttnn.reshape(patch, (merged_h * merged_w, d))
                    patch = ttnn.to_layout(patch, ttnn.TILE_LAYOUT)
                    projected = ttnn.linear(
                        patch,
                        self.merging_weights[weight_index],
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=merge_program_config,
                    )
                    merged = projected if merged is None else ttnn.add(merged, projected)
                    weight_index += 1

            permuted_tensor.append(merged)

        image_features = ttnn.concat(permuted_tensor, dim=0)

        return image_features
