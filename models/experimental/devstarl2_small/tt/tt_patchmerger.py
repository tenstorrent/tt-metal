from models.common.lightweightmodule import LightweightModule
import ttnn
import torch
from ttnn import ConcatMeshToTensor


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

        def get_bias(name):
            return state_dict[f"{state_dict_prefix}{name}.bias"]

        def as_tensor(name, dtype, is_bias=False):
            tensor_data = get_bias(name) if is_bias else get_weight(name)
            return ttnn.as_tensor(
                tensor_data,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.merging_weights = as_tensor("merging_layer", dtype)
        self.merging_bias = as_tensor("merging_layer", ttnn.bfloat16, is_bias=False)

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

            image_tokens = ttnn.to_layout(image_tokens, ttnn.ROW_MAJOR_LAYOUT)

            image_grid = ttnn.view(image_tokens, (h, w, d))
            # Permute the grid to have channels last
            image_grid = ttnn.permute(image_grid, (2, 0, 1))  # Channels first
            image_grid = ttnn.unsqueeze(image_grid, dim=0)  # Add batch dimension
            # Reshape the grid to merge patches
            if self.args.num_devices > 1:
                image_grid_torch = ttnn.to_torch(image_grid, mesh_composer=ConcatMeshToTensor(self.device, dim=0))
                image_grid_torch = image_grid_torch[0].unsqueeze(0)  # shape: [1, 1024, 30, 44]
                image_grid_torch = image_grid_torch.to(dtype=torch.bfloat16)
            else:
                image_grid_torch = ttnn.to_torch(image_grid).to(dtype=torch.bfloat16)

            grid = torch.nn.functional.unfold(
                image_grid_torch, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size
            )

            grid = ttnn.from_torch(grid, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            grid = ttnn.view(grid, (d * self.spatial_merge_size**2, -1))
            grid = ttnn.transpose(grid, 0, 1)  # Transpose to have features first

            permuted_tensor.append(grid)

        image_features = ttnn.concat(permuted_tensor, dim=0)
        # Apply merging layer
        image_features = ttnn.linear(
            image_features, self.merging_weights, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return image_features
