import math
import ttnn
import torch
from models.common.lightweightmodule import LightweightModule
from models.experimental.mochi.tt.common import (
    to_tt_tensor,
)
from loguru import logger


class GroupNorm(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        state_dict_prefix: str,
        num_groups: int,
        channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        self.mesh_device = mesh_device
        self.channels = channels
        self.num_groups = num_groups
        self.affine = affine
        self.eps = eps

        # Get grid size from mesh device
        self.grid_size = mesh_device.compute_with_storage_grid_size()
        self.grid_size_y = 4 if channels == 128 else 8
        assert self.grid_size_y <= self.grid_size.y

        # Initialize weights and bias
        self.weight = state_dict[f"{state_dict_prefix}weight"]
        self.bias = state_dict[f"{state_dict_prefix}bias"]

        # Create input mask
        self.input_mask = ttnn.create_group_norm_input_mask(channels, num_groups, self.grid_size_y)
        self.input_mask = ttnn.get_device_tensors(
            ttnn.from_torch(
                self.input_mask,
                dtype=ttnn.DataType.BFLOAT8_B,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        )

        # Prepare gamma and beta
        self.gamma = ttnn.create_group_norm_weight_bias_rm(self.weight, channels, self.grid_size_y)
        self.beta = ttnn.create_group_norm_weight_bias_rm(self.bias, channels, self.grid_size_y)

        self.gamma_t = ttnn.get_device_tensors(
            ttnn.from_torch(
                self.gamma,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        )

        self.beta_t = ttnn.get_device_tensors(
            ttnn.from_torch(
                self.beta,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        )

    def forward(self, x_NTHWC):
        # Reshape to format expected by group_norm [N, 1, H*W, C]
        # print(f"x_NTHWC: {x_NTHWC.shape}")
        # x_reshaped = ttnn.reshape(x_NTHWC, [N * T, 1, H * W, C])
        # print(f"x_reshaped: {x_reshaped.shape}")

        # # Tilize the input tensor
        # x_tilized = ttnn.tilize_with_zero_padding(x_reshaped, use_multicore=True)
        # print(f"x_tilized: {x_tilized.shape}")
        # Determine proper num_out_blocks based on input dimensions
        # This is a simplified heuristic - adjust based on your model's specific needs

        tensors = ttnn.get_device_tensors(x_NTHWC)
        outputs = []
        for i, tensor in enumerate(tensors):
            N, T, H, W, C = tensor.shape
            tensor = ttnn.reshape(tensor, [N * T, 1, H * W, C])
            tensor = ttnn.tilize_with_zero_padding(tensor, use_multicore=True)
            HW = tensor.shape[2]
            T = tensor.shape[0]
            num_out_blocks_map = {
                60 * 106: 8,
                120 * 212: 10,
                240 * 424: 40,
                480 * 848: 135,
            }
            num_out_blocks = num_out_blocks_map[HW] if HW in num_out_blocks_map else math.ceil(HW / 2000)

            grid_size_x = min(T, self.grid_size.x)
            # Apply group_norm
            core_grid = ttnn.CoreGrid(y=self.grid_size_y, x=grid_size_x)
            output = ttnn.group_norm(
                tensor,
                num_groups=self.num_groups,
                epsilon=self.eps,
                input_mask=self.input_mask[i],
                weight=self.gamma_t[i],
                bias=self.beta_t[i],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_layout=ttnn.TILE_LAYOUT,
                core_grid=core_grid,
                inplace=False,
                num_out_blocks=num_out_blocks,
            )
            output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
            output = ttnn.reshape(output, [N, T, H, W, C])
            outputs.append(output)

        output_tiled_NTHWC = ttnn.aggregate_as_tensor(outputs)
        return output_tiled_NTHWC
