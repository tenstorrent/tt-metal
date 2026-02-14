# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch

import ttnn


def find_closest_largest_divisor(num: int, start_divisor: int):
    """Return the largest divisor of num that is <= start_divisor.

    Used to choose a core count that divides a work quota. Assumes
    1 <= start_divisor <= num. Decrements until a divisor is found.
    """
    divisor = start_divisor
    while num % divisor != 0:
        divisor = divisor - 1
    return divisor


# group norm helper function
def determine_expected_group_norm_sharded_config_and_grid_size(
    *, device, num_channels, num_groups, input_nhw, is_height_sharded, is_row_major=False
):
    """Derive sharded memory config and grid for group norm.

    - num_channels must be divisible by num_groups and 32 (tile width).
    - input_nhw is N*H*W in logical units; padded to core multiples.
    - If is_height_sharded: shard along NHW only; channels per core is all C.
      Otherwise: shard across channels and NHW (BLOCK_SHARDED).
    - is_row_major toggles shard shape orientation.

    Returns: (MemoryConfig, CoreGrid)
    """
    assert num_channels % num_groups == 0
    assert num_channels % 32 == 0  # TODO: remove this later
    group_size = num_channels // num_groups
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = [compute_with_storage_grid_size.x, compute_with_storage_grid_size.y]
    if is_row_major:
        device_grid_size = [compute_with_storage_grid_size.y, compute_with_storage_grid_size.x]

    max_num_cores = device_grid_size[0] * device_grid_size[1]
    input_nhw_paddedto32 = math.ceil(input_nhw / 32) * 32
    num_cores_nhw = find_closest_largest_divisor(
        input_nhw_paddedto32 // 32, max_num_cores if is_height_sharded else device_grid_size[0]
    )
    if is_height_sharded:
        num_cores_channels = 1
    else:
        num_cores_channels = device_grid_size[1]
        # num_channels_tiles = num_channels // 16
        num_channels_tiles = num_channels // 8
        while (num_channels_tiles % num_cores_channels != 0) or (
            ((num_channels // num_cores_channels) % group_size) != 0
        ):
            num_cores_channels -= 1
            assert num_cores_channels > 0
    input_nhw_padded_to_ncores = math.ceil(input_nhw / (num_cores_nhw * 32)) * (num_cores_nhw * 32)
    gn_in_channels_per_core = num_channels // num_cores_channels
    # multiply gn_in_channels_per_core by 2 until it is multiple of 32
    while gn_in_channels_per_core % 32 != 0:
        gn_in_channels_per_core += gn_in_channels_per_core
        num_cores_channels = num_cores_channels // 2
    # assert gn_in_channels_per_core % 16 == 0
    assert gn_in_channels_per_core % 8 == 0
    gn_nhw_per_core = input_nhw_padded_to_ncores // num_cores_nhw
    if is_height_sharded:
        grid_size = [
            device_grid_size[0] if num_cores_nhw >= device_grid_size[0] else num_cores_nhw,
            math.ceil(num_cores_nhw / device_grid_size[0]),
        ]  # for 1d systolic array, grid size is the tightest bound of num_cores_nhw as a rectangle (x,y)
        assert (
            num_cores_nhw <= grid_size[0] * grid_size[1]
        ), "Error: For height sharding, num_cores_nhw must be <= grid size"
    else:
        grid_size = [num_cores_channels, num_cores_nhw] if is_row_major else [num_cores_nhw, num_cores_channels]
    shard_shape = (
        (1, 1, gn_nhw_per_core, gn_in_channels_per_core)
        if is_row_major
        else (1, 1, gn_in_channels_per_core, gn_nhw_per_core)
    )
    shard_strategy = ttnn.ShardStrategy.HEIGHT if is_height_sharded else ttnn.ShardStrategy.BLOCK
    shard_orientation = (
        ttnn.ShardOrientation.ROW_MAJOR if is_height_sharded or is_row_major else ttnn.ShardOrientation.COL_MAJOR
    )
    return ttnn.create_sharded_memory_config_(
        shard_shape,
        ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        shard_strategy,
        shard_orientation,
        use_height_and_width_as_shard_shape=True,
    ), ttnn.CoreGrid(y=grid_size[1], x=grid_size[0])


def is_height_sharded_gn_from_dims(N, D, H, W, C):
    physical_height_to_width_ratio = (N * D * H * W) / C
    threshold = 4

    return physical_height_to_width_ratio >= threshold


class GroupNorm3D:
    def __init__(self, device, num_channels: int, num_groups: int):
        self.num_channels = num_channels
        self.channels_padding = (32 - (num_channels % 32)) % 32
        assert (
            self.num_channels + self.channels_padding
        ) % self.num_channels == 0, "padded channels must be divisible by channels"
        assert self.num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        # assert self.num_channels % 32 == 0, "num_channels must be divisible by 32 for height sharded group norm"
        self.num_groups = num_groups
        self.num_groups_padding = self.num_groups * (self.channels_padding // self.num_channels)

        self.grid_size = device.compute_with_storage_grid_size()
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.input_mask_tensor_dict = {
            i: ttnn.to_device(
                ttnn.create_group_norm_input_mask(
                    num_channel=self.num_channels + self.channels_padding,
                    num_groups=self.num_groups + self.num_groups_padding,
                    num_cores_across_channel=i,
                    data_type=ttnn.bfloat8_b,
                ),
                device,
            )
            for i in [1, 2, 4, 8]
        }

    def load_state_dict(self, device, params_dict: dict[str, torch.Tensor], module_prefix: Optional[str] = None):
        weight_str = f"{module_prefix}.weight" if module_prefix else "weight"
        bias_str = f"{module_prefix}.bias" if module_prefix else "bias"
        weight_tensor_padded = torch.nn.functional.pad(
            params_dict[weight_str], (0, self.channels_padding), "constant", 0
        )
        bias_tensor_padded = torch.nn.functional.pad(params_dict[bias_str], (0, self.channels_padding), "constant", 0)

        weight = ttnn.create_group_norm_weight_bias_rm(
            input_tensor=weight_tensor_padded, num_channels=self.num_channels + self.channels_padding, num_cores_x=1
        )
        bias = ttnn.create_group_norm_weight_bias_rm(
            input_tensor=bias_tensor_padded, num_channels=self.num_channels + self.channels_padding, num_cores_x=1
        )

        self.weight = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.from_torch(
            bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # TODO: Negative mask seems buggy, disable for now
        # self.input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

        # self.negative_mask = ttnn.create_group_norm_input_negative_mask(
        #     num_channel=self.num_channels + self.channels_padding,
        #     num_groups=self.num_groups + self.num_groups_padding,
        #     num_cores_across_channel=1,  # As explained in the Limitations, supply 1 for height sharded input tensors
        #     data_type=ttnn.bfloat8_b,
        # )
        # self.negative_mask = ttnn.to_device(self.negative_mask, device)

    def __call__(self, x: ttnn.Tensor, device) -> ttnn.Tensor:
        N, D, H, W, C = x.shape
        is_height_sharded = is_height_sharded_gn_from_dims(N, D, H, W, C)
        sharded_mem_config, grid_size = determine_expected_group_norm_sharded_config_and_grid_size(
            device=device,
            num_channels=self.num_channels + self.channels_padding,
            num_groups=self.num_groups + self.num_groups_padding,
            input_nhw=N * H * W * D,
            is_height_sharded=is_height_sharded,
            is_row_major=True,
        )
        assert x.layout == ttnn.ROW_MAJOR_LAYOUT, "Input tensor must be in ROW_MAJOR_LAYOUT"
        x0 = ttnn.reshape(x, (N, 1, D * H * W, C))
        x1 = ttnn.to_memory_config(x0, sharded_mem_config)
        if x1.buffer_address() != x0.buffer_address():
            ttnn.deallocate(x0)

        x2 = ttnn.group_norm(
            x1,
            input_mask=self.input_mask_tensor_dict[1 if is_height_sharded else grid_size.x],
            num_groups=self.num_groups + self.num_groups_padding,
            weight=self.weight,
            bias=self.bias,
            epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=grid_size,
            memory_config=sharded_mem_config,
            inplace=True,
        )
        x3 = ttnn.sharded_to_interleaved(x2, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x2)
        return ttnn.reshape(x3, (N, D, H, W, C))
