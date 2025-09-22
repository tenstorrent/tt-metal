# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def get_DRAM_GN_config(module_path, idx):
    core_y = 8
    core_x = 8
    if module_path is None:
        num_out_blocks = 32
    elif "encoder" == module_path:
        num_out_blocks = -1  # BS VAE group norm
    elif "mid_block" in module_path or "up_blocks.0" in module_path:
        num_out_blocks = -1  # BS VAE group norm
    elif "down_blocks.0" in module_path:
        num_out_blocks = 32
    elif "down_blocks.1" in module_path:
        if "resnets.0" in module_path and idx == 1:
            num_out_blocks = 8
        else:
            num_out_blocks = 12
    elif "down_blocks.2" in module_path:
        num_out_blocks = 4
    elif "down_blocks.3" in module_path:
        num_out_blocks = -1  # BS VAE group norm
    else:
        parts = module_path.split(".")
        block_id = int(parts[parts.index("up_blocks") + 1])
        resnet_id = int(parts[parts.index("resnets") + 1])

        if block_id == 1:
            num_out_blocks = 4
        elif block_id == 2:
            num_out_blocks = 12
        else:
            if idx == 1 and resnet_id == 0:
                num_out_blocks = 48
            else:
                num_out_blocks = 32

    return core_x, core_y, num_out_blocks


def get_DRAM_conv_config(module_path, idx):
    num_slices = 0
    is_encoder = True if module_path is not None and "encoder" == module_path else False
    if module_path is None:
        if idx == 1:
            return None
        else:
            num_slices = 8
    elif is_encoder:
        if idx == 1:
            num_slices = 8
        else:
            num_slices = 2
    elif "mid_block" in module_path:
        return None
    else:
        parts = module_path.split(".")
        if "down_blocks" in module_path:
            is_encoder = True
            block_id = int(parts[parts.index("down_blocks") + 1])
        else:
            is_encoder = False
            block_id = int(parts[parts.index("up_blocks") + 1])

        if block_id == 0:
            if "downsamplers" in module_path:
                num_slices = 8
            elif is_encoder:
                num_slices = 8
            elif "upsamplers" not in module_path:
                return None
            else:
                num_slices = 2
        if block_id == 1:
            if is_encoder:
                num_slices = 4
            elif "downsamplers" in module_path:
                num_slices = 4
            elif "upsamplers" not in module_path:
                num_slices = 2
            else:
                num_slices = 8
        if block_id == 2:
            if "downsamplers" in module_path:
                num_slices = 2
            elif is_encoder:
                num_slices = 2
            elif "upsamplers" not in module_path:
                if "resnets.0" in module_path and idx == 1:
                    num_slices = 8
                else:
                    num_slices = 4
            else:
                num_slices = 16
        if block_id == 3 and "upsamplers" not in module_path:
            if is_encoder:
                return None
            if "resnets.0" in module_path and idx == 1:
                num_slices = 16
            else:
                num_slices = 8

    slice_type = ttnn.Conv2dSliceWidth
    return ttnn.Conv2dSliceConfig(
        slice_type=slice_type,
        num_slices=num_slices,
    )
