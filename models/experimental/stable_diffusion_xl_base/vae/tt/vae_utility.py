# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def get_DRAM_GN_config(module_path, idx):
    core_x = 8
    if module_path is None:
        core_x = 4
        core_y = 4
        num_out_blocks = 96
    elif "mid_block" in module_path or "up_blocks.0" in module_path:
        core_y = 4
        num_out_blocks = 4
    else:
        parts = module_path.split(".")
        block_id = int(parts[parts.index("up_blocks") + 1])
        resnet_id = int(parts[parts.index("resnets") + 1])

        if block_id == 1:
            core_y = 8
            num_out_blocks = 4
        elif block_id == 2:
            core_y = 8
            num_out_blocks = 12
        else:
            if idx == 1 and resnet_id == 0:
                core_y = 8
                num_out_blocks = 48
            else:
                core_x = 4
                core_y = 4
                num_out_blocks = 96

    return core_x, core_y, num_out_blocks


def get_DRAM_conv_config(module_path, idx):
    if module_path is None:
        if idx == 1:
            return None
    elif "mid_block" in module_path:
        return None
    else:
        parts = module_path.split(".")
        block_id = int(parts[parts.index("up_blocks") + 1])
        if "upsamplers" not in module_path and block_id == 0:
            return None
    slice_type = ttnn.Conv2dSliceWidth
    return ttnn.Conv2dSliceConfig(
        slice_type=slice_type,
    )
