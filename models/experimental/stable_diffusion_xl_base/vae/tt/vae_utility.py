# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def get_DRAM_GN_shape(module_path, idx):
    if module_path is None:
        return (1, 128, 1024, 1024)
    elif "down_blocks.0" in module_path:
        return (1, 128, 1024, 1024)
    elif "down_blocks.1" in module_path:
        if idx == 1 and "resnets.0" in module_path:
            return (1, 128, 512, 512)
        else:
            return (1, 256, 512, 512)
    elif "down_blocks.2" in module_path:
        if idx == 1 and "resnets.0" in module_path:
            return (1, 256, 256, 256)
        else:
            return (1, 512, 256, 256)
    elif "up_blocks.1" in module_path:
        return (1, 512, 256, 256)
    elif "up_blocks.2" in module_path:
        if idx == 1 and "resnets.0" in module_path:
            return (1, 512, 512, 512)
        else:
            return (1, 256, 512, 512)
    elif "up_blocks.3" in module_path:
        if idx == 1 and "resnets.0" in module_path:
            return (1, 256, 1024, 1024)
        else:
            return (1, 128, 1024, 1024)
    else:
        return None


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
