# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def get_DRAM_GN_config(module_path, idx):
    if "mid_block" in module_path:
        core_y = 4
        num_out_blocks = 4
    else:
        assert "Only mid_block is implemented for VAE"

    return core_y, num_out_blocks


def get_DRAM_conv_config(module_path, idx):
    if "mid_block" in module_path:
        slice_type = None
        num_slices = 1
    else:
        assert "Only mid_block is implemented for VAE"

    slice_config = (
        ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        )
        if num_slices > 1 and slice_type is not None
        else None
    )
    return slice_config
