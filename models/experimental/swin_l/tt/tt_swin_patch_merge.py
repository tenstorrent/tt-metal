# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Patch Merging (downsampling) for Swin-L backbone.
Adapted from models/experimental/swin_s/tt/tt_patchmerging.py.
Initial version: generic (no hardcoded sharding configs).
"""

import ttnn


class TtSwinPatchMerge:
    """Patch merging: 2x2 spatial downsample -> concat -> LN -> linear (4C -> 2C)."""

    def __init__(self, device, parameters, dim):
        self.device = device
        self.parameters = parameters
        self.dim = dim

    def __call__(self, input_tensor):
        B, H, W, C = input_tensor.shape
        pad_h = H & 1
        pad_w = W & 1
        if pad_h | pad_w:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], value=0.0)
        x0 = input_tensor[..., 0::2, 0::2, :]
        x1 = input_tensor[..., 1::2, 0::2, :]
        x2 = input_tensor[..., 0::2, 1::2, :]
        x3 = input_tensor[..., 1::2, 1::2, :]
        ttnn.deallocate(input_tensor)
        output = ttnn.concat([x0, x1, x2, x3], -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x0)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)

        output = ttnn.layer_norm(
            ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            weight=self.parameters["norm"]["weight"],
            bias=self.parameters["norm"]["bias"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(
            output,
            self.parameters["reduction"]["weight"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
