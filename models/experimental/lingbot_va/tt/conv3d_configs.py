# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Conv3d blocking config overrides for the Lingbot-VA VAE decoder.

The decoder uses decoder_base_dim=256 with dim_mult=(1,2,4,4), producing channel
dimensions up to 1024. Each entry is tuned so that the per-core L1 usage fits
within the ~1.5 MB limit on Wormhole.

Usage: call ``override_conv3d_configs(module)`` after construction but **before**
weight loading (``load_torch_state_dict`` / ``load_model``).
"""

from __future__ import annotations

import ttnn
from loguru import logger

from models.tt_dit.models.vae.vae_wan2_1 import WanCausalConv3d

_LINGBOT_VA_BLOCKING = {
    # (in_channels, out_channels, kernel_size)
    #   -> (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)
    #
    # conv_in: z_dim=48 (padded to 64) -> 1024
    (64, 1024, (3, 3, 3)): (64, 128, 1, 8, 8),
    # mid_block and up_block resnets: 1024 -> 1024
    (1024, 1024, (3, 3, 3)): (128, 128, 1, 4, 4),
    # up_block cross-dim resnets
    (1024, 512, (3, 3, 3)): (128, 128, 1, 4, 4),
    (512, 512, (3, 3, 3)): (128, 128, 1, 8, 4),
    (512, 256, (3, 3, 3)): (128, 128, 1, 8, 4),
    (256, 256, (3, 3, 3)): (128, 128, 1, 16, 2),
    # conv_out: 256 -> 12 (padded to 32)
    (256, 32, (3, 3, 3)): (256, 32, 1, 2, 2),
    # WanResample upsampler convs (1x3x3 kernel)
    (1024, 1024, (1, 3, 3)): (128, 128, 1, 8, 4),
    (512, 512, (1, 3, 3)): (128, 128, 1, 16, 2),
    # WanResample temporal convs (3x1x1 kernel)
    (1024, 2048, (3, 1, 1)): (128, 256, 1, 8, 4),
}


def override_conv3d_configs(module) -> None:
    """Walk all WanCausalConv3d children and replace conv_config with tuned
    Lingbot-VA blocking where a match exists.

    Must be called after construction but before weight loading so that
    ``_prepare_torch_state`` reshapes weights with the correct ``C_in_block``.
    """
    for name, child in module.named_children():
        if isinstance(child, WanCausalConv3d):
            key = (child.in_channels, child.out_channels, child.kernel_size)
            blocking = _LINGBOT_VA_BLOCKING.get(key)
            if blocking is not None:
                C_in, C_out, T, H, W = blocking
                child.conv_config = ttnn.Conv3dConfig(
                    weights_dtype=ttnn.bfloat16,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                    C_in_block=C_in,
                    C_out_block=C_out,
                    T_out_block=T,
                    H_out_block=H,
                    W_out_block=W,
                    compute_with_storage_grid_size=child.mesh_device.compute_with_storage_grid_size(),
                )
                logger.debug(
                    "Overrode conv3d config for {} ({}): C_in={}, C_out={}, T={}, H={}, W={}",
                    name,
                    key,
                    C_in,
                    C_out,
                    T,
                    H,
                    W,
                )
        else:
            override_conv3d_configs(child)
