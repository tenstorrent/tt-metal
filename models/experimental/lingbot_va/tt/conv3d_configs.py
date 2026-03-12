# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Conv3d blocking configs for the Lingbot-VA VAE decoder.

The decoder uses decoder_base_dim=256 with dim_mult=(1,2,4,4), producing channel
dimensions up to 1024. Each entry is tuned so that the per-core weight buffer
(C_in_block * C_out_block * kernel_volume * 2 bytes) fits within the ~1.5 MB L1
limit on Wormhole.
"""

from models.tt_dit.utils.conv3d import register_conv3d_blocking

LINGBOT_VA_CONV3D_BLOCKING = {
    # (in_channels, out_channels, kernel_size) -> (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)
    #
    # conv_in: z_dim=48 (padded to 64) → 1024
    (64, 1024, (3, 3, 3)): (64, 128, 1, 8, 8),
    # mid_block and up_block resnets: 1024 → 1024
    (1024, 1024, (3, 3, 3)): (128, 128, 1, 4, 4),
    # up_block cross-dim resnets
    (1024, 512, (3, 3, 3)): (128, 128, 1, 4, 4),
    (512, 512, (3, 3, 3)): (128, 128, 1, 8, 4),
    (512, 256, (3, 3, 3)): (128, 128, 1, 8, 4),
    (256, 256, (3, 3, 3)): (128, 128, 1, 16, 2),
    # conv_out: 256 → 12 (padded to 32)
    (256, 32, (3, 3, 3)): (256, 32, 1, 2, 2),
    # WanResample upsampler convs (1x3x3 kernel)
    (1024, 1024, (1, 3, 3)): (128, 128, 1, 8, 4),
    (512, 512, (1, 3, 3)): (128, 128, 1, 16, 2),
    # WanResample temporal convs (3x1x1 kernel)
    (1024, 2048, (3, 1, 1)): (128, 256, 1, 8, 4),
}


def register_lingbot_va_conv3d_configs():
    """Register Lingbot-VA blocking configs. Call before building the VAE decoder."""
    register_conv3d_blocking(LINGBOT_VA_CONV3D_BLOCKING)
