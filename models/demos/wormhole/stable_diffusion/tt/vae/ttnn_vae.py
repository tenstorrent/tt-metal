# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import (
    GROUPNORM_DECODER_NUM_BLOCKS,
    MIDBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
    MIDBLOCK_RESNET_NORM_NUM_BLOCKS,
    UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
    UPBLOCK_RESNET_NORM_NUM_BLOCKS,
    UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS,
)
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_decoder import ConvBlock, VaeDecoder


# This is a wrapper class for the VAE decoder that is the equivalent of the AutoencoderKL
# class in the original Stable Diffusion codebase, but used exclusively for the VAE decoder.
# It uses AutoencoderKL vae for its weights and biases, and the VaeDecoder class
# for the actual decoding process.
class Vae:
    def __init__(self, torch_vae, device):
        input_height = 64
        input_width = 64
        in_channels = 4
        out_channels = 3
        output_height = 512
        output_width = 512

        self.device = device

        self.decoder = VaeDecoder(
            torch_decoder=torch_vae.decoder,
            device=device,
            in_channels=in_channels,
            input_height=input_height,
            input_width=input_width,
            out_channels=out_channels,
            midblock_in_channels=512,
            output_height=output_height,
            output_width=output_width,
            midblock_norm_blocks=MIDBLOCK_RESNET_NORM_NUM_BLOCKS,
            midblock_conv_channel_split_factors=MIDBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
            upblock_out_channels=[512, 512, 256, 128],
            upblock_out_dimensions=[128, 256, 512, 512],
            upblock_norm_blocks=UPBLOCK_RESNET_NORM_NUM_BLOCKS,
            upblock_resnet_conv_channel_split_factors=UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
            upblock_upsample_conv_channel_split_factors=UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS,
            norm_num_blocks=GROUPNORM_DECODER_NUM_BLOCKS,
        )

        self.post_quant_conv = ConvBlock(
            torch_vae.post_quant_conv,
            device,
            in_channels=in_channels,
            out_channels=in_channels,
            input_height=input_width,
            input_width=input_width,
            padding=0,
            kernel_size=1,
        )

    def decode(self, x):
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x
