# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Weight loading for VAE decoder TTNN model."""

import torch

import ttnn
from models.demos.z_image_turbo.tt.vae.consteval import (
    make_ones_scalars,
    make_scalar,
    make_upsample_matrix,
    prepare_conv_bias,
    prepare_conv_weights,
    reshape_gn_weight,
    reshape_gn_weight_attn,
)

# fmt: off
WEIGHT_SPECS = {
    "conv_in.weight": ("conv_weight", (16, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "conv_in.bias": ("conv_bias", (512, 16, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "conv_out.weight": ("conv_weight", (128, 3, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 192)),
    "conv_out.bias": ("conv_bias", (3, 128, 3, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 192)),
    "conv_norm_out.weight": ("gn_weight", 128),
    "conv_norm_out.bias": ("gn_weight", 128),
    "mid_block.attentions.0.group_norm.weight": ("gn_weight_attn", 512),
    "mid_block.attentions.0.group_norm.bias": ("gn_weight_attn", 512),
    "mid_block.resnets.0.conv1.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.0.conv1.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.0.conv2.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.0.conv2.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.0.norm1.weight": ("gn_weight", 512),
    "mid_block.resnets.0.norm1.bias": ("gn_weight", 512),
    "mid_block.resnets.0.norm2.weight": ("gn_weight", 512),
    "mid_block.resnets.0.norm2.bias": ("gn_weight", 512),
    "mid_block.resnets.1.conv1.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.1.conv1.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.1.conv2.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.1.conv2.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "mid_block.resnets.1.norm1.weight": ("gn_weight", 512),
    "mid_block.resnets.1.norm1.bias": ("gn_weight", 512),
    "mid_block.resnets.1.norm2.weight": ("gn_weight", 512),
    "mid_block.resnets.1.norm2.bias": ("gn_weight", 512),
    "up_blocks.0.resnets.0.conv1.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.0.conv1.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.0.conv2.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.0.conv2.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.0.norm1.weight": ("gn_weight", 512),
    "up_blocks.0.resnets.0.norm1.bias": ("gn_weight", 512),
    "up_blocks.0.resnets.0.norm2.weight": ("gn_weight", 512),
    "up_blocks.0.resnets.0.norm2.bias": ("gn_weight", 512),
    "up_blocks.0.resnets.1.conv1.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.1.conv1.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.1.conv2.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.1.conv2.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.1.norm1.weight": ("gn_weight", 512),
    "up_blocks.0.resnets.1.norm1.bias": ("gn_weight", 512),
    "up_blocks.0.resnets.1.norm2.weight": ("gn_weight", 512),
    "up_blocks.0.resnets.1.norm2.bias": ("gn_weight", 512),
    "up_blocks.0.resnets.2.conv1.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.2.conv1.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.2.conv2.weight": ("conv_weight", (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.2.conv2.bias": ("conv_bias", (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0)),
    "up_blocks.0.resnets.2.norm1.weight": ("gn_weight", 512),
    "up_blocks.0.resnets.2.norm1.bias": ("gn_weight", 512),
    "up_blocks.0.resnets.2.norm2.weight": ("gn_weight", 512),
    "up_blocks.0.resnets.2.norm2.bias": ("gn_weight", 512),
    "up_blocks.0.upsamplers.0.conv.weight": ("conv_weight", (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.0.upsamplers.0.conv.bias": ("conv_bias", (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.0.conv1.weight": ("conv_weight", (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.0.conv1.bias": ("conv_bias", (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.0.conv2.weight": ("conv_weight", (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.0.conv2.bias": ("conv_bias", (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.0.norm1.weight": ("gn_weight", 512),
    "up_blocks.1.resnets.0.norm1.bias": ("gn_weight", 512),
    "up_blocks.1.resnets.0.norm2.weight": ("gn_weight", 512),
    "up_blocks.1.resnets.0.norm2.bias": ("gn_weight", 512),
    "up_blocks.1.resnets.1.conv1.weight": ("conv_weight", (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.1.conv1.bias": ("conv_bias", (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.1.conv2.weight": ("conv_weight", (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.1.conv2.bias": ("conv_bias", (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.1.norm1.weight": ("gn_weight", 512),
    "up_blocks.1.resnets.1.norm1.bias": ("gn_weight", 512),
    "up_blocks.1.resnets.1.norm2.weight": ("gn_weight", 512),
    "up_blocks.1.resnets.1.norm2.bias": ("gn_weight", 512),
    "up_blocks.1.resnets.2.conv1.weight": ("conv_weight", (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.2.conv1.bias": ("conv_bias", (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.2.conv2.weight": ("conv_weight", (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.2.conv2.bias": ("conv_bias", (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.resnets.2.norm1.weight": ("gn_weight", 512),
    "up_blocks.1.resnets.2.norm1.bias": ("gn_weight", 512),
    "up_blocks.1.resnets.2.norm2.weight": ("gn_weight", 512),
    "up_blocks.1.resnets.2.norm2.bias": ("gn_weight", 512),
    "up_blocks.1.upsamplers.0.conv.weight": ("conv_weight", (512, 512, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.1.upsamplers.0.conv.bias": ("conv_bias", (512, 512, 512, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.0.conv1.weight": ("conv_weight", (512, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.0.conv1.bias": ("conv_bias", (256, 512, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.0.conv2.weight": ("conv_weight", (256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.0.conv2.bias": ("conv_bias", (256, 256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.0.conv_shortcut.weight": ("conv_weight", (512, 256, 256, 256, [1, 1], [1, 1], [0, 0, 0, 0], 0)),
    "up_blocks.2.resnets.0.conv_shortcut.bias": ("conv_bias", (256, 512, 256, 256, 256, [1, 1], [1, 1], [0, 0, 0, 0], 0)),
    "up_blocks.2.resnets.0.norm1.weight": ("gn_weight", 512),
    "up_blocks.2.resnets.0.norm1.bias": ("gn_weight", 512),
    "up_blocks.2.resnets.0.norm2.weight": ("gn_weight", 256),
    "up_blocks.2.resnets.0.norm2.bias": ("gn_weight", 256),
    "up_blocks.2.resnets.1.conv1.weight": ("conv_weight", (256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.1.conv1.bias": ("conv_bias", (256, 256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.1.conv2.weight": ("conv_weight", (256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.1.conv2.bias": ("conv_bias", (256, 256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.1.norm1.weight": ("gn_weight", 256),
    "up_blocks.2.resnets.1.norm1.bias": ("gn_weight", 256),
    "up_blocks.2.resnets.1.norm2.weight": ("gn_weight", 256),
    "up_blocks.2.resnets.1.norm2.bias": ("gn_weight", 256),
    "up_blocks.2.resnets.2.conv1.weight": ("conv_weight", (256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.2.conv1.bias": ("conv_bias", (256, 256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.2.conv2.weight": ("conv_weight", (256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.2.conv2.bias": ("conv_bias", (256, 256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.resnets.2.norm1.weight": ("gn_weight", 256),
    "up_blocks.2.resnets.2.norm1.bias": ("gn_weight", 256),
    "up_blocks.2.resnets.2.norm2.weight": ("gn_weight", 256),
    "up_blocks.2.resnets.2.norm2.bias": ("gn_weight", 256),
    "up_blocks.2.upsamplers.0.conv.weight": ("conv_weight", (256, 256, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.2.upsamplers.0.conv.bias": ("conv_bias", (256, 256, 256, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.0.conv1.weight": ("conv_weight", (256, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.0.conv1.bias": ("conv_bias", (128, 256, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.0.conv2.weight": ("conv_weight", (128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.0.conv2.bias": ("conv_bias", (128, 128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.0.conv_shortcut.weight": ("conv_weight", (256, 128, 512, 512, [1, 1], [1, 1], [0, 0, 0, 0], 0)),
    "up_blocks.3.resnets.0.conv_shortcut.bias": ("conv_bias", (128, 256, 128, 512, 512, [1, 1], [1, 1], [0, 0, 0, 0], 0)),
    "up_blocks.3.resnets.0.norm1.weight": ("gn_weight", 256),
    "up_blocks.3.resnets.0.norm1.bias": ("gn_weight", 256),
    "up_blocks.3.resnets.0.norm2.weight": ("gn_weight", 128),
    "up_blocks.3.resnets.0.norm2.bias": ("gn_weight", 128),
    "up_blocks.3.resnets.1.conv1.weight": ("conv_weight", (128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.1.conv1.bias": ("conv_bias", (128, 128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.1.conv2.weight": ("conv_weight", (128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.1.conv2.bias": ("conv_bias", (128, 128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.1.norm1.weight": ("gn_weight", 128),
    "up_blocks.3.resnets.1.norm1.bias": ("gn_weight", 128),
    "up_blocks.3.resnets.1.norm2.weight": ("gn_weight", 128),
    "up_blocks.3.resnets.1.norm2.bias": ("gn_weight", 128),
    "up_blocks.3.resnets.2.conv1.weight": ("conv_weight", (128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.2.conv1.bias": ("conv_bias", (128, 128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.2.conv2.weight": ("conv_weight", (128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.2.conv2.bias": ("conv_bias", (128, 128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024)),
    "up_blocks.3.resnets.2.norm1.weight": ("gn_weight", 128),
    "up_blocks.3.resnets.2.norm1.bias": ("gn_weight", 128),
    "up_blocks.3.resnets.2.norm2.weight": ("gn_weight", 128),
    "up_blocks.3.resnets.2.norm2.bias": ("gn_weight", 128),
}
# fmt: on

ATTN_WEIGHTS = [
    "mid_block.attentions.0.to_q.weight",
    "mid_block.attentions.0.to_q.bias",
    "mid_block.attentions.0.to_k.weight",
    "mid_block.attentions.0.to_k.bias",
    "mid_block.attentions.0.to_v.weight",
    "mid_block.attentions.0.to_v.bias",
    "mid_block.attentions.0.to_out.0.weight",
    "mid_block.attentions.0.to_out.0.bias",
]


def load_weights(state_dict, device):
    """Load all VAE decoder weights and produce processed TTNN tensors."""
    weights = {}

    for name, (transform, cfg) in WEIGHT_SPECS.items():
        raw = ttnn.from_torch(
            state_dict[name].to(torch.bfloat16),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
        )
        if transform == "conv_weight":
            weights[name] = prepare_conv_weights(
                raw,
                device,
                in_channels=cfg[0],
                out_channels=cfg[1],
                batch_size=1,
                input_height=cfg[2],
                input_width=cfg[3],
                kernel_size=cfg[4],
                stride=cfg[5],
                padding=cfg[6],
                dilation=[1, 1],
                groups=1,
                act_block_h_override=cfg[7],
            )
        elif transform == "conv_bias":
            weights[name] = prepare_conv_bias(
                raw,
                device,
                channels=cfg[0],
                in_channels=cfg[1],
                out_channels=cfg[2],
                batch_size=1,
                input_height=cfg[3],
                input_width=cfg[4],
                kernel_size=cfg[5],
                stride=cfg[6],
                padding=cfg[7],
                dilation=[1, 1],
                groups=1,
                act_block_h_override=cfg[8],
            )
        elif transform == "gn_weight":
            weights[name] = reshape_gn_weight(raw, device, cfg)
        elif transform == "gn_weight_attn":
            weights[name] = reshape_gn_weight_attn(raw, device, cfg)

    for name in ATTN_WEIGHTS:
        weights[name] = ttnn.from_torch(
            state_dict[name].to(torch.bfloat16),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    weights["_gn_eps_large"] = make_scalar(device, 1.9073486328125e-06)
    ones_4d, ones_2d = make_ones_scalars(device)
    weights["_ones_4d"] = ones_4d
    weights["_ones_2d"] = ones_2d
    weights["_gn_eps"] = make_scalar(device, 9.9999999747524271e-07)
    weights["_gn_inv_256x512"] = make_scalar(device, 4.76837158203125e-07)
    weights["_gn_inv_512x128"] = make_scalar(device, 3.814697265625e-06)
    weights["_upsample_256_512"] = make_upsample_matrix(device, 256, 512)
    weights["_gn_inv_512x64"] = make_scalar(device, 1.52587890625e-05)
    weights["_upsample_128_256"] = make_upsample_matrix(device, 128, 256)
    weights["_gn_inv_512x256"] = make_scalar(device, 9.5367431640625e-07)
    weights["_upsample_64_128"] = make_upsample_matrix(device, 64, 128)

    return weights
