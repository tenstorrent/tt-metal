# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional
from models.common.lightweightmodule import LightweightModule
from dataclasses import dataclass, asdict


@dataclass
class EncoderLayerArgs:
    d_model: int = None
    nhead: int = 4
    dim_feedforward: int = 128
    activation: str = "relu"
    normalize_before: bool = True
    norm_name: str = "ln"
    use_ffn: bool = True
    ffn_use_bias: bool = True


class TtMaskedTransformerEncoder(LightweightModule):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        masking_radius,
        interim_downsampling,
        norm=None,
        device=None,
        encoder_args=EncoderLayerArgs(),
        parameters=None,
    ):
        self.layers = []

        for i in range(num_layers):
            self.layers.append(
                encoder_layer(
                    device,
                    **asdict(encoder_args),
                    parameters=parameters.layers[i],
                )
            )

        self.num_layers = num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling
        self.norm = norm
        self.device = device

        assert len(masking_radius) == num_layers

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        mask_ttnn = torch.zeros_like(mask, dtype=torch.float).masked_fill_(mask, float("-inf"))
        mask_ttnn = ttnn.from_torch(mask_ttnn, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
        return mask_ttnn, dist

    def forward(
        self,
        src,
        mask: Optional[ttnn.Tensor] = None,
        src_key_padding_mask: Optional[ttnn.Tensor] = None,
        pos: Optional[ttnn.Tensor] = None,
        xyz: Optional[ttnn.Tensor] = None,
        transpose_swap: Optional[bool] = False,
    ):
        # Convert inputs to ttnn tensors if needed
        if not isinstance(src, ttnn.Tensor):
            src = ttnn.from_torch(src, device=self.device)

        if transpose_swap:
            bs, c, h, w = src.shape
            # Flatten and permute: (bs, c, h, w) -> (h*w, bs, c)
            src = ttnn.reshape(src, (bs, c, h * w))
            src = ttnn.transpose(src, 1, 2)  # (bs, h*w, c)
            src = ttnn.transpose(src, 0, 1)  # (h*w, bs, c)

            if pos is not None:
                if not isinstance(pos, ttnn.Tensor):
                    pos = ttnn.from_torch(pos, device=self.device)
                pos = ttnn.reshape(pos, (bs, c, h * w))
                pos = ttnn.transpose(pos, 1, 2)
                pos = ttnn.transpose(pos, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None

        for idx, layer in enumerate(self.layers):
            attn_mask = None
            if self.masking_radius[idx] > 0:
                attn_mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)
                attn_mask = ttnn.unsqueeze(attn_mask, 1)

            output = ttnn.permute(output, (1, 0, 2))
            output = layer(output, src_mask=attn_mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            output = ttnn.permute(output, (1, 0, 2))

            if idx == 0 and self.interim_downsampling:
                output = ttnn.permute(output, (1, 2, 0))
                xyz, output, xyz_inds = self.interim_downsampling(xyz, output)
                output = ttnn.permute(output, (2, 0, 1))

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            # Reshape back to original format
            output = ttnn.transpose(output, 0, 1)  # (bs, h*w, c)
            output = ttnn.transpose(output, 1, 2)  # (bs, c, h*w)
            output = ttnn.reshape(output, (bs, c, h, w))

        return xyz, output, xyz_inds
