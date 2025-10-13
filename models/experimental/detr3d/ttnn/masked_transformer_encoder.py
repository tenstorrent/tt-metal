# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.multihead_attention import TTNNMultiheadAttention
from dataclasses import dataclass, asdict


@dataclass
class EncoderLayerArgs:
    d_model: int = None
    nhead: int = 4
    dim_feedforward: int = 128
    normalize_before: bool = True
    use_ffn: bool = True


class TTTransformerEncoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        d_model,
        nhead=4,
        dim_feedforward=128,
        normalize_before=True,
        use_ffn=True,
        parameters=None,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.normalize_before = normalize_before
        self.use_ffn = use_ffn

        self.self_attn = TTNNMultiheadAttention(d_model, nhead, device)

        # Load preprocessed parameters
        if parameters is not None:
            self.load_parameters(parameters)
        else:
            # Initialize weights as None (for backward compatibility)
            self.self_attn_weights = None
            self.ff_weights1 = None
            self.ff_weights2 = None
            self.norm1_weights = None
            self.norm2_weights = None

    def load_parameters(self, parameters):
        """Load preprocessed parameters from the preprocessor"""
        # Self-attention weights
        if "self_attn" in parameters:
            self.self_attn.q_weight = parameters["self_attn"].get("q_weight")
            self.self_attn.k_weight = parameters["self_attn"].get("k_weight")
            self.self_attn.v_weight = parameters["self_attn"].get("v_weight")
            self.self_attn.q_bias = parameters["self_attn"].get("q_bias")
            self.self_attn.k_bias = parameters["self_attn"].get("k_bias")
            self.self_attn.v_bias = parameters["self_attn"].get("v_bias")
            self.self_attn.out_weight = parameters["self_attn"].get("out_weight")
            self.self_attn.out_bias = parameters["self_attn"].get("out_bias")

        # Feedforward weights
        if "linear1" in parameters:
            self.ff_weights1 = parameters["linear1"]["weight"]
            self.ff1_bias = parameters["linear1"].get("bias")
        if "linear2" in parameters:
            self.ff_weights2 = parameters["linear2"]["weight"]
            self.ff2_bias = parameters["linear2"].get("bias")

        # Normalization weights
        for i, norm_name in enumerate(["norm1", "norm2"], 1):
            if norm_name in parameters:
                setattr(self, f"norm{i}_weights", parameters[norm_name]["weight"])
                setattr(self, f"norm{i}_bias", parameters[norm_name].get("bias"))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else ttnn.add(tensor, pos)

    def forward(
        self,
        src,
        src_mask=None,
        pos=None,
        return_attn_weights=False,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, pos, return_attn_weights)
        else:
            return self.forward_post(src, src_mask, pos, return_attn_weights)

    def forward_post(
        self,
        src,
        src_mask=None,
        pos=None,
        return_attn_weights=False,
    ):
        q = k = self.with_pos_embed(src, pos)
        value = src

        # Self-attention
        src2 = self.self_attn(q, k, value, src_mask)
        src = ttnn.add(src, src2)
        src = ttnn.layer_norm(src, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))

        # Feedforward network
        if self.use_ffn:
            src2 = ttnn.linear(src, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
            src2 = ttnn.relu(src2)
            src2 = ttnn.linear(src2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
            src = ttnn.add(src, src2)
            src = ttnn.layer_norm(src, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))

        if return_attn_weights:
            return src, None  # TTNN doesn't return attention weights
        return src

    def forward_pre(
        self,
        src,
        src_mask=None,
        pos=None,
        return_attn_weights=False,
    ):
        src = ttnn.to_layout(src, ttnn.TILE_LAYOUT)
        # Pre-norm self-attention
        src2 = ttnn.layer_norm(src, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value, src_mask)
        src = ttnn.add(src, src2)

        # Pre-norm feedforward
        if self.use_ffn:
            src2 = ttnn.layer_norm(src, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))
            src2 = ttnn.linear(src2, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
            src2 = ttnn.relu(src2)
            src2 = ttnn.linear(src2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
            src = ttnn.add(src, src2)

        if return_attn_weights:
            return src, None
        return src


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
            output = layer(output, src_mask=attn_mask, pos=pos)
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
