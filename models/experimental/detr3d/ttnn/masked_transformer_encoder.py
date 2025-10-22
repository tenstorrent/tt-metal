# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.multihead_attention import TtnnMultiheadAttention
from dataclasses import dataclass, asdict


@dataclass
class EncoderLayerArgs:
    d_model: int = None
    nhead: int = 4
    dim_feedforward: int = 128
    normalize_before: bool = True
    use_ffn: bool = True


class TtnnTransformerEncoderLayer(LightweightModule):
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

        # Initialize self-attention with parameters
        attn_params = parameters.get("self_attn") if parameters is not None else None
        self.self_attn = TtnnMultiheadAttention(d_model, nhead, device, parameters=attn_params)

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
            src = ttnn.to_memory_config(src, ttnn.L1_MEMORY_CONFIG)
            src2 = ttnn.linear(
                src, self.ff_weights1, bias=getattr(self, "ff1_bias", None), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            src2 = ttnn.relu(src2)
            src2 = ttnn.to_memory_config(src2, ttnn.L1_MEMORY_CONFIG)
            src2 = ttnn.linear(
                src2, self.ff_weights2, bias=getattr(self, "ff2_bias", None), memory_config=ttnn.L1_MEMORY_CONFIG
            )
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
            src = ttnn.to_memory_config(src, ttnn.L1_MEMORY_CONFIG)
            src2 = ttnn.layer_norm(src, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))
            src2 = ttnn.to_memory_config(src2, ttnn.L1_MEMORY_CONFIG)
            src2 = ttnn.linear(
                src2, self.ff_weights1, bias=getattr(self, "ff1_bias", None), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            src2 = ttnn.relu(src2)
            src2 = ttnn.to_memory_config(src2, ttnn.L1_MEMORY_CONFIG)
            src2 = ttnn.linear(
                src2, self.ff_weights2, bias=getattr(self, "ff2_bias", None), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            src = ttnn.add(src, src2)

        if return_attn_weights:
            return src, None
        return src


class TtnnMaskedTransformerEncoder(LightweightModule):
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

    def compute_mask_ttnn(self, xyz, radius, dist=None):
        """
        Compute attention mask and distance matrix using ttnn operations.

        Args:
            xyz_ttnn: ttnn tensor of shape (batch_size, seq_len, 3)
            radius: masking radius threshold
            dist: optional precomputed distance tensor

        Returns:
            mask_ttnn: boolean mask tensor where True means distance >= radius
            dist_ttnn: distance matrix
        """
        # Compute pairwise distances using ttnn operations
        # Using the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b

        tt_xyz = ttnn.from_torch(
            xyz,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if dist is None or dist.shape[1] != xyz.shape[1]:
            # Compute squared norms: ||a||^2 for each point
            # xyz_ttnn shape: (batch_size, seq_len, 3)
            xyz_squared = ttnn.pow(tt_xyz, 2)  # (batch_size, seq_len, 3)
            norms_squared = ttnn.sum(xyz_squared, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)

            # Compute dot products: xyz @ xyz^T
            xyz_transpose = ttnn.permute(tt_xyz, (0, 2, 1))  # (batch_size, 3, seq_len)
            dot_products = ttnn.matmul(tt_xyz, xyz_transpose)  # (batch_size, seq_len, seq_len)

            # Compute squared distances: ||a||^2 + ||b||^2 - 2*a·b
            # norms_squared: (batch_size, seq_len, 1)
            # norms_squared^T: (batch_size, 1, seq_len)
            norms_squared_t = ttnn.permute(norms_squared, (0, 2, 1))

            # dist^2 = ||a||^2 + ||b||^2 - 2*a·b
            dist_squared_ttnn = ttnn.add(norms_squared, norms_squared_t)  # broadcast to (batch_size, seq_len, seq_len)
            dist_squared_ttnn = ttnn.subtract(dist_squared_ttnn, ttnn.multiply(dot_products, 2.0))

            # Take sqrt to get actual distances
            # Add small epsilon to avoid numerical issues with sqrt(0)
            epsilon = 1e-8
            dist_squared_ttnn = ttnn.add(dist_squared_ttnn, epsilon)
            dist_ttnn = ttnn.sqrt(dist_squared_ttnn)
        else:
            dist_ttnn = dist

        ttnn.deallocate(tt_xyz)

        # Create mask: distance >= radius
        # Convert radius to ttnn scalar and compare
        radius_threshold = ttnn.full_like(dist_ttnn, radius)

        # Create mask where True means distance >= radius (should be masked)
        mask_ttnn = ttnn.ge(dist_ttnn, radius_threshold)  # boolean-like (0 or 1)
        mask_ttnn = mask_ttnn * float("-inf")

        return mask_ttnn, dist_ttnn

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
                attn_mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)  # Torch based mask gen
                # attn_mask, xyz_dist = self.compute_mask_ttnn(xyz, self.masking_radius[idx], xyz_dist) # FIXME: Minor pcc drop
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
