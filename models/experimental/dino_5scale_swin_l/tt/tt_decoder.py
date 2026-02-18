# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the DINO-5scale DinoTransformerDecoder.

Architecture per layer (x6):
  Self-Attention (MHA) -> LN -> Cross-Attention (MSDeformAttn) -> LN -> FFN -> LN

DINO-specific:
  - ref_point_head MLP converts reference points -> query_pos
  - coordinate_to_encoding generates sinusoidal pos from reference points
  - Iterative box refinement via reg_branches
  - Final layer norm on outputs
"""

import math
import torch
import ttnn
from loguru import logger

from models.experimental.dino_5scale_swin_l.tt.tt_encoder import TtMSDeformAttn, TtFFN


def coordinate_to_encoding_torch(coord_tensor, num_feats=128, temperature=10000):
    """
    Convert coordinate tensor to positional encoding (on CPU).
    coord_tensor: [B, num_queries, 4] with (cx, cy, w, h) in [0,1].
    Returns: [B, num_queries, num_feats * 2 * coord_dim] torch tensor.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(2)

    if coord_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(2)
        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(2)
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError(f"Unsupported coord_tensor last dim: {coord_tensor.size(-1)}")


def inverse_sigmoid_torch(x, eps=1e-3):
    """Inverse sigmoid on torch tensor."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class TtMultiheadAttention:
    """
    Standard multi-head attention for decoder self-attention.
    Follows UniAD's pattern: all 3D tensors, no 4D reshape/permute.
    """

    def __init__(self, params, device, embed_dims=256, num_heads=8):
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.device = device

        # Pre-split QKV weights and transpose for ttnn.linear: [in, out]
        in_proj_w = ttnn.to_layout(params["in_proj_weight"], ttnn.TILE_LAYOUT)
        in_proj_b = ttnn.squeeze(ttnn.to_layout(params["in_proj_bias"], ttnn.TILE_LAYOUT), 0)
        E = embed_dims
        self.q_w = ttnn.permute(in_proj_w[:E, :], (1, 0))
        self.k_w = ttnn.permute(in_proj_w[E : 2 * E, :], (1, 0))
        self.v_w = ttnn.permute(in_proj_w[2 * E :, :], (1, 0))
        self.q_b = in_proj_b[:E]
        self.k_b = in_proj_b[E : 2 * E]
        self.v_b = in_proj_b[2 * E :]

        self.out_proj_weight = params["out_proj"]["weight"]
        self.out_proj_bias = params["out_proj"]["bias"]

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
            query_pos = ttnn.to_layout(query_pos, ttnn.TILE_LAYOUT)
            query = query + query_pos
        if key_pos is not None:
            key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
            key_pos = ttnn.to_layout(key_pos, ttnn.TILE_LAYOUT)
            key = key + key_pos

        bs, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        q = ttnn.linear(query, self.q_w, bias=self.q_b)
        k = ttnn.linear(key, self.k_w, bias=self.k_b)
        v = ttnn.linear(value, self.v_w, bias=self.v_b)

        # 3D head reshape (UniAD pattern): [bs, seq, E] → [seq, bs*heads, head_dim] → [bs*heads, seq, head_dim]
        q = ttnn.reshape(q, (tgt_len, bs * self.num_heads, self.head_dim))
        q = ttnn.permute(q, (1, 0, 2))
        k = ttnn.reshape(k, (src_len, bs * self.num_heads, self.head_dim))
        k = ttnn.permute(k, (1, 0, 2))
        v = ttnn.reshape(v, (src_len, bs * self.num_heads, self.head_dim))
        v = ttnn.permute(v, (1, 0, 2))

        # Scaled dot-product attention — all 3D
        q_scaled = q * math.sqrt(1.0 / float(self.head_dim))
        k_t = ttnn.permute(k, (0, 2, 1))
        attn_weights = ttnn.matmul(q_scaled, k_t)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = ttnn.softmax(attn_weights, dim=-1)
        attn_out = ttnn.matmul(attn_weights, v)

        # Merge heads: [bs*heads, seq, head_dim] → [seq, bs*heads, head_dim] → [bs*seq, E] → [bs, seq, E]
        attn_out = ttnn.permute(attn_out, (1, 0, 2))
        attn_out = ttnn.reshape(attn_out, (tgt_len * bs, embed_dim))
        attn_out = ttnn.linear(attn_out, self.out_proj_weight, bias=self.out_proj_bias)
        attn_out = ttnn.reshape(attn_out, (bs, tgt_len, embed_dim))

        identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)
        return attn_out + identity


class TtRefPointHeadMLP:
    """
    ref_point_head: MLP(embed_dims*2, embed_dims, embed_dims, 2)
    = Linear(512, 256) -> ReLU -> Linear(256, 256)
    """

    def __init__(self, params, device):
        self.device = device
        self.w0 = params["layers"][0]["weight"]
        self.b0 = params["layers"][0]["bias"]
        self.w1 = params["layers"][1]["weight"]
        self.b1 = params["layers"][1]["bias"]

    def __call__(self, x):
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.linear(x, self.w0, bias=self.b0)
        x = ttnn.relu(x)
        x = ttnn.linear(x, self.w1, bias=self.b1)
        return x


class TtRegBranch:
    """
    Regression branch: 3 linear layers with ReLU between the first two.
    Linear(256, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 4)
    """

    def __init__(self, params, device):
        self.device = device
        self.layers = params

    def __call__(self, x):
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        for i, layer_params in enumerate(self.layers):
            x = ttnn.linear(x, layer_params["weight"], bias=layer_params["bias"])
            if i < len(self.layers) - 1:
                x = ttnn.relu(x)
        return x


class TtDINODecoderLayer:
    """
    Single DINO decoder layer:
    Self-Attention (MHA) -> LN -> Cross-Attention (MSDeformAttn) -> LN -> FFN -> LN
    """

    def __init__(self, params, device, embed_dims=256, num_heads=8, num_levels=5, num_points=4):
        self.self_attn = TtMultiheadAttention(
            params["self_attn"],
            device,
            embed_dims=embed_dims,
            num_heads=num_heads,
        )
        self.cross_attn = TtMSDeformAttn(
            params["cross_attn"],
            device,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        self.ffn = TtFFN(params["ffn"], device)
        self.norm1_w = params["norms"][0]["weight"]
        self.norm1_b = params["norms"][0]["bias"]
        self.norm2_w = params["norms"][1]["weight"]
        self.norm2_b = params["norms"][1]["bias"]
        self.norm3_w = params["norms"][2]["weight"]
        self.norm3_b = params["norms"][2]["bias"]

    def __call__(
        self,
        query,
        query_pos,
        value,
        key_padding_mask,
        self_attn_mask,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        reference_points,
        **kwargs,
    ):
        logger.info("  DecoderLayer: self-attention...")
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
        )
        query = ttnn.layer_norm(query, weight=self.norm1_w, bias=self.norm1_b)

        logger.info("  DecoderLayer: cross-attention...")
        query = self.cross_attn(
            query=query,
            value=value,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = ttnn.layer_norm(query, weight=self.norm2_w, bias=self.norm2_b)

        query = self.ffn(query)
        query = ttnn.layer_norm(query, weight=self.norm3_w, bias=self.norm3_b)

        return query


class TtDINODecoder:
    """
    DINO DinoTransformerDecoder (6 layers).

    Takes encoder memory + initial queries/reference points,
    returns intermediate hidden states and reference points for each layer.
    """

    def __init__(self, params, device, num_layers=6, embed_dims=256, num_heads=8, num_levels=5, num_points=4):
        self.device = device
        self.num_layers = num_layers
        self.embed_dims = embed_dims

        self.layers = [
            TtDINODecoderLayer(
                params["layers"][i],
                device,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
            )
            for i in range(num_layers)
        ]

        self.ref_point_head = TtRefPointHeadMLP(params["ref_point_head"], device)
        self.norm_w = params["norm"]["weight"]
        self.norm_b = params["norm"]["bias"]

        self.reg_branches = None
        if "reg_branches" in params:
            self.reg_branches = [TtRegBranch(params["reg_branches"][i], device) for i in range(num_layers)]

    def __call__(
        self,
        query,
        value,
        key_padding_mask,
        self_attn_mask,
        reference_points,
        spatial_shapes,
        level_start_index,
        valid_ratios,
    ):
        """
        Args:
            query: [B, num_queries, 256] ttnn
            value: [B, N, 256] encoder memory ttnn
            key_padding_mask: [B, N] or None
            self_attn_mask: [num_queries, num_queries] or None
            reference_points: torch.Tensor [B, num_queries, 4] (cx, cy, w, h) in [0,1]
            spatial_shapes: torch.Tensor [num_levels, 2]
            level_start_index: torch.Tensor [num_levels]
            valid_ratios: torch.Tensor [B, num_levels, 2]

        Returns:
            intermediate: list of [B, num_queries, 256] (one per layer, after norm)
            intermediate_reference_points: list of [B, num_queries, 4] (initial + per layer)
        """
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]

        for lid, layer in enumerate(self.layers):
            logger.info(f"Decoder layer {lid} starting...")
            ref_pts = reference_points
            if ref_pts.shape[-1] == 4:
                valid_ratios_4d = torch.cat([valid_ratios, valid_ratios], -1)
                reference_points_input = ref_pts[:, :, None] * valid_ratios_4d[:, None]
            else:
                reference_points_input = ref_pts[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding_torch(
                reference_points_input[:, :, 0, :],
                num_feats=self.embed_dims // 2,
                temperature=10000,
            )
            query_sine_embed_tt = ttnn.from_torch(
                query_sine_embed, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            query_pos = self.ref_point_head(query_sine_embed_tt)

            self_attn_mask_tt = None
            if self_attn_mask is not None:
                self_attn_mask_tt = ttnn.from_torch(
                    self_attn_mask.unsqueeze(0).unsqueeze(0).float().masked_fill(self_attn_mask, -1e9),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )

            output = layer(
                query=output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask_tt,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
            )

            if self.reg_branches is not None:
                tmp = self.reg_branches[lid](output)
                tmp_torch = ttnn.to_torch(tmp).float()
                assert ref_pts.shape[-1] == 4
                new_ref = tmp_torch + inverse_sigmoid_torch(ref_pts, eps=1e-3)
                new_ref = new_ref.sigmoid()
                reference_points = new_ref.detach()

            normed = ttnn.layer_norm(output, weight=self.norm_w, bias=self.norm_b)
            intermediate.append(normed)
            intermediate_reference_points.append(reference_points)
            logger.info(f"Decoder layer {lid} done.")

        return intermediate, intermediate_reference_points
