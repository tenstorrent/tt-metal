# ------------------------------------------------------------------------
# RF-DETR-base faithful CPU reference implementation (PyTorch).
#
# Numerically reproduces the official transformers RfDetrForObjectDetection
# while keeping module/parameter names IDENTICAL to the published
# Roboflow/rf-detr-base safetensors checkpoint (487 tensors), so that
# load_state_dict(strict=True) succeeds with zero missing/unexpected keys.
#
# Math adapted verbatim from:
#   - transformers (main) models/rf_detr/modeling_rf_detr.py  (oracle)
#   - rfdetr (develop) models/{backbone/dinov2_with_windowed_attn,transformer,
#       backbone/projector,lwdetr,ops}.py  (naming + lite-refine/bbox-reparam)
#   - transformers Dinov2 (per-layer block internals)
#
# Key architectural facts (RF-DETR-base):
#   backbone: windowed DINOv2-S/14, hidden 384, 12 layers, 6 heads,
#             num_windows=4 (=> 16 windows of 10x10 patches at 560/14=40 grid),
#             window_block_indexes=[0,1,3,4,6,7,9,10], global/out blocks at
#             [2,5,8,11]; 4 feature maps [B,384,40,40].
#   projector: concat 4 maps (1536ch) -> C2f -> channels-first LayerNorm
#             -> single level [B,256,40,40].
#   transformer: two-stage query selection (group 0 at inference) + 3 deformable
#             decoder layers (self_attn = nn.MultiheadAttention 8 heads;
#             cross_attn = MSDeformAttn 16 heads, 1 level, 2 points).
#   lite_refpoint_refine=True: reference points are NOT refined inside the
#             decoder; the final shared bbox_embed is applied on the last hidden
#             state with bbox_reparam refinement against the initial refpoints.
# ------------------------------------------------------------------------
"""RF-DETR-base reference model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .configuration_rf_detr import RfDetrBackboneConfig, RfDetrConfig

ACT = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}


# ============================================================================
# Backbone: windowed DINOv2 (per-layer internals identical to HF Dinov2)
# ============================================================================
class DinoPatchEmbeddings(nn.Module):
    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.num_channels = cfg.num_channels
        self.projection = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class DinoEmbeddings(nn.Module):
    """CLS token, (mask token), position embeddings, plus window partitioning."""

    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.num_windows = cfg.num_windows
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        if cfg.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, cfg.hidden_size))
        self.patch_embeddings = DinoPatchEmbeddings(cfg)
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.hidden_size))

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(round(num_positions**0.5))
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        # NOTE: RF-DETR uses antialias=True (differs from vanilla Dinov2).
        patch_pos_embed = F.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def window_partition(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        batch_size = embeddings.shape[0]
        num_windows = self.num_windows
        patch_size = self.patch_size
        num_h = height // patch_size
        num_w = width // patch_size
        num_w_per_window = num_w // num_windows
        num_h_per_window = num_h // num_windows

        cls_token_with_pos = embeddings[:, :1]
        pixel_tokens = embeddings[:, 1:]
        pixel_tokens = pixel_tokens.view(batch_size, num_h, num_w, -1)
        # (B, nw, h_pw, nw, w_pw, C) -> transpose(2,3) -> (B*nw^2, h_pw*w_pw, C)
        windowed = pixel_tokens.view(
            batch_size, num_windows, num_w_per_window, num_windows, num_h_per_window, -1
        )
        windowed = windowed.transpose(2, 3)
        windowed = windowed.reshape(
            batch_size * num_windows**2, num_h_per_window * num_w_per_window, -1
        )
        windowed_cls = cls_token_with_pos.repeat(num_windows**2, 1, 1)
        return torch.cat((windowed_cls, windowed), dim=1)

    def forward(self, pixel_values: Tensor) -> Tensor:
        _, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        if self.num_windows > 1:
            embeddings = self.window_partition(embeddings, height, width)
        return embeddings


class DinoSelfAttention(nn.Module):
    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.num_attention_heads = cfg.num_attention_heads
        self.attention_head_size = cfg.hidden_size // cfg.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.query = nn.Linear(cfg.hidden_size, self.all_head_size, bias=cfg.qkv_bias)
        self.key = nn.Linear(cfg.hidden_size, self.all_head_size, bias=cfg.qkv_bias)
        self.value = nn.Linear(cfg.hidden_size, self.all_head_size, bias=cfg.qkv_bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        b = hidden_states.shape[0]
        new_shape = (b, -1, self.num_attention_heads, self.attention_head_size)
        q = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        k = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        v = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous()
        return ctx.reshape(b, -1, self.all_head_size)


class DinoSelfOutput(nn.Module):
    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, hidden_states: Tensor, _input: Tensor) -> Tensor:
        return self.dense(hidden_states)


class DinoAttention(nn.Module):
    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.attention = DinoSelfAttention(cfg)
        self.output = DinoSelfOutput(cfg)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.output(self.attention(hidden_states), hidden_states)


class DinoLayerScale(nn.Module):
    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.lambda1 = nn.Parameter(cfg.layerscale_value * torch.ones(cfg.hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.lambda1


class DinoMLP(nn.Module):
    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        hidden = int(cfg.hidden_size * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.hidden_size, hidden)
        self.activation = ACT[cfg.hidden_act]()
        self.fc2 = nn.Linear(hidden, cfg.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class DinoLayer(nn.Module):
    """One DINOv2 block with windowed/global attention switching."""

    def __init__(self, cfg: RfDetrBackboneConfig, layer_idx: int):
        super().__init__()
        self.num_windows = cfg.num_windows
        self.global_attention = layer_idx not in cfg.window_block_indexes
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attention = DinoAttention(cfg)
        self.layer_scale1 = DinoLayerScale(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = DinoMLP(cfg)
        self.layer_scale2 = DinoLayerScale(cfg)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        if self.global_attention:
            # merge windows back into one sequence per image for global attention
            b, s, c = hidden_states.shape
            nw2 = self.num_windows**2
            hidden_states = hidden_states.view(b // nw2, nw2 * s, c)

        attn_out = self.attention(self.norm1(hidden_states))

        if self.global_attention:
            b, s, c = hidden_states.shape  # merged shape
            nw2 = self.num_windows**2
            attn_out = attn_out.view(b * nw2, s // nw2, c)

        attn_out = self.layer_scale1(attn_out)
        hidden_states = attn_out + residual
        residual = hidden_states
        h = self.norm2(hidden_states)
        h = self.mlp(h)
        h = self.layer_scale2(h)
        return h + residual


class DinoEncoderInner(nn.Module):
    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.layer = nn.ModuleList([DinoLayer(cfg, i) for i in range(cfg.num_hidden_layers)])

    def forward(self, hidden_states: Tensor) -> list[Tensor]:
        # Returns list of hidden states starting with the embedding output,
        # matching transformers' output_hidden_states ordering.
        all_hidden = [hidden_states]
        for layer in self.layer:
            hidden_states = layer(hidden_states)
            all_hidden.append(hidden_states)
        return all_hidden


class WindowedDinoBackbone(nn.Module):
    """Maps the published key prefix backbone.0.encoder.encoder.* exactly."""

    def __init__(self, cfg: RfDetrBackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = DinoEmbeddings(cfg)
        self.encoder = DinoEncoderInner(cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.stage_names = ["stem"] + [f"stage{i}" for i in range(1, cfg.num_hidden_layers + 1)]
        self.out_features = [f"stage{i}" for i in cfg.out_indices]

    def window_unpartition(self, hidden_state: Tensor, height: int, width: int) -> Tensor:
        num_windows = self.cfg.num_windows
        patch_size = self.cfg.patch_size
        num_h = height // patch_size
        num_w = width // patch_size
        hb, s, c = hidden_state.shape
        nw2 = num_windows**2
        num_h_pw = num_h // num_windows
        num_w_pw = num_w // num_windows
        hidden_state = hidden_state.reshape(hb // nw2, nw2 * s, c)
        hidden_state = hidden_state.view(
            hb // nw2, num_windows, num_windows, num_h_pw, num_w_pw, c
        )
        hidden_state = hidden_state.transpose(2, 3)
        return hidden_state

    def forward(self, pixel_values: Tensor) -> list[Tensor]:
        _, _, height, width = pixel_values.shape
        embedding_output = self.embeddings(pixel_values)
        hidden_states = self.encoder(embedding_output)

        feature_maps = []
        for stage, hs in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.cfg.apply_layernorm:
                    hs = self.layernorm(hs)
                if self.cfg.reshape_hidden_states:
                    hs = hs[:, 1:]  # drop CLS token (per window)
                    num_h = height // self.cfg.patch_size
                    num_w = width // self.cfg.patch_size
                    if self.cfg.num_windows > 1:
                        hs = self.window_unpartition(hs, height, width)
                    hs = hs.reshape(pixel_values.shape[0], num_h, num_w, -1)
                    hs = hs.permute(0, 3, 1, 2).contiguous()
                feature_maps.append(hs)
        return feature_maps


# ============================================================================
# Projector: C2f. Names match published checkpoint (cv1/cv2/m/bn).
# ConvNorm uses a channels-first LayerNorm (only affine params in checkpoint).
# ============================================================================
class ChannelsFirstLayerNorm(nn.Module):
    """LayerNorm over channels for (B, C, H, W) tensors (stages.0.1 + every bn)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.normalized_shape = (num_channels,)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvX(nn.Module):
    """Conv (bias=False) + channels-first LayerNorm (`bn`) + activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int,
                 act: str, norm_eps: float):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding=padding, bias=False)
        self.bn = ChannelsFirstLayerNorm(out_ch, eps=norm_eps)
        self.act = ACT[act]()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x.contiguous())))


class Bottleneck(nn.Module):
    def __init__(self, ch: int, act: str, norm_eps: float):
        super().__init__()
        self.cv1 = ConvX(ch, ch, 3, 1, act, norm_eps)
        self.cv2 = ConvX(ch, ch, 3, 1, act, norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv2(self.cv1(x))  # add=False (in_ch==out_ch but shortcut disabled)


class C2f(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int, expansion: float, act: str, norm_eps: float):
        super().__init__()
        self.c = int(out_ch * expansion)
        self.cv1 = ConvX(in_ch, 2 * self.c, 1, 1, act, norm_eps)
        self.cv2 = ConvX((2 + n) * self.c, out_ch, 1, 1, act, norm_eps)
        self.m = nn.ModuleList(Bottleneck(self.c, act, norm_eps) for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        cur = y[-1].contiguous()
        for bottleneck in self.m:
            cur = bottleneck(cur)
            y.append(cur)
        return self.cv2(torch.cat(y, 1))


class Projector(nn.Module):
    """backbone.0.projector. stages = ModuleList([Sequential(C2f, LayerNorm)])."""

    def __init__(self, cfg: RfDetrConfig):
        super().__init__()
        in_ch = cfg.backbone_config.hidden_size * len(cfg.backbone_config.out_indices)
        c2f = C2f(
            in_ch,
            cfg.d_model,
            cfg.c2f_num_blocks,
            cfg.hidden_expansion,
            cfg.activation_function,
            cfg.projector_norm_eps,
        )
        ln = ChannelsFirstLayerNorm(cfg.d_model, eps=cfg.projector_norm_eps)
        self.stages = nn.ModuleList([nn.Sequential(c2f, ln)])

    def forward(self, feature_maps: list[Tensor]) -> Tensor:
        fused = torch.cat(feature_maps, dim=1)
        return self.stages[0](fused)


class Backbone(nn.Module):
    """backbone.0 = encoder (windowed DINOv2) + projector (C2f)."""

    def __init__(self, cfg: RfDetrConfig):
        super().__init__()
        # encoder.encoder gives the published prefix backbone.0.encoder.encoder.*
        self.encoder = nn.Module()
        self.encoder.encoder = WindowedDinoBackbone(cfg.backbone_config)
        self.projector = Projector(cfg)

    def forward(self, pixel_values: Tensor) -> tuple[list[Tensor], Tensor]:
        feature_maps = self.encoder.encoder(pixel_values)
        projected = self.projector(feature_maps)
        return feature_maps, projected


# ============================================================================
# Multi-scale deformable attention (grid_sample core).
# ============================================================================
def ms_deform_attn_core(value, value_spatial_shapes_list, sampling_locations, attention_weights):
    b, _, num_heads, hidden_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in value_spatial_shapes_list], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (h, w) in enumerate(value_spatial_shapes_list):
        value_l = value_list[level_id].flatten(2).transpose(1, 2).reshape(b * num_heads, hidden_dim, h, w)
        grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampled)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        b * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(b, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, cfg: RfDetrConfig, num_heads: int, n_points: int):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_levels = cfg.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(cfg.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(cfg.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.output_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, hidden_states, encoder_hidden_states, reference_points,
                spatial_shapes, spatial_shapes_list, position_embeddings=None,
                attention_mask=None):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        b, num_queries, _ = hidden_states.shape
        _, seq_len, _ = encoder_hidden_states.shape

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], 0.0)
        value = value.view(b, seq_len, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            b, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            b, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            b, num_queries, self.n_heads, self.n_levels, self.n_points
        )

        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {num_coordinates}")

        output = ms_deform_attn_core(value, spatial_shapes_list, sampling_locations, attention_weights)
        return self.output_proj(output)


# ============================================================================
# Decoder.
# ============================================================================
def encode_sinusoidal_position_embedding(pos_tensor: Tensor, num_pos_feats: int = 128,
                                         temperature: int = 10000) -> Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    coords = pos_tensor.unbind(-1)
    embeddings = [coord[..., None] * scale / dim_t for coord in coords]
    embeddings = [
        torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=-1).flatten(-2) for e in embeddings
    ]
    if len(embeddings) >= 2:
        embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
    return torch.cat(embeddings, dim=-1).to(pos_tensor.dtype)


class MLP(nn.Module):
    """ref_point_head / bbox_embed / enc_out_bbox_embed predictor."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DecoderLayer(nn.Module):
    """Post-LN decoder layer. self_attn = nn.MultiheadAttention (in_proj_weight)."""

    def __init__(self, cfg: RfDetrConfig, layer_idx: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=cfg.decoder_self_attention_heads, dropout=0.0,
            batch_first=True,
        )
        self.cross_attn = MSDeformAttn(
            cfg, num_heads=cfg.decoder_cross_attention_heads, n_points=cfg.decoder_n_points
        )
        self.linear1 = nn.Linear(cfg.d_model, cfg.decoder_ffn_dim)
        self.linear2 = nn.Linear(cfg.decoder_ffn_dim, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.activation = ACT[cfg.decoder_activation_function]()

    def forward(self, hidden_states, query_pos, reference_points, encoder_hidden_states,
                spatial_shapes, spatial_shapes_list, encoder_attention_mask=None):
        # self-attention: q=k=tgt+query_pos, v=tgt
        q = k = hidden_states + query_pos
        tgt2 = self.self_attn(q, k, hidden_states, need_weights=False)[0]
        hidden_states = hidden_states + tgt2
        hidden_states = self.norm1(hidden_states)

        # deformable cross-attention
        tgt2 = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            position_embeddings=query_pos,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + tgt2
        hidden_states = self.norm2(hidden_states)

        # FFN
        tgt2 = self.linear2(self.activation(self.linear1(hidden_states)))
        hidden_states = hidden_states + tgt2
        hidden_states = self.norm3(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, cfg: RfDetrConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.decoder_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ref_point_head = MLP(2 * cfg.d_model, cfg.d_model, cfg.d_model, num_layers=2)

    def get_reference(self, reference_points, valid_ratios):
        obj_center = reference_points[..., :4]
        ref_inputs = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        query_sine = encode_sinusoidal_position_embedding(
            ref_inputs[:, :, 0, :], num_pos_feats=self.cfg.d_model // 2
        )
        query_pos = self.ref_point_head(query_sine)
        return ref_inputs, query_pos

    def forward(self, target, reference_points, valid_ratios, encoder_hidden_states,
                spatial_shapes, spatial_shapes_list, encoder_attention_mask=None,
                collect_intermediate=False):
        ref_inputs, query_pos = self.get_reference(reference_points, valid_ratios)
        hidden_states = target
        intermediate = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                query_pos=query_pos,
                reference_points=ref_inputs,
                encoder_hidden_states=encoder_hidden_states,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                encoder_attention_mask=encoder_attention_mask,
            )
            intermediate.append(self.norm(hidden_states))
        intermediate = torch.stack(intermediate)
        return intermediate, (intermediate if collect_intermediate else None)


# ============================================================================
# Transformer container (two-stage query selection + decoder).
# ============================================================================
def refine_bboxes(reference_points: Tensor, deltas: Tensor) -> Tensor:
    """bbox_reparam refinement: cxcy = delta_xy*ref_wh + ref_xy; wh = exp(delta_wh)*ref_wh."""
    new_cxcy = deltas[..., :2] * reference_points[..., 2:] + reference_points[..., :2]
    new_wh = deltas[..., 2:].exp() * reference_points[..., 2:]
    return torch.cat((new_cxcy, new_wh), -1)


class Transformer(nn.Module):
    """transformer.* — two-stage selection heads + decoder."""

    def __init__(self, cfg: RfDetrConfig):
        super().__init__()
        self.cfg = cfg
        g = cfg.group_detr
        self.enc_output = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model) for _ in range(g)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(g)])
        self.enc_out_bbox_embed = nn.ModuleList(
            [MLP(cfg.d_model, cfg.d_model, 4, num_layers=3) for _ in range(g)]
        )
        self.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.num_labels) for _ in range(g)]
        )
        self.decoder = Decoder(cfg)


# ============================================================================
# Top-level model.
# ============================================================================
@dataclass
class RfDetrOutput:
    logits: Tensor
    pred_boxes: Tensor
    init_reference_points: Tensor | None = None
    backbone_feature_maps: list | None = None
    projector_out: Tensor | None = None
    decoder_hidden_states: list | None = None
    enc_outputs_class: Tensor | None = None
    enc_outputs_coord_logits: Tensor | None = None


class RfDetrForObjectDetection(nn.Module):
    """Faithful RF-DETR-base reference. Keys match Roboflow/rf-detr-base exactly."""

    def __init__(self, cfg: RfDetrConfig):
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.num_queries
        self.group_detr = cfg.group_detr
        self.d_model = cfg.d_model

        # backbone.0.* container
        self.backbone = nn.ModuleList([Backbone(cfg)])
        self.transformer = Transformer(cfg)

        self.refpoint_embed = nn.Embedding(cfg.num_queries * cfg.group_detr, 4)
        self.query_feat = nn.Embedding(cfg.num_queries * cfg.group_detr, cfg.d_model)
        self.class_embed = nn.Linear(cfg.d_model, cfg.num_labels)
        self.bbox_embed = MLP(cfg.d_model, cfg.d_model, 4, num_layers=3)

    @torch.no_grad()
    def forward(self, pixel_values: Tensor, pixel_mask: Tensor | None = None,
                collect_intermediates: bool = False) -> RfDetrOutput:
        b, _, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((b, height, width), dtype=torch.long, device=device)

        # --- backbone + projector ---
        backbone = self.backbone[0]
        feature_maps, source = backbone(pixel_values)
        # downsample mask to feature resolution
        mask = F.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]

        source_flatten = source.flatten(2).transpose(1, 2)  # (B, HW, C)
        mask_flatten = mask.flatten(1)  # (B, HW) — True where valid
        spatial_shapes_list = [tuple(source.shape[2:])]
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=device)
        valid_ratios = self._get_valid_ratio(mask).unsqueeze(1)  # (B, 1, 2)

        # --- two-stage proposal generation ---
        object_query_embedding, output_proposals, invalid_mask = self._gen_proposals(
            source_flatten, ~mask_flatten, spatial_shapes_list
        )

        # inference uses only group 0
        topk = self.num_queries
        tf = self.transformer
        object_query = tf.enc_output[0](object_query_embedding)
        object_query = tf.enc_output_norm[0](object_query)
        enc_class = tf.enc_out_class_embed[0](object_query)
        enc_class = enc_class.masked_fill(invalid_mask, float("-inf"))
        delta_bbox = tf.enc_out_bbox_embed[0](object_query)
        enc_coord = refine_bboxes(output_proposals, delta_bbox)
        topk_proposals = torch.topk(enc_class.max(-1)[0], topk, dim=1)[1]
        topk_coords = torch.gather(enc_coord, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, 4))
        object_query_topk = torch.gather(
            object_query, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        # --- reference points (group 0) ---
        reference_points = self.refpoint_embed.weight[: self.num_queries]
        reference_points = reference_points.unsqueeze(0).expand(b, -1, -1)
        reference_points = refine_bboxes(topk_coords, reference_points)
        init_reference_points = reference_points

        target = self.query_feat.weight[: self.num_queries].unsqueeze(0).expand(b, -1, -1)

        # --- decoder (lite refine: reference points fixed across layers) ---
        intermediate, decoder_hidden = tf.decoder(
            target=target,
            reference_points=reference_points,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            encoder_attention_mask=mask_flatten,
            collect_intermediate=collect_intermediates,
        )
        last_hidden_state = intermediate[-1]

        # --- heads ---
        logits = self.class_embed(last_hidden_state)
        boxes_delta = self.bbox_embed(last_hidden_state)
        pred_boxes = refine_bboxes(init_reference_points, boxes_delta)

        decoder_hidden_states = None
        if collect_intermediates:
            decoder_hidden_states = [intermediate[i] for i in range(intermediate.shape[0])]

        return RfDetrOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            init_reference_points=init_reference_points,
            backbone_feature_maps=feature_maps if collect_intermediates else None,
            projector_out=source if collect_intermediates else None,
            decoder_hidden_states=decoder_hidden_states,
            enc_outputs_class=object_query_topk if collect_intermediates else None,
            enc_outputs_coord_logits=topk_coords if collect_intermediates else None,
        )

    @staticmethod
    def _get_valid_ratio(mask: Tensor, dtype=torch.float32) -> Tensor:
        _, h, w = mask.shape
        valid_h = torch.sum(mask[:, :, 0], 1)
        valid_w = torch.sum(mask[:, 0, :], 1)
        return torch.stack([valid_w.to(dtype) / w, valid_h.to(dtype) / h], -1)

    @staticmethod
    def _gen_proposals(enc_output, padding_mask, spatial_shapes_list):
        """padding_mask: True where padded (invalid). Returns proposals in [0,1] cxcywh."""
        b = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (h, w) in enumerate(spatial_shapes_list):
            mask_flatten_ = padding_mask[:, _cur:_cur + h * w].view(b, h, w, 1)
            valid_h = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, h - 1, h, dtype=enc_output.dtype, device=enc_output.device),
                torch.linspace(0, w - 1, w, dtype=enc_output.dtype, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(b, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(b, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, wh), -1).view(b, -1, 4)
            proposals.append(proposal)
            _cur += h * w
        output_proposals = torch.cat(proposals, 1)
        valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        invalid_mask = padding_mask.unsqueeze(-1) | ~valid
        output_proposals = output_proposals.masked_fill(invalid_mask, 0.0)
        object_query = enc_output.masked_fill(invalid_mask, 0.0)
        return object_query, output_proposals, invalid_mask
