# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone Swin Transformer backbone for ATSS detection.
# Extracted from MMDetection v3.3.0 (mmdet.models.backbones.swin)
# and converted to dependency-free PyTorch for PCC validation.

from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    During inference this is an identity operation.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class AdaptivePadding(nn.Module):
    """Pads input so that it is fully covered by kernel/stride."""

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding="corner"):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using a convolution layer."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        kernel_size: int = 16,
        stride: int = 16,
        padding: str = "corner",
        dilation: int = 1,
        bias: bool = True,
        norm_cfg: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
            )
            padding_val = 0
        else:
            self.adap_padding = None
            padding_val = padding

        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=to_2tuple(padding_val),
            dilation=dilation,
            bias=bias,
        )
        self.norm = nn.LayerNorm(embed_dims) if norm_cfg else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.adap_padding:
            x = self.adap_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(nn.Module):
    """Merge patch feature map (downsample 2x) using nn.Unfold."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: Optional[int] = None,
        padding: str = "corner",
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = stride or kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
            )
            padding_val = 0
        else:
            self.adap_padding = None
            padding_val = padding

        padding_val = to_2tuple(padding_val)
        self.sampler = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding_val, stride=stride)
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        self.norm = nn.LayerNorm(sample_dim)
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, input_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]
        x = self.sampler(x)  # B, 4*C, H/2*W/2
        out_h = (
            H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1
        ) // self.sampler.stride[1] + 1
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x, (out_h, out_w)


class WindowMSA(nn.Module):
    """Window-based Multi-head Self-Attention with relative position bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        Wh, Ww = self.window_size
        rel_index_coords = self._double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multi-head Self-Attention."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = DropPath(drop_path_rate)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = self.window_size
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
        return windows

    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        ws = self.window_size
        B = int(windows.shape[0] / (H * W / ws / ws))
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, query: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W
        query = query.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size**2, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x


class FFN(nn.Module):
    """Feed-Forward Network (MLP) with residual connection."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        ffn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, feedforward_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)
        self.drop = nn.Dropout(ffn_drop)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.drop_path(out)
        if identity is None:
            identity = x
        return identity + out


class SwinBlock(nn.Module):
    """A single Swin Transformer block (W-MSA/SW-MSA + FFN)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x


class SwinBlockSequence(nn.Module):
    """One stage of Swin Transformer (multiple SwinBlocks + optional downsample)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=(i % 2 == 1),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
            )
            self.blocks.append(block)
        self.downsample = downsample

    def forward(
        self, x: torch.Tensor, hw_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor, Tuple[int, int]]:
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class SwinTransformer(nn.Module):
    """Swin Transformer backbone for object detection.

    Produces multi-scale feature maps at selected stages (out_indices).
    """

    def __init__(
        self,
        pretrain_img_size: int = 224,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: int = 4,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        strides: Tuple[int, ...] = (4, 2, 2, 2),
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_abs_pos_embed: bool = False,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        num_layers = len(depths)

        assert strides[0] == patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            padding="corner",
            norm_cfg="LN" if patch_norm else None,
        )

        if self.use_abs_pos_embed:
            if isinstance(pretrain_img_size, int):
                pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        in_ch = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_ch,
                    out_channels=2 * in_ch,
                    stride=strides[i + 1],
                )
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_ch,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_ch,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=downsample,
            )
            self.stages.append(stage)
            if downsample:
                in_ch = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]

        for i in out_indices:
            layer = nn.LayerNorm(self.num_features[i])
            layer_name = f"norm{i}"
            self.add_module(layer_name, layer)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


def build_swin_l_backbone(
    window_size: int = 12,
    out_indices: Tuple[int, ...] = (1, 2, 3),
    drop_path_rate: float = 0.2,
) -> SwinTransformer:
    """Instantiate Swin-L backbone matching the ATSS config."""
    return SwinTransformer(
        pretrain_img_size=384,
        in_channels=3,
        embed_dims=192,
        patch_size=4,
        window_size=window_size,
        mlp_ratio=4,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        strides=(4, 2, 2, 2),
        out_indices=out_indices,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        use_abs_pos_embed=False,
    )
