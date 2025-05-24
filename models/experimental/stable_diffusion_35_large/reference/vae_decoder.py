# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/autoencoders/vae.py
class VaeDecoder(torch.nn.Module):
    def __init__(
        self,
        block_out_channels: list[int] | tuple[int, ...] = (128, 256, 512, 512),
        in_channels: int = 16,
        out_channels: int = 3,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ) -> None:
        super().__init__()

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, padding=1)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        self.up_blocks = torch.nn.ModuleList([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        prev_output_channel = reversed_block_out_channels[0]
        for i, output_channel in enumerate(reversed_block_out_channels):
            is_final_block = i == len(reversed_block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                num_layers=layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_groups=norm_num_groups,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = torch.nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        x = self.mid_block(x)
        x = x.to(upscale_dtype)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        return self.conv_out(x)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/unets/unet_2d_blocks.py
class UpDecoderBlock2D(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        resnet_groups: int,
        add_upsample: bool,
    ) -> None:
        super().__init__()

        self.resnets = torch.nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    groups=resnet_groups,
                )
                for i in range(num_layers)
            ]
        )

        if add_upsample:
            self.upsamplers = torch.nn.ModuleList([Upsample2D(channels=out_channels, out_channels=out_channels)])
        else:
            self.upsamplers = torch.nn.ModuleList([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)

        for upsampler in self.upsamplers:
            x = upsampler(x)

        return x


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/unets/unet_2d_blocks.py
class UNetMidBlock2D(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        resnet_groups: int,
        attention_head_dim: int,
    ) -> None:
        super().__init__()

        self.attentions = torch.nn.ModuleList(
            [
                Attention(
                    query_dim=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    norm_num_groups=resnet_groups,
                )
            ]
        )

        self.resnets = torch.nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    groups=resnet_groups,
                )
                for _ in range(2)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/resnet.py
class ResnetBlock2D(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        groups: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = torch.nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return residual + x


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/upsampling.py
class Upsample2D(torch.nn.Module):
    def __init__(self, *, channels: int, out_channels: int) -> None:
        super().__init__()

        self.channels = channels
        self.conv = torch.nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class Attention(torch.nn.Module):
    def __init__(self, *, query_dim: int, norm_num_groups: int, heads: int, dim_head: int) -> None:
        super().__init__()

        self.heads = query_dim // dim_head

        self.group_norm = torch.nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=1e-6, affine=True)

        inner_dim = dim_head * heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim)
        self.to_k = torch.nn.Linear(query_dim, inner_dim)
        self.to_v = torch.nn.Linear(query_dim, inner_dim)

        self.to_out = torch.nn.ModuleList([torch.nn.Linear(inner_dim, query_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4  # noqa: PLR2004

        residual = x

        batch_size, features, height, width = x.shape
        x = x.view(batch_size, features, height * width).transpose(1, 2)

        x = self.group_norm(x.transpose(1, 2)).transpose(1, 2)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        inner_dim = k.shape[-1]
        head_dim = inner_dim // self.heads

        q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

        x = self.to_out[0](x)

        x = x.transpose(-1, -2).reshape(batch_size, features, height, width)

        return x + residual
