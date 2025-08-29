# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ...utils.check import assert_quality
from ...models.vae import vae_sd35
from ...parallel.manager import CCLManager
from ...parallel.config import vae_all_gather, VAEParallelConfig, ParallelFactor
from time import time
from loguru import logger


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


class ResnetBlock2D(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels, groups, eps=1e-6):
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


# Custom pytest mark for shared VAE device configuration
def vae_device_config(func):
    """Decorator to apply standard VAE device configuration to tests"""
    func = pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)(func)
    func = pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
        indirect=True,
    )(func)
    return func


@vae_device_config
@pytest.mark.parametrize(
    (
        "batch",
        "height",
        "width",
        "in_channels",
        "out_channels",
        "groups",
    ),
    [(1, 1024, 1024, 256, 128, 32)],
)
def test_sd35_vae_resnet_block(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    groups: int,
) -> None:
    torch_model = ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, groups=groups)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_sd35.ResnetBlock.from_torch(
        torch_ref=torch_model, mesh_device=mesh_device, parallel_config=vae_parallel_config, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_out = vae_all_gather(ccl_manager, tt_out)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)


@vae_device_config
@pytest.mark.parametrize(
    ("batch", "in_channels", "out_channels", "height", "width", "num_layers", "num_groups", "add_upsample"),
    [
        (1, 512, 512, 128, 128, 2, 32, False),
        (1, 512, 512, 128, 128, 2, 32, True),
    ],
)
def test_sd35_vae_up_decoder_block(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    num_layers: int,
    num_groups: int,
    add_upsample: bool,
) -> None:
    torch_model = UpDecoderBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        resnet_groups=num_groups,
        add_upsample=add_upsample,
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_sd35.UpDecoderBlock2D.from_torch(
        torch_ref=torch_model, mesh_device=mesh_device, parallel_config=vae_parallel_config, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
    )

    # TODO: Refactor common test components
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_out = vae_all_gather(ccl_manager, tt_out)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)


@vae_device_config
@pytest.mark.parametrize(
    ("batch", "in_channels", "height", "width", "num_groups", "num_heads"),
    [
        (1, 512, 128, 128, 32, 4),  # slice 128, output blocks 32. Need to parametize
        # (1, 512, 128, 128, 32, 4, False),  # slice 128, output blocks 32. Need to parametize
    ],
)
def test_sd35_vae_attention(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    in_channels: int,
    height: int,
    width: int,
    num_groups: int,
    num_heads: int,
):
    torch_model = Attention(
        query_dim=in_channels, heads=num_heads, dim_head=in_channels // num_heads, norm_num_groups=num_groups
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_sd35.Attention.from_torch(
        torch_ref=torch_model, mesh_device=mesh_device, parallel_config=vae_parallel_config, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_out = vae_all_gather(ccl_manager, tt_out)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)


@vae_device_config
@pytest.mark.parametrize(
    ("batch", "in_channels", "height", "width", "num_groups", "num_heads"),
    [
        (1, 512, 128, 128, 32, 4),  # slice 128, output blocks 32. Need to parametize
    ],
)
def test_sd35_vae_unet_mid_block2d(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    in_channels: int,
    height: int,
    width: int,
    num_groups: int,
    num_heads: int,
):
    torch_model = UNetMidBlock2D(
        in_channels=in_channels, resnet_groups=num_groups, attention_head_dim=in_channels // num_heads
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_sd35.UnetMidBlock2D.from_torch(
        torch_ref=torch_model, mesh_device=mesh_device, parallel_config=vae_parallel_config, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_out = vae_all_gather(ccl_manager, tt_out)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.999_000)


@vae_device_config
@pytest.mark.parametrize(
    (
        "batch",
        "in_channels",
        "out_channels",
        "layers_per_block",
        "height",
        "width",
        "norm_num_groups",
        "block_out_channels",
    ),
    [
        (1, 16, 3, 2, 128, 128, 32, (128, 256, 512, 512)),
    ],
)
def test_sd35_vae_vae_decoder(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    height: int,
    width: int,
    norm_num_groups: int,
    block_out_channels: list[int] | tuple[int, ...],
):
    torch_model = VaeDecoder(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_sd35.VAEDecoder.from_torch(
        torch_ref=torch_model, mesh_device=mesh_device, parallel_config=vae_parallel_config, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.99_000)

    start = time()
    tt_out = tt_model(tt_input_tensor)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"VAE Time taken: {time() - start}")
