# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from ...layers.conv2d import Conv2d
from ...layers.normalization import GroupNorm
from ...layers.linear import ColParallelLinear, Linear
from ...utils.substate import substate, indexed_substates
from ...parallel.config import vae_all_gather

if TYPE_CHECKING:
    pass


# TODO: Cleanup the use of torch ref
class ResnetBlock:
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        num_groups=None,
        eps=None,
        norm_out_blocks=-1,
        mesh_device=None,
        norm_core_grid=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        self.norm1 = GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=eps,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            core_grid=norm_core_grid,
            num_out_blocks=norm_out_blocks,
            torch_ref=torch_ref.norm1 if torch_ref is not None else None,
        )
        self.norm2 = GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
            eps=eps,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            core_grid=norm_core_grid,
            num_out_blocks=norm_out_blocks,
            torch_ref=torch_ref.norm2 if torch_ref is not None else None,
        )
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            torch_ref=torch_ref.conv1 if torch_ref is not None else None,
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            torch_ref=torch_ref.conv2 if torch_ref is not None else None,
        )
        self.conv_shortcut = None
        if in_channels != out_channels or torch_ref.conv_shortcut is not None:
            self.conv_shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
                torch_ref=torch_ref.conv_shortcut,
            )
        else:
            self.conv_shortcut = None

    def load_state_dict(self, state_dict):
        self.norm1.load_state_dict(state_dict["norm1"])
        self.norm2.load_state_dict(state_dict["norm2"])
        self.conv1.load_state_dict(state_dict["conv1"])
        self.conv2.load_state_dict(state_dict["conv2"])

        if "conv_shortcut" in state_dict:
            self.conv_shortcut.load_state_dict(state_dict["conv_shortcut"])

    # TODO: Update to use defined members within the class for portability
    @classmethod
    def from_torch(
        cls,
        torch_ref,
        norm_out_blocks=-1,
        mesh_device=None,
        norm_core_grid=None,
        parallel_config=None,
        ccl_manager=None,
    ):
        resnet_block = cls(
            torch_ref=torch_ref,
            norm_out_blocks=norm_out_blocks,
            mesh_device=mesh_device,
            norm_core_grid=norm_core_grid,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        return resnet_block

    def __call__(self, x):
        residual = ttnn.clone(x)
        x = self.norm1(x)
        x = ttnn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = ttnn.silu(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # Following binary op requires tile layout
        return x + residual


class Upsample2D:
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            torch_ref=torch_ref.conv if torch_ref is not None else None,
        )

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, mesh_axis=None, parallel_manager=None):
        layer = cls(
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            parallel_manager=parallel_manager,
            torch_ref=torch_ref,
        )
        return layer

    def load_state_dict(self, state_dict):
        self.conv.load_state_dict(state_dict["conv"])

    def __call__(self, x):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Upsample requires row major.
        x = ttnn.upsample(x, scale_factor=2)
        x = self.conv(x)
        return x


class UpDecoderBlock2D:
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        num_layers=None,
        resnet_groups=None,
        add_upsample=None,
        mesh_device=None,
        norm_core_grid=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        if torch_ref is None:
            self.resnets = [
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_groups=resnet_groups,
                    mesh_device=mesh_device,
                    norm_core_grid=norm_core_grid,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for _ in range(num_layers)
            ]
            self.upsamplers = (
                [Upsample2D(in_channels, out_channels, mesh_device, parallel_config, ccl_manager)]
                if add_upsample
                else []
            )
        else:
            self.resnets = [
                ResnetBlock(
                    torch_ref=resnet,
                    mesh_device=mesh_device,
                    norm_core_grid=norm_core_grid,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for resnet in torch_ref.resnets
            ]

            self.upsamplers = [
                Upsample2D(
                    torch_ref=upsampler,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for upsampler in torch_ref.upsamplers or []
            ]

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, norm_core_grid=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            torch_ref=torch_ref,
            mesh_device=mesh_device,
            norm_core_grid=norm_core_grid,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        return layer

    # TODO: Fix state dict
    def load_state_dict(self, state_dict):
        for i, state in enumerate(indexed_substates(state_dict, "resnets")):
            self.resnets[i].load_state_dict(state)

        for i, state in enumerate(indexed_substates(state_dict, "upsamplers")):
            self.upsamplers[i].load_state_dict(state)

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


# TODO: Add support for coll and row parallel linear. Fuse qkv computation
class Attention:
    def __init__(
        self,
        query_dim=None,
        head_dim=None,
        num_heads=None,
        norm_num_groups=None,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        self.query_dim = query_dim or torch_ref.to_q.in_features
        self.num_heads = num_heads or torch_ref.heads
        self.head_dim = head_dim or torch_ref.to_q.out_features // self.num_heads
        self.inner_dim = self.head_dim * self.num_heads
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.to_q = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_k = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_v = Linear(in_features=self.query_dim, out_features=self.inner_dim, mesh_device=mesh_device)
        self.to_out = [
            ColParallelLinear(
                in_features=self.inner_dim,
                out_features=self.query_dim,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            )
        ]
        self.group_norm = GroupNorm(
            num_groups=(norm_num_groups or torch_ref.group_norm.num_groups),
            num_channels=self.query_dim,
            eps=torch_ref.group_norm.eps,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        if torch_ref is not None:
            self.load_state_dict(torch_ref.state_dict())

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            torch_ref=torch_ref, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return layer

    @staticmethod
    def reorder_for_attention(x, batch_size, n_heads, head_dim):
        return ttnn.permute(ttnn.reshape(x, (batch_size, -1, n_heads, head_dim)), (0, 2, 1, 3))

    def load_state_dict(self, state_dict):
        self.to_q.load_state_dict(substate(state_dict, "to_q"))
        self.to_k.load_state_dict(substate(state_dict, "to_k"))
        self.to_v.load_state_dict(substate(state_dict, "to_v"))
        for i, state in enumerate(indexed_substates(state_dict, "to_out")):
            self.to_out[i].load_state_dict(state)
        self.group_norm.load_state_dict(substate(state_dict, "group_norm"))

    # TODO: Standardize this usage
    def gather_if_sharded(self, x):
        if x.shape[3] < self.to_q.in_features:
            x = vae_all_gather(self.ccl_manager, x)
        return x

    def __call__(self, x):
        assert len(x.shape) == 4
        residual = x
        # elementwise required to be tilized
        in_layout = x.layout
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        [b, h, w, c] = list(x.shape)

        # No need to transpose like reference. x is alredy channel last
        x = self.group_norm(x)
        x = self.gather_if_sharded(x)

        # output will be bxhxwx(num_heads*head_dims)
        q = self.to_q(x, core_grid=self.mesh_device.core_grid)
        k = self.to_k(x, core_grid=self.mesh_device.core_grid)
        v = self.to_v(x, core_grid=self.mesh_device.core_grid)
        inner_dim = k.shape[-1]
        head_dim = inner_dim // self.num_heads

        q = self.reorder_for_attention(q, b, self.num_heads, head_dim)
        k = self.reorder_for_attention(k, b, self.num_heads, head_dim)
        v = self.reorder_for_attention(v, b, self.num_heads, head_dim)

        x = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (b, h, w, inner_dim))

        for to_out in self.to_out:
            x = to_out(x, core_grid=self.mesh_device.core_grid)

        x = x + residual

        x = ttnn.to_layout(x, in_layout)
        return x


class UnetMidBlock2D:
    def __init__(
        self,
        in_channels=None,
        resnet_groups=None,
        attention_head_dim=None,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
    ):
        if torch_ref is None:
            self.attentions = [
                Attention(
                    query_dim=in_channels,
                    head_dim=attention_head_dim,
                    num_heads=in_channels // attention_head_dim,
                    norm_num_groups=resnet_groups,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
            ]
            self.resnets = [
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_groups=resnet_groups,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for _ in range(2)
            ]
        else:
            self.attentions = [
                Attention(
                    torch_ref=attention,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for attention in torch_ref.attentions
            ]
            self.resnets = [
                ResnetBlock(
                    torch_ref=resnet, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
                )
                for resnet in torch_ref.resnets
            ]

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        layer = cls(
            torch_ref=torch_ref, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return layer

    def load_state_dict(self, state_dict):
        for i, state in enumerate(indexed_substates(state_dict, "attentions")):
            self.attentions[i].load_state_dict(state)
        for i, state in enumerate(indexed_substates(state_dict, "resnets")):
            self.resnets[i].load_state_dict(state)

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


# TODO: Clean up, and factor out duplicate code
class VAEDecoder:
    def __init__(
        self,
        block_out_channels=(128, 256, 512, 512),
        in_channels=16,
        out_channels=3,
        layers_per_block=2,
        norm_num_groups=32,
        torch_ref=None,
        mesh_device=None,
        parallel_config=None,
        ccl_manager=None,
    ):
        """
        Initialize the VAEDecoder.
        Args:
            block_out_channels: The number of channels for the updecoder blocks. They are also used to support other layers and blocks
            in_channels: The number of channels in the input image.
            out_channels: The number of channels in the output image.
            layers_per_block: The number of Resnet layers (blocks) in each updecoder.
            norm_num_groups: The number of groups in the normalization layer.
            torch_ref: The reference to the torch model.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        # TODO: Add support for torch_ref
        if torch_ref is None:
            self.conv_in = Conv2d(
                in_channels,
                block_out_channels[-1],
                kernel_size=(3, 3),
                padding=(1, 1),
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )
            self.mid_block = UnetMidBlock2D(
                in_channels=block_out_channels[-1],
                attention_head_dim=block_out_channels[-1],
                resnet_groups=norm_num_groups,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )

            self.up_blocks = []
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

            self.conv_norm_out = GroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_out_channels[0],
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            )

            self.conv_out = Conv2d(
                block_out_channels[0],
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )

        else:
            self.conv_in = Conv2d.from_torch(
                torch_ref.conv_in,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )
            self.mid_block = UnetMidBlock2D.from_torch(
                torch_ref=torch_ref.mid_block,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )

            self.up_blocks = [
                UpDecoderBlock2D.from_torch(
                    torch_ref=up_block,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for up_block in torch_ref.up_blocks
            ]

            self.conv_norm_out = GroupNorm.from_torch(
                torch_ref=torch_ref.conv_norm_out,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            )

            self.conv_out = Conv2d.from_torch(
                torch_ref.conv_out, mesh_device=mesh_device, mesh_axis=None, ccl_manager=ccl_manager
            )

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, parallel_config=None, ccl_manager=None):
        vae_model = cls(
            torch_ref=torch_ref, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return vae_model

    def load_state_dict(self, state_dict):
        self.conv_in.load_state_dict(substate(state_dict, "conv_in"))
        self.mid_block.load_state_dict(substate(state_dict, "mid_block"))
        for i, state in enumerate(indexed_substates(state_dict, "up_blocks")):
            self.up_blocks[i].load_state_dict(state)
        self.conv_norm_out.load_state_dict(substate(state_dict, "conv_norm_out"))
        self.conv_out.load_state_dict(substate(state_dict, "conv_out"))

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = ttnn.silu(x)
        x = self.conv_out(x)
        return x
