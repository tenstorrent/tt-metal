# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from ...layers.normalization import GroupNorm
from ...layers.conv3d import ContextParallelConv3d
from ...utils.tensor import bf16_tensor

if TYPE_CHECKING:
    pass


# TODO REIMPLEMENT WITH THE LINEAR LAYER
class Conv1x1:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        swizzle_weight: Callable = None,
        mesh_device=None,
        torch_ref=None,
    ):
        """
        A 1x1 convolution implemented as a linear operation for ttnn.

        Can be instantiated from either:
        - nn.Conv3d with kernel_size=(1,1,1)
        - Conv1x1 (which is implemented as nn.Linear)

        Args:
            mesh_device: TTNN mesh device
            state_dict: Dictionary containing weights
            state_dict_prefix: Prefix for loading weights from state_dict
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include bias
            swizzle_weight: Function to swizzle weights, useful for channel expansion
        """
        self.in_features = in_channels
        self.out_features = out_channels
        self.use_bias = bias
        self.swizzle_weight = swizzle_weight
        self.mesh_device = mesh_device
        self.weight = None
        self.bias = None

        # Configure compute kernel
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if torch_ref is not None:
            self.load_state_dict(torch_ref.state_dict())
        else:
            self.weight = bf16_tensor(
                torch.randn(self.in_features, self.out_features),
                device=self.mesh_device,
                mesh_axis=None,
                shard_dim=None,
            )
            if self.use_bias:
                self.bias = bf16_tensor(
                    torch.randn(1, self.out_features), device=self.mesh_device, mesh_axis=None, shard_dim=None
                )
            else:
                self.bias = None

    def load_state_dict(self, state_dict):
        # Load weights - supports both Conv3d(1,1,1) and Conv1x1 format
        weight = state_dict["weight"]
        if weight.ndim == 5:  # Conv3d weight
            # Convert from (out_channels, in_channels, 1, 1, 1) to (out_channels, in_channels)
            weight = weight.squeeze()
        weight = weight.transpose(0, 1)  # (out_channels, in_channels) -> (in_channels, out_channels)
        if self.swizzle_weight:
            weight = self.swizzle_weight(weight)

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=[None, None]
        )
        self.weight = ttnn.from_torch(
            weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=self.mesh_device, mesh_mapper=mesh_mapper
        )
        assert self.use_bias == ("bias" in state_dict)
        if self.use_bias:
            bias_weight = state_dict["bias"]
            if self.swizzle_weight:
                bias_weight = self.swizzle_weight(bias_weight)
            self.bias = ttnn.from_torch(
                bias_weight.reshape(1, -1),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
            )
        else:
            self.bias = None

    @classmethod
    def from_torch(
        cls, torch_ref, in_channels=None, out_channels=None, bias=None, swizzle_weight=None, mesh_device=None
    ):
        layer = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            swizzle_weight=siwzzle_weight,
            mesh_device=mesh_device,
            torch_ref=torch_ref,
        )
        return layer

    def __call__(self, x_NTHWC):
        """
        Forward pass for Conv1x1.

        Args:
            x_NTHWC: Input tensor in NTHWC layout

        Returns:
            Output tensor in NTHWC layout
        """
        # Convert to tile layout for efficient computation
        x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_NTHWC)

        # Apply linear transformation
        x_tile_NTHWO = ttnn.linear(
            x_tile_NTHWC,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=self.mesh_device.core_grid,
        )
        ttnn.deallocate(x_tile_NTHWC)

        # Convert back to row major layout
        x_NTHWO = ttnn.to_layout(x_tile_NTHWO, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tile_NTHWO)

        return x_NTHWO


class ResBlock:
    def __init__(
        self,
        causal: bool = True,
        padding_mode: str = "replicate",
        bias: bool = True,
        mesh_device=None,
        mesh_axis=None,
        input_shape=None,
        torch_ref=None,
    ):
        self.num_out_blocks_map = {
            60 * 106: 8,
            120 * 212: 10,
            240 * 424: 40,
            480 * 848: 135,
        }

        grid_size_y = 4 if torch_ref.channels == 128 else mesh_device.core_grid.y
        grid_size_x = (
            min(32 // mesh_device.get_num_devices(), mesh_device.core_grid.x)
            if torch_ref.channels == 768
            else mesh_device.core_grid.x
        )
        self.grid_size = ttnn.CoreGrid(y=grid_size_y, x=grid_size_x)

        self.norm1 = GroupNorm(
            num_groups=32,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            core_grid=self.grid_size,
            torch_ref=torch_ref.stack[0] if torch_ref is not None else None,
        )
        self.norm2 = GroupNorm(
            num_groups=32,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            core_grid=self.grid_size,
            torch_ref=torch_ref.stack[3] if torch_ref is not None else None,
        )
        self.conv1 = ContextParallelConv3d(
            mesh_device=mesh_device,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
            input_shape=input_shape,
            torch_ref=torch_ref.stack[2] if torch_ref is not None else None,
        )
        self.conv2 = ContextParallelConv3d(
            mesh_device=mesh_device,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
            input_shape=input_shape,
            torch_ref=torch_ref.stack[5] if torch_ref is not None else None,
        )

    def get_tensor_shapes(self, x):
        return x.shape

    def reshape_tilize(self, x, shape):
        N, T, H, W, C = shape
        output = ttnn.tilize_with_zero_padding(
            ttnn.reshape(x, [N * T, 1, H * W, C]),
            use_multicore=True,
        )
        return output

    def untilize_reshape(self, x, shape):
        N, T, H, W, C = shape
        output = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [N, T, H, W, C])
        return output

    def __call__(self, x_NTHWC):
        shapes = self.get_tensor_shapes(x_NTHWC)

        x_tiled_NTHWC = self.reshape_tilize(x_NTHWC, shapes)
        residual_tiled_NTHWC = x_tiled_NTHWC
        ttnn.deallocate(x_NTHWC)

        HW = x_tiled_NTHWC.shape[2]
        T = x_tiled_NTHWC.shape[0]
        num_out_blocks = self.num_out_blocks_map[HW] if HW in self.num_out_blocks_map else math.ceil(HW / 2000)
        x_norm_tiled_NTHWC = self.norm1(x_tiled_NTHWC, num_out_blocks)

        # TODO: Investigate packing more data into a tile
        x_norm_tiled_NTHWC = ttnn.silu(x_norm_tiled_NTHWC, output_tensor=x_norm_tiled_NTHWC)  # in-place
        x_NTHWC = self.untilize_reshape(x_norm_tiled_NTHWC, shapes)
        ttnn.deallocate(x_norm_tiled_NTHWC)

        x_conv1_NTHWC = self.conv1(x_NTHWC)
        ttnn.deallocate(x_NTHWC)
        x_conv1_tiled_NTHWC = self.reshape_tilize(x_conv1_NTHWC, shapes)
        ttnn.deallocate(x_conv1_NTHWC)

        HW = x_conv1_tiled_NTHWC.shape[2]
        T = x_conv1_tiled_NTHWC.shape[0]
        num_out_blocks = self.num_out_blocks_map[HW] if HW in self.num_out_blocks_map else math.ceil(HW / 2000)
        x_tiled_NTHWC = self.norm2(x_conv1_tiled_NTHWC, num_out_blocks)
        ttnn.deallocate(x_conv1_tiled_NTHWC)

        # TODO: Investigate packing more data into a tile
        x_tiled_NTHWC = ttnn.silu(x_tiled_NTHWC, output_tensor=x_tiled_NTHWC)  # in-place
        x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, shapes)
        ttnn.deallocate(x_tiled_NTHWC)

        x_conv2_NTHWC = self.conv2(x_NTHWC)
        ttnn.deallocate(x_NTHWC)
        x_conv2_tiled_NTHWC = self.reshape_tilize(x_conv2_NTHWC, shapes)
        ttnn.deallocate(x_conv2_NTHWC)

        x_tiled_NTHWC = ttnn.add(x_conv2_tiled_NTHWC, residual_tiled_NTHWC)
        ttnn.deallocate(x_conv2_tiled_NTHWC)
        ttnn.deallocate(residual_tiled_NTHWC)
        x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, shapes)
        ttnn.deallocate(x_tiled_NTHWC)
        return x_NTHWC


class CausalUpsampleBlock:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        input_shape=None,
        torch_ref=None,
        num_res_blocks: int = 0,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        has_attention: bool = False,
        affine: bool = True,
        attn_block=None,
        causal: bool = True,
        prune_bottleneck: bool = False,
        padding_mode: str = "replicate",
        bias: bool = True,
    ):
        assert causal
        assert not prune_bottleneck
        assert not has_attention
        self.mesh_device = mesh_device
        self.blocks = []
        for i in range(num_res_blocks):
            self.blocks.append(
                ResBlock(
                    mesh_device=mesh_device,
                    causal=causal,
                    padding_mode=padding_mode,
                    bias=bias,
                    input_shape=input_shape,
                    torch_ref=torch_ref.blocks[i],
                )
            )

        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion
        self.out_channels = out_channels

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Swizzle conv1x1 weights
        def swizzle_weight(w):
            # X (C texp sexp sexp) -> X (texp sexp sexp C)
            w = w.reshape(-1, out_channels, temporal_expansion, spatial_expansion, spatial_expansion)
            w = w.permute(0, 2, 3, 4, 1)
            w = w.reshape(-1, temporal_expansion * spatial_expansion * spatial_expansion * out_channels)
            return w.squeeze()

        self.proj = Conv1x1(
            mesh_device=mesh_device,
            in_channels=in_channels,
            out_channels=out_channels * temporal_expansion * (spatial_expansion**2),
            bias=bias,
            swizzle_weight=swizzle_weight,
            torch_ref=torch_ref.proj,
        )

    def depth_to_spacetime(self, x_NTHWC):
        texp, sexp = self.temporal_expansion, self.spatial_expansion
        if self.mesh_device.get_num_devices() == 1:
            B, T, H, W, C = x_NTHWC.shape
            x = ttnn.reshape(x_NTHWC, [B, T, H, W, texp, sexp, sexp, self.out_channels])
            x = ttnn.permute(x, [0, 1, 4, 2, 5, 3, 6, 7])  # (B T texp H sexp W sexp C_out)

            x = ttnn.reshape(x, [B, T * texp, H * sexp, W * sexp, self.out_channels])
            if texp > 1:
                # Drop the first texp - 1 frames.
                x = ttnn.slice(x, [0, texp - 1, 0, 0, 0], [B, T * texp, H * sexp, W * sexp, self.out_channels])
            return x
        else:
            # Workaround for 1) issue #17535 for multi-device reshape,
            # and 2) slicing only the first shard.
            B, T, H, W, C = x_NTHWC.shape
            x_NTHWC = ttnn.reshape(x_NTHWC, [B, T, H, W, texp, sexp, sexp, self.out_channels])
            x_NTHWC = ttnn.permute(x_NTHWC, [0, 1, 4, 2, 5, 3, 6, 7])  # (B T texp H sexp W sexp C_out)

            x_NTHWC = ttnn.reshape(x_NTHWC, [B, T * texp, H * sexp, W * sexp, self.out_channels])

            # if texp > 1 and i == 0:
            #     x_NTHWC = ttnn.slice(
            #         x_NTHWC, [0, texp - 1, 0, 0, 0], [B, T * texp, H * sexp, W * sexp, self.out_channels]
            #     )
            #     # TODO: This messes up the shape of the tensor...
            # TODO fix depth_to_spaceitme multi tensor

            return x_NTHWC

    def __call__(self, x_NTHWC):
        for block in self.blocks:
            x_NTHWC = block(x_NTHWC)

        x_NTHWO = self.proj(x_NTHWC)
        x_NTHWC = self.depth_to_spacetime(x_NTHWO)
        ttnn.deallocate(x_NTHWO)
        return x_NTHWC


class Decoder:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        input_shape=None,
        torch_ref=None,
        out_channels=3,
        base_channels=128,
        channel_multipliers=[1, 2, 4, 6],
        temporal_expansions=[1, 2, 3],
        spatial_expansions=[2, 2, 2],
        num_res_blocks=[3, 3, 4, 6, 3],
        latent_dim=12,
        has_attention=[False, False, False, False, False],
        output_norm=False,
        nonlinearity="silu",
        output_nonlinearity="silu",
        causal=True,
    ):
        """
        TTNN implementation of the VAE Decoder.
        """
        self.input_channels = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.output_nonlinearity = output_nonlinearity
        self.mesh_device = mesh_device
        assert nonlinearity == "silu"
        assert causal
        assert not any(has_attention), "Attention is not supported in the decoder"
        attn_block = None
        # Calculate channels for each level
        ch = [mult * base_channels for mult in channel_multipliers]
        self.num_up_blocks = len(ch) - 1
        assert len(num_res_blocks) == self.num_up_blocks + 2

        assert len(temporal_expansions) == len(spatial_expansions) == self.num_up_blocks
        assert len(num_res_blocks) == len(has_attention) == self.num_up_blocks + 2

        first_block_torch_ref = torch_ref.blocks[0]
        # Create the initial projection from latent space
        self.input_proj = Conv1x1(
            mesh_device=mesh_device,
            in_channels=latent_dim,
            out_channels=ch[-1],
            torch_ref=first_block_torch_ref[0],
        )

        # First set of residual blocks
        self.first_blocks = []
        for i in range(num_res_blocks[-1]):
            self.first_blocks.append(
                ResBlock(
                    mesh_device=mesh_device,
                    causal=causal,
                    padding_mode="replicate",
                    input_shape=input_shape,
                    torch_ref=first_block_torch_ref[i + 1],
                )
            )

        # Create upsampling blocks
        self.up_blocks = []
        for i in range(self.num_up_blocks):
            upsample_block_torch_ref = torch_ref.blocks[i + 1]
            self.up_blocks.append(
                CausalUpsampleBlock(
                    mesh_device=mesh_device,
                    in_channels=ch[-i - 1],
                    out_channels=ch[-i - 2],
                    num_res_blocks=num_res_blocks[-i - 2],
                    attn_block=attn_block,
                    temporal_expansion=temporal_expansions[-i - 1],
                    spatial_expansion=spatial_expansions[-i - 1],
                    causal=causal,
                    padding_mode="replicate",
                    input_shape=input_shape,
                    torch_ref=upsample_block_torch_ref,
                )
            )

        last_block_torch_ref = torch_ref.blocks[1 + self.num_up_blocks]
        # Last set of residual blocks
        self.last_blocks = []
        for i in range(num_res_blocks[0]):
            self.last_blocks.append(
                ResBlock(
                    mesh_device=mesh_device,
                    causal=causal,
                    padding_mode="replicate",
                    input_shape=input_shape,
                    torch_ref=last_block_torch_ref[i],
                )
            )

        # Final output projection
        self.output_proj = Conv1x1(
            mesh_device=mesh_device,
            in_channels=ch[0],
            out_channels=out_channels,
            bias=True,
            torch_ref=torch_ref.output_proj,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def dealloc(self):
        self.input_proj.dealloc()
        for block in self.first_blocks:
            block.dealloc()
        for block in self.up_blocks:
            block.dealloc()
        for block in self.last_blocks:
            block.dealloc()
        self.output_proj.dealloc()

    def prepare_input(self, z_BCTHW):
        z_BTHWC = ttnn.from_torch(
            z_BCTHW.permute(0, 2, 3, 4, 1),
            device=self.mesh_device,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )
        return z_BTHWC

    def postprocess_output(self, x_NTHWC):
        x_NTHWC = ttnn.to_torch(x_NTHWC, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1))
        x_NCTHW = x_NTHWC.permute(0, 4, 1, 2, 3)
        return x_NCTHW

    def __call__(self, x_NTHWC):
        """
        Forward pass for the decoder.

        Args:
            x_NTHWC: Input tensor in NTHWC layout

        Returns:
            Output tensor in NTHWC layout
        """
        # Initial projection
        x_NTHWC = self.input_proj(x_NTHWC)

        # First set of residual blocks
        for block in self.first_blocks:
            x_res_NTHWC = block(x_NTHWC)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Upsampling blocks
        for block in self.up_blocks:
            x_res_NTHWC = block(x_NTHWC)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Last set of residual blocks
        for block in self.last_blocks:
            x_res_NTHWC = block(x_NTHWC)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Apply output nonlinearity if needed
        if self.output_nonlinearity == "silu":
            x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
            ttnn.deallocate(x_NTHWC)
            x_tile_NTHWC = ttnn.silu(x_tile_NTHWC, output_tensor=x_tile_NTHWC)  # in-place
            x_NTHWC = ttnn.to_layout(x_tile_NTHWC, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_tile_NTHWC)
        else:
            assert not self.output_nonlinearity  # StyleGAN3 omits the to-RGB nonlinearity.

        # Final projection
        x_NTHWC = self.output_proj(x_NTHWC)

        return x_NTHWC

    @classmethod
    def from_pretrained(cls, mesh_device, **kwargs):
        """
        Create a TtDecoder from pretrained weights.

        Args:
            mesh_device: TTNN mesh device
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            TtDecoder: Initialized decoder
        """
        state_dict = load_decoder_weights()
        if state_dict is None:
            logger.error("Failed to load decoder weights")
            return None

        return cls(mesh_device=mesh_device, state_dict=state_dict, **kwargs)
