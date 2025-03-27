import ttnn
from .groupnorm_torch import GroupNorm as GroupNormTorch
from .groupnorm import GroupNorm
from .conv3d import ContextParallelConv3d
from models.common.lightweightmodule import LightweightModule


class ResBlock(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        state_dict_prefix: str,
        channels: int,
        affine: bool = True,
        attn_block=None,
        causal: bool = True,
        prune_bottleneck: bool = False,
        padding_mode: str = "replicate",
        bias: bool = True,
    ):
        assert attn_block is None
        assert causal
        assert not prune_bottleneck

        self.channels = channels

        # TODO: Figure out how these are actually named in the Sequential module

        self.norm1 = GroupNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}stack.0.",
            num_groups=32,
            channels=channels,
            affine=affine,
        )

        self.conv1 = ContextParallelConv3d(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}stack.2.",
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
        )

        self.norm2 = GroupNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}stack.3.",
            num_groups=32,
            channels=channels,
            affine=affine,
        )

        self.conv2 = ContextParallelConv3d(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}stack.5.",
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
        )

    def forward(self, x_NTHWC):
        residual_NTHWC = x_NTHWC

        x_NTHWC = self.norm1(x_NTHWC)

        # Note that SILU requires TILE_LAYOUT, so we convert to and from.
        x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_NTHWC)

        # TODO: Investigate packing more data into a tile
        x_tile_NTHWC = ttnn.silu(x_tile_NTHWC, output_tensor=x_tile_NTHWC)  # in-place
        x_NTHWC = ttnn.to_layout(x_tile_NTHWC, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tile_NTHWC)

        x_conv1_NTHWC = self.conv1(x_NTHWC)
        ttnn.deallocate(x_NTHWC)

        x_NTHWC = self.norm2(x_conv1_NTHWC)
        ttnn.deallocate(x_conv1_NTHWC)

        x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_NTHWC)
        # TODO: Investigate packing more data into a tile
        x_tile_NTHWC = ttnn.silu(x_tile_NTHWC, output_tensor=x_tile_NTHWC)  # in-place
        x_NTHWC = ttnn.to_layout(x_tile_NTHWC, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tile_NTHWC)

        x_conv2_NTHWC = self.conv2(x_NTHWC)
        ttnn.deallocate(x_NTHWC)

        x_NTHWC = ttnn.to_layout(
            ttnn.add(ttnn.to_layout(x_conv2_NTHWC, ttnn.TILE_LAYOUT), ttnn.to_layout(residual_NTHWC, ttnn.TILE_LAYOUT)),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        return x_NTHWC
