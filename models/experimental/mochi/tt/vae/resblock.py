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

    def get_tensor_shapes(self, x):
        tensors = ttnn.get_device_tensors(x)
        return [t.shape for t in tensors]

    def reshape_tilize(self, x, shapes):
        tensors = ttnn.get_device_tensors(x)
        outputs = []
        for t, shape in zip(tensors, shapes):
            N, T, H, W, C = shape
            outputs.append(
                ttnn.tilize_with_zero_padding(
                    ttnn.reshape(t, [N * T, 1, H * W, C]),
                    use_multicore=True,
                )
            )
        return ttnn.aggregate_as_tensor(outputs)

    def untilize_reshape(self, x, shapes):
        tensors = ttnn.get_device_tensors(x)
        outputs = []
        for t, shape in zip(tensors, shapes):
            N, T, H, W, C = shape
            outputs.append(ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [N, T, H, W, C]))
        return ttnn.aggregate_as_tensor(outputs)

    def forward(self, x_NTHWC):
        shapes = self.get_tensor_shapes(x_NTHWC)

        x_tiled_NTHWC = self.reshape_tilize(x_NTHWC, shapes)
        residual_tiled_NTHWC = x_tiled_NTHWC
        ttnn.deallocate(x_NTHWC)

        x_norm_tiled_NTHWC = self.norm1(x_tiled_NTHWC)

        # TODO: Investigate packing more data into a tile
        x_norm_tiled_NTHWC = ttnn.silu(x_norm_tiled_NTHWC, output_tensor=x_norm_tiled_NTHWC)  # in-place
        x_NTHWC = self.untilize_reshape(x_norm_tiled_NTHWC, shapes)
        ttnn.deallocate(x_norm_tiled_NTHWC)

        x_conv1_NTHWC = self.conv1(x_NTHWC)
        ttnn.deallocate(x_NTHWC)
        x_conv1_tiled_NTHWC = self.reshape_tilize(x_conv1_NTHWC, shapes)
        ttnn.deallocate(x_conv1_NTHWC)

        x_tiled_NTHWC = self.norm2(x_conv1_tiled_NTHWC)
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
