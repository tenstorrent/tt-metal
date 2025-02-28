import ttnn
import torch

from models.common.lightweightmodule import LightweightModule


from typing import Union, Tuple
from .common import prepare_conv3d_weights, get_conv3d_config

from loguru import logger


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class TtContextParallelConv3d(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        state_dict_prefix: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        causal: bool = True,
        context_parallel: bool = True,
        groups: int = 1,
        **kwargs,
    ):
        assert causal
        assert context_parallel
        self.mesh_device = mesh_device
        self.grid_size = mesh_device.compute_with_storage_grid_size()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.context_parallel = context_parallel

        # Conv3d parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.has_bias = kwargs["bias"]
        self.padding_mode = kwargs["padding_mode"]
        assert self.padding_mode in ["zeros", "replicate"]
        assert groups == 1
        self.groups = groups

        # Calculate padding
        height_pad = (self.kernel_size[1] - 1) // 2
        width_pad = (self.kernel_size[2] - 1) // 2
        self.padding = (0, height_pad, width_pad)

        # Load weights and bias
        self.torch_weight = state_dict[f"{state_dict_prefix}weight"]
        self.torch_bias = state_dict[f"{state_dict_prefix}bias"] if self.has_bias else None

        self.weight = None
        self.bias = None

        self.conv_config = None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x_NTHWC):
        logger.warning("TtContextParallelConv3d does not support T parallelism")

        """
        On the first run, use the input shape to determine the conv3d config
        and preprocess the weight.
        """
        if self.weight is None:
            logger.warning("Setting up conv3d config")
            self.cached_input_shape = x_NTHWC.shape
            conv_config = get_conv3d_config(
                x_NTHWC.shape,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.padding_mode,
                self.grid_size,
            )
            self.conv_config = conv_config
            self.weight, self.bias = prepare_conv3d_weights(
                self.mesh_device, self.torch_weight, self.torch_bias, conv_config
            )
        else:
            assert (
                self.cached_input_shape == x_NTHWC.shape
            ), f"Input shape mismatch. Expected {self.cached_input_shape} but got {x_NTHWC.shape}"

        N, T, H, W, C = x_NTHWC.shape
        # Compute temporal padding amounts
        context_size = self.kernel_size[0] - 1
        if self.causal:
            pad_front = context_size
            pad_back = 0
        else:
            pad_front = context_size // 2
            pad_back = context_size - pad_front

        if self.padding_mode == "zeros":
            # Do front and back padding
            x_pad_NTHWC = ttnn.pad(x_NTHWC, (0, 0, 0, 0, pad_front, pad_back), value=0.0)
        elif self.padding_mode == "replicate":
            # Use concat to pad
            front_slice = x_NTHWC[:, 0:1, :, :, :]
            back_slice = x_NTHWC[:, -1:, :, :, :]
            x_pad_NTHWC = ttnn.concat(
                [front_slice] * pad_front + [x_NTHWC] + [back_slice] * pad_back,
                dim=1,
            )
            ttnn.deallocate(x_NTHWC)
            T_pad = x_pad_NTHWC.shape[1]
            assert T_pad == T + pad_front + pad_back

        out_NTHWC = ttnn.conv3d(
            input_tensor=x_pad_NTHWC,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            config=self.conv_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        return out_NTHWC
