import torch
import tt_lib

from loguru import logger
from tt_lib.fallback_ops import fallback_ops
from typing import Optional, Sequence, Tuple, Union

from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.conv_on_device_utils_new import (
    run_conv_on_tt_device,
    run_conv_on_device_wrapper,
    is_conv_supported_on_device,
)



class TtEfficientnetConv2dNormActivation(torch.nn.Module):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (bool): True if to use activation (Silu), false othervise.
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        state_dict,
        base_address,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        activation_layer: bool = True,
        dilation: Union[int, Tuple[int, ...]] = 1,
        conv_on_device=False,
    ):

        super().__init__()

        self.conv_weight = state_dict[f"{base_address}.0.weight"]
        bias_key = f"{base_address}.0.bias"

        if bias_key in state_dict:
            self.conv_bias = state_dict[bias_key]
        else:
            self.conv_bias = None

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))

        # self.conv =
        #     torch.nn.Conv2d(
        #         in_channels,
        #         out_channels,
        #         kernel_size,
        #         stride,
        #         padding,
        #         dilation=dilation,
        #         groups=groups,
        #         bias=bias,)

        self.device = device
        self.conv_on_device = conv_on_device

        # conv_params = [out_channels, in_channels, kernel_size, kernel_size, stride, stride, padding, padding, dilation, groups]
        self.conv_params = [out_channels, in_channels, kernel_size, kernel_size, stride, stride, padding, padding, dilation, groups]

        if self.conv_on_device and is_conv_supported_on_device(self.conv_params):
            logger.debug(f"Using TtConv for params {self.conv_params}")

            self.conv = run_conv_on_device_wrapper(
                self.conv_weight.reshape(-1).tolist(),
                self.conv_params,
                self.device,
                self.host,
                conv_bias=None,
            )

        else:
            self.conv_on_device = False
            logger.debug(f"Using fallback_ops.Conv2d for params {self.conv_params}")

            # self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, dilation=d, bias=False)
            self.conv = fallback_ops.Conv2d(
                weights=self.conv_weight,
                biases=self.conv_bias,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=self.conv_bias is not None,
            )

        self.bnorm = torch.nn.BatchNorm2d(out_channels)
        self.bnorm.weight = torch.nn.Parameter(state_dict[f"{base_address}.1.weight"])
        self.bnorm.bias = torch.nn.Parameter(state_dict[f"{base_address}.1.bias"])
        #self.bnorm.running_mean = torch.nn.Parameter(state_dict[f"{base_address}.1.running_mean"])
        #self.bnorm.running_var = torch.nn.Parameter(state_dict[f"{base_address}.1.running_var"])
        self.bnorm.num_batches_tracked = state_dict[f"{base_address}.1.num_batches_tracked"]

        #self.bnorm.running_mean.requires_grad = False
        #self.bnorm.running_var.requires_grad = False

        # print(f"self.bnorm out_channels {out_channels}")
        # print(f"self.bnorm eps {self.bnorm.eps}")
        # print(f"self.bnorm momentum {self.bnorm.momentum}")
        # print(f"self.bnorm affine {self.bnorm.affine}")
        # print(f"self.bnorm track_running_stats {self.bnorm.track_running_stats}")

        self.activation_layer = activation_layer

    def forward(self, x):
        x = self.conv(x)

        x = tt2torch_tensor(x)
        x = self.bnorm(x)
        x = torch2tt_tensor(x, self.device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

        if self.activation_layer is True:
            x = fallback_ops.silu(x)

        return x
