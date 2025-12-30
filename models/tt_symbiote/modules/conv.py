"""Linear layer implementations for TTNN."""

import torch
from torch import nn

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d
from models.tt_symbiote.core.module import TTNNModule
from models.tt_symbiote.modules.activation import TTNNReLU
from models.tt_symbiote.modules.tensor import TTNNPermute, TTNNReshape


def fold_batch_norm2d_into_conv2d(weight, bias, scale, shift, running_mean, running_var, eps):
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    return weight, bias


class NHWCConvPytorch(nn.Module):
    """A wrapper around nn.Conv2d to handle NHWC input/output."""

    def __init__(
        self,
        conv: nn.Conv2d,
    ) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class NHWCConvBNPytorch(nn.Module):
    """A wrapper around nn.Conv2d to handle NHWC input/output."""

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class NHWCConvBNActivationPytorch(nn.Module):
    """A wrapper around nn.Conv2d to handle NHWC input/output."""

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class TTNNConv2dNHWC(TTNNModule):
    """TTNN-accelerated Conv layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups: int = 1,
        slice_config=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.slice_config = slice_config
        self.reshape = TTNNReshape()

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, slice_config=None) -> "TTNNConv2d":
        """Create TTNNConv2d from PyTorch Conv2d layer."""
        new_conv = TTNNConv2dNHWC(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvPytorch(conv)
        return new_conv

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvPytorch(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            self.torch_layer.conv.weight, self.torch_layer.conv.bias
        )
        super().preprocess_weights_impl()

    def move_weights_to_host_impl(self):
        """Move weights back to host."""
        self.tt_weight = self.tt_weight.cpu()
        if self.tt_bias is not None:
            self.tt_bias = self.tt_bias.cpu()
        super().move_weights_to_host_impl()

    def deallocate_weights_impl(self):
        """Deallocate weights from device."""
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        batch_size, input_height, input_width, _ = input_tensor.shape
        config = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
            weight=self.tt_weight,
            bias=self.tt_bias,
            slice_strategy=self.slice_config,
        )
        layer = TtConv2d(config, input_tensor.device())
        if reshape_output:
            out, h_w = layer(input_tensor, return_output_dim=reshape_output)
            out = self.reshape(out, [batch_size, h_w[0], h_w[1], -1])
            return out
        return layer(input_tensor)


class TTNNConv2dBNNHWC(TTNNConv2dNHWC):
    """TTNN-accelerated Fused Conv BN layer."""

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, bn: nn.BatchNorm2d, slice_config=None) -> "TTNNConv2d":
        """Create TTNNConv2d from PyTorch Conv2d layer."""
        new_conv = TTNNConv2dBNNHWC(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvBNPytorch(conv, bn)
        return new_conv

    def _preprocess_weights_local(self):
        torch_weight, torch_bias = fold_batch_norm2d_into_conv2d(
            self.torch_layer.conv.weight,
            self.torch_layer.conv.bias,
            self.torch_layer.bn.weight,
            self.torch_layer.bn.bias,
            self.torch_layer.bn.running_mean,
            self.torch_layer.bn.running_var,
            self.torch_layer.bn.eps,
        )
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            torch_weight, torch_bias
        )

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvBNPytorch(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                ),
                nn.BatchNorm2d(self.out_channels),
            )
        self._preprocess_weights_local()
        # call method from TTNNConv2dNHWC's grandparent TTNNModule
        TTNNModule.preprocess_weights_impl(self)


class TTNNConv2dBNActivationNHWC(TTNNConv2dBNNHWC):
    """TTNN-accelerated Fused Conv BN layer."""

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, bn: nn.BatchNorm2d, activation, slice_config=None) -> "TTNNConv2d":
        """Create TTNNConv2d from PyTorch Conv2d layer."""
        new_conv = TTNNConv2dBNActivationNHWC(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvBNActivationPytorch(conv, bn, nn.ReLU())
        assert isinstance(activation, nn.ReLU), "Only ReLU activation is supported in TTNNConv2dBNActivationNHWC."
        return new_conv

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvBNPytorch(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(),
            )
        self._preprocess_weights_local()
        # call method from TTNNConv2dNHWC's grandparent TTNNModule
        TTNNModule.preprocess_weights_impl(self)

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        batch_size, input_height, input_width, _ = input_tensor.shape
        config = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
            weight=self.tt_weight,
            bias=self.tt_bias,
            slice_strategy=self.slice_config,
            activation=ttnn.UnaryOpType.RELU,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
        )
        layer = TtConv2d(config, input_tensor.device())
        if reshape_output:
            out, h_w = layer(input_tensor, return_output_dim=reshape_output)
            out = self.reshape(out, [batch_size, h_w[0], h_w[1], -1])
            return out
        return layer(input_tensor)


class TTNNBottleneck(TTNNModule):
    """TTNN-accelerated ResNet Bottleneck block."""

    def __init__(
        self,
        downsample=None,
    ) -> None:
        super().__init__()
        self.downsample = downsample

    def initilize_submodules(self):
        assert (
            self._fallback_torch_layer is not None
        ), "Fallback torch layer must be set before initializing submodules."
        assert isinstance(self.torch_layer.bn1, nn.BatchNorm2d), "Only BatchNorm2d is supported in Bottleneck blocks."
        self.conv1 = TTNNConv2dBNActivationNHWC.from_torch(
            self.torch_layer.conv1, self.torch_layer.bn1, self.torch_layer.relu
        )
        self.conv2 = TTNNConv2dBNActivationNHWC.from_torch(
            self.torch_layer.conv2, self.torch_layer.bn2, self.torch_layer.relu
        )
        self.conv3 = TTNNConv2dBNNHWC.from_torch(self.torch_layer.conv3, self.torch_layer.bn3)
        self.relu = TTNNReLU()
        self.permute = TTNNPermute()

    @classmethod
    def from_torch(cls, bottleneck: "torchvision.models.resnet.Bottleneck") -> "TTNNBottleneck":
        """Create TTNNBottleneck from PyTorch Bottleneck layer."""
        new_bottleneck = TTNNBottleneck(
            downsample=bottleneck.downsample,
        )
        new_bottleneck._fallback_torch_layer = bottleneck
        new_bottleneck.initilize_submodules()
        return new_bottleneck

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Bottleneck block."""
        if self.downsample is not None:
            identity = x
            x = self.permute(x, perm=[0, 2, 3, 1])
        else:
            x = self.permute(x, perm=[0, 2, 3, 1])
            identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            from models.tt_symbiote.core.tensor import TorchTTNNTensor

            identity = self.downsample(TorchTTNNTensor(identity, dtype=torch.bfloat16))
            if identity.to_ttnn.device() != out.to_ttnn.device():
                identity = ttnn.to_device(identity.to_ttnn, out.to_ttnn.device())

            identity = self.permute(identity, perm=[0, 2, 3, 1])
        out = out + identity
        out = self.relu(out)
        out = self.permute(out, perm=[0, 3, 1, 2])
        return out
