"""Linear layer implementations for TTNN."""

import torch
from torch import nn

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d
from models.tt_symbiote.core.module import TTNNModule
from models.tt_symbiote.modules.tensor import TTNNPermute


class TTNNConv2dNHWC(TTNNModule):
    """TTNN-accelerated linear layer."""

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
        new_conv.torch_weight = conv.weight
        new_conv.torch_bias = conv.bias
        new_conv._fallback_torch_layer = conv
        return new_conv

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            self.torch_weight = self._fallback_torch_layer.weight
            self.torch_bias = self._fallback_torch_layer.bias
        torch_weight, torch_bias = self.torch_weight, self.torch_bias
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            torch_weight, torch_bias
        )

    def move_weights_to_host_impl(self):
        """Move weights back to host."""
        self.tt_weight = self.tt_weight.cpu()
        if self.tt_bias is not None:
            self.tt_bias = self.tt_bias.cpu()

    def deallocate_weights_impl(self):
        """Deallocate weights from device."""
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)

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
            out = ttnn.reshape(
                out,
                [batch_size, h_w[0], h_w[1], -1],
            )
            return out
        return layer(input_tensor)


def fold_batch_norm2d_into_conv2d(weight, scale, shift, running_mean, running_var, eps):
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    bias = torch.reshape(bias, (-1,))
    return weight, bias


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
        self.conv1 = TTNNConv2dNHWC.from_torch(self.torch_layer.conv1)
        assert isinstance(self.torch_layer.bn1, nn.BatchNorm2d), "Only BatchNorm2d is supported in Bottleneck blocks."
        self.batch_norm_args1 = {
            "scale": self.torch_layer.bn1.weight,
            "shift": self.torch_layer.bn1.bias,
            "running_mean": self.torch_layer.bn1.running_mean,
            "running_var": self.torch_layer.bn1.running_var,
            "eps": self.torch_layer.bn1.eps,
        }
        self.conv2 = TTNNConv2dNHWC.from_torch(self.torch_layer.conv2)
        self.batch_norm_args2 = {
            "scale": self.torch_layer.bn2.weight,
            "shift": self.torch_layer.bn2.bias,
            "running_mean": self.torch_layer.bn2.running_mean,
            "running_var": self.torch_layer.bn2.running_var,
            "eps": self.torch_layer.bn2.eps,
        }
        self.conv3 = TTNNConv2dNHWC.from_torch(self.torch_layer.conv3)
        self.batch_norm_args3 = {
            "scale": self.torch_layer.bn3.weight,
            "shift": self.torch_layer.bn3.bias,
            "running_mean": self.torch_layer.bn3.running_mean,
            "running_var": self.torch_layer.bn3.running_var,
            "eps": self.torch_layer.bn3.eps,
        }
        self.relu = nn.ReLU()
        self.nchw_to_nhwc_permute = TTNNPermute(perm=[0, 2, 3, 1])
        self.nhwc_to_nchw_permute = TTNNPermute(perm=[0, 3, 1, 2])

    @classmethod
    def from_torch(cls, bottleneck: "torchvision.models.resnet.Bottleneck") -> "TTNNBottleneck":
        """Create TTNNBottleneck from PyTorch Bottleneck layer."""
        new_bottleneck = TTNNBottleneck(
            downsample=bottleneck.downsample,
        )
        new_bottleneck._fallback_torch_layer = bottleneck
        new_bottleneck.initilize_submodules()
        return new_bottleneck

    def to_device(self, device: str):
        """Set device for the module and its submodules."""
        super().to_device(device)
        self.conv1.to_device(device)
        self.conv2.to_device(device)
        self.conv3.to_device(device)
        self.nhwc_to_nchw_permute.to_device(device)
        self.nchw_to_nhwc_permute.to_device(device)

    def preprocess_weights_impl(self):
        """Preprocess attention weights for TTNN."""
        assert (
            self.torch_layer is not None
        ), "Torch layer must be set before preprocessing weights. This layer can only be created from a torch layer (e.g. from_torch method)."

        if self.batch_norm_args1 is not None:
            self.conv1.torch_weight, self.conv1.torch_bias = fold_batch_norm2d_into_conv2d(
                self.conv1.torch_weight,
                self.batch_norm_args1["scale"],
                self.batch_norm_args1["shift"],
                self.batch_norm_args1["running_mean"],
                self.batch_norm_args1["running_var"],
                self.batch_norm_args1["eps"],
            )
        self.conv1.preprocess_weights()
        if self.batch_norm_args2 is not None:
            self.conv2.torch_weight, self.conv2.torch_bias = fold_batch_norm2d_into_conv2d(
                self.conv2.torch_weight,
                self.batch_norm_args2["scale"],
                self.batch_norm_args2["shift"],
                self.batch_norm_args2["running_mean"],
                self.batch_norm_args2["running_var"],
                self.batch_norm_args2["eps"],
            )
        self.conv2.preprocess_weights()
        if self.batch_norm_args3 is not None:
            self.conv3.torch_weight, self.conv3.torch_bias = fold_batch_norm2d_into_conv2d(
                self.conv3.torch_weight,
                self.batch_norm_args3["scale"],
                self.batch_norm_args3["shift"],
                self.batch_norm_args3["running_mean"],
                self.batch_norm_args3["running_var"],
                self.batch_norm_args3["eps"],
            )
        self.conv3.preprocess_weights()

    def move_weights_to_device_impl(self):
        """Move attention weights to TTNN device."""
        assert self.device is not None, "Device must be set before moving weights to device."
        self.conv1.move_weights_to_device()
        self.conv2.move_weights_to_device()
        self.conv3.move_weights_to_device()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Bottleneck block."""
        if self.downsample is not None:
            identity = x
            x = self.nchw_to_nhwc_permute(x)
        else:
            x = self.nchw_to_nhwc_permute(x)
            identity = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            from models.tt_symbiote.core.tensor import TorchTTNNTensor

            identity = self.downsample(TorchTTNNTensor(identity))
            if identity.to_ttnn.device() != out.to_ttnn.device():
                identity = ttnn.to_device(identity.to_ttnn, out.to_ttnn.device())
            out = self.nhwc_to_nchw_permute(out)
            out += identity
        else:
            out += identity
        out = self.relu(out)
        if self.downsample is None:
            out = self.nhwc_to_nchw_permute(out)
        return out
