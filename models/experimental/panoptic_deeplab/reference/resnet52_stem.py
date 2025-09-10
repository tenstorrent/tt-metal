# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


class DeepLabStem(CNNBlockBase):
    """
    DeepLab ResNet stem module that processes the input image before the first residual block.

    This stem consists of three 3x3 convolution layers with intermediate Batch Normalization and ReLU,
    followed by a 3x3 max pooling operation. The design reduces spatial resolution while increasing
    the number of channels to prepare features for deeper layers.

    Attributes:
        conv1 (nn.Conv2d): First convolution layer with stride 2 and out_channels // 2 filters.
        bn1 (nn.BatchNorm2d): Batch normalization applied after conv1.
        conv2 (nn.Conv2d): Second convolution layer maintaining channel size.
        bn2 (nn.BatchNorm2d): Batch normalization applied after conv2.
        conv3 (nn.Conv2d): Third convolution layer increasing to final out_channels.
        bn3 (nn.BatchNorm2d): Batch normalization applied after conv3.
        relu (nn.ReLU): In-place ReLU activation function.
        maxpool (nn.MaxPool2d): 3x3 max pooling with stride 2 and padding 1.

    Args:
        in_channels (int): Number of input channels. Default is 3 for RGB images.
        out_channels (int): Number of output channels. Default is 128.
        stride (int): Stride used for conv2 and conv3. Default is 1.

    Returns:
        torch.Tensor: Output feature map after the stem block with reduced spatial resolution
        and increased channel depth.
    """

    def __init__(self, in_channels=3, out_channels=128, stride=1):
        """
        Initialize the DeepLabStem module.

        Args:
            in_channels (int): Number of input channels. Default is 3 (e.g., RGB image).
            out_channels (int): Number of output channels after the stem. Default is 128.
            stride (int): Stride value for the second and third convolutions. Default is 1.
        """
        super().__init__(in_channels, out_channels, 1)
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
