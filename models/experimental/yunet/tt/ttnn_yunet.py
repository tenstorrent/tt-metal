# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN YUNet Face Detection Model.

Architecture:
- Backbone: 5 stages producing features at 3 scales [p3, p4, p5]
- Neck: FPN-style feature fusion with upsampling
- Head: Multi-scale detection heads (cls, box, obj, kpt)

Key optimizations:
- No Python dispatch overhead
- All operations are TTNN native
- Supports trace capture for maximum performance
"""

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    TtConv2d,
    TtMaxPool2d,
    HeightShardedStrategyConfiguration,
)


class TTNNDPUnit:
    """TTNN DPUnit: Pointwise 1x1 → Depthwise 3x3 with ReLU."""

    def __init__(self, device, in_channels: int, out_channels: int, name: str):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.conv1_weight = self.conv1_bias = None
        self.conv2_weight = self.conv2_bias = None
        self._conv1_cache = {}
        self._conv2_cache = {}

    def load_weights(self, conv1_weight, conv1_bias, conv2_weight, conv2_bias):
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias

    def _get_conv1(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv1_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                dilation=(1, 1),
                weight=self.conv1_weight,
                bias=self.conv1_bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
            )
            self._conv1_cache[key] = TtConv2d(config, self.device)
        return self._conv1_cache[key]

    def _get_conv2(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv2_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=self.out_channels,
                dilation=(1, 1),
                weight=self.conv2_weight,
                bias=self.conv2_bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
                activation=ttnn.UnaryOpType.RELU,
            )
            self._conv2_cache[key] = TtConv2d(config, self.device)
        return self._conv2_cache[key]

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_height, input_width, channels = x.shape
        conv1 = self._get_conv1(batch_size, input_height, input_width)
        x, h_w = conv1(x, return_output_dim=True)
        x = ttnn.reshape(x, [batch_size, h_w[0], h_w[1], -1], memory_config=ttnn.L1_MEMORY_CONFIG)
        conv2 = self._get_conv2(batch_size, h_w[0], h_w[1])
        x, h_w = conv2(x, return_output_dim=True)
        x = ttnn.reshape(x, [batch_size, h_w[0], h_w[1], -1], memory_config=ttnn.L1_MEMORY_CONFIG)
        return x


class TTNNConvBN:
    """TTNN Conv + BatchNorm (fused) with ReLU."""

    def __init__(self, device, in_ch: int, out_ch: int, kernel: int, stride: int, pad: int, name: str):
        self.device = device
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel, self.stride, self.pad = kernel, stride, pad
        self.name = name
        self.weight = self.bias = None
        self._conv_cache = {}

    def load_weights(self, weight, bias):
        self.weight, self.bias = weight, bias

    def _get_conv(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                batch_size=batch_size,
                kernel_size=(self.kernel, self.kernel),
                stride=(self.stride, self.stride),
                padding=(self.pad, self.pad),
                groups=1,
                dilation=(1, 1),
                weight=self.weight,
                bias=self.bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
                activation=ttnn.UnaryOpType.RELU,
            )
            self._conv_cache[key] = TtConv2d(config, self.device)
        return self._conv_cache[key]

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_height, input_width, channels = x.shape
        conv = self._get_conv(batch_size, input_height, input_width)
        x, h_w = conv(x, return_output_dim=True)
        x = ttnn.reshape(x, [batch_size, h_w[0], h_w[1], -1], memory_config=ttnn.L1_MEMORY_CONFIG)
        return x


class TTNNConv1x1:
    """TTNN 1x1 convolution for head outputs."""

    def __init__(self, device, in_ch: int, out_ch: int, name: str):
        self.device = device
        self.in_ch, self.out_ch = in_ch, out_ch
        self.name = name
        self.weight = self.bias = None
        self._conv_cache = {}

    def load_weights(self, weight, bias):
        self.weight, self.bias = weight, bias

    def _get_conv(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                dilation=(1, 1),
                weight=self.weight,
                bias=self.bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
            )
            self._conv_cache[key] = TtConv2d(config, self.device)
        return self._conv_cache[key]

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_height, input_width, channels = x.shape
        conv = self._get_conv(batch_size, input_height, input_width)
        x, h_w = conv(x, return_output_dim=True)
        x = ttnn.reshape(x, [batch_size, h_w[0], h_w[1], -1], memory_config=ttnn.L1_MEMORY_CONFIG)
        return x


class TTNNMaxPool:
    """TTNN MaxPool2d layer."""

    def __init__(self, device, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        self.device = device
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self._pool_cache = {}

    def _get_pool(self, batch_size: int, h: int, w: int, c: int):
        key = (batch_size, h, w, c)
        if key not in self._pool_cache:
            config = MaxPool2dConfiguration(
                input_height=h,
                input_width=w,
                channels=c,
                batch_size=batch_size,
                kernel_size=[self.kernel_size, self.kernel_size],
                stride=[self.stride, self.stride],
                padding=[self.padding, self.padding],
                dilation=[1, 1],
            )
            self._pool_cache[key] = TtMaxPool2d(config, self.device)
        return self._pool_cache[key]

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, h, w, c = x.shape
        pool = self._get_pool(batch_size, h, w, c)
        x = pool(x)
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        return ttnn.reshape(x, [batch_size, out_h, out_w, c], memory_config=ttnn.L1_MEMORY_CONFIG)


class TTNNUpsample:
    """TTNN Upsample (nearest neighbor 2x)."""

    def __init__(self, scale_factor: int = 2):
        self.scale_factor = scale_factor

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.upsample(x, scale_factor=self.scale_factor, mode="nearest")


class TtYUNet:
    """
    TTNN YUNet Face Detection Model.

    Input: [B, 320, 320, 3] NHWC format (bfloat16)
    Output: (cls, box, obj, kpt) - lists of 3 scale tensors each
    """

    def __init__(self, device, filters=(3, 16, 64, 64, 64, 64)):
        self.device = device
        self.filters = filters

        # Backbone
        self.p1_conv = TTNNConvBN(device, filters[0], filters[1], 3, 2, 1, "backbone.p1.0")
        self.p1_dpunit = TTNNDPUnit(device, filters[1], filters[1], "backbone.p1.1")

        self.p2_pool = TTNNMaxPool(device, 2, 2)
        self.p2_dpunit1 = TTNNDPUnit(device, filters[1], filters[2], "backbone.p2.1")
        self.p2_dpunit2 = TTNNDPUnit(device, filters[2], filters[2], "backbone.p2.2")
        self.p2_dpunit3 = TTNNDPUnit(device, filters[2], filters[2], "backbone.p2.3")

        self.p3_pool = TTNNMaxPool(device, 2, 2)
        self.p3_dpunit1 = TTNNDPUnit(device, filters[2], filters[3], "backbone.p3.1")
        self.p3_dpunit2 = TTNNDPUnit(device, filters[3], filters[3], "backbone.p3.2")

        self.p4_pool = TTNNMaxPool(device, 2, 2)
        self.p4_dpunit1 = TTNNDPUnit(device, filters[3], filters[4], "backbone.p4.1")
        self.p4_dpunit2 = TTNNDPUnit(device, filters[4], filters[4], "backbone.p4.2")

        self.p5_pool = TTNNMaxPool(device, 2, 2)
        self.p5_dpunit1 = TTNNDPUnit(device, filters[4], filters[5], "backbone.p5.1")
        self.p5_dpunit2 = TTNNDPUnit(device, filters[5], filters[5], "backbone.p5.2")

        # Neck
        self.up = TTNNUpsample(2)
        self.neck_conv1 = TTNNDPUnit(device, filters[5], filters[4], "neck.conv1")
        self.neck_conv2 = TTNNDPUnit(device, filters[4], filters[3], "neck.conv2")
        self.neck_conv3 = TTNNDPUnit(device, filters[3], filters[3], "neck.conv3")

        # Head (3 scales)
        self.head_m, self.head_cls, self.head_box, self.head_obj, self.head_kpt = [], [], [], [], []
        head_filters = (filters[3], filters[3], filters[4])
        for i, f in enumerate(head_filters):
            self.head_m.append(TTNNDPUnit(device, f, f, f"head.m.{i}"))
            self.head_cls.append(TTNNConv1x1(device, f, 1, f"head.cls.{i}"))
            self.head_box.append(TTNNConv1x1(device, f, 4, f"head.box.{i}"))
            self.head_obj.append(TTNNConv1x1(device, f, 1, f"head.obj.{i}"))
            self.head_kpt.append(TTNNConv1x1(device, f, 10, f"head.kpt.{i}"))

    def load_weights_from_torch(self, torch_model):
        """Load weights from fused PyTorch YUNet model."""

        def to_ttnn(w, b):
            return Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(w, b)

        def load_dpunit(ttnn_dp, torch_dp):
            w1, b1 = to_ttnn(torch_dp.conv1.weight, torch_dp.conv1.bias)
            w2, b2 = to_ttnn(torch_dp.conv2.conv.weight, torch_dp.conv2.conv.bias)
            ttnn_dp.load_weights(w1, b1, w2, b2)

        # Backbone
        w, b = to_ttnn(torch_model.backbone.p1[0].conv.weight, torch_model.backbone.p1[0].conv.bias)
        self.p1_conv.load_weights(w, b)
        load_dpunit(self.p1_dpunit, torch_model.backbone.p1[1])

        load_dpunit(self.p2_dpunit1, torch_model.backbone.p2[1])
        load_dpunit(self.p2_dpunit2, torch_model.backbone.p2[2])
        load_dpunit(self.p2_dpunit3, torch_model.backbone.p2[3])

        load_dpunit(self.p3_dpunit1, torch_model.backbone.p3[1])
        load_dpunit(self.p3_dpunit2, torch_model.backbone.p3[2])

        load_dpunit(self.p4_dpunit1, torch_model.backbone.p4[1])
        load_dpunit(self.p4_dpunit2, torch_model.backbone.p4[2])

        load_dpunit(self.p5_dpunit1, torch_model.backbone.p5[1])
        load_dpunit(self.p5_dpunit2, torch_model.backbone.p5[2])

        # Neck
        load_dpunit(self.neck_conv1, torch_model.neck.conv1)
        load_dpunit(self.neck_conv2, torch_model.neck.conv2)
        load_dpunit(self.neck_conv3, torch_model.neck.conv3)

        # Head
        for i in range(3):
            load_dpunit(self.head_m[i], torch_model.head.m[i])
            for head_list, torch_head in [
                (self.head_cls, torch_model.head.cls),
                (self.head_box, torch_model.head.box),
                (self.head_obj, torch_model.head.obj),
                (self.head_kpt, torch_model.head.kpt),
            ]:
                w, b = to_ttnn(torch_head[i].weight, torch_head[i].bias)
                head_list[i].load_weights(w, b)

    def __call__(self, x: ttnn.Tensor):
        """Forward pass. Returns (cls, box, obj, kpt) outputs."""
        # Backbone
        p1 = self.p1_dpunit(self.p1_conv(x))
        p2 = self.p2_dpunit3(self.p2_dpunit2(self.p2_dpunit1(self.p2_pool(p1))))
        p3 = self.p3_dpunit2(self.p3_dpunit1(self.p3_pool(p2)))
        p4 = self.p4_dpunit2(self.p4_dpunit1(self.p4_pool(p3)))
        p5 = self.p5_dpunit2(self.p5_dpunit1(self.p5_pool(p4)))

        # Neck
        p5_out = self.neck_conv1(p5)
        p4 = ttnn.add(p4, self.up(p5_out))
        p4_out = self.neck_conv2(p4)
        p3 = ttnn.add(p3, self.up(p4_out))
        p3_out = self.neck_conv3(p3)

        features = [p3_out, p4_out, p5_out]

        # Head
        cls_out, box_out, obj_out, kpt_out = [], [], [], []
        for i, feat in enumerate(features):
            feat = self.head_m[i](feat)
            cls_out.append(self.head_cls[i](feat))
            box_out.append(self.head_box[i](feat))
            obj_out.append(self.head_obj[i](feat))
            kpt_out.append(self.head_kpt[i](feat))

        return cls_out, box_out, obj_out, kpt_out


def create_yunet_model(device, torch_model=None) -> TtYUNet:
    """Factory function to create TtYUNet model."""
    model = TtYUNet(device)
    if torch_model is not None:
        model.load_weights_from_torch(torch_model)
    return model
