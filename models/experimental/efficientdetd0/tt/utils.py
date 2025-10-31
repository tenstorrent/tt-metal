# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

# from models.experimental.efficientnetb0.tt.efficientnetb0 import Conv2dDynamicSamePadding


class BatchNorm2d:
    def __init__(self, parameters, device, cache={}, memory_config=None, compute_kernel_config=None):
        self.device = device
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.running_mean = parameters.running_mean
        self.running_var = parameters.running_var
        self.eps = parameters.eps
        self.cache = cache
        self.memory_config = memory_config
        self.compute_kernel_config = compute_kernel_config

    def __call__(self, x):
        return ttnn.batch_norm(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            training=False,
            eps=self.eps,
            weight=self.weight,
            bias=self.bias,
            output=None,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )


class EfficientDetd0Maxpool2D:
    def __init__(
        self,
        maxpool_params,
        device,
        cache={},
        dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=ttnn.TILE_LAYOUT,
        dilation=1,
        ceil_mode=False,
        memory_config=None,
        in_place_halo=False,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch_size = maxpool_params.batch_size
        self.input_height = maxpool_params.input_height
        self.input_width = maxpool_params.input_width
        self.channels = maxpool_params.channels
        self.kernel_size = maxpool_params.kernel_size
        self.padding = maxpool_params.padding
        self.stride = maxpool_params.stride
        self.ceil_mode = maxpool_params.ceil_mode
        # self.ceil_mode = ceil_mode
        self.deallocate_activation = deallocate_activation
        self.cache = cache
        self.dtype = dtype
        self.memory_config = (memory_config,)
        self.in_place_halo = (in_place_halo,)
        self.shard_layout = shard_layout
        self.output_layout = output_layout
        self.dilation = dilation

    def __call__(self, x):
        return ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=self.input_height,
            input_w=self.input_width,
            channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=(self.dilation, self.dilation),
            ceil_mode=self.ceil_mode,
            memory_config=self.memory_config,
            applied_shard_scheme=self.shard_layout,
            in_place_halo=self.in_place_halo,
            deallocate_input=self.deallocate_activation,
            reallocate_halo_output=True,
            dtype=self.dtype,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )


class MaxPool2dDynamicSamePadding:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        maxpool_params,
        batch=1,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch = batch
        self.parameters = parameters
        self.in_channels = maxpool_params.in_channels
        self.kernel_size = maxpool_params.kernel_size
        self.stride = maxpool_params.stride
        self.dilation = maxpool_params.dilation
        self.groups = maxpool_params.groups
        self.input_height = maxpool_params.input_height
        self.input_width = maxpool_params.input_width
        self.shard_layout = shard_layout
        ih, iw = self.input_height, self.input_width
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if self.pad_h > 0 or self.pad_w > 0:
            pad_offset_width = self.pad_w // 2 + self.pad_w - self.pad_w // 2
            pad_offset_height = self.pad_h // 2 + self.pad_h - self.pad_h // 2
            if pad_offset_width % 2 == 0 and pad_offset_height % 2 == 0:
                maxpool_params.padding = (pad_offset_height // 2, pad_offset_width // 2)
            else:
                pad_top = pad_offset_height // 2
                pad_bottom = pad_top + pad_offset_height % 2
                pad_left = pad_offset_width // 2
                pad_right = pad_left + pad_offset_width % 2
                maxpool_params.padding = (pad_top, pad_bottom, pad_left, pad_right)

        self.dynamic_conv = EfficientDetd0Maxpool2D(
            maxpool_params,
            device=device,
            shard_layout=self.shard_layout,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, x):
        return self.dynamic_conv(x)


class EfficientNetb0Conv2D:
    def __init__(
        self,
        parameters,
        conv,
        device,
        cache={},
        activation=None,
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        groups=1,
        output_layout=ttnn.TILE_LAYOUT,
        dilation=1,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch_size = 1
        self.conv_params = conv
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.deallocate_activation = deallocate_activation
        self.cache = cache
        self.parameters = parameters
        self.shard_layout = shard_layout
        self.output_layout = output_layout
        self.shard_layout = shard_layout
        self.dilation = dilation
        self.weights_dtype = weights_dtype
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters["weight"], self.parameters["bias"]

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True if self.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED else False,
            reshard_if_not_optimal=True,
        )

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x):
        [x, [out_h, out_w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=ttnn.bfloat16,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )

        return x


class Conv2dDynamicSamePadding:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        conv_params,
        batch=1,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch = batch
        self.parameters = parameters
        self.in_channels = conv_params.in_channels
        self.kernel_size = conv_params.kernel_size
        self.stride = conv_params.stride
        self.dilation = conv_params.dilation
        self.groups = conv_params.groups
        self.input_height = conv_params.input_height
        self.input_width = conv_params.input_width
        self.shard_layout = shard_layout
        ih, iw = self.input_height, self.input_width
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        self.pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        self.pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if self.pad_h > 0 or self.pad_w > 0:
            pad_offset_width = self.pad_w // 2 + self.pad_w - self.pad_w // 2
            pad_offset_height = self.pad_h // 2 + self.pad_h - self.pad_h // 2
            if pad_offset_width % 2 == 0 and pad_offset_height % 2 == 0:
                conv_params.padding = (pad_offset_height // 2, pad_offset_width // 2)
            else:
                pad_top = pad_offset_height // 2
                pad_bottom = pad_top + pad_offset_height % 2
                pad_left = pad_offset_width // 2
                pad_right = pad_left + pad_offset_width % 2
                conv_params.padding = (pad_top, pad_bottom, pad_left, pad_right)

        self.dynamic_conv = EfficientNetb0Conv2D(
            parameters,
            conv_params,
            device=device,
            shard_layout=self.shard_layout,
            deallocate_activation=deallocate_activation,
        )

        self.parameters_conv = conv_params

    def __call__(self, x):
        if self.pad_h > 0 or self.pad_w > 0:
            padded_shape = [self.batch, self.parameters_conv.input_height, self.parameters_conv.input_width, x.shape[3]]
            input_height = int(math.sqrt((x.shape[2] // self.batch)))
            input_width = int(math.sqrt((x.shape[2] // self.batch)))

        return self.dynamic_conv(x)


class SeparableConvBlock:
    def __init__(
        self,
        device,
        parameters,
        shard_layout,
        conv_params,
        activation=False,
        batch=1,
        deallocate_activation=False,
    ):
        self.activation = activation
        self.depthwise_conv = Conv2dDynamicSamePadding(
            device,
            parameters.depthwise_conv,
            shard_layout,
            conv_params.depthwise_conv,
            batch=batch,
            deallocate_activation=deallocate_activation,
        )
        self.pointwise_conv = Conv2dDynamicSamePadding(
            device,
            parameters.pointwise_conv,
            shard_layout,
            conv_params.pointwise_conv,
            batch=batch,
            deallocate_activation=True,
        )

    def __call__(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.activation:
            x = x * ttnn.sigmoid_accurate(x, True)

        return x
