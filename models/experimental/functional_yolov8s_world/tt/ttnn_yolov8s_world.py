# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math


class ttnn_Conv:
    def __init__(
        self,
        device,
        parameters,
        input_params,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=None,
        bfloat8=False,
        change_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        is_fused=True,
        is_dfl=False,
        is_detect_cv2=False,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=False,
        cache={},
        batch_size=1,
    ):
        self.device = device
        self.parameters = parameters
        self.input_params = input_params
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.bfloat8 = bfloat8
        self.change_shard = change_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.is_fused = is_fused
        self.is_dfl = is_dfl
        self.is_detect_cv2 = is_detect_cv2
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.cache = cache
        self.batch_size = batch_size

        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters["conv"]["weight"], self.parameters["conv"]["bias"]

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation="",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            input_channels_alignment=16 if self.input_params[4] < 16 else 32,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=False,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
        )

        if self.deallocate_activation:
            conv_config.deallocate_activation = self.deallocate_activation

        if self.change_shard:
            conv_config.shard_layout = None

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        if self.width_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        if self.bfloat8:
            conv_config.weights_dtype = ttnn.bfloat8_b

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt(x.shape[2]) // self.batch_size)
            input_width = int(math.sqrt(x.shape[2]) // self.batch_size)

        [x, [out_height, out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=self.input_params[4],
            out_channels=self.input_params[3],
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            conv_op_cache=self.cache,
            debug=False,
            groups=self.groups,
            memory_config=None,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        if self.is_detect_cv2:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            return x, out_height, out_width

        x = ttnn.silu(x)
        return x, out_height, out_width


class ttnn_Bottleneck:
    def __init__(
        self,
        device,
        parameters,
        shortcut,
        change_shard,
        input_params,
        act_block_h=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        tilize=False,
    ):
        self.device = device
        self.tilize = tilize
        self.shortcut = shortcut
        self.cv1 = ttnn_Conv(
            device,
            parameters["cv1"],
            input_params,
            change_shard=change_shard,
            deallocate_activation=deallocate_activation,
            output_layout=output_layout,
        )
        self.cv2 = ttnn_Conv(
            device,
            parameters["cv2"],
            input_params,
            act_block_h=act_block_h,
            change_shard=change_shard,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        cv2, out_h, out_w = self.cv2(cv1)  # pass cv1
        ttnn.deallocate(cv1)

        if self.tilize:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv2


class ttnn_C2f:
    def __init__(
        self,
        device,
        parameters,
        n=1,
        shortcut=False,
        change_shard=None,
        input_params=None,
        act_block_h=False,
        bfloat8=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    ):
        self.device = device
        self.parameters = parameters
        self.n = n
        self.shortcut = shortcut
        self.change_shard = change_shard
        self.input_params = input_params
        self.act_block_h = act_block_h
        self.bfloat8 = bfloat8
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout

        self.cv1 = ttnn_Conv(
            device,
            self.parameters["cv1"],
            input_params=self.input_params[0],
            bfloat8=self.bfloat8,
            change_shard=self.change_shard,
            deallocate_activation=self.deallocate_activation,
            output_layout=self.output_layout,
        )
        self.cv2 = ttnn_Conv(
            self.device,
            self.parameters["cv2"],
            input_params=self.input_params[1],
            bfloat8=self.bfloat8,
            block_shard=self.block_shard,
            change_shard=True,
            deallocate_activation=self.deallocate_activation,
        )
        self.bottleneck_modules = []
        for i in range(self.n):
            if i == 0:
                self.tilize = True
            else:
                self.tilize = False
            self.bottleneck_modules.append(
                ttnn_Bottleneck(
                    self.device,
                    self.parameters["m"][i],
                    self.shortcut,
                    self.change_shard,
                    input_params=self.input_params[2],
                    act_block_h=self.act_block_h,
                    deallocate_activation=self.deallocate_activation,
                    tilize=self.tilize,
                )
            )

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # y = list(
        #     ttnn.split(cv1, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        # )  # use this for 320 resolution, but ttnn.split is not supprted by trace

        # split is not supported by trace, hence using this this
        y = []
        y.append(cv1[:, :, :, : cv1.shape[-1] // 2])
        y.append(cv1[:, :, :, cv1.shape[-1] // 2 :])

        ttnn.deallocate(cv1)

        to_tile = True
        for i in range(self.n):
            z = self.bottleneck_modules[i](y[-1])

            y.append(z)
            to_tile = False

        y[0] = ttnn.to_layout(y[0], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        y[1] = ttnn.to_layout(y[1], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        if not self.shortcut:
            for i in range(2, len(y)):
                y[i] = ttnn.sharded_to_interleaved(y[i], ttnn.L1_MEMORY_CONFIG)

        x = ttnn.concat(y, 3)

        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x, out_h, out_w = self.cv2(x)
        return x, out_h, out_w


class ttnn_SPPF:
    def __init__(self, device, parameters, input_params, batch_size):
        self.device = device
        self.parameters = parameters
        self.input_params = input_params
        self.batch_size = batch_size
        self.cv1 = ttnn_Conv(device, parameters["cv1"], input_params=input_params[0], change_shard=True)
        self.cv2 = ttnn_Conv(device, parameters["cv2"], input_params=input_params[1], change_shard=True)

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        p = 5 // 2
        cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = [cv1]
        for i in range(3):
            output = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=self.batch_size,
                input_h=out_h,
                input_w=out_w,
                channels=y[-1].shape[-1],
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[p, p],
                dilation=[1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            y.append(output)

        x = ttnn.concat(y, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x, out_h, out_w = self.cv2(x)
        return x, out_h, out_w
