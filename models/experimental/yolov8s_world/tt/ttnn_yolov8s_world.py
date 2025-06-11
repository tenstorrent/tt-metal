# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
import torch
from torch import nn
from models.experimental.yolov8s_world.tt.ttnn_yolov8s_world_utils import (
    ttnn_decode_bboxes,
    concat,
    determine_num_cores_for_upsample,
    get_core_grid_from_num_cores,
    tt_adaptive_to_max_pool2d,
)


class TtConv:
    def __init__(
        self,
        device,
        parameters,
        input_params,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=None,
        bfloat8=True,
        conv_alone=False,
        reshape_tensor=False,
        change_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        is_fused=True,
        is_dfl=False,
        is_act_false=False,
        width_shard=False,
        act_blocks=False,
        enable_act_double_buffer=True,
        enable_split_reader=False,
        reshard_if_not_optimal=True,
        batch_size=1,
        conv_math_fidelity=None,
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
        self.is_act_false = is_act_false
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.batch_size = batch_size
        self.reshape_tensor = reshape_tensor

        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config(conv_math_fidelity)
        if conv_alone:
            self.weights, self.bias = self.parameters["weight"], self.parameters["bias"]
        else:
            self.weights, self.bias = self.parameters["conv"]["weight"], self.parameters["conv"]["bias"]

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation="",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            act_block_w_div=1,
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

        if self.is_act_false != True:
            conv_config.activation = "silu"

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        if self.width_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        if self.bfloat8:
            conv_config.weights_dtype = ttnn.bfloat8_b

        return conv_config

    def _initialize_compute_config(self, math_fidelity):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=math_fidelity if math_fidelity is not None else ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
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
            groups=self.groups,
            memory_config=ttnn.L1_MEMORY_CONFIG if self.change_shard == True else None,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        if self.is_act_false:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            if self.reshape_tensor:
                x = ttnn.reshape(x, (self.batch_size, out_height, out_width, x.shape[-1]))
            return x, out_height, out_width

        return x, out_height, out_width


class TtBottleneck:
    def __init__(
        self,
        device,
        parameters,
        shortcut,
        input_params,
        act_block_h=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        tilize=False,
    ):
        self.device = device
        self.tilize = tilize
        self.shortcut = shortcut
        self.cv1 = TtConv(
            device,
            parameters["cv1"],
            input_params,
            deallocate_activation=deallocate_activation,
            output_layout=output_layout,
        )
        self.cv2 = TtConv(
            device,
            parameters["cv2"],
            input_params,
            act_block_h=act_block_h,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        cv2, out_h, out_w = self.cv2(cv1)  # pass cv1
        ttnn.deallocate(cv1)

        if self.tilize:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.to_memory_config(x, memory_config=cv2.memory_config())

        return ttnn.add(x, cv2, memory_config=x.memory_config()) if self.shortcut else cv2


class TtC2f:
    def __init__(
        self,
        device,
        parameters,
        n=1,
        shortcut=False,
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
        self.input_params = input_params
        self.act_block_h = act_block_h
        self.bfloat8 = bfloat8
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout

        self.cv1 = TtConv(
            device,
            self.parameters["cv1"],
            input_params=self.input_params[0],
            deallocate_activation=self.deallocate_activation,
            output_layout=self.output_layout,
            block_shard=True,
        )
        self.cv2 = TtConv(
            self.device,
            self.parameters["cv2"],
            input_params=self.input_params[1],
            block_shard=self.block_shard,
            deallocate_activation=self.deallocate_activation,
        )
        self.bottleneck_modules = []
        for i in range(self.n):
            if i == 0:
                self.tilize = True
            else:
                self.tilize = False
            self.bottleneck_modules.append(
                TtBottleneck(
                    self.device,
                    self.parameters["m"][i],
                    self.shortcut,
                    input_params=self.input_params[2],
                    act_block_h=self.act_block_h,
                    deallocate_activation=self.deallocate_activation,
                    tilize=self.tilize,
                )
            )

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)

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

        y[0] = ttnn.to_memory_config(y[0], memory_config=y[-1].memory_config())
        y[1] = ttnn.to_memory_config(y[1], memory_config=y[-1].memory_config())

        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [y[1].memory_config().shard_spec.shape[0], len(y) * y[1].memory_config().shard_spec.shape[1]],
            core_grid=y[1].memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        x = ttnn.concat(y, 3, memory_config=output_sharded_memory_config)

        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x, out_h, out_w = self.cv2(x)
        return x, out_h, out_w


class TtSPPF:
    def __init__(self, device, parameters, input_params, batch_size):
        self.device = device
        self.parameters = parameters
        self.input_params = input_params
        self.batch_size = batch_size
        self.cv1 = TtConv(device, parameters["cv1"], input_params=input_params[0], deallocate_activation=True)
        self.cv2 = TtConv(
            device,
            parameters["cv2"],
            input_params=input_params[1],
            change_shard=True,
            deallocate_activation=True,
            conv_math_fidelity=ttnn.MathFidelity.HiFi2,
        )

    def __call__(self, x):
        cv1, out_h, out_w = self.cv1(x)
        p = 5 // 2
        y = [cv1]
        for i in range(3):
            output = ttnn.max_pool2d(
                input_tensor=(
                    ttnn.to_layout(ttnn.sharded_to_interleaved(y[-1]), layout=ttnn.ROW_MAJOR_LAYOUT)
                    if y[-1].is_sharded()
                    else y[-1]
                ),
                batch_size=self.batch_size,
                input_h=out_h,
                input_w=out_w,
                channels=y[-1].shape[-1],
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[p, p],
                dilation=[1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                applied_shard_scheme=None if y[-1].is_sharded() else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            )
            y.append(output)

        y[0] = ttnn.to_layout(y[0], ttnn.ROW_MAJOR_LAYOUT)

        x = concat(y, 3, True)
        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        x, out_h, out_w = self.cv2(x)
        return x, out_h, out_w


class TtMaxSigmoidAttnBlock:
    """Max Sigmoid attention block."""

    def __init__(
        self,
        device,
        parameters,
        input_params,
        c1,
        c2,
        nh=1,
        ec=128,
        gc=512,
        scale=False,
        block_shard=False,
    ):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.block_shard = block_shard
        self.deallocate_activation = True
        self.nh = nh
        self.hc = c2 // nh
        self.deallocate = False
        self.ec = (
            TtConv(
                self.device,
                self.parameters["ec"],
                input_params=None,
                block_shard=self.block_shard,
                change_shard=True,
                deallocate_activation=self.deallocate_activation,
            )
            if c1 != ec
            else None
        )
        self.gl = ttnn.linear
        self.bias = self.parameters["bias"]
        self.proj_conv = TtConv(
            self.device,
            self.parameters["proj_conv"],
            input_params=input_params,
            block_shard=self.block_shard,
            change_shard=True,
            deallocate_activation=self.deallocate_activation,
            is_act_false=True,
        )
        self.scale = ttnn.ones(1, nh, 1, 1) if scale else 1.0

    def __call__(self, x, guide):
        """Forward process."""
        bs, h, w, _ = x.shape

        guide = self.gl(
            guide,
            self.parameters["gl"]["weight"],
            bias=self.parameters["gl"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        guide = ttnn.reshape(guide, (bs, -1, self.nh, self.hc))
        x = ttnn.permute(x, (0, 3, 1, 2))  # nhwc->nchw
        embed = self.ec(x) if self.ec is not None else x
        embed = ttnn.reshape(embed, (bs, self.nh, self.hc, h, w))

        ## Replacement for torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        batch, m, channel, height, width = embed.shape
        _, n, _, _ = guide.shape
        embed = ttnn.permute(embed, (0, 1, 3, 4, 2))
        embed = ttnn.reshape(embed, (batch, m, -1, channel))
        guide = ttnn.permute(guide, (0, 2, 3, 1))
        aw = ttnn.matmul(embed, guide, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(embed)
        ttnn.deallocate(guide)
        aw = ttnn.reshape(aw, (batch, m, height, width, n))

        aw = ttnn.max(aw, dim=-1, keepdim=True)
        aw = ttnn.permute(aw, (0, 1, 2, 4, 3))  # To increase the perfomance of squeeze operation
        aw = ttnn.squeeze(aw, -2)  # If the above permute is removed use ttnn.squeeze(aw, -1)
        aw = ttnn.div(aw, (self.hc**0.5))
        aw = aw + ttnn.reshape(self.bias, (1, -1, 1, 1))
        aw = ttnn.sigmoid(aw) * self.scale

        x = ttnn.permute(x, (0, 2, 3, 1))  # nchw->nhwc
        x, _, _ = self.proj_conv(x)
        x = ttnn.reshape(x, (bs, h, w, self.nh, -1))
        x = ttnn.permute(x, (0, 3, 4, 1, 2))  # nhwc->nchw
        aw = ttnn.to_memory_config(
            aw, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # added as we are facing OOM issue in next line.
        x = x * ttnn.unsqueeze(aw, 2)
        x = ttnn.to_memory_config(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(aw)
        x = ttnn.reshape(x, (bs, -1, h, w))
        x = ttnn.permute(x, (0, 2, 3, 1))
        h, w = x.shape[1], x.shape[2]
        return ttnn.reshape(x, (1, 1, x.shape[1] * x.shape[2], x.shape[3])), h, w


class TtC2fAttn:
    """C2f module with an additional attn module."""

    def __init__(
        self,
        device,
        parameters,
        c1,
        c2,
        n=1,
        shortcut=False,
        input_params=None,
        deallocate_activation=False,
        ec=128,
        nh=1,
        gc=512,
        g=1,
        e=0.5,
    ):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.device = device
        self.n = n
        self.parameters = parameters
        self.shortcut = shortcut
        self.input_params = input_params
        self.deallocate_activation = deallocate_activation
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = TtConv(
            self.device,
            self.parameters["cv1"],
            input_params=input_params[0],
            deallocate_activation=self.deallocate_activation,
        )
        self.cv2 = TtConv(
            self.device,
            self.parameters["cv2"],
            input_params=input_params[1],
            change_shard=True,
            deallocate_activation=self.deallocate_activation,
            conv_math_fidelity=ttnn.MathFidelity.HiFi2,
        )
        self.m = [
            TtBottleneck(
                self.device,
                self.parameters["m"][i],
                self.shortcut,
                input_params=self.input_params[2],
                deallocate_activation=self.deallocate_activation,
                tilize=True if i == 0 else False,
            )
            for i in range(n)
        ]
        self.attn = TtMaxSigmoidAttnBlock(
            self.device, self.parameters["attn"], self.input_params[2], c1=self.c, c2=self.c, gc=gc, ec=ec, nh=nh
        )

    def __call__(self, x, guide):
        """Forward pass through C2f layer."""
        cv1, out_h, out_w = self.cv1(x)
        cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)

        # y = list(
        #     ttnn.split(cv1, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        # )  # use this for 320 resolution, but ttnn.split is not supprted by trace

        # split is not supported by trace, hence using this this
        y = []
        y.append(cv1[:, :, :, : cv1.shape[-1] // 2])
        y.append(cv1[:, :, :, cv1.shape[-1] // 2 :])

        ttnn.deallocate(cv1)

        for i in range(self.n):
            z = self.m[i](y[-1])

            y.append(z)

        y[0] = ttnn.to_layout(y[0], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        y[1] = ttnn.to_layout(y[1], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        if not self.shortcut:
            for i in range(2, len(y)):
                y[i] = ttnn.sharded_to_interleaved(y[i], ttnn.L1_MEMORY_CONFIG)
        y.append(self.attn(y[-1], guide)[0])
        y[-1] = ttnn.to_layout(y[-1], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.concat(y, 3)  # Decrease in Fps if we use shard concat
        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x, out_h, out_w = self.cv2(x)
        return x, out_h, out_w


class TtImagePoolingAttn:
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, device, parameters, input_params, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.device = device
        self.parameters = parameters
        self.query = [ttnn.layer_norm, ttnn.linear]
        self.key = [ttnn.layer_norm, ttnn.linear]
        self.value = [ttnn.layer_norm, ttnn.linear]
        self.proj = ttnn.linear
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = [
            TtConv(
                self.device,
                self.parameters["projections"][i],
                input_params=input_params[i],
                is_act_false=True,
                conv_alone=True,
            )
            for i, in_channels in enumerate(ch)
        ]
        self.im_pools = [ttnn.max_pool2d for i in range(nf)]
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def __call__(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2

        x = [proj(x)[0] for (x, proj) in zip(x, self.projections)]
        x = [
            ttnn.to_layout(
                ttnn.reshape(
                    pool(
                        input_tensor=ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT),
                        batch_size=1,
                        input_h=int(math.sqrt(x.shape[2])),
                        input_w=int(math.sqrt(x.shape[2])),
                        channels=x.shape[-1],
                        kernel_size=[
                            tt_adaptive_to_max_pool2d(
                                torch.randn(
                                    x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]
                                ),
                                (3, 3),
                            )[0],
                            tt_adaptive_to_max_pool2d(
                                torch.randn(
                                    x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]
                                ),
                                (3, 3),
                            )[0],
                        ],
                        stride=[
                            tt_adaptive_to_max_pool2d(
                                torch.randn(
                                    x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]
                                ),
                                (3, 3),
                            )[1],
                            tt_adaptive_to_max_pool2d(
                                torch.randn(
                                    x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]
                                ),
                                (3, 3),
                            )[1],
                        ],
                        padding=[0, 0],
                        dilation=[1, 1],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        applied_shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ),
                    (bs, num_patches, -1),
                ),
                layout=ttnn.TILE_LAYOUT,
            )
            for (x, pool) in zip(x, self.im_pools)
        ]
        x = ttnn.concat(x, dim=1)
        q = ttnn.clone(text)
        for index, module in enumerate(self.query):
            if module == ttnn.linear:
                q = module(
                    q,
                    self.parameters["query"][index]["weight"],
                    bias=self.parameters["query"][index]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                q = module(
                    q,
                    weight=self.parameters["query"][index]["weight"],
                    bias=self.parameters["query"][index]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

        k = ttnn.clone(x)
        for index, module in enumerate(self.key):
            if module == ttnn.linear:
                k = module(
                    k,
                    self.parameters["key"][index]["weight"],
                    bias=self.parameters["key"][index]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                k = module(
                    k,
                    weight=self.parameters["key"][index]["weight"],
                    bias=self.parameters["key"][index]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

        v = ttnn.clone(x)
        for index, module in enumerate(self.value):
            if module == ttnn.linear:
                v = module(
                    v,
                    self.parameters["value"][index]["weight"],
                    bias=self.parameters["value"][index]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                v = module(
                    v,
                    weight=self.parameters["value"][index]["weight"],
                    bias=self.parameters["value"][index]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        ## Replacement for torch.einsum("bnmc,bkmc->bmnk", q, k))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 3, 1))
        aw = ttnn.matmul(q, k, memory_config=ttnn.L1_MEMORY_CONFIG)

        ttnn.deallocate(q)
        ttnn.deallocate(k)

        aw = ttnn.div(aw, (self.hc**0.5))
        aw = ttnn.softmax(aw, dim=-1)

        ## Replacement for  torch.einsum("bmnk,bkmc->bnmc", aw, v)
        v = ttnn.permute(v, (0, 2, 1, 3))
        x = ttnn.matmul(aw, v, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.permute(x, (0, 2, 1, 3))

        ttnn.deallocate(aw)
        ttnn.deallocate(v)

        x = self.proj(
            ttnn.reshape(x, (bs, -1, self.ec)),
            self.parameters["proj"]["weight"],
            bias=self.parameters["proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = x * self.scale + text
        # Note: The x value is not permuted, torch module return and ttnn module return is same
        return x


class TtDFL:
    def __init__(self, device, parameters, input_params):
        self.device = device
        self.parameters = parameters
        self.input_params = input_params
        self.conv = TtConv(device, parameters, input_params, is_act_false=True, deallocate_activation=True)

    def __call__(self, x, c1=16):
        b, _, a = x.shape

        x = ttnn.reshape(x, (b, 4, c1, a), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.permute(x, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.softmax(x, dim=-1)

        x, _, _ = self.conv(x)
        x = ttnn.permute(x, (0, 1, 3, 2))  # To increase the perfomance of below reshape operation
        x = ttnn.reshape(x, (x.shape[0], 4, -1))
        return x


class TtContrastiveHead:
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self, device, parameters):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.device = device
        self.parameters = parameters
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = self.parameters["bias"]
        self.logit_scale = self.parameters["logit_scale"]

    def __call__(self, x, w):
        """Forward function of contrastive learning."""
        # x = ttnn.permute(x, (0, 3, 1, 2))
        import torch.nn.functional as F

        x = ttnn.to_torch(x).to(torch.float32)
        w = ttnn.to_torch(w).to(torch.float32)
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = ttnn.from_torch(
            x, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        w = ttnn.from_torch(
            w, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )

        ## Replacement for  x = torch.einsum("bchw,bkc->bkhw", x, w)
        batch, height, width, channel = x.shape
        _, k, _ = w.shape
        # x = ttnn.permute(x, (0, 2, 3, 1))  # bchw->bhwc
        x = ttnn.reshape(x, (batch, -1, channel))  # bhwc->b(hw)c
        w = ttnn.permute(w, (0, 2, 1))  # bkc->bck
        x = ttnn.matmul(x, w, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (batch, height, width, k))
        # x = ttnn.permute(x, (0, 3, 1, 2))

        ttnn.deallocate(w)

        return (
            x * ttnn.exp(self.logit_scale) + self.bias
        )  # ttnn.permute(x * ttnn.exp(self.logit_scale) + self.bias, (0, 2, 3, 1))


class TtWorldDetect:
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    dynamic = False
    export = False  # export mode
    shape = None

    def __init__(self, device, parameters, input_params, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        # self.stride = ttnn.Tensor([8.0, 16.0, 32.0],device=self.device,layout=ttnn.TILE_LAYOUT,dtype=ttnn.bfloat16)
        self.c2, self.c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(nc, 100))

        self.cv2 = [
            [
                TtConv(
                    self.device,
                    self.parameters["cv2"][i][0],
                    input_params=input_params["cv2_params"][i]["input_params"][0],
                ),
                TtConv(
                    self.device,
                    self.parameters["cv2"][i][1],
                    input_params=input_params["cv2_params"][i]["input_params"][1],
                ),
                TtConv(
                    self.device,
                    self.parameters["cv2"][i][2],
                    input_params=input_params["cv2_params"][i]["input_params"][2],
                    is_act_false=True,
                    conv_alone=True,
                    reshape_tensor=True,
                ),
            ]
            for i in range(self.nl)
        ]

        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = [
            [
                TtConv(
                    self.device,
                    self.parameters["cv3"][i][0],
                    input_params=input_params["cv3_params"][i]["input_params"][0],
                ),
                TtConv(
                    self.device,
                    self.parameters["cv3"][i][1],
                    input_params=input_params["cv3_params"][i]["input_params"][1],
                ),
                TtConv(
                    self.device,
                    self.parameters["cv3"][i][2],
                    input_params=input_params["cv3_params"][i]["input_params"][2],
                    is_act_false=True,
                    reshape_tensor=True,
                    conv_alone=True,
                ),
            ]
            for i, x in enumerate(ch)
        ]

        self.dfl = TtDFL(
            device=self.device,
            parameters=self.parameters["dfl"],
            input_params=input_params["dfl_params"]["input_params"],
        )
        self.anchors = None
        self.strides = None
        self.self_shape = None
        self.cv4 = [
            (
                BNContrastiveHead(embed) if with_bn else TtContrastiveHead(self.device, self.parameters["cv4"][i])
            )  # BNContrastiveHead(embed) is not invoked So, Not defined.
            for i, _ in enumerate(ch)
        ]

    def __call__(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = ttnn.sharded_to_interleaved(x[i], memory_config=ttnn.L1_MEMORY_CONFIG)
            cv_2 = ttnn.clone(x[i])
            for module in self.cv2[i]:
                cv_2, _, _ = module(cv_2)

            cv3 = ttnn.clone(x[i])
            for module in self.cv3[i]:
                cv3, _, _ = module(cv3)

            x[i] = ttnn.concat((cv_2, self.cv4[i](cv3, text)), -1)

        # Inference path
        shape = x[0].shape
        anchors, strides = self.parameters["anchors"], self.parameters["strides"]

        xi = []
        for i in x:
            i = ttnn.reshape(i, (shape[0], -1, self.no), memory_config=ttnn.L1_MEMORY_CONFIG)
            xi.append(i)

        x_cat = ttnn.concat(xi, 1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_cat = ttnn.permute(x_cat, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

        box = ttnn.slice(x_cat, [0, 0, 0], [1, 64, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)
        cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)
        dfl = self.dfl(box)
        dbox = ttnn_decode_bboxes(self.device, dfl, anchors)
        dbox = dbox * strides

        return [ttnn.concat((dbox, ttnn.sigmoid(cls)), dim=1), x]


class TtWorldModel:
    def __init__(self, device, parameters, ch=3, nc=None, verbose=True):
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.txt_feats = self.parameters["txt_feats"]
        self.clip_model = None  # CLIP model placeholder

        sppf_configs = {"input_params": ((1, 1, 0, 256, 512), (1, 1, 0, 512, 1024))}
        ImagePoolingAttn_configs = {"input_params": ((1, 1, 0, 256, 128), (1, 1, 0, 256, 256), (1, 1, 0, 256, 512))}

        c2f_configs = {
            "model.2": {"input_params": ((1, 1, 0, 64, 64), (1, 1, 0, 64, 96), (3, 1, 1, 32, 32))},
            "model.4": {"input_params": ((1, 1, 0, 128, 128), (1, 1, 0, 128, 256), (3, 1, 1, 64, 64))},
            "model.6": {"input_params": ((1, 1, 0, 256, 256), (1, 1, 0, 256, 512), (3, 1, 1, 128, 128))},
            "model.8": {"input_params": ((1, 1, 0, 512, 512), (1, 1, 0, 512, 768), (3, 1, 1, 256, 256))},
            "model.12": {"input_params": ((1, 1, 0, 256, 768), (1, 1, 0, 256, 512), (3, 1, 1, 128, 128))},
            "model.15": {"input_params": ((1, 1, 0, 128, 384), (1, 1, 0, 128, 256), (3, 1, 1, 64, 64))},
            "model.19": {"input_params": ((1, 1, 0, 256, 384), (1, 1, 0, 256, 512), (3, 1, 1, 128, 128))},
            "model.22": {"input_params": ((1, 1, 0, 512, 768), (1, 1, 0, 512, 1024), (3, 1, 1, 256, 256))},
        }

        world_detect_configs = {
            "cv2_params": [
                {
                    "input_params": [
                        (3, 1, 1, 64, 128),
                        (3, 1, 1, 64, 64),
                        (1, 1, 0, 64, 64),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 64, 256),
                        (3, 1, 1, 64, 64),
                        (1, 1, 0, 64, 64),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 64, 512),
                        (3, 1, 1, 64, 64),
                        (1, 1, 0, 64, 64),
                    ]
                },
            ],
            "cv3_params": [
                {
                    "input_params": [
                        (3, 1, 1, 128, 128),
                        (3, 1, 1, 128, 128),
                        (1, 1, 0, 512, 128),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 128, 256),
                        (3, 1, 1, 128, 128),
                        (1, 1, 0, 512, 128),
                    ]
                },
                {
                    "input_params": [
                        (3, 1, 1, 128, 512),
                        (3, 1, 1, 128, 128),
                        (1, 1, 0, 512, 128),
                    ]
                },
            ],
            "dfl_params": {
                "input_params": (1, 1, 0, 1, 16),
            },
        }

        self.model = [
            TtConv(
                device,
                parameters["model"][0],
                input_params=[3, 2, 1, 32, 3],
                deallocate_activation=True,
            ),
            TtConv(
                device,
                parameters["model"][1],
                input_params=[3, 2, 1, 64, 32],
                deallocate_activation=True,
            ),
            TtC2f(
                device, parameters["model"][2], n=1, shortcut=True, input_params=c2f_configs["model.2"]["input_params"]
            ),
            TtConv(
                device,
                parameters["model"][3],
                input_params=[3, 2, 1, 128, 64],
                deallocate_activation=True,
            ),
            TtC2f(
                device, parameters["model"][4], n=2, shortcut=True, input_params=c2f_configs["model.4"]["input_params"]
            ),
            TtConv(
                device,
                parameters["model"][5],
                input_params=[3, 2, 1, 256, 128],
                # deallocate_activation=True,
            ),
            TtC2f(
                device, parameters["model"][6], n=2, shortcut=True, input_params=c2f_configs["model.6"]["input_params"]
            ),
            TtConv(
                device,
                parameters["model"][7],
                input_params=[3, 2, 1, 512, 256],
                # deallocate_activation=True,
            ),
            TtC2f(
                device, parameters["model"][8], n=1, shortcut=True, input_params=c2f_configs["model.8"]["input_params"]
            ),
            TtSPPF(device, parameters["model"][9], input_params=sppf_configs["input_params"], batch_size=1),
            ttnn.upsample,
            ttnn.concat,
            TtC2fAttn(
                device,
                parameters["model"][12],
                input_params=c2f_configs["model.12"]["input_params"],
                c1=768,
                c2=256,
                n=1,
                ec=128,
                nh=4,
            ),
            ttnn.upsample,
            ttnn.concat,
            TtC2fAttn(
                device,
                parameters["model"][15],
                input_params=c2f_configs["model.15"]["input_params"],
                c1=384,
                c2=128,
                n=1,
                ec=64,
                nh=2,
            ),
            TtImagePoolingAttn(
                device,
                parameters["model"][16],
                input_params=ImagePoolingAttn_configs["input_params"],
                ec=256,
                ch=[128, 256, 512],
            ),
            TtConv(
                device,
                parameters["model"][17],
                input_params=[3, 2, 1, 128, 128],
                # deallocate_activation=True,
            ),
            ttnn.concat,
            TtC2fAttn(
                device,
                parameters["model"][19],
                input_params=c2f_configs["model.19"]["input_params"],
                c1=384,
                c2=256,
                n=1,
                ec=128,
                nh=4,
            ),
            TtConv(
                device,
                parameters["model"][20],
                input_params=[3, 2, 1, 256, 256],
                # deallocate_activation=True,
            ),
            ttnn.concat,
            TtC2fAttn(
                device,
                parameters["model"][22],
                input_params=c2f_configs["model.22"]["input_params"],
                c1=768,
                c2=512,
                n=1,
                ec=256,
                nh=8,
            ),
            TtWorldDetect(
                device,
                parameters["model"][23],
                input_params=world_detect_configs,
                nc=80,
                embed=512,
                with_bn=False,
                ch=[128, 256, 512],
            ),
        ]

    def __call__(self, x, txt_feats=None):  # 4,6,9,12,15,19
        txt_feats = self.txt_feats if txt_feats is None else txt_feats
        # if len(txt_feats) != len(x) or False: #not invoked
        #     txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = ttnn.clone(txt_feats)
        y, dt, embeddings = [], [], []  # outputs
        save = [4, 6, 9, 9, 12, 12, 15, 15, 15, 19, 22]
        f_info = [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            [-1, 6],
            -1,
            -1,
            [-1, 4],
            -1,
            [15, 12, 9],
            15,
            [-1, 12],
            -1,
            -1,
            [-1, 9],
            -1,
            [15, 19, 22],
        ]
        for index, (f, m) in enumerate(zip(f_info, self.model)):  # except the head part
            if index == 12:
                ttnn.deallocate(y[6])
            if index == 15:
                ttnn.deallocate(y[4])
            if index == 19:
                ttnn.deallocate(y[12])

            if f != -1:  # if not from previous layer
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
            if isinstance(m, TtC2fAttn):
                x, _, _ = m(x, ttnn.clone(txt_feats))
            elif isinstance(m, TtWorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, TtImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                if m == ttnn.concat:
                    for i_1, i in enumerate(x):
                        x[i_1] = ttnn.sharded_to_interleaved(x[i_1], memory_config=ttnn.L1_MEMORY_CONFIG)
                        x[i_1] = ttnn.to_layout(x[i_1], layout=ttnn.TILE_LAYOUT)
                    x = m(x, dim=-1)
                elif m == ttnn.upsample:
                    if index == 10:
                        x = ttnn.reshape(x, (1, 20, 20, 512))
                    else:
                        x = ttnn.reshape(x, (1, 40, 40, 256))
                    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

                    nhw = x.shape[0] * x.shape[1] * x.shape[2]
                    num_cores = determine_num_cores_for_upsample(nhw, x.shape[2])
                    core_grid = get_core_grid_from_num_cores(num_cores)
                    shardspec = ttnn.create_sharded_memory_config_(
                        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
                    )
                    if x.is_sharded():
                        x = ttnn.reshard(x, shardspec)
                    else:
                        x = ttnn.interleaved_to_sharded(x, shardspec)

                    x = m(x, scale_factor=2)
                    x = ttnn.reshape(x, (1, 1, -1, x.shape[-1]))
                else:
                    x, _, _ = m(x)  # run

            y.append(x if index in save else None)  # save output

        ttnn.deallocate(y[15])
        ttnn.deallocate(y[19])
        ttnn.deallocate(y[22])
        return x


class TtYOLOWorld:
    def __init__(self, device=None, parameters=None):
        super().__init__()
        self.model = TtWorldModel(device=device, parameters=parameters)

    def __call__(self, x):
        return self.model(x)
