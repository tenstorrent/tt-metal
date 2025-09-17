# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import torch.nn as nn
from typing import Tuple, Union, Optional
from torch.autograd import Function
from torch.nn.modules.utils import _pair, _single

from models.experimental.uniad.tt.common import TtnnConv2D

from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext("_ext", ["modulated_deform_conv_forward"])


# TODO Raised issue for this operation - <https://github.com/tenstorrent/tt-metal/issues/25526>
class ModulatedDeformConv2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
    ) -> torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead."
            )
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)  # type: ignore
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConv2dFunction._output_size(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias,
        )
        return output

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be " + "x".join(map(str, output_size)) + ")")
        return output_size


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


class TtModulatedDeformConv2dPack:
    _version = 2

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: Union[bool, str] = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.device = device
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = conv_pth.weight  # torch weight
        self.bias = conv_pth.bias  # torch bias, None

        self.conv_offset = TtnnConv2D(conv_args.conv_offset, conv_pth.conv_offset, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out, out_h, out_w = self.conv_offset(x)
        out = ttnn.sharded_to_interleaved(out)
        out = ttnn.reshape(out, (6, out_h, out_w, out.shape[3]))
        o1, o2, mask = ttnn.chunk(out, 3, dim=3)
        ttnn.deallocate(out)
        offset = ttnn.concat((o1, o2), dim=3)
        ttnn.deallocate(o1)
        ttnn.deallocate(o2)
        mask = ttnn.sigmoid_accurate(mask)  # low pcc if we use ttnn sigmoid for mask
        mask = ttnn.permute(mask, (0, 3, 1, 2))

        mask = ttnn.to_torch(mask).to(dtype=torch.float)

        x = ttnn.to_torch(x).permute(0, 3, 1, 2).to(dtype=torch.float)
        offset = ttnn.to_torch(offset).permute(0, 3, 1, 2).to(dtype=torch.float)

        result = modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )
        out_h, out_w = result.shape[2], result.shape[3]
        result = result.permute(0, 2, 3, 1)
        result = result.reshape(1, 1, result.shape[0] * result.shape[1] * result.shape[2], result.shape[3])

        return ttnn.from_torch(result, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), out_h, out_w


class TtResLayer:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        inplanes,
        num_blocks,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat16,
        conv3_blk_sharded=False,
        planes=None,
        stride=1,
        dilation=1,
        style="pytorch",
        conv_cfg=None,
        dcn=None,
    ):
        expansion = 4

        if stride != 1 or inplanes != planes * expansion:
            is_downsample = True

        layers = []

        layers.append(
            TtBottleneck(
                conv_args[0],
                conv_pth[0],
                device,
                is_downsample=is_downsample,
                blk_sharded=False,
                activation_dtype=activation_dtype,
                conv3_blk_sharded=conv3_blk_sharded,
                planes=planes,
                stride=stride,
                dilation=dilation,
                style=style,
                conv_cfg=None,
                dcn=dcn,
            )
        )
        inplanes = planes * expansion
        for j in range(1, num_blocks):
            layers.append(
                TtBottleneck(
                    conv_args[j],
                    conv_pth[j],
                    device,
                    is_downsample=False,
                    blk_sharded=False,
                    activation_dtype=activation_dtype,
                    conv3_blk_sharded=conv3_blk_sharded,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    style=style,
                    conv_cfg=None,
                    dcn=dcn,
                )
            )
        self.layer = layers

    def __call__(self, x):
        for i in self.layer:
            x = i(x)
        return x


class TtBottleneck:
    expansion = 4

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat16,
        conv3_blk_sharded=False,
        planes=None,
        stride=1,
        dilation=1,
        style="pytorch",
        conv_cfg=None,
        dcn=None,
    ):
        assert style in ["pytorch", "caffe"]
        self.device = device

        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.activation_dtype = activation_dtype
        self.is_downsample = is_downsample

        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.conv1 = TtnnConv2D(
            conv_args.conv1, conv_pth.conv1, device=device, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        )

        if not self.with_dcn:
            self.conv2 = TtnnConv2D(
                conv_args.conv2,
                conv_pth.conv2,
                device=device,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                act_block_h=32,
                dealloc_act=True,
            )
        else:
            assert self.conv_cfg is None, "conv_cfg must be None for DCN"
            self.conv2 = TtModulatedDeformConv2dPack(
                conv_args.conv2,
                conv_pth.conv2,
                device,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
            self.bn_parameters = conv_pth.bn2

        self.conv3 = TtnnConv2D(
            conv_args.conv3, conv_pth.conv3, device=device, activation=None, is_blk=conv3_blk_sharded, dealloc_act=True
        )

        if is_downsample:
            self.downsample = TtnnConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                activation=None,
                is_blk=True if self.dcn else False,
                activation_dtype=activation_dtype,
            )

    def __call__(self, x_identity):
        x, out_h, out_w = self.conv1(x_identity)
        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if self.dcn == True:
            x = ttnn.sharded_to_interleaved(x)
            x = ttnn.reshape(x, (6, out_h, out_w, x.shape[3]))
            x, out_h, out_w = self.conv2(x)
            x = ttnn.reshape(x, (6, out_h, out_w, x.shape[3]))
            x = ttnn.permute(x, (0, 3, 1, 2))
            x = ttnn.batch_norm(
                x,
                running_mean=ttnn.to_device(self.bn_parameters.running_mean, device=self.device),
                running_var=ttnn.to_device(self.bn_parameters.running_var, device=self.device),
                eps=self.bn_parameters.eps,
                weight=ttnn.to_device(self.bn_parameters.weight, device=self.device),
                bias=ttnn.to_device(self.bn_parameters.bias, device=self.device),
            )
            x = ttnn.relu(x)
            x = ttnn.permute(x, (0, 2, 3, 1))
        else:
            x, _, _ = self.conv2(x)
        x, _, _ = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity, _, _ = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x


class TtResNet:
    arch_settings = {
        # 18: (BasicBlock, (2, 2, 2, 2)),
        # 34: (BasicBlock, (3, 4, 6, 3)),
        50: (TtBottleneck, (3, 4, 6, 3)),
        101: (TtBottleneck, (3, 4, 23, 3)),
        152: (TtBottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        pretrained=None,
        init_cfg=None,
    ):
        self.conv_args = conv_args
        self.device = device
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        assert not (init_cfg and pretrained), "init_cfg and pretrained cannot be specified at the same time"

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self.conv1 = TtnnConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            activation_dtype=ttnn.bfloat16,
            act_block_h=64,
            dealloc_act=True,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = base_channels * 2**i
            res_layer = TtResLayer(
                conv_args=conv_args[f"layer{i+1}"],
                conv_pth=conv_pth[f"layer{i+1}"],
                device=device,
                inplanes=self.inplanes,
                num_blocks=num_blocks,
                is_downsample=False,
                blk_sharded=False,
                activation_dtype=ttnn.bfloat8_b if i == 1 else ttnn.bfloat16,
                conv3_blk_sharded=False,
                planes=planes,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=None,
                dcn=dcn,
            )
            self.inplanes = planes * self.block.expansion
            self.res_layers.append(res_layer)

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def __call__(self, x):
        """Forward function."""
        x, _, _ = self.conv1(x)

        x = ttnn.sharded_to_interleaved(x)
        x = ttnn.add(x, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=6,
            input_h=320,
            input_w=180,
            channels=x.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
            in_place_halo=True,
        )

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = layer_name(x)
            if i == 0:
                x = ttnn.add(x, 0.0, dtype=ttnn.bfloat8_b)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
