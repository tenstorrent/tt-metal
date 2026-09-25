# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import torch.nn as nn
from typing import Tuple, Union
from torch.nn.modules.utils import _pair, _single

from models.experimental.bevformer.tt.tt_common import TtnnConv2D
from models.experimental.bevformer.tt.tt_modulated_deform_conv import TtModulatedDeformConv2dDevice


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
        input_dtype=ttnn.bfloat16,
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
        # The device DCN samples x at the output grid and reshapes x to (B, out_h, out_w,
        # C_in), which only holds at stride 1. ResNet101 puts every stride on conv1 or the
        # downsample shortcut, never on the DCN conv2.
        assert self.stride == (1, 1), f"device DCN supports stride 1 only, got {self.stride}"
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = conv_pth.weight  # torch weight
        self.bias = conv_pth.bias  # torch bias, None

        self.conv_offset = TtnnConv2D(
            conv_args.conv_offset, conv_pth.conv_offset, device=device, input_dtype=input_dtype
        )

        self.device_dcn = TtModulatedDeformConv2dDevice(
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            deform_groups=self.deform_groups,
            device=device,
            input_shape=(
                conv_args.conv_offset.batch_size,
                conv_args.conv_offset.input_height,
                conv_args.conv_offset.input_width,
            ),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out, out_h, out_w = self.conv_offset(x)
        out = ttnn.sharded_to_interleaved(out)
        # conv_offset reports logical shape (1, 1, B*H*W, C); recover B from
        # the total volume so this path doesn't pin the batch dimension.
        last = out.shape[-1]
        out_volume = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]
        B = out_volume // (out_h * out_w * last)
        out = ttnn.reshape(out, (B, out_h, out_w, last))
        o1, o2, mask = ttnn.chunk(out, 3, dim=3)
        ttnn.deallocate(out)
        offset = ttnn.concat((o1, o2), dim=3)  # NHWC (B, H_out, W_out, 2*K*K), DCNv2 (y,x) interleaved layout
        ttnn.deallocate(o1)
        ttnn.deallocate(o2)
        mask = ttnn.sigmoid(mask)  # low pcc if we use ttnn sigmoid for mask

        # The caller's reshape to (B, H, W, C) doesn't always make x.shape[0] == B (the
        # underlying tile-layout tensor can still report logical shape (1, 1, B*H*W, C));
        # reshape unconditionally so the device DCN sees a proper 4D NHWC tensor.
        C_in = x.shape[-1]
        x_nhwc = ttnn.reshape(x, (B, out_h, out_w, C_in))
        out_nhwc = self.device_dcn(x_nhwc, offset, mask)  # (B, H_out, W_out, C_out) tile
        ttnn.deallocate(offset)
        ttnn.deallocate(mask)
        C_out = out_nhwc.shape[-1]
        return ttnn.reshape(out_nhwc, (1, 1, B * out_h * out_w, C_out)), out_h, out_w


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
        dram_activation=False,
        dram_input=False,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
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
                dram_activation=dram_activation,
                dram_input=dram_input,
                input_dtype=input_dtype,
                input_layout=input_layout,
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
                    dram_activation=dram_activation,
                    input_dtype=layers[-1].output_dtype,
                )
            )
        self.layer = layers
        self.output_dtype = layers[-1].output_dtype

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
        dram_activation=False,
        dram_input=False,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
    ):
        """``input_dtype`` and ``input_layout`` describe the block input; the dtype and layout
        each conv sees follow from them. ``dram_activation`` keeps the activations of conv1, conv3 and the downsample in
        DRAM; a DCN conv2 is unaffected. ``dram_input`` only covers the convs that read the
        block input, for a block fed by a DRAM stage whose own activations fit in L1."""
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

        # conv2d and ttnn.linear keep their input's dtype and emit TILE, the DCN branch emits
        # bfloat16, and a bfloat8_b block casts its identity before the downsample.
        conv2_dtype = input_dtype
        conv3_dtype = ttnn.bfloat16 if self.with_dcn else input_dtype
        downsample_dtype = ttnn.bfloat8_b if activation_dtype == ttnn.bfloat8_b else input_dtype
        downsample_layout = ttnn.TILE_LAYOUT if activation_dtype == ttnn.bfloat8_b else input_layout
        self.output_dtype = conv3_dtype

        self.conv1 = TtnnConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            dram_activation=dram_activation or dram_input,
            input_dtype=input_dtype,
            input_layout=input_layout,
        )

        if not self.with_dcn:
            self.conv2 = TtnnConv2D(
                conv_args.conv2,
                conv_pth.conv2,
                device=device,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                act_block_h=32,
                dealloc_act=True,
                dram_activation=dram_activation,
                input_dtype=conv2_dtype,
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
                input_dtype=conv2_dtype,
            )
            # The DCN branch runs its BatchNorm as a separate op. Its parameters go to the
            # device once here, so a forward writes nothing from the host.
            bn = conv_pth.bn2
            self.bn_running_mean = ttnn.to_device(bn.running_mean, device=device)
            self.bn_running_var = ttnn.to_device(bn.running_var, device=device)
            self.bn_weight = None if bn.weight is None else ttnn.to_device(bn.weight, device=device)
            self.bn_bias = None if bn.bias is None else ttnn.to_device(bn.bias, device=device)
            self.bn_eps = bn.eps

        self.conv3 = TtnnConv2D(
            conv_args.conv3,
            conv_pth.conv3,
            device=device,
            activation=None,
            is_blk=conv3_blk_sharded,
            dealloc_act=True,
            dram_activation=dram_activation,
            input_dtype=conv3_dtype,
        )

        if is_downsample:
            self.downsample = TtnnConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                activation=None,
                is_blk=True if self.dcn else False,
                activation_dtype=activation_dtype,
                dram_activation=dram_activation or dram_input,
                input_dtype=downsample_dtype,
                input_layout=downsample_layout,
            )

    def __call__(self, x_identity):
        x, out_h, out_w = self.conv1(x_identity)
        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if self.dcn == True:
            x = ttnn.sharded_to_interleaved(x)
            batch_size = self.conv1.conv.batch_size
            x = ttnn.reshape(x, (batch_size, out_h, out_w, x.shape[3]))
            x, out_h, out_w = self.conv2(x)
            x = ttnn.reshape(x, (batch_size, out_h, out_w, x.shape[3]))
            x = ttnn.permute(x, (0, 3, 1, 2))
            x = ttnn.batch_norm(
                x,
                running_mean=self.bn_running_mean,
                running_var=self.bn_running_var,
                eps=self.bn_eps,
                weight=self.bn_weight,
                bias=self.bn_bias,
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
        dram_activation_stages=(),
    ):
        """``dram_activation_stages`` lists the stage indices whose activations are kept in
        DRAM and computed in pieces, for stages whose convs do not fit in L1. A stage that
        follows one of them and is not listed itself reads its input from DRAM the same way."""
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
            input_dtype=ttnn.bfloat16,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The max pool emits a bfloat16 ROW_MAJOR tensor; forward casts stage 1's output to
        # bfloat8_b; every stage emits TILE.
        stage_input_dtype = ttnn.bfloat16
        stage_input_layout = ttnn.ROW_MAJOR_LAYOUT
        self.output_dtypes = []
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
                dram_activation=i in dram_activation_stages,
                dram_input=i not in dram_activation_stages and i - 1 in dram_activation_stages,
                input_dtype=stage_input_dtype,
                input_layout=stage_input_layout,
            )
            self.inplanes = planes * self.block.expansion
            self.res_layers.append(res_layer)
            stage_input_dtype = ttnn.bfloat8_b if i == 0 else res_layer.output_dtype
            stage_input_layout = ttnn.TILE_LAYOUT
            if i in out_indices:
                self.output_dtypes.append(stage_input_dtype)

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def __call__(self, x):
        """Forward function."""
        x, out_h, out_w = self.conv1(x)
        x = ttnn.sharded_to_interleaved(x)
        x = ttnn.add(x, 0.0, dtype=ttnn.bfloat8_b)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv1.conv.batch_size,
            input_h=out_h,
            input_w=out_w,
            channels=x.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = layer_name(x)
            if i == 0:
                x = ttnn.add(x, 0.0, dtype=ttnn.bfloat8_b)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
