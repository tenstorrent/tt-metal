# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import torchvision.ops
import ttnn

import torch.nn as nn
from typing import Tuple, Union, Optional
from torch.nn.modules.utils import _pair, _single

from models.experimental.uniad.tt.common import TtnnConv2D
from models.experimental.uniad.tt.ttnn_modulated_deform_conv import TtModulatedDeformConv2dDevice

# Diagnostic harness — set TT_DCN_TIMING=1 to print, on every TtResNet
# forward, the accumulated time spent in TtModulatedDeformConv2dPack
# split into:
#   transfer: device->host reads for x / offset / mask
#   cpu:      torchvision deform_conv2d on the host
#   back:     host->device write of the result tensor
# This is the diagnostic we used to confirm CPU compute (not PCIe) is
# the dominant cost of the modulated deformable conv path.
_DCN_TIMING = os.environ.get("TT_DCN_TIMING") == "1"
_dcn_accum = {"transfer": 0.0, "cpu": 0.0, "back": 0.0, "device": 0.0, "n": 0}

# Route TtModulatedDeformConv2dPack through the device-side
# TtModulatedDeformConv2dDevice (grid_sample + fused matmul) instead of
# the host CPU path. On by default — validated end-to-end against
# the UniAD PCC gate (sdc_traj 0.9910, gate 0.99) and worth ~3.4 sec of
# img_backbone wall time on Blackhole (CPU ~3.57 s → device ~0.35 s
# across 26 DCN blocks in ResNet101 layer3/layer4). Set TT_DCN_DEVICE=0
# to fall back to the host path for debugging or numerical bisection.
_USE_DEVICE_DCN = os.environ.get("TT_DCN_DEVICE", "1") == "1"


def modulated_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    deform_groups: int = 1,
) -> torch.Tensor:
    """DCNv2 host reference via `torchvision.ops.deform_conv2d`.

    Offset is `(N, 2*deform_groups*K*K, H_out, W_out)` in interleaved
    `(y, x)` order — same layout torchvision and mmcv both expect.
    `groups` / `deform_groups` are derived by torchvision from tensor
    shapes, so they're accepted here only for call-site compatibility.
    """
    return torchvision.ops.deform_conv2d(
        input,
        offset,
        weight,
        bias=bias,
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        mask=mask,
    )


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

        # Device-side modulated_deform_conv. Uploads per-chunk weights /
        # base grids lazily, so initialisation here is cheap and warm-path
        # cost is amortised across forward calls. Created unconditionally
        # — `_USE_DEVICE_DCN` only gates whether __call__ uses it.
        self.device_dcn = TtModulatedDeformConv2dDevice(
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            deform_groups=self.deform_groups,
            device=device,
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

        if _USE_DEVICE_DCN:
            # The device path samples x at the output grid and reshapes x to
            # (B, out_h, out_w, C_in), which only holds when input and output
            # share spatial dims — i.e. stride 1. UniAD's ResNet101 runs all
            # 26 DCN convs at stride 1 (downsampling happens at conv1 / the
            # downsample shortcut, never at the DCN conv2), so this is always
            # true here. Assert it so a future config that puts DCN on a
            # strided conv fails with a clear message instead of a cryptic
            # reshape volume error.
            assert self.stride == (1, 1), (
                f"device DCN path supports stride-1 only (got stride={self.stride}); "
                "set TT_DCN_DEVICE=0 for the host fallback to run a strided DCN."
            )
            if _DCN_TIMING:
                ttnn.synchronize_device(self.device)
                _t0 = time.perf_counter()
            # The caller's reshape to (B, H, W, C) doesn't always make
            # x.shape[0] == B (the underlying tile-layout tensor can still
            # report logical shape (1, 1, B*H*W, C)); reshape unconditionally
            # so the scaffold sees a proper 4D NHWC tensor.
            C_in = x.shape[-1]
            x_nhwc = ttnn.reshape(x, (B, out_h, out_w, C_in))
            out_nhwc = self.device_dcn(x_nhwc, offset, mask)  # (B, H_out, W_out, C_out) tile
            ttnn.deallocate(offset)
            ttnn.deallocate(mask)
            C_out = out_nhwc.shape[-1]
            result_ttnn = ttnn.reshape(out_nhwc, (1, 1, B * out_h * out_w, C_out))
            if _DCN_TIMING:
                ttnn.synchronize_device(self.device)
                _dcn_accum["device"] += time.perf_counter() - _t0
                _dcn_accum["n"] += 1
            return result_ttnn, out_h, out_w

        # Host fallback: pull x / offset / mask back, run torchvision
        # deform_conv2d (DCNv2) on CPU.
        mask = ttnn.permute(mask, (0, 3, 1, 2))

        if _DCN_TIMING:
            ttnn.synchronize_device(self.device)
            _t0 = time.perf_counter()
        mask = ttnn.to_torch(mask).to(dtype=torch.float)

        x = ttnn.to_torch(x).permute(0, 3, 1, 2).to(dtype=torch.float)
        offset = ttnn.to_torch(offset).permute(0, 3, 1, 2).to(dtype=torch.float)
        if _DCN_TIMING:
            _t1 = time.perf_counter()
            _dcn_accum["transfer"] += _t1 - _t0

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
        if _DCN_TIMING:
            _t2 = time.perf_counter()
            _dcn_accum["cpu"] += _t2 - _t1
        out_h, out_w = result.shape[2], result.shape[3]
        result = result.permute(0, 2, 3, 1)
        result = result.reshape(1, 1, result.shape[0] * result.shape[1] * result.shape[2], result.shape[3])

        result_ttnn = ttnn.from_torch(result, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if _DCN_TIMING:
            ttnn.synchronize_device(self.device)
            _dcn_accum["back"] += time.perf_counter() - _t2
            _dcn_accum["n"] += 1
        return result_ttnn, out_h, out_w


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
        if _DCN_TIMING:
            _dcn_accum["transfer"] = 0.0
            _dcn_accum["cpu"] = 0.0
            _dcn_accum["back"] = 0.0
            _dcn_accum["device"] = 0.0
            _dcn_accum["n"] = 0

        # Per-stage timing markers — gated by the same TT_UNIAD_TIMING harness
        # used in ttnn_uniad. Imported lazily to avoid a circular dependency
        # at module-load time (ttnn_uniad ultimately imports this module).
        from models.experimental.uniad.tt.ttnn_uniad import _timing_phase as _phase

        with _phase("        bb_stem (conv1+maxpool)", self.device):
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
            )

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            with _phase(f"        bb_layer{i+1}", self.device):
                x = layer_name(x)
                if i == 0:
                    x = ttnn.add(x, 0.0, dtype=ttnn.bfloat8_b)
            if i in self.out_indices:
                outs.append(x)
        if _DCN_TIMING and _dcn_accum["n"] > 0:
            if _USE_DEVICE_DCN:
                print(f"  [DCN device] count={_dcn_accum['n']:>3d} " f"device={_dcn_accum['device']*1000:.1f}ms")
            else:
                print(
                    f"  [DCN host] count={_dcn_accum['n']:>3d} "
                    f"transfer={_dcn_accum['transfer']*1000:.1f}ms "
                    f"cpu={_dcn_accum['cpu']*1000:.1f}ms "
                    f"back={_dcn_accum['back']*1000:.1f}ms "
                    f"total={(_dcn_accum['transfer']+_dcn_accum['cpu']+_dcn_accum['back'])*1000:.1f}ms"
                )
        return tuple(outs)
