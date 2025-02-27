# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import functools

from typing import Optional, Tuple, Union, Type, List


from models.experimental.vovnet.vovnet_utils import *

model_cfgs = dict(
    stem_chs=[64, 64, 64],
    stage_conv_chs=[128, 160, 192, 224],
    stage_out_chs=[256, 512, 768, 1024],
    layer_per_block=3,
    block_per_stage=[1, 1, 1, 1],
    residual=True,
    depthwise=True,
    attn="ese",
)


class EffectiveSEModule(nn.Module):
    """'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, add_maxpool=False, gate_layer=nn.Hardsigmoid, **_):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_layer=nn.ReLU,
        act_kwargs=None,
        inplace=True,
    ):
        super(BatchNormAct2d, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

        self.act = create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        assert x.ndim == 4, f"expected 4D input (got {x.ndim}D input)"

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            exponential_average_factor,
            self.eps,
        )

        x = self.act(x)
        return x


_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
)
_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}
# has act_layer arg to define act type

_NORM_ACT_REQUIRES_ARG = {
    BatchNormAct2d,
}


class SeparableConvNormAct(nn.Module):
    """Separable Conv w/ trailing Norm and Activation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding="",
        bias=False,
        channel_multiplier=1.0,
        pw_kernel_size=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        apply_act=True,
    ):
        super(SeparableConvNormAct, self).__init__()

        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True,
        )

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias,
        )

        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_kwargs = {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.bn(x)
        return x


class SequentialAppendList(nn.Sequential):
    def __init__(self, *args):
        super(SequentialAppendList, self).__init__(*args)

    def forward(self, x: torch.Tensor, concat_list: List[torch.Tensor]) -> torch.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x


class OsaBlock(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layer_per_block,
        residual=False,
        depthwise=False,
        attn="",
        norm_layer=BatchNormAct2d,
        act_layer=nn.ReLU,
    ):
        super(OsaBlock, self).__init__()

        self.residual = residual
        self.depthwise = depthwise
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)

        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = ConvNormAct(next_in_chs, mid_chs, 1, **conv_kwargs)
        else:
            self.conv_reduction = None

        mid_convs = []
        for i in range(layer_per_block):
            if self.depthwise:
                conv = SeparableConvNormAct(mid_chs, mid_chs, **conv_kwargs)
            else:
                conv = ConvNormAct(next_in_chs, mid_chs, 3, **conv_kwargs)
            next_in_chs = mid_chs
            mid_convs.append(conv)
        self.conv_mid = SequentialAppendList(*mid_convs)

        # feature aggregation
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = ConvNormAct(next_in_chs, out_chs, **conv_kwargs)

        self.attn = create_attn(attn, out_chs) if attn else None

    def forward(self, x):
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.residual:
            x = x + output[0]
        return x


def create_attn(attn_type, channels, **kwargs):
    module_cls = get_attn(attn_type)
    if module_cls is not None:
        # NOTE: it's expected the first (positional) argument of all attention layers is the # input channels
        return module_cls(channels, **kwargs)
    return None


def get_attn(attn_type):
    if isinstance(attn_type, torch.nn.Module):
        return attn_type
    module_cls = None
    if attn_type:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            # Lightweight attention modules (channel and/or coarse spatial).
            # Typically added to existing network architecture blocks in addition to existing convolutions.
            if attn_type == "ese":
                module_cls = EffectiveSEModule
            else:
                assert False, "Invalid attn module (%s)" % attn_type
        else:
            module_cls = attn_type
    return module_cls


class OsaStage(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_per_stage,
        layer_per_block,
        downsample=True,
        residual=True,
        depthwise=False,
        attn="ese",
        norm_layer=BatchNormAct2d,
        act_layer=nn.ReLU,
    ):
        super(OsaStage, self).__init__()

        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None

        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            blocks += [
                OsaBlock(
                    in_chs,
                    mid_chs,
                    out_chs,
                    layer_per_block,
                    residual=residual and i > 0,
                    depthwise=depthwise,
                    attn=attn if last_block else "",
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            ]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.blocks(x)
        return x


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size"""

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = 1,
        pool_type: str = "fast",
        flatten: bool = False,
        input_fmt: str = "NCHW",
    ):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ("NCHW", "NHWC")
        self.pool_type = pool_type or ""  # convert other falsy values to empty string for consistent TS typing
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        else:
            assert input_fmt == "NCHW"
            if pool_type == "max":
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            else:
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + " (" + "pool_type=" + self.pool_type + ", flatten=" + str(self.flatten) + ")"


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=False,
        apply_act=True,
        norm_layer=nn.BatchNorm2d,
        norm_kwargs=None,
        act_layer=nn.ReLU,
        act_kwargs=None,
    ):
        super(ConvNormAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`

        self.bn = norm_act_layer(
            out_channels,
            apply_act=apply_act,
            act_kwargs=act_kwargs,
            **norm_kwargs,
        )

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def get_norm_act_layer(norm_layer, act_layer=None):
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        layer_name = norm_layer.replace("_", "").lower().split("-")[0]
        norm_act_layer = _NORM_ACT_MAP.get(layer_name, None)

    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer

    elif isinstance(norm_layer, types.FunctionType):
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith("batchnorm"):
            norm_act_layer = BatchNormAct2d
        else:
            assert False, f"No equivalent norm_act layer for {type_name}"

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault("act_layer", act_layer)

    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args
    return norm_act_layer


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: str = "avg",
        use_conv: bool = False,
        input_fmt: str = "NCHW",
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes, pool_type=None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else nn.Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


def create_classifier(
    num_features: int,
    num_classes: int,
    pool_type: str = "avg",
    use_conv: bool = False,
    input_fmt: str = "NCHW",
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    return global_pool, fc


def _create_pool(
    num_features: int,
    num_classes: int,
    pool_type: str = "avg",
    use_conv: bool = False,
    input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert (
            num_classes == 0 or use_conv
        ), "Pooling can only be disabled if classifier is also removed or conv classifier is used"
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


class VoVNet(nn.Module):
    def __init__(
        self,
        cfg=model_cfgs,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        output_stride=32,
        norm_layer=BatchNormAct2d,
        act_layer=nn.ReLU,
        **kwargs,
    ):
        """
        Args:
            cfg (dict): Model architecture configuration
            in_chans (int): Number of input channels (default: 3)
            num_classes (int): Number of classifier classes (default: 1000)
            global_pool (str): Global pooling type (default: 'avg')
            output_stride (int): Output stride of network, one of (8, 16, 32) (default: 32)
            norm_layer (Union[str, nn.Module]): normalization layer
            act_layer (Union[str, nn.Module]): activation layer
            kwargs (dict): Extra kwargs overlayed onto cfg
        """
        super(VoVNet, self).__init__()
        self.num_classes = num_classes
        assert output_stride == 32  # FIXME support dilation
        cfg = dict(cfg, **kwargs)
        stem_stride = cfg.get("stem_stride", 4)
        stem_chs = cfg["stem_chs"]
        stage_conv_chs = cfg["stage_conv_chs"]
        stage_out_chs = cfg["stage_out_chs"]
        block_per_stage = cfg["block_per_stage"]
        layer_per_block = cfg["layer_per_block"]
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)

        # Stem module
        last_stem_stride = stem_stride // 2
        conv_type = SeparableConvNormAct if cfg["depthwise"] else ConvNormAct
        self.stem = nn.Sequential(
            *[
                ConvNormAct(in_chans, stem_chs[0], 3, stride=2, **conv_kwargs),
                conv_type(stem_chs[0], stem_chs[1], 3, stride=1, **conv_kwargs),
                conv_type(stem_chs[1], stem_chs[2], 3, stride=last_stem_stride, **conv_kwargs),
            ]
        )

        self.feature_info = [
            dict(
                num_chs=stem_chs[1],
                reduction=2,
                module=f"stem.{1 if stem_stride == 4 else 2}",
            )
        ]
        current_stride = stem_stride

        in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]
        stage_args = dict(
            residual=cfg["residual"],
            depthwise=cfg["depthwise"],
            attn=cfg["attn"],
            **conv_kwargs,
        )
        stages = []
        for i in range(4):  # num_stages
            downsample = stem_stride == 2 or i > 0  # first stage has no stride/downsample if stem_stride is 4
            stages += [
                OsaStage(
                    in_ch_list[i],
                    stage_conv_chs[i],
                    stage_out_chs[i],
                    block_per_stage[i],
                    layer_per_block,
                    downsample=downsample,
                    **stage_args,
                )
            ]
            self.num_features = stage_out_chs[i]
            current_stride *= 2 if downsample else 1
            self.feature_info += [
                dict(
                    num_chs=self.num_features,
                    reduction=current_stride,
                    module=f"stages.{i}",
                )
            ]

        self.stages = nn.Sequential(*stages)

        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
