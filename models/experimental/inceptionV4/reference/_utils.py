# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, Tuple
import torch.nn as nn

from models.experimental.inceptionV4.reference.basicconv import BasicConv2d


_int_tuple_2_t = Union[int, Tuple[int, int]]


def adaptive_pool_feat_mult(pool_type="avg"):
    if pool_type.endswith("catavgmax"):
        return 2
    else:
        return 1


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size"""

    def __init__(
        self,
        output_size: _int_tuple_2_t = 1,
        pool_type: str = "fast",
        flatten: bool = False,
        input_fmt: str = "NCHW",
    ):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ("NCHW", "NHWC")
        self.pool_type = pool_type or ""  # convert other falsy values to empty string for consistent TS typing

        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.flatten = nn.Flatten(1)

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + " (" + "pool_type=" + self.pool_type + ", flatten=" + str(self.flatten) + ")"


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


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


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
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    return global_pool, fc


def assign_weight_conv(conv: nn.Conv2d, state_dict, key_w: str):
    conv.weight = nn.Parameter(state_dict[f"{key_w}.weight"])
    if conv.bias is not None:
        conv.bias = nn.Parameter(state_dict[f"{key_w}.bias"])


def assign_weight_batchnorm(norm: nn.BatchNorm2d, state_dict, key_w: str):
    norm.weight = nn.Parameter(state_dict[f"{key_w}.weight"])
    norm.bias = nn.Parameter(state_dict[f"{key_w}.bias"])
    norm.running_mean = nn.Parameter(state_dict[f"{key_w}.running_mean"])
    norm.running_var = nn.Parameter(state_dict[f"{key_w}.running_var"])
    norm.num_batches_tracked = nn.Parameter(state_dict[f"{key_w}.num_batches_tracked"], requires_grad=False)
    norm.eval()


def assign_weight_linear(linear: nn.Linear, state_dict, key_w: str):
    linear.weight = nn.Parameter(state_dict[f"{key_w}.weight"])

    if linear.bias is not None:
        linear.bias = nn.Parameter(state_dict[f"{key_w}.bias"])


def assign_weight_basic_conv(item, state_dict, key_w: str):
    assign_weight_conv(item.conv, state_dict, f"{key_w}.conv")
    assign_weight_batchnorm(item.bn, state_dict, f"{key_w}.bn")


def assign_weight_seq(seq: nn.Sequential, state_dict, key_w: str):
    for index, item in enumerate(seq):
        if isinstance(item, nn.Conv2d):
            assign_weight_conv(item, state_dict, f"{key_w}.{index}")
        elif isinstance(item, nn.Linear):
            assign_weight_linear(item, state_dict, f"{key_w}.{index}")
        elif isinstance(item, nn.BatchNorm2d):
            assign_weight_batchnorm(item, state_dict, f"{key_w}.{index}")
        elif isinstance(item, BasicConv2d):
            assign_weight_basic_conv(item, state_dict, f"{key_w}.{index}")
        elif isinstance(item, nn.Sequential):
            assign_weight_seq(item, state_dict, f"{key_w}.{index}")


def assign_weight(item, state_dict, key_w: str):
    if isinstance(item, nn.Conv2d):
        assign_conv_weight(item, state_dict, f"{key_w}")
    elif isinstance(item, nn.Linear):
        assign_linear_weights(item, state_dict, f"{key_w}")
    elif isinstance(item, nn.BatchNorm2d):
        assign_batchnorm_weight(item, state_dict, f"{key_w}")
    elif isinstance(item, nn.Sequential):
        assign_weight_seq(item, state_dict, f"{key_w}")
