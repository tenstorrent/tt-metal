# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from typing import List, Tuple, Union, Optional
from packaging import version
from collections import OrderedDict
from PIL import Image
import os
import glob
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

is_torch_greater_or_equal_than_1_10 = parsed_torch_version_base >= version.parse("1.10")


def meshgrid(
    *tensors: Union[torch.Tensor, List[torch.Tensor]], indexing: Optional[str] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    if is_torch_greater_or_equal_than_1_10:
        return torch.meshgrid(*tensors, indexing=indexing)
    else:
        if indexing != "ij":
            raise ValueError('torch.meshgrid only supports `indexing="ij"` for torch<1.10.')
        return torch.meshgrid(*tensors)


def window_partition(input_feature, window_size, device, put_on_device=True):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape.with_tile_padding()
    input_feature = tt_to_torch_tensor(input_feature)
    input_feature = input_feature.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        num_channels,
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)

    windows = torch_to_tt_tensor_rm(windows, device, put_on_device=put_on_device)
    return windows


def window_reverse(windows, window_size, height, width, device, put_on_device=True):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape.with_tile_padding()[-1]
    windows = tt_to_torch_tensor(windows)
    windows = windows.view(
        -1,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        num_channels,
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)

    windows = torch_to_tt_tensor_rm(windows, device, put_on_device=put_on_device)
    return windows


def get_shape(shape):
    """Insert 1's in the begining of shape list until the len(shape) = 4"""
    if len(shape) <= 4:
        new_shape = [1 for i in range(4 - len(shape))]
        new_shape.extend(shape)
    else:
        new_shape = shape
    return new_shape


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_input(image_path):
    img = Image.open(image_path)
    return img


def get_label(image_path):
    _, image_name = image_path.rsplit("/", 1)
    image_name_exact, _ = image_name.rsplit(".", 1)
    _, label_id = image_name_exact.rsplit("_", 1)
    label = list(IMAGENET2012_CLASSES).index(label_id)
    return label


def get_data(input_loc):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = sorted(glob.glob(data_path))
    examples = []
    for f1 in files:
        examples.append(
            InputExample(
                image=get_input(f1),
                label=get_label(f1),
            )
        )
    image_examples = examples

    return image_examples
