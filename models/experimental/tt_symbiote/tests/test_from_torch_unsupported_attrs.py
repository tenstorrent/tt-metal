# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks that from_torch converters reject torch attributes the device path does not implement."""

import pytest
from torch import nn

from models.experimental.tt_symbiote.modules.activation import TTNNGelu
from models.experimental.tt_symbiote.modules.conv import (
    TTNNConv2dNHWC,
    TTNNMaxPool2dNHWC,
    TTNNUpsampleNHWC,
    _pool_output_size,
)
from models.tt_cnn.tt.builder import Conv2dConfiguration, MaxPool2dConfiguration, UpsampleConfiguration


def test_gelu_keeps_approximate():
    assert TTNNGelu.from_torch(nn.GELU()).approximate == "none"
    assert TTNNGelu.from_torch(nn.GELU(approximate="tanh")).approximate == "tanh"


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_conv_rejects_padding_mode(padding_mode, expect_error):
    conv = nn.Conv2d(3, 4, 3, padding=1, padding_mode=padding_mode)
    with expect_error(ValueError, "padding_mode"):
        TTNNConv2dNHWC.from_torch(conv)
    with expect_error(ValueError, "padding_mode"):
        Conv2dConfiguration.from_torch(conv, input_height=8, input_width=8, batch_size=1)


@pytest.mark.parametrize("padding", ["same", "valid"])
def test_conv_rejects_string_padding(padding, expect_error):
    conv = nn.Conv2d(3, 4, 3, padding=padding)
    with expect_error(ValueError, "String padding"):
        TTNNConv2dNHWC.from_torch(conv)
    with expect_error(ValueError, "String padding"):
        Conv2dConfiguration.from_torch(conv, input_height=8, input_width=8, batch_size=1)


def test_maxpool_rejects_return_indices(expect_error):
    pool = nn.MaxPool2d(3, 2, return_indices=True)
    with expect_error(NotImplementedError, "return_indices"):
        TTNNMaxPool2dNHWC.from_torch(pool)
    with expect_error(NotImplementedError, "return_indices"):
        MaxPool2dConfiguration.from_torch(pool, input_height=8, input_width=8, batch_size=1, channels=32)


def test_maxpool_ceil_mode_is_forwarded():
    assert TTNNMaxPool2dNHWC.from_torch(nn.MaxPool2d(3, 2, ceil_mode=True)).ceil_mode is True
    assert TTNNMaxPool2dNHWC.from_torch(nn.MaxPool2d(3, 2)).ceil_mode is False
    cfg = MaxPool2dConfiguration.from_torch(
        nn.MaxPool2d(3, 2, ceil_mode=True), input_height=10, input_width=10, batch_size=1, channels=32
    )
    assert cfg.ceil_mode is True


@pytest.mark.parametrize("size", range(4, 20))
@pytest.mark.parametrize("kernel,stride,padding,dilation", [(3, 2, 0, 1), (3, 2, 1, 1), (2, 2, 0, 1), (3, 3, 1, 2)])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_pool_output_size_matches_torch(size, kernel, stride, padding, dilation, ceil_mode):
    import torch

    pool = nn.MaxPool2d(kernel, stride, padding, dilation, ceil_mode=ceil_mode)
    expected = pool(torch.zeros(1, 1, size, size)).shape[-1]
    assert _pool_output_size(size, kernel, stride, padding, dilation, ceil_mode) == expected


def test_upsample_rejects_unsupported(expect_error):
    with expect_error(NotImplementedError, "align_corners"):
        TTNNUpsampleNHWC.from_torch(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
    with expect_error(NotImplementedError, "align_corners"):
        UpsampleConfiguration.from_torch(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            input_height=8,
            input_width=8,
            batch_size=1,
            channels=32,
        )
    with expect_error(ValueError, "integer scale_factor"):
        TTNNUpsampleNHWC.from_torch(nn.Upsample(scale_factor=2.5, mode="nearest"))
    assert TTNNUpsampleNHWC.from_torch(nn.Upsample(scale_factor=2, mode="nearest")).scale_factor == 2
