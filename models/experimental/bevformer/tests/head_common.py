# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration and helpers for the detection head tests."""

from loguru import logger
import torch
import torch.nn as nn

from models.experimental.bevformer.config.decoder_config import CODE_SIZE, CODE_XY, CODE_Z
from models.experimental.bevformer.config.head_config import BOX_CENTER, BOX_SIZE, BOX_VELOCITY, BOX_YAW
from models.experimental.bevformer.reference.head import BEVFormerHead
from models.experimental.bevformer.tests.decoder_common import (
    EMBED_DIMS,
    NUM_QUERY,
    assert_channels_close,
    build_reference_decoder,
    init_reg_branches,
)


def build_reference_head(bev_shape, seed=0):
    """BEVFormer's head with dummy weights over a ``bev_shape`` BEV map, all drawn from ``seed``.

    The decoder is ``build_reference_decoder``'s, the reg branches get ``init_reg_branches``'
    scale, and ``reference_points`` gets BEVFormer's xavier init, which spreads the initial
    points over the BEV grid. The rest keeps PyTorch's default init.
    """
    torch.manual_seed(seed)
    model = BEVFormerHead(*bev_shape, num_query=NUM_QUERY, embed_dims=EMBED_DIMS)
    nn.init.xavier_uniform_(model.reference_points.weight)
    nn.init.zeros_(model.reference_points.bias)
    init_reg_branches(model.reg_branches)
    model.decoder = build_reference_decoder(seed)
    return model.eval().requires_grad_(False)


# Box center error bounds in metres, about twice what the head's dummy-weight tests measure: PCC
# alone allows metres of error on x and y, which span 102.4 m, where nuScenes matches boxes from
# 0.5 m.
CENTER_MEAN_ATOL = 0.01
CENTER_MAX_ATOL = 0.15


def center_channels():
    """Indexes of cx, cy and cz in the box code, which are not contiguous."""
    channels = list(range(CODE_SIZE))
    return channels[CODE_XY] + channels[CODE_Z]


def assert_centers_close(expected, actual):
    """The box predictions' center channels within ``CENTER_MEAN_ATOL`` and ``CENTER_MAX_ATOL``."""
    centers = center_channels()
    error = (expected[..., centers] - actual[..., centers]).abs()
    logger.info(f"box center error: mean {error.mean():.4f} m, max {error.max():.4f} m")
    assert error.mean() <= CENTER_MEAN_ATOL, f"mean box center error {error.mean():.4f} m"
    assert error.max() <= CENTER_MAX_ATOL, f"max box center error {error.max():.4f} m"


def assert_boxes_close(expected, actual, pcc=0.99):
    """PCC of every decoded box channel apart, yaw through its sine and cosine, as atan2
    may land on either side of +-pi."""
    for channels in (BOX_CENTER, BOX_SIZE, BOX_VELOCITY):
        assert_channels_close(expected[..., channels], actual[..., channels], pcc)
    yaw_expected, yaw_actual = expected[..., BOX_YAW], actual[..., BOX_YAW]
    assert_channels_close(
        torch.cat([yaw_expected.sin(), yaw_expected.cos()], dim=-1),
        torch.cat([yaw_actual.sin(), yaw_actual.cos()], dim=-1),
        pcc,
    )
