# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration and helpers for the detection head tests."""

import torch
import torch.nn as nn

from models.experimental.bevformer.config.decoder_config import CODE_SIZE, CODE_XY, CODE_Z
from models.experimental.bevformer.config.head_config import BOX_CENTRE, BOX_SIZE, BOX_VELOCITY, BOX_YAW
from models.experimental.bevformer.reference.head import BEVFormerHead
from models.experimental.bevformer.tests.decoder_common import (
    EMBED_DIMS,
    NUM_QUERY,
    assert_channels_close,
    build_reference_decoder,
)


def build_reference_head(bev_shape, seed=0):
    """BEVFormer's head with dummy weights over a ``bev_shape`` BEV map, all drawn from ``seed``.

    The decoder is ``build_reference_decoder``'s, and ``reference_points`` gets BEVFormer's
    xavier init, which spreads the initial points over the BEV grid. The rest keeps
    PyTorch's default init.
    """
    torch.manual_seed(seed)
    model = BEVFormerHead(*bev_shape, num_query=NUM_QUERY, embed_dims=EMBED_DIMS)
    nn.init.xavier_uniform_(model.reference_points.weight)
    nn.init.zeros_(model.reference_points.bias)
    model.decoder = build_reference_decoder(seed)
    return model.eval().requires_grad_(False)


def centre_channels():
    """The box code's cx, cy and cz channels."""
    channels = list(range(CODE_SIZE))
    return channels[CODE_XY] + channels[CODE_Z]


def assert_boxes_close(expected, actual, pcc=0.99):
    """PCC of every decoded box channel apart, yaw through its sine and cosine, as atan2
    may land on either side of +-pi."""
    for channels in (BOX_CENTRE, BOX_SIZE, BOX_VELOCITY):
        assert_channels_close(expected[..., channels], actual[..., channels], pcc)
    yaw_expected, yaw_actual = expected[..., BOX_YAW], actual[..., BOX_YAW]
    assert_channels_close(
        torch.cat([yaw_expected.sin(), yaw_expected.cos()], dim=-1),
        torch.cat([yaw_actual.sin(), yaw_actual.cos()], dim=-1),
        pcc,
    )
