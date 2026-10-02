# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration and helpers for the detection head tests."""

import torch
import torch.nn as nn

from models.experimental.bevformer.config.decoder_config import CODE_SIZE, REG_XY, REG_Z
from models.experimental.bevformer.config.head_config import BOX_CENTRE, BOX_SIZE, BOX_VELOCITY, BOX_YAW
from models.experimental.bevformer.reference.head import BEVFormerHead
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.decoder_common import EMBED_DIMS, NUM_QUERY, build_reference_decoder


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


def _centre_channels():
    channels = list(range(CODE_SIZE))
    return channels[REG_XY] + channels[REG_Z]


def assert_box_codes_close(expected, actual, pcc=0.99):
    """PCC of the centres (metres) and of the other channels apart: the centres span
    ``PC_RANGE`` and would carry a joint PCC alone."""
    centre = _centre_channels()
    rest = [c for c in range(CODE_SIZE) if c not in centre]
    assert_pcc(expected[..., centre], actual[..., centre], pcc)
    assert_pcc(expected[..., rest], actual[..., rest], pcc)


def assert_boxes_close(expected, actual, pcc=0.99):
    """PCC of the decoded boxes' centres, sizes, yaw and velocities apart. Yaw is compared
    through its sine and cosine, as atan2 may land on either side of +-pi."""
    for channels in (BOX_CENTRE, BOX_SIZE, BOX_VELOCITY):
        assert_pcc(expected[..., channels], actual[..., channels], pcc)
    yaw_expected, yaw_actual = expected[..., BOX_YAW], actual[..., BOX_YAW]
    assert_pcc(
        torch.cat([yaw_expected.sin(), yaw_expected.cos()], dim=-1),
        torch.cat([yaw_actual.sin(), yaw_actual.cos()], dim=-1),
        pcc,
    )
