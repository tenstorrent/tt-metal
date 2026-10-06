# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Inputs shared by the perception-transformer and detector tests."""

import math

import torch

from models.experimental.bevformer.reference.bevformer import relative_can_bus
from models.experimental.bevformer.reference.perception_transformer import CAN_BUS_DIMS, PerceptionTransformer
from models.experimental.bevformer.tests.encoder_common import (
    EMBED_DIMS,
    NUM_CAMS,
    SPATIAL_SHAPES,
    _smooth,
    build_reference_encoder,
    img_metas,
)

# Ego motion between consecutive frames, sample ``b`` scaled by ``b + 1``: a few BEV cells of
# translation and a heading change large enough that the previous BEV's rotation moves most cells.
EGO_STEP_M = (2.0, -1.0)
EGO_TURN_DEG = 5.0


def build_reference_transformer(num_layers, seed=0):
    """``PerceptionTransformer`` over ``build_reference_encoder``'s encoder; the embeddings and the
    CAN-bus MLP keep upstream's and PyTorch's init."""
    torch.manual_seed(seed)
    return PerceptionTransformer(build_reference_encoder(num_layers, seed)).eval().requires_grad_(False)


def random_fpn_levels(batch_size, generator, spatial_shapes=SPATIAL_SHAPES):
    """Unit-variance FPN-like levels ``(bs, num_cams, C, h, w)``, smooth as the encoder tests'."""
    levels = []
    for h, w in spatial_shapes:
        level = _smooth(batch_size * NUM_CAMS, EMBED_DIMS, h, w, generator)
        levels.append((level / level.std()).view(batch_size, NUM_CAMS, EMBED_DIMS, h, w))
    return levels


def fpn_rows(level):
    """A ``(bs, num_cams, C, h, w)`` level as the TTNN FPN emits it, ``(1, 1, bs * num_cams * h * w, C)``."""
    return level.permute(0, 1, 3, 4, 2).reshape(1, 1, -1, level.shape[2])


def frame_metas(batch_size, num_frames, generator, yaw_step_deg=0.0):
    """Per frame, ``img_metas`` with the relative CAN bus of an ego that moves by ``EGO_STEP_M`` and
    turns by ``EGO_TURN_DEG`` per frame, both times ``b + 1`` for sample ``b``. The CAN bus's other
    readings are random."""
    frames, previous = [], [None] * batch_size
    for i in range(num_frames):
        metas = img_metas(batch_size, yaw_step_deg=yaw_step_deg)
        for b, meta in enumerate(metas):
            can_bus = torch.randn(CAN_BUS_DIMS, generator=generator, dtype=torch.float64)
            heading = 0.3 + 0.2 * b + i * math.radians(EGO_TURN_DEG) * (b + 1)
            can_bus[0] = i * EGO_STEP_M[0] * (b + 1)
            can_bus[1] = i * EGO_STEP_M[1] * (b + 1)
            can_bus[2] = 0.0
            can_bus[-2] = heading
            can_bus[-1] = math.degrees(heading)
            meta["can_bus"] = relative_can_bus(can_bus, previous[b]).numpy()
            previous[b] = can_bus
        frames.append(metas)
    return frames
