# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The weight format is a fixed property of the model: every projection and the LM head are BF8, with no switch."""

import ttnn
from models.demos.gemma4_31b_qb2.tt import decoder, model


def test_every_projection_and_the_head_are_bf8():
    assert set(decoder.WEIGHT_DTYPES) == {"qkv", "o", "gate_up", "down"}
    assert all(dtype == ttnn.bfloat8_b for dtype in decoder.WEIGHT_DTYPES.values())
    assert model.HEAD_DTYPE == ttnn.bfloat8_b
