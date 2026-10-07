# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Elementwise add through the Quasar copy of binary_ng, checked against torch."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(32, 32), (64, 128)])
def test_quasar_add(device, shape):
    torch.manual_seed(0)
    a, b = torch.rand(shape), torch.rand(shape)

    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.to_torch(ttnn.experimental.quasar.add(tt_a, tt_b)).float()

    assert_with_pcc(a + b, out, pcc=0.999)
