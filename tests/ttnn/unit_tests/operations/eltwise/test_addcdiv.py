# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_addcdiv_fp32_overflow_order(device):
    """Regression test for #54055: addcdiv must divide before scaling by value.

    When |in1 * value| > FLT_MAX but value*(in1/in2) is representable,
    the old multiply-first order produced inf instead of a finite result.
    """
    in0 = torch.tensor([0.0], dtype=torch.float32)
    in1 = torch.tensor([3.0e38], dtype=torch.float32)
    in2 = torch.tensor([8.0], dtype=torch.float32)
    value = 4.0

    golden = torch.addcdiv(in0, in1, in2, value=value)

    in0_dev = ttnn.from_torch(
        in0.reshape(1, 1, 1, 1),
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    in1_dev = ttnn.from_torch(
        in1.reshape(1, 1, 1, 1),
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    in2_dev = ttnn.from_torch(
        in2.reshape(1, 1, 1, 1),
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    result = ttnn.to_torch(ttnn.addcdiv(in0_dev, in1_dev, in2_dev, value=value))

    assert not torch.isinf(result).any(), f"expected finite, got {result.item()}"
    assert_with_pcc(golden, result, 0.99)
