# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-measurement shapes for g6_to_layout (run under run_safe_pytest.sh --profile)."""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout


@pytest.mark.parametrize(
    "shape",
    [(1, 1024, 1024, 3), (1, 512, 2048, 128), (8, 1024, 2560), (1, 1, 131072), (1, 32, 2560)],
    ids=["p4_w3", "p2_half", "qwen_8x1024", "w131072", "c2003"],
)
def test_perf(device, shape):
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = g6_to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(out).reshape(shape).view(torch.int16), x.view(torch.int16))
