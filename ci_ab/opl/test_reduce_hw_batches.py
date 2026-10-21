# Scratch copy of tests/ttnn/unit_tests/operations/reduce/test_reduce_hw_batches.py, not for merge.
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# The H and W reduction accumulates each batch's tiles into one DEST tile, one block call per batch on one init, with odd and even tile counts.

import pytest
import torch

import ttnn


def assert_with_pcc(expected, actual, pcc):
    got = torch.corrcoef(torch.stack([expected.flatten().float(), actual.flatten().float()]))[0, 1].item()
    assert got >= pcc, got


@pytest.mark.parametrize("shape", [(1, 8, 96, 96), (2, 3, 64, 160), (4, 1, 32, 672), (1, 3, 128, 128), (1, 5, 32, 64)])
@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_reduce_hw_batches(device, shape, op, dtype):
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = getattr(ttnn, op)(x, dim=[2, 3], keepdim=True)
    expected = getattr(torch, op)(t, dim=(2, 3), keepdim=True)
    assert_with_pcc(expected, ttnn.to_torch(out).to(torch.float32), 0.999)
