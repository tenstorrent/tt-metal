# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Functional test for `ttnn.experimental.quasar.simple_add`: C = A + B on a single node, one reader, one
compute and one writer thread, with bfloat16 TILE-layout DRAM-interleaved tensors.

The op is built with the Metal 2.0 host API, so the same test runs on Wormhole/Blackhole (CB-backed DFBs)
and on Quasar (overlay-backed DFBs):
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_simple_add.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # one tile
        (1, 1, 256, 256),  # 64 tiles
        (2, 3, 64, 128),  # 48 tiles over batch dims
    ],
)
def test_simple_add(device, shape):
    torch.manual_seed(0)
    a_pt = torch.randn(shape, dtype=torch.bfloat16)
    b_pt = torch.randn(shape, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(
        a_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_tt = ttnn.from_torch(
        b_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    c_tt = ttnn.experimental.quasar.simple_add(a_tt, b_tt)

    assert c_tt.shape == a_tt.shape
    assert c_tt.dtype == ttnn.bfloat16
    assert c_tt.layout == ttnn.TILE_LAYOUT

    golden = a_pt + b_pt
    c_pt = ttnn.to_torch(c_tt)
    assert_with_pcc(c_pt, golden, 0.9999)
    # The FPU adds in bfloat16, so the result is the bfloat16-rounded sum.
    assert torch.allclose(c_pt.float(), golden.float(), rtol=1e-2, atol=1e-2)


def test_simple_add_rejects_mismatched_shapes(device, expect_error):
    a_tt = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(torch.randn((1, 1, 32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "same shape"):
        ttnn.experimental.quasar.simple_add(a_tt, b_tt)
