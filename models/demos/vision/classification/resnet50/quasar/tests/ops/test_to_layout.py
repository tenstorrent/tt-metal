# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for ttnn.experimental.quasar.to_layout as used by resnet50/quasar.

to_layout is a value-preserving layout conversion: it re-packs the same elements between TILE and
ROW_MAJOR without changing shape or values, so a to_torch of the output must round-trip the input
exactly (PCC ~1.0). This isolates the to_layout call-site so the LLK team can test/fix it alone.

Call-site mirrored (see ttnn_functional_resnet50.py, resnet50Bottleneck.__call__, ~line 280):
    x_rm = ttnn.experimental.quasar.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
  -> TILE_LAYOUT -> ROW_MAJOR_LAYOUT (a defrag step before reallocate). We cover both directions.

Run (craq-sim example):
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
    pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_to_layout.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 1, 32, 32), id="one_tile"),
        pytest.param((1, 1, 128, 256), id="multi_tile"),
        pytest.param((1, 2, 96, 128), id="rank4"),
    ],
)
@pytest.mark.parametrize(
    "src_layout, dst_layout",
    [
        # The resnet call-site: TILE -> ROW_MAJOR.
        pytest.param(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, id="tile_to_rm"),
        # Reverse direction for completeness (also value-preserving).
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, id="rm_to_tile"),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_to_layout(mesh_device, input_shape, src_layout, dst_layout):
    torch.manual_seed(0)
    device = mesh_device

    x = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=src_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.to_layout(tt_in, dst_layout)

    # The output must be in the requested layout, same shape.
    assert out.layout == dst_layout
    assert tuple(out.shape) == tuple(input_shape)

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)

    # Pure layout change -> exact round-trip of the data.
    assert_with_pcc(x, got, pcc=0.999)
