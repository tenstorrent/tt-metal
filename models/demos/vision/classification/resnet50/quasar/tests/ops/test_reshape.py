# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for ttnn.experimental.quasar.reshape as used by resnet50/quasar.

reshape is a pure metadata/value-preserving op: it changes the logical shape but must round-trip
every element unchanged. This isolates the two reshape call-sites in
models/demos/vision/classification/resnet50/quasar/tt/ttnn_functional_resnet50.py so the LLK team
can test/fix reshape alone with a PCC check (expect ~1.0).

Call-sites mirrored (see ttnn_functional_resnet50.py):
  1. fold-output collapse (run(), ~line 711):
       reshape(fold_output, (1, 1, n*c*h, w))   # (n, c, h, w) -> (1, 1, n*c*h, w), ROW_MAJOR
  2. fc-output split (run(), ~line 1037):
       reshape(x, (batch, 1, x.shape[2] // batch, 1000))  # (1,1,batch*32,1000) -> (batch,1,32,1000)
     (fc output arrives ROW_MAJOR from untilize_with_unpadding.)

Run (craq-sim example):
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
    pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_reshape.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape, target_shape",
    [
        # call-site 1: fold-output collapse (n, c, h, w) -> (1, 1, n*c*h, w)
        pytest.param((4, 1, 64, 16), (1, 1, 4 * 1 * 64, 16), id="fold_collapse"),
        pytest.param((16, 1, 60, 16), (1, 1, 16 * 1 * 60, 16), id="fold_collapse_b16"),
        # call-site 2: fc-output split (1, 1, batch*32, 1000) -> (batch, 1, 32, 1000)
        pytest.param((1, 1, 4 * 32, 1000), (4, 1, 32, 1000), id="fc_split_b4"),
        pytest.param((1, 1, 16 * 32, 1000), (16, 1, 32, 1000), id="fc_split_b16"),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_reshape(mesh_device, input_shape, target_shape):
    torch.manual_seed(0)
    device = mesh_device

    x = torch.rand(input_shape, dtype=torch.bfloat16)

    # Both resnet reshape call-sites operate on ROW_MAJOR tensors (fold output / untilize output).
    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.reshape(tt_in, target_shape)

    # Shape must be exactly the requested target.
    assert tuple(out.shape) == tuple(target_shape)

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(target_shape)

    # reshape preserves values -> the torch golden is just the input reshaped the same way.
    golden = x.reshape(target_shape)
    assert_with_pcc(golden, got, pcc=0.999)
