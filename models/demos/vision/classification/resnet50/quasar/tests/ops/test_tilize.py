# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the Quasar resnet50 stem `ttnn.experimental.quasar.tilize`.

WHERE IT COMES FROM
-------------------
resnet50 (models/.../quasar/tt/ttnn_functional_resnet50.py, run()) tilizes the max-pool stem
activation before layer1:

    x = ttnn.experimental.quasar.tilize(x, dtype=self.model_config["ACTIVATIONS_DTYPE"])

The stem activation after max_pool2d is 56x56 spatial with 64 channels, so for batch 1 the tensor
is (1, 1, batch*56*56, 64) = (1, 1, 3136, 64) -> 98 row-tiles x 2 col-tiles. The shapes below cover
that representative geometry plus a couple of smaller multi-tile cases.

WHAT IT VALIDATES
-----------------
tilize is a pure layout change (ROW_MAJOR -> TILE): the values are unchanged, only the in-L1 layout
differs. So `ttnn.to_torch` of the output must round-trip the original data exactly through the full
reader -> compute -> writer pipeline. A wrong Quasar LLK binding (tilize pack/unpack, DFB in/out)
corrupts the data or hangs. The golden is therefore just the input tensor itself. tilize requires the
last two dims to be tile-aligned (multiples of 32); all shapes below satisfy this.

RUN
---
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_tilize.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 1, 64, 64), id="4_tiles"),
        pytest.param((1, 1, 128, 256), id="wide"),
        pytest.param((1, 1, 3136, 64), id="stem_maxpool_b1"),  # resnet50 stem, batch 1
    ],
)
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_quasar_tilize(mesh_device, input_shape, use_multicore):
    device = mesh_device
    torch.manual_seed(0)

    x = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.tilize(tt_in, dtype=ttnn.bfloat16, use_multicore=use_multicore)

    assert out.layout == ttnn.TILE_LAYOUT
    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)

    # tilize is a pure layout change -> exact round-trip of the data through every LLK binding.
    assert_with_pcc(x, got, pcc=0.999)
