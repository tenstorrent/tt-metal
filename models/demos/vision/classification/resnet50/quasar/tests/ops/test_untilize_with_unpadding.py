# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the Quasar resnet50 `ttnn.experimental.quasar.untilize_with_unpadding`.

WHERE IT COMES FROM
-------------------
resnet50 (models/.../quasar/tt/ttnn_functional_resnet50.py, run()) uses it as the final op to strip
the fc output back to the real 1000-class logits and return to ROW_MAJOR:

    desired_shape = list(x.shape)
    desired_shape[-1] = 1000
    x = ttnn.experimental.quasar.untilize_with_unpadding(
        x,
        output_tensor_end=(desired_shape[0]-1, desired_shape[1]-1, desired_shape[2]-1, desired_shape[3]-1),
        memory_config=self.final_output_mem_config,
    )

The fc matmul output is TILE_LAYOUT with N padded from 1000 up to 1024 (32 tiles), so the op removes
the last 24 columns of W (1024 -> 1000) while untilizing. The `fc_1000` cases below reproduce exactly
that (W 1024 -> 1000); a small `unpad_both` case additionally trims the height to exercise the
row-unpadding path.

WHAT IT VALIDATES
-----------------
untilize_with_unpadding is a pure layout change + slice (TILE -> ROW_MAJOR, then drop the padding
region). The kept values are unchanged, so `ttnn.to_torch` of the output must equal the input sliced
to the unpadded shape. A wrong Quasar LLK binding (untilize pack/unpack, DFB in/out, unpad offsets)
corrupts the data or hangs. Input is bfloat16 TILE_LAYOUT (op requirement).

RUN
---
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_untilize_with_unpadding.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (input_shape [tile-padded, logical], unpadded_shape)
_CASES = [
    # resnet50 fc: N padded 1000 -> 1024; op trims W back to 1000. (rows = batch folded into height)
    pytest.param((1, 1, 32, 1024), (1, 1, 32, 1000), id="fc_1000_1tilerow"),
    pytest.param((1, 1, 64, 1024), (1, 1, 64, 1000), id="fc_1000_2tilerows"),
    # small case that also trims the height (row-unpadding path).
    pytest.param((1, 1, 64, 64), (1, 1, 30, 62), id="unpad_both"),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_shape, unpadded_shape", _CASES)
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_quasar_untilize_with_unpadding(mesh_device, input_shape, unpadded_shape, use_multicore):
    device = mesh_device
    torch.manual_seed(0)

    x = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor_end = tuple(d - 1 for d in unpadded_shape)
    out = ttnn.experimental.quasar.untilize_with_unpadding(
        tt_in,
        output_tensor_end=output_tensor_end,
        use_multicore=use_multicore,
    )

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(unpadded_shape)

    # Golden: the input sliced to the unpadded region (data preserved, padding dropped).
    golden = x[: unpadded_shape[0], : unpadded_shape[1], : unpadded_shape[2], : unpadded_shape[3]]
    assert_with_pcc(golden, got, pcc=0.999)
