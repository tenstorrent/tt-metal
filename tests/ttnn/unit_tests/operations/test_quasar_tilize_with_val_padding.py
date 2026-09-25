# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Metal 2.0 binding validation for ttnn.experimental.quasar.tilize_with_val_padding /
tilize_with_zero_padding.

Each parametrization is steered (via use_multicore + output width) at a *specific*
ported program factory, so a wrong Metal 2.0 binding (DFB in/out, TensorParameter
input/output, named CTA/RTA, or the reader's BlockRep runtime-varargs) shows up as
either wrong numerics or a hang:

  - use_multicore=False                         -> TilizeWithValPaddingSingleCoreFactory
  - use_multicore=True, padded W <= 1024 (<=32  -> TilizeWithValPaddingMultiCoreDefaultFactory
    tiles/row, so it does not route to block-       (exercises the per-core named RTAs + the
    interleaved); >1 block so work splits)          BlockRep RLE varargs on the reader, and
                                                     full + cliff compute KernelSpecs)

The other two factories (block_interleaved, sharded) are not yet ported to Metal 2.0;
shapes here deliberately avoid routing to them (narrow width, interleaved/DRAM input).
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _nearest_32(x):
    return math.ceil(x / 32) * 32


# (use_multicore, input_shape, output_shape) — output_shape tile-aligned in last 2 dims, W<=1024.
_VAL_PADDING_CASES = [
    # ---- single_core (use_multicore=False) ----
    pytest.param(False, (1, 1, 30, 62), (1, 1, 32, 64), id="single_core-pad_both"),
    pytest.param(False, (1, 1, 1, 1), (1, 1, 32, 32), id="single_core-heavy_pad"),
    pytest.param(False, (1, 1, 64, 64), (1, 1, 64, 64), id="single_core-no_pad"),
    pytest.param(False, (1, 2, 33, 31), (1, 2, 64, 32), id="single_core-rank4"),
    # ---- multi_core_default (use_multicore=True, narrow width -> not block-interleaved) ----
    # height + width padding with non-tile-aligned input rows -> varied BlockReps (data/mixed/pad)
    pytest.param(True, (1, 1, 400, 62), (1, 1, 416, 64), id="mc_default-mixed_blocks"),
    pytest.param(True, (1, 1, 4096, 64), (1, 1, 4096, 64), id="mc_default-many_blocks_no_pad"),
    # block count chosen high + non-round to make a compute cliff likely (grid-dependent)
    pytest.param(True, (1, 1, 4130, 62), (1, 1, 4160, 64), id="mc_default-cliff_likely"),
    pytest.param(True, (1, 1, 100, 30), (1, 1, 128, 32), id="mc_default-small_pad"),
]


@pytest.mark.parametrize("use_multicore, input_shape, output_shape", _VAL_PADDING_CASES)
@pytest.mark.parametrize("pad_value", [0.0, 7.0])
def test_quasar_tilize_with_val_padding(device, use_multicore, input_shape, output_shape, pad_value):
    torch.manual_seed(0)
    x = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.tilize_with_val_padding(
        tt_in,
        ttnn.Shape(output_shape),
        pad_value,
        use_multicore=use_multicore,
    )

    assert tuple(out.padded_shape) == tuple(output_shape)
    assert out.layout == ttnn.TILE_LAYOUT

    # tilize_with_val_padding keeps the output's LOGICAL shape = input shape; the padding lives in
    # the tile-padding region (padded_shape == output_shape, asserted above). ttnn.to_torch returns
    # the logical region, so it must round-trip the original data through the full
    # reader -> compute -> writer pipeline (every Metal 2.0 binding exercised; a wrong binding
    # corrupts the data or hangs). tilize is a pure layout change, so the round-trip is exact.
    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    assert_with_pcc(x, got, 0.9999)


# tilize_with_zero_padding routes through the same device op / factories (output padded to the
# nearest tile, pad value 0), so it re-exercises the same bindings via its own pybind entry point.
@pytest.mark.parametrize(
    "use_multicore, input_shape",
    [
        pytest.param(False, (1, 1, 30, 62), id="single_core"),
        pytest.param(True, (1, 1, 400, 62), id="mc_default"),
    ],
)
def test_quasar_tilize_with_zero_padding(device, use_multicore, input_shape):
    torch.manual_seed(0)
    x = torch.rand(input_shape, dtype=torch.bfloat16)
    output_shape = list(input_shape)
    output_shape[-1] = _nearest_32(output_shape[-1])
    output_shape[-2] = _nearest_32(output_shape[-2])

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.tilize_with_zero_padding(tt_in, use_multicore=use_multicore)

    assert tuple(out.padded_shape) == tuple(output_shape)
    assert out.layout == ttnn.TILE_LAYOUT

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    assert_with_pcc(x, got, 0.9999)
