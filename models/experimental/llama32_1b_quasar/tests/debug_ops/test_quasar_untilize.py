# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Metal 2.0 binding validation for ttnn.experimental.quasar.untilize (single-core factory).

use_multicore=False routes to UntilizeSingleCoreProgramFactory (the only untilize factory ported
to Metal 2.0 so far). untilize is a pure layout change (TILE -> ROW_MAJOR), so to_torch of the
output must round-trip the original data through the full reader -> compute -> writer pipeline;
a wrong Metal 2.0 binding (dfb in/out, TensorParameter input/output, named CTA/RTA) corrupts the
data or hangs. Other untilize factories remain on the legacy ProgramDescriptor concept
(use_multicore=True), so this test pins use_multicore=False.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 1, 32, 32), id="one_tile"),
        pytest.param((1, 1, 64, 64), id="4_tiles"),
        pytest.param((1, 2, 96, 64), id="rank4"),
        pytest.param((1, 1, 128, 256), id="wide"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_quasar_untilize_single_core(device, input_shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # use_multicore=False -> UntilizeSingleCoreProgramFactory (the ported Metal 2.0 factory)
    out = ttnn.experimental.quasar.untilize(tt_in, use_multicore=False)

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT

    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    # untilize is a pure layout change -> exact round-trip of the data through every binding.
    assert_with_pcc(x, got, 0.9999)


# MultiCoreBlock factory (the WH block path): a wide, multi-tile-per-row tensor with use_multicore=True
# routes here on a typical grid (num_tiles_per_row > 32 and num_compute_cores < ncores_wh). Now ported
# to Metal 2.0 -- it binds uwu's m2 WH reader/compute/writer kernels (reader_unary_interleaved_wh_multicore,
# untilize_wh, writer_unary_stick_layout_wh_multicore). wide_1tilerow exercises the cliff_row sub-region
# (full_cores_per_col == 0); wide_2tilerows adds a 2-tile-high column (cliff_col). If a given grid routes
# elsewhere the round-trip still validates correctness, just via a different factory.
#
# NOTE: a much taller+wider shape (e.g. (1,1,256,4096), num_tiles_per_col == 8) trips a compile-time
# assert inside the shared untilize_wh compute kernel: with that geometry split_blocks_for_tilize_wh
# produces a large square block whose block_width_tiles exceeds the pack_untilize DEST limit. This is a
# constraint of the shared kernel (identical CTA binding to untilize_with_unpadding's proven
# block_interleaved factory; reproducible there) -- NOT a host-side defect of this port (the assert is
# unchanged across every host RTA/region variant tried). See METAL2_PORT_REPORT.md. The cliff_row and
# small cliff_col paths the resnet model exercises are covered below.
@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 1, 32, 8192), id="wide_1tilerow"),
        pytest.param((1, 1, 64, 8192), id="wide_2tilerows"),
    ],
)
def test_quasar_untilize_multi_core_block(device, input_shape):
    torch.manual_seed(0)
    x = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.untilize(tt_in, use_multicore=True)

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    assert_with_pcc(x, got, 0.9999)


# MultiCore factory (the default interleaved multi-core path): use_multicore=True with a moderate width
# (num_tiles_per_row <= 32, so the block/parallelize_column checks don't fire) and >1 tile row. Now ported
# to Metal 2.0 -- interleaved mode binds reader_unary_start_id_metal2 + writer_..._multi_core_metal2 +
# untilize_variable_num_blocks_metal2. Multi-row shapes exercise the per-core block distribution and the
# interleaved cliff core. (The two sharded modes -- even-shard zero-copy DFB borrowed_from the input, and
# the block reader -- are build-validated; they require sharded input not reachable from this plain call,
# and are exercised via the resnet demo / integration.)
@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 1, 128, 256), id="mc_8x4tiles"),
        pytest.param((1, 1, 256, 512), id="mc_16x8tiles"),
        pytest.param((1, 2, 96, 128), id="mc_rank4"),
    ],
)
def test_quasar_untilize_multi_core(device, input_shape):
    torch.manual_seed(0)
    x = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.untilize(tt_in, use_multicore=True)

    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out).to(torch.bfloat16)
    assert tuple(got.shape) == tuple(input_shape)
    assert_with_pcc(x, got, 0.9999)


# NOTE: the parallelize_column factory (get_pf_type==0) cannot be deterministically selected from
# Python. Its precondition (a full row of tiles too wide for one CB) is always also wide enough to
# trip the earlier MultiCoreBlock check, so which of the two wins is grid-dependent. Its Metal 2.0
# bindings are validated at build time (kernels compile + named tokens resolve) and exercised at
# runtime via the resnet demo / integration, not by this unit test.
