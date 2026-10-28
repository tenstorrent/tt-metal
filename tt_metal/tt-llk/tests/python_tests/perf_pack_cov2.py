# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Coverage sample of the pack perf population: what perf_pack and perf_pack_dest_bank never run.

perf_pack sweeps Float16_b, Float32, Int32 and Bfp8_b in and out at dest index 0 on 32x32 to 64x64 (the block rows to 128x128). This module adds
  * Bfp4_b and Float16 on either side of the pack, every valid pair of the six formats;
  * dest indices 1 and 3 for the per-tile pack (the pack starts mid DEST);
  * the 32x128 shape (four tiles in a row);
for both the per-tile pack (_llk_pack_) and the block packs (pack_block=True): the dense fast block for 16-bit and 32-bit outputs and the closed
fast block, replayed per tile, for the block-float outputs. Tiny tiles and partial faces have no perf kernel (pack_tiny_tile_block_test.cpp is
functional only)."""

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes, get_valid_dest_indices
from helpers.format_config import DataFormat
from helpers.llk_params import DestSync, PackerReluType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_pack import test_pack as run_pack

F = DataFormat
COV2_FORMATS = input_output_formats([F.Float16_b, F.Float16, F.Float32, F.Int32, F.Bfp8_b, F.Bfp4_b])
COV2_DIMS = [[64, 64], [32, 128], [128, 64]]


def _dest_indices(dest_acc, dest_sync, formats, input_dimensions):
    return [i for i in get_valid_dest_indices(dest_sync, dest_acc, formats, input_dimensions) if i in (0, 1, 3)]


COV2_SWEEP = dict(
    formats=COV2_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    input_dimensions=COV2_DIMS,
    relu_type=[PackerReluType.NoRelu],
    dest_sync=[DestSync.Half, DestSync.Full],
    dest_index=_dest_indices,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**COV2_SWEEP)
def test_perf_pack_cov2(perf_report, formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types, loop_factor, is_perf):
    run_pack(
        formats,
        dest_acc,
        input_dimensions,
        relu_type,
        dest_sync,
        dest_index,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


# The block packs run at dest index 0: from index 3 the harness splits the tiles into one tile calls, which pack with the per-tile program behind the
# block init and measure that mix, not the block code.
COV2_BLOCK_SWEEP = {**COV2_SWEEP, "dest_index": [0]}


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**COV2_BLOCK_SWEEP)
def test_perf_pack_cov2_block(
    perf_report, formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types, loop_factor, is_perf
):
    run_pack(
        formats,
        dest_acc,
        input_dimensions,
        relu_type,
        dest_sync,
        dest_index,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        pack_block=True,
    )
