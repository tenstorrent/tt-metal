# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The whole Blackhole pack sweep space for counter attribution: every valid pair of the six formats, every tile shape from 32x32 to 128x128, dest indices
0, 1, 3 and 7 (the first and the last tile of a DEST section, the sweep of perf_pack), both syncs, for the per-tile pack, the block pack, the W offset block pack and the blocks of two and four tiles from a dest index.
perf_pack and perf_pack_cov2 run the part of it that the other pack perf modules do not."""

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes, get_valid_dest_indices
from helpers.format_config import DataFormat
from helpers.llk_params import DestSync, PackerReluType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_pack import test_pack as run_pack

F = DataFormat
COV3_FORMATS = input_output_formats([F.Float16_b, F.Float16, F.Float32, F.Int32, F.Bfp8_b, F.Bfp4_b])
TILE_DIMS = [[32, 32], [32, 64], [64, 32], [64, 64], [32, 128], [128, 64], [128, 128]]
BLOCK_DIMS = TILE_DIMS


def _indices(wanted, all_indices=False):
    return lambda dest_acc, dest_sync, formats, input_dimensions: [
        i for i in get_valid_dest_indices(dest_sync, dest_acc, formats, input_dimensions, all_indices=all_indices) if i in wanted
    ]


COV3_SWEEP = dict(
    formats=COV3_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    input_dimensions=TILE_DIMS,
    relu_type=[PackerReluType.NoRelu],
    dest_sync=[DestSync.Half, DestSync.Full],
    dest_index=_indices((0, 1, 3, 7)),
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**COV3_SWEEP)
def test_perf_pack_cov3(perf_report, formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types, loop_factor, is_perf):
    run_pack(formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types=run_types, loop_factor=loop_factor, is_perf=is_perf, perf_report=perf_report)


COV3_BLOCK_SWEEP = {**COV3_SWEEP, "input_dimensions": BLOCK_DIMS, "dest_index": [0]}


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**COV3_BLOCK_SWEEP)
def test_perf_pack_cov3_block(perf_report, formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types, loop_factor, is_perf):
    run_pack(
        formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types=run_types, loop_factor=loop_factor, is_perf=is_perf, perf_report=perf_report, pack_block=True
    )


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**COV3_BLOCK_SWEEP)
def test_perf_pack_cov3_block_w(perf_report, formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types, loop_factor, is_perf):
    """The block packs with the W offset forms (no destination write per call)."""
    run_pack(
        formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, run_types=run_types, loop_factor=loop_factor, is_perf=is_perf, perf_report=perf_report, pack_block=3
    )


COV3_BLOCK_IDX_SWEEP = {
    **COV3_SWEEP,
    "input_dimensions": [[64, 64], [32, 128], [128, 64]],
    "dest_index": _indices((1, 3, 7), all_indices=True),
    "block_tiles": [2, 4],
    "pack_block": [False, 2, 3],
}


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**COV3_BLOCK_IDX_SWEEP)
def test_perf_pack_cov3_block_idx(
    perf_report, formats, dest_acc, input_dimensions, relu_type, dest_sync, dest_index, block_tiles, pack_block, run_types, loop_factor, is_perf
):
    if pack_block == 2 and formats.output_format.is_block_float() and block_tiles == 2 and dest_index % 4 == 3:
        pytest.skip("two block-float tiles that straddle a four tile boundary of DEST: llk_pack_block takes the per-tile path")
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
        pack_block=pack_block,
        block_tiles=block_tiles,
    )
