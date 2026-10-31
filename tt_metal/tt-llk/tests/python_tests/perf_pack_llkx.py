# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every packer conversion the Blackhole pack LLK accepts (llk_pack_space.llk_accepts) that perf_pack, perf_pack_cov2, perf_pack_cov3 and perf_pack_grid do not run:
Tf32, Bfp8 (A exponent), Bfp2_b, Fp8_e4m3, Int8 and UInt8 outputs from float and Int32 DEST, Float32 and Float16_b outputs from Int32, UInt8 and Int8 into each other,
and Tf32 and Bfp8_b inputs, with the per-tile pack (pack_block False), the default block (pack_block 2) and the W offset block (pack_block 3) of 1, 2, 4, 8 and
16 tiles from dest index 0 and the last tile that fits."""

import pytest
from conftest import skip_for_wormhole
from helpers.llk_params import DestSync, PackerReluType
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from llk_pack_space import llkx_accs, llkx_formats
from perf_pack_grid import _grid_dims, _grid_indices
from test_pack import test_pack as run_pack

LLKX_BLOCKS = [1, 2, 4, 8, 16]


def _llkx_blocks(dest_acc, dest_sync, formats):
    from perf_pack_grid import _section_tiles

    section = _section_tiles(dest_sync, dest_acc, formats)
    return [n for n in LLKX_BLOCKS if n <= section]


def _llkx_kinds(block_tiles):
    return [False] if block_tiles < 4 else [False, 2, 3]


def _llkx_indices(dest_acc, dest_sync, formats, block_tiles):
    idx = _grid_indices(dest_acc, dest_sync, formats, block_tiles)
    return [i for i in idx if i in (0, max(idx))] if idx else []


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=llkx_formats(),
    dest_acc=llkx_accs,
    relu_type=[PackerReluType.NoRelu],
    dest_sync=[DestSync.Half, DestSync.Full],
    block_tiles=_llkx_blocks,
    input_dimensions=_grid_dims,
    dest_index=_llkx_indices,
    pack_block=_llkx_kinds,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_llkx(
    perf_report, formats, dest_acc, relu_type, dest_sync, block_tiles, input_dimensions, dest_index, pack_block, run_types, loop_factor, is_perf
):
    """Per-tile, default block and W offset packs of block_tiles tiles from dest_index for the packer conversions of the LLK matrix."""
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
        allow_int32_mix=True,
    )
