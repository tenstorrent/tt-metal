# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Blackhole-only experimental SFPU op sdpa_reduce_row
(tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_reduce_row.h, promoted through
experimental/llk_sfpu/{ckernel,llk_math}_sfpu_sdpa_reduce_row.h and consumed by
hw/inc/api/compute/experimental/sdpa.h).

The op is a ROW reduction over an "8x32" tile held in one 16x16 Dest face (sdpa.h: "Each tile is
8x32, which is the same as a full 16x16 face"): each of the 8 logical rows is reduced across its 32
columns down to one scalar (MAX or SUM pool), written into column 0 of that row.

Which 32 Dest datums form a logical row follows from the op's loads (see the golden derivation
block at the top of sources/sdpa_reduce_row_test.cpp): an SFPLOAD at Dest offset o reads the
aligned group of four rows o & ~3, one row per 8-lane subvector, taking the even columns when
(o & 2) == 0 and the odd columns otherwise
(https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md).
The op folds offsets 0, 2, 8, 10 into one register and 4, 6, 12, 14 into another, so subvector s
of those registers holds all 16 columns of face rows s and s + 8 (resp. 4 + s and 12 + s). The
epilogue then reduces each 8-lane subvector into its lane 0 and the stores (offsets 0 and 4) land
that lane in column 0 of face rows 0-7. So, in row-major 32x32 tile coordinates:

    logical row r (r = 0..7) = tile[r, 0:16] ++ tile[r + 8, 0:16]
    output                   = tile_out[r, 0]       MAX -> max(row),  SUM -> sum(row)

Only those eight output lanes are defined; every other lane of the tile is left as the op found
it (the A2D datacopy of the input) or holds the epilogue's partial reductions, and is not
asserted.

MAX stimuli. A max is one of its inputs, so a dropped fold only shows when the row's maximum sat
in the dropped load group (one in four with random data). The MAX arm therefore plants a unique
maximum per logical row, rotated through the four load groups of a row -- (r, even columns),
(r, odd), (r + 8, even), (r + 8, odd), i.e. offsets 0/2/8/10 for rows 0-3 and 4/6/12/14 for rows
4-7 -- so each of the eight load groups holds exactly one row maximum (see plant_row_maxima). SUM
touches every datum by construction and keeps the random stimuli.
"""

import torch
from conftest import blackhole_only, skip_for_coverage
from helpers.format_config import DataFormat
from helpers.golden_generators import FACE_DIM, TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    SDPA_REDUCE_ROW_POOL,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# Blackhole-only: sources include experimental/llk_sfpu/ckernel_sfpu_sdpa_reduce_row.h,
# which exists only in the Blackhole tree, so WH/Quasar cannot even build it.
pytestmark = blackhole_only

# The reduce span: 8 logical rows of 32 columns, packed into Dest face 0 (16 rows of 16 lanes).
LOGICAL_ROWS = 8

# Pinned stimuli range (the tolerance reasoning at the assert depends on it): bf16 values in
# [0.1, 1.1) are multiples of 2^-11 and every partial sum of 32 of them stays below 2^6, so the
# op's fp32 add chain is exact and the row maxima cluster near the top of the range.
STIMULI_SPEC = StimuliSpec.uniform(low=0.1, high=1.1)

# Planted MAX values: above STIMULI_SPEC's range, distinct per row, exact in bf16.
PLANTED_MAX_BASE = 1.5
PLANTED_MAX_STEP = 0.125


def plant_row_maxima(src_rowmajor: torch.Tensor) -> torch.Tensor:
    """Give each logical row a unique maximum in the load group its index selects.

    Logical row r is face rows r and r + 8; load group g = r % 4 picks the half (r for g < 2,
    r + 8 otherwise) and the column parity (g % 2), so rows 0-3 cover offsets 0, 2, 8, 10 and
    rows 4-7 cover offsets 4, 6, 12, 14 once each. The column inside the group is spread so the
    maxima do not share a lane.
    """
    src = src_rowmajor.clone()
    for r in range(LOGICAL_ROWS):
        group = r % 4
        face_row = r if group < 2 else r + LOGICAL_ROWS
        col = 2 * ((3 * r) % (FACE_DIM // 2)) + (group % 2)
        src[face_row, col] = PLANTED_MAX_BASE + PLANTED_MAX_STEP * r
    return src


def sdpa_reduce_row_golden(
    src_rowmajor: torch.Tensor, reduce_pool: ReducePool
) -> torch.Tensor:
    """Per-logical-row reduction of a 32x32 row-major tile, as the op defines its rows
    (module docstring). Returns the LOGICAL_ROWS defined outputs, i.e. column 0 of rows 0-7.
    """
    face0 = src_rowmajor[:FACE_DIM, :FACE_DIM].to(torch.float32)
    span = torch.cat(
        [face0[:LOGICAL_ROWS, :], face0[LOGICAL_ROWS : 2 * LOGICAL_ROWS, :]], dim=1
    )  # [8, 32]: logical row r = face rows r and r + 8
    if reduce_pool == ReducePool.Max:
        return span.amax(dim=1)
    return span.sum(dim=1)


@skip_for_coverage
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],  # Only Float16_b is supported for SDPA reduce row
        same=True,
    ),
    dest_acc=[DestAccumulation.No],  # Only Float16_b / non-fp32 dest is supported
    reduce_pool=[ReducePool.Max, ReducePool.Sum],
)
def test_sdpa_reduce_row(
    formats,
    dest_acc,
    reduce_pool,
):
    input_dimensions = [TILE_DIM, TILE_DIM]  # single tile

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=STIMULI_SPEC,
        spec_B=STIMULI_SPEC,
    )

    torch_format = format_dict[formats.input_format]

    src_rowmajor = src_A.to(torch_format).reshape(input_dimensions)
    if reduce_pool == ReducePool.Max:
        src_rowmajor = plant_row_maxima(src_rowmajor)

    # GOLDEN
    # *******************************************************
    golden = sdpa_reduce_row_golden(src_rowmajor, reduce_pool).to(
        format_dict[formats.output_format]
    )
    # *******************************************************

    # Tilize the tile the way the kernel reads it.
    src_A = tilize_block(
        src_rowmajor.flatten(), input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sdpa_reduce_row_test.cpp",
        formats,
        templates=[
            SDPA_REDUCE_ROW_POOL(reduce_pool=reduce_pool),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=False,  # MATH kernel does an A2D datacopy, so the input must land in SrcA
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = torch.as_tensor(
        untilize_block(res_tensor, formats.output_format, input_dimensions)
    ).reshape(input_dimensions)

    # Validate ONLY the op's defined output lanes: column 0 of rows 0..LOGICAL_ROWS-1.
    # MAX returns one of its inputs, so it must be bit-exact: with STIMULI_SPEC the row maxima
    # cluster near the top of the range, where the default Float16_b tolerance (atol + rtol * |x|
    # = 0.05 + 0.05 * |x|) would also accept a max taken over only part of the row.
    # SUM: with STIMULI_SPEC the fp32 SFPADD chain is exact (multiples of 2^-11 summing to less
    # than 2^6), so the only device error is the single fp32 -> bf16 conversion at the store,
    # at most 1 ULP (measured 1 ULP on p100a); the default tolerance (about +-1.0 on row sums
    # of ~19, one whole input element) would miss a dropped addend.
    assert passed_test(
        golden,
        res_tensor[:LOGICAL_ROWS, 0],
        formats.output_format,
        max_ulp=0 if reduce_pool == ReducePool.Max else 1,
    )
