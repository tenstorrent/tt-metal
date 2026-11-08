# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    ALL_MATH_FIDELITIES,
    DestAccumulation,
    DestSync,
    Transpose,
)
from helpers.matmul_sweep import (
    DEST_RT_CT_BLOCKS,
    generate_face_layout_config,
    generate_tile_dims,
    mid_fill_rt_ct_pairs,
    tiny_tile_face_layout,
)
from helpers.param_config import input_output_formats, max_dest_tiles, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Bfp8_b,
        DataFormat.Float16_b,
        DataFormat.Float32,
    ]
)
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]

# ((in0 tile rows, in0 tile cols), (in1 tile rows, in1 tile cols)), the Quasar
# ``input_tile_dimensions`` layout. Full tiles first, then tiny in0 tiles (rows < 32).
FULL_TILE = ((TILE_DIM, TILE_DIM), (TILE_DIM, TILE_DIM))
TINY_IN0_TILE_ROWS = (1, 2, 4, 8, 16)
INPUT_TILE_DIMENSIONS = [FULL_TILE] + [
    ((rows, TILE_DIM), (TILE_DIM, TILE_DIM)) for rows in TINY_IN0_TILE_ROWS
]

# Dest-fill RT x CT from dest capacity, dest occupancy blocks, and half-dest
# mid-fill when they fit. KT 1 and 4 (long-K lives on perf_matmul).
PERF_KT_DIMS = (1, 4)
THROTTLE_LEVELS = [0, 5]
DEST_HANDOFF_NUM_BLOCKS = 4


def _is_tiny(input_tile_dimensions) -> bool:
    return input_tile_dimensions[0][0] < TILE_DIM


def _dest_fill_rt_ct_pairs(max_tiles):
    """Power-of-two dest-fill (rt, ct) pairs, dest occupancy blocks, and mid-fill when they fit dest."""
    pairs = []
    rt_dim = 1
    while rt_dim <= max_tiles:
        if max_tiles % rt_dim == 0:
            pairs.append((rt_dim, max_tiles // rt_dim))
        rt_dim *= 2
    pairs.extend((rt, ct) for rt, ct in DEST_RT_CT_BLOCKS if rt * ct <= max_tiles)
    pairs.extend(mid_fill_rt_ct_pairs(max_tiles))
    return list(dict.fromkeys(pairs))


def perf_math_matmul_dest_acc_modes(formats: FormatConfig) -> List[DestAccumulation]:
    if is_dest_acc_needed(formats):
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def perf_math_matmul_tile_dims(
    input_tile_dimensions, dest_acc: DestAccumulation, dest_sync: DestSync
) -> List[tuple]:
    """(ct, rt, kt), the Quasar ``matmul_tile_dims`` order.

    Full tiles: dest-filling RT x CT grids, dest occupancy blocks, and mid-fill with KT
    in {1, 4}. Tiny tiles: one row of ct=1, half dest, or dest-fill tiles at KT=1.
    """
    max_tiles = max_dest_tiles(dest_acc, dest_sync)
    if _is_tiny(input_tile_dimensions):
        return [
            (ct_dim, 1, 1) for ct_dim in dict.fromkeys((1, max_tiles // 2, max_tiles))
        ]
    return [
        (ct_dim, rt_dim, kt_dim)
        for rt_dim, ct_dim in _dest_fill_rt_ct_pairs(max_tiles)
        for kt_dim in PERF_KT_DIMS
    ]


def perf_math_matmul_transpose(input_tile_dimensions) -> List[Transpose]:
    """Tiny in0 tiles have no transposed face layout."""
    if _is_tiny(input_tile_dimensions):
        return [Transpose.No]
    return [Transpose.No, Transpose.Yes]


def perf_math_matmul_throttle(input_tile_dimensions) -> List[int]:
    return [0] if _is_tiny(input_tile_dimensions) else THROTTLE_LEVELS


def perf_math_matmul_num_blocks(
    input_tile_dimensions, matmul_tile_dims, throttle
) -> List[int]:
    """Dest handoff: full-tile KT=1 grids repeat over DEST_HANDOFF_NUM_BLOCKS blocks, unthrottled."""
    _, _, kt_dim = matmul_tile_dims
    if not _is_tiny(input_tile_dimensions) and kt_dim == 1 and throttle == 0:
        return [1, DEST_HANDOFF_NUM_BLOCKS]
    return [1]


def _face_layout(input_tile_dimensions, transpose: Transpose):
    if _is_tiny(input_tile_dimensions):
        return tiny_tile_face_layout(input_tile_dimensions[0][0])
    return next(
        layout
        for layout in generate_face_layout_config(4)
        if layout.unpack_transpose_faces == transpose
    )


@pytest.mark.perf
@parametrize(
    formats=MATMUL_FORMATS,
    dest_acc=perf_math_matmul_dest_acc_modes,
    dest_sync=DEST_SYNC_MODES,
    input_tile_dimensions=INPUT_TILE_DIMENSIONS,
    matmul_tile_dims=perf_math_matmul_tile_dims,
    transpose=perf_math_matmul_transpose,
    throttle=perf_math_matmul_throttle,
    num_blocks=perf_math_matmul_num_blocks,
    math_fidelity=ALL_MATH_FIDELITIES,
)
def test_perf_math_matmul(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    input_tile_dimensions,
    matmul_tile_dims,
    transpose,
    throttle,
    num_blocks,
    math_fidelity,
):
    """
    Performance test for matmul operations.

    Full tiles use dest-filling RT x CT grids sized to dest capacity, dest occupancy
    blocks (1×N through dest Half, 2×N, 4×1 / 4×2), and half-dest mid-fill (2×2,
    1×(cap/2), (cap/2)×1) when they fit, with KT in {1, 4} and throttle 0 or 5. Tiny
    in0 tiles (1-16 rows) cover ct=1, cap/2, and dest-fill ct. A dest-handoff slice
    repeats the full-tile KT=1 grids (NUM_BLOCKS=4, throttle=0).
    """
    ct_dim, rt_dim, kt_dim = matmul_tile_dims
    (in0_tile_r_dim, _), _ = input_tile_dimensions
    tile_dimensions = generate_tile_dims(
        (
            [rt_dim * TILE_DIM, kt_dim * TILE_DIM],
            [kt_dim * TILE_DIM, ct_dim * TILE_DIM],
        ),
        in0_tile_r_dim=in0_tile_r_dim,
    )
    face_layout = _face_layout(input_tile_dimensions, transpose)

    variant_tile_count = rt_dim * ct_dim * kt_dim

    configuration = PerfConfig(
        "sources/math_matmul_test.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(throttle),
        ],
        runtimes=[
            DEST_INDEX(0),
            UNPACK_TRANS_FACES(face_layout.unpack_transpose_faces),
            UNPACK_TRANS_WITHIN_FACE(face_layout.unpack_transpose_faces),
            TILE_COUNT(variant_tile_count * num_blocks),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(rt_dim * ct_dim),
            NUM_FACES(
                face_layout.num_faces,
                face_layout.num_faces_in0,
                face_layout.num_faces_in1,
            ),  # In0 -> Input A, In1 -> Input B
            PARTIAL_FACE(  # In0 -> Input A, In1 -> Input B
                partial_a=face_layout.partial_face_in0,
                partial_face_pack=face_layout.partial_face_pack,
                partial_b=face_layout.partial_face_in1,
                partial_face_math=face_layout.partial_face_math,
            ),
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
            IN_TILE_DIMS(
                tile_dimensions.in0_tile_r_dim,
                tile_dimensions.in0_tile_c_dim,
                tile_dimensions.in1_tile_r_dim,
                tile_dimensions.in1_tile_c_dim,
            ),
            LOOP_FACTOR(1024),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_dimensions.tile_cnt_in0,
            tile_count_B=tile_dimensions.tile_cnt_in1,
            tile_count_res=tile_dimensions.output_tile_cnt * num_blocks,
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
