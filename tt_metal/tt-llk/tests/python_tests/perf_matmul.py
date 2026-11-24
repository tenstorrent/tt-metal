# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Union

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    ALL_MATH_FIDELITIES,
    DestAccumulation,
    DestSync,
    Transpose,
)
from helpers.matmul_prod_census import (
    is_production_key,
    perf_gate_applies,
    perf_gate_selection,
)
from helpers.matmul_sweep import (
    DEST_RT_CT_BLOCKS,
    PERF_RING_TILES,
    generate_tile_dims,
    matmul_pack_form,
    mid_fill_rt_ct_pairs,
    unpack_matmul_fits_perf_ring,
)
from helpers.param_config import (
    input_output_formats,
    max_dest_tiles,
    parametrize,
)
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_K_LOOP,
    MATMUL_PACK_BLOCK,
    MATMUL_PHASE_ELIDE,
    MATMUL_REUSE_DIR,
    MATMUL_ROW_END_FOLD,
    MATMUL_ROW_MOP,
    MATMUL_STREAM_NARROW,
    MATMUL_STREAM_Z,
    MATMUL_UNPACK_TTSYNC,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    matmul_two_row_off,
)

# Cold start, functional-max inner dim, and reuse/bandwidth.
KT_DIMS = [1, 4, 32]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]


def dest_corner_mn(max_tiles: int) -> List[tuple]:
    """1×1, dest-fill vectors/square, dest occupancy blocks, and half-dest mid-fill when they fit."""
    square = int(max_tiles**0.5)
    corners = [(1, 1), (1, max_tiles), (max_tiles, 1), (square, square)]
    corners.extend((rt, ct) for rt, ct in DEST_RT_CT_BLOCKS if rt * ct <= max_tiles)
    corners.extend(mid_fill_rt_ct_pairs(max_tiles))
    return list(dict.fromkeys(corners))


def generate_dest_corner_combinations(max_tiles: int, kt_dims=KT_DIMS) -> List[tuple]:
    return [
        (
            [mt_dim * TILE_DIM, kt_dim * TILE_DIM],
            [kt_dim * TILE_DIM, nt_dim * TILE_DIM],
        )
        for mt_dim, nt_dim in dest_corner_mn(max_tiles)
        for kt_dim in kt_dims
        if unpack_matmul_fits_perf_ring(mt_dim, nt_dim, kt_dim)
    ]


def matmul_combos(
    formats: List[FormatConfig],
    dest_acc: Union[
        List[DestAccumulation], Callable[[FormatConfig], List[DestAccumulation]]
    ],
):
    def _acc_modes(fmt: FormatConfig) -> List[DestAccumulation]:
        return dest_acc(fmt) if callable(dest_acc) else dest_acc

    def _max_tiles(fmt: FormatConfig, acc: DestAccumulation, sync: DestSync) -> int:
        return max_dest_tiles(
            is_dest_acc_needed(fmt) or acc == DestAccumulation.Yes, sync
        )

    unique_max_tiles = set(
        _max_tiles(fmt, acc, sync)
        for fmt in formats
        for acc in _acc_modes(fmt)
        for sync in DEST_SYNC_MODES
    )
    dimensions = {
        max_tiles: generate_dest_corner_combinations(max_tiles, kt_dims=KT_DIMS)
        for max_tiles in unique_max_tiles
    }

    return [
        (format, accumulation, dest_sync, dims)
        for format in formats
        for accumulation in _acc_modes(format)
        for dest_sync in DEST_SYNC_MODES
        for dims in dimensions[_max_tiles(format, accumulation, dest_sync)]
    ]


PERF_MATMUL_COMBOS = matmul_combos(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=lambda formats: (
        [DestAccumulation.Yes]
        if is_dest_acc_needed(formats)
        else [DestAccumulation.No, DestAccumulation.Yes]
    ),
)


def _gate_key(p):
    (formats, dest_acc, dest_sync, (matrix_a, matrix_b)), fid = p
    return is_production_key(
        fid,
        dest_sync == DestSync.Half,
        dest_acc == DestAccumulation.Yes,
        formats.input_format,
        formats.input_format,
        formats.output_format,
        matrix_a[0] // TILE_DIM,
        matrix_b[1] // TILE_DIM,
        matrix_a[1] // TILE_DIM,
    )


def _gate_family(p):
    (formats, dest_acc, dest_sync, _dims), fid = p
    return (fid, dest_sync, dest_acc, formats.input_format, formats.output_format)


_PERF_GATE = perf_gate_selection(
    [(c, f) for c in PERF_MATMUL_COMBOS for f in ALL_MATH_FIDELITIES],
    _gate_key,
    _gate_family,
    lambda p: (id(p[0]), p[1]),
)


def perf_gate_keep(params) -> bool:
    """The perf CI runner's ids on Blackhole (helpers/matmul_prod_census.py)."""
    return (
        not perf_gate_applies()
        or (id(params["combos"]), params["math_fidelity"]) in _PERF_GATE
    )


@pytest.mark.perf
@parametrize(
    combos=PERF_MATMUL_COMBOS,
    math_fidelity=ALL_MATH_FIDELITIES,
)
def test_perf_matmul(
    perf_report,
    combos,
    math_fidelity,
):

    formats, dest_acc, dest_sync, (matrix_a, matrix_b) = combos

    # Calculate all matmul dimensions using helper function
    dims = generate_tile_dims((matrix_a, matrix_b))

    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    # PERF_ADDRESS rings are PERF_RING_TILES (tests/helpers/include/perf.h).
    # Cap L1 stimuli so K=32 dest-fill Float32 does not overflow Tensix L1.
    # Unpack MOP occupancy is filtered in generate_dest_corner_combinations.
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)
    two_row_off = matmul_two_row_off(
        formats,
        dest_sync,
        dims.kt_dim,
        k_loop=MATMUL_K_LOOP.for_block(
            formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim
        ),
    )

    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_FIDELITY(math_fidelity),
            *MATMUL_PHASE_ELIDE.for_block(formats, math_fidelity, dest_acc, 0),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            *MATMUL_STREAM_NARROW.for_block(
                formats,
                dest_acc,
                dest_sync,
                dims.ct_dim,
                dims.rt_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_REUSE_DIR.for_block(
                formats, math_fidelity, dims.ct_dim, dims.rt_dim, dest_sync, two_row_off
            ),
            *MATMUL_STREAM_Z.for_block(
                formats,
                dest_acc,
                dest_sync,
                dims.ct_dim,
                dims.rt_dim,
                math_fidelity=math_fidelity,
                kt_dim=dims.kt_dim,
                two_row_off=two_row_off,
            ),
            *MATMUL_UNPACK_TTSYNC.for_block(
                formats,
                dest_acc,
                math_fidelity,
                dims.ct_dim,
                dims.rt_dim,
                dims.kt_dim,
                dest_sync,
                two_row_off,
            ),
            *MATMUL_K_LOOP.for_block(
                formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim
            ),
            *MATMUL_ROW_END_FOLD.for_block(dims.ct_dim, dims.rt_dim),
            *MATMUL_ROW_MOP.for_block(
                formats,
                dest_acc,
                math_fidelity,
                dest_sync,
                dims.ct_dim,
                dims.rt_dim,
                dims.kt_dim,
                two_row_off=two_row_off,
            ),
            *MATMUL_FORM_KEYS.for_block(
                formats,
                dims.kt_dim,
                dims.ct_dim,
                dims.rt_dim,
                math_fidelity=math_fidelity,
                dest_sync=dest_sync,
                dest_acc=dest_acc,
                unfuse_one_tile=True,
                two_row_off=two_row_off,
            ),
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    dest_sync,
                    math_fidelity,
                    dims.rt_dim,
                    dims.ct_dim,
                    dims.kt_dim,
                )
            ),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(variant_tile_count),
            CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=stimuli_tiles,
            tile_count_B=stimuli_tiles,
            tile_count_res=min(dims.rt_dim * dims.ct_dim, PERF_RING_TILES),
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
