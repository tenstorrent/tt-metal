# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Coverage of the matmul format and mode cells that no other perf module ran: 366 cells of (SrcA, SrcB, output, dest accumulate, sync, fidelity) over Float16_b, Bfp8_b, Bfp4_b and Float32,
two block shapes each (one K tile block and one four tile K block, rt x ct of at least two tiles), throttle 0, on the shapes perf_matmul's dest corner grid yields (cov3_cases.json).
SrcA is input_format (unpacker 0, matmul in1), SrcB is input_format_B (unpacker 1, matmul in0)."""

from typing import Callable, List, Union

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
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
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATMUL_PHASE_ELIDE,
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_K_LOOP,
    MATMUL_ROW_END_FOLD,
    MATMUL_PACK_BLOCK,
    MATMUL_ROW_MOP,
    MATMUL_STREAM_NARROW,
    MATMUL_REUSE_DIR,
    MATMUL_STREAM_Z,
    MATMUL_UNPACK_TTSYNC,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
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

    def _dest_bank_max_tiles(
        format: FormatConfig, dest_acc: DestAccumulation, dest_sync: DestSync
    ):
        capacity_divisor = (
            2 if is_dest_acc_needed(format) or dest_acc == DestAccumulation.Yes else 1
        )
        return DEST_SYNC_TILE_LIMITS[dest_sync] // capacity_divisor

    unique_max_tiles = set(
        _dest_bank_max_tiles(fmt, acc, sync)
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
        for dims in dimensions[_dest_bank_max_tiles(format, accumulation, dest_sync)]
    ]



import json,os
from conftest import skip_for_wormhole
from helpers.format_config import InputOutputFormat
from helpers.test_variant_parameters import PACK_SRC_FROM_OUTPUT
_F=lambda n:getattr(DataFormat,n)
_J=json.load(open(os.path.join(os.path.dirname(__file__),"cov3_cases.json")))
def _cases():
    out=[]
    for a,b,o,acc,sy,fi,rt,ct,kt in _J:
        fmt=InputOutputFormat(_F(a),_F(o),input_format_B=_F(b))
        out.append((fmt,getattr(DestAccumulation,acc.split(".")[1]),getattr(DestSync,sy.split(".")[1]),([rt*TILE_DIM,kt*TILE_DIM],[kt*TILE_DIM,ct*TILE_DIM]),getattr(MathFidelity,fi.split(".")[1]),0))
    return out

@pytest.mark.perf
@skip_for_wormhole
@parametrize(case=_cases())
def test_perf_matmul_cov3(
    perf_report,
    case,
):

    case = case[0] if len(case) == 1 else case
    formats, dest_acc, dest_sync, (matrix_a, matrix_b), math_fidelity, throttle = case

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    # Calculate all matmul dimensions using helper function
    dims = generate_tile_dims((matrix_a, matrix_b))

    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    # PERF_ADDRESS rings are PERF_RING_TILES (tests/helpers/include/perf.h).
    # Cap L1 stimuli so K=32 dest-fill Float32 does not overflow Tensix L1.
    # Unpack MOP occupancy is filtered in generate_dest_corner_combinations.
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)


    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            PACK_SRC_FROM_OUTPUT(),
            *MATMUL_PHASE_ELIDE.for_block(formats, math_fidelity, dest_acc, throttle),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(throttle),
            *MATMUL_STREAM_NARROW.for_block(
                formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_REUSE_DIR.for_block(formats, math_fidelity, dims.ct_dim, dims.rt_dim, dest_sync),
            *MATMUL_STREAM_Z.for_block(
                formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity,
                kt_dim=dims.kt_dim,
            ),
            *MATMUL_UNPACK_TTSYNC.for_block(
                formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim, dest_sync
            ),
            *MATMUL_K_LOOP.for_block(
                formats, dest_acc, math_fidelity, dims.ct_dim, dims.rt_dim, dims.kt_dim
            ),
            *MATMUL_ROW_END_FOLD.for_block(dims.ct_dim, dims.rt_dim),
            *MATMUL_ROW_MOP.for_block(
                formats, dest_acc, math_fidelity, dest_sync, dims.ct_dim, dims.rt_dim, dims.kt_dim
            ),
            *MATMUL_FORM_KEYS.for_block(
                formats, dims.kt_dim, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity, dest_sync=dest_sync, dest_acc=dest_acc, unfuse_one_tile=True,
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
            formats.input_format_B,
            formats.output_format,
            tile_count_A=stimuli_tiles,
            tile_count_B=stimuli_tiles,
            tile_count_res=min(dims.rt_dim * dims.ct_dim, PERF_RING_TILES),
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
