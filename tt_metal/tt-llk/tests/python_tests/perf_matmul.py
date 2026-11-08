# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.llk_params import (
    ALL_MATH_FIDELITIES,
    DestAccumulation,
    DestSync,
    Transpose,
)
from helpers.matmul_sweep import (
    PERF_RING_TILES,
    dest_corner_mn,
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
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

# Cold start, functional-max inner dim, and reuse/bandwidth.
KT_DIMS = [1, 4, 32]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]


def perf_matmul_dest_acc_modes(formats: FormatConfig) -> List[DestAccumulation]:
    if is_dest_acc_needed(formats):
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def perf_matmul_tile_dims(
    formats: FormatConfig, dest_acc: DestAccumulation, dest_sync: DestSync
) -> List[tuple]:
    """(ct, rt, kt) dest corners that keep unpack inside the perf ring.

    Same order as the Quasar ``CRK_TILE_DIMM`` axis, so the two perf sweeps compare
    shape for shape. ``is_dest_acc_needed`` formats run on the 32-bit Dest whatever
    ``dest_acc`` says, so their capacity is halved here too.
    """
    max_tiles = max_dest_tiles(
        is_dest_acc_needed(formats) or dest_acc == DestAccumulation.Yes, dest_sync
    )
    return [
        (ct_dim, rt_dim, kt_dim)
        for rt_dim, ct_dim in dest_corner_mn(max_tiles)
        for kt_dim in KT_DIMS
        if unpack_matmul_fits_perf_ring(rt_dim, ct_dim, kt_dim)
    ]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=perf_matmul_dest_acc_modes,
    dest_sync=DEST_SYNC_MODES,
    matmul_tile_dims=perf_matmul_tile_dims,
    math_fidelity=ALL_MATH_FIDELITIES,
)
def test_perf_matmul(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    matmul_tile_dims,
    math_fidelity,
):
    ct_dim, rt_dim, kt_dim = matmul_tile_dims

    variant_tile_count = rt_dim * ct_dim * kt_dim
    # PERF_ADDRESS rings are PERF_RING_TILES (tests/helpers/include/perf.h).
    # Cap L1 stimuli so K=32 dest-fill Float32 does not overflow Tensix L1.
    # Unpack MOP occupancy is filtered in perf_matmul_tile_dims.
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)

    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(variant_tile_count),
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=stimuli_tiles,
            tile_count_B=stimuli_tiles,
            tile_count_res=min(rt_dim * ct_dim, PERF_RING_TILES),
        ),
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
