# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every perf_matmul Half cell with every form (math MOP kind x hoisted unpack), no fused post flip, stream_z or TTSync: the forms
alone, so a key (helpers/matmul_prod_keys.py) can be judged against main by lookup."""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.matmul_sweep import PERF_RING_TILES, generate_tile_dims, matmul_pack_form
from helpers.param_config import input_output_formats
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORCE_FORM,
    MATMUL_FORM_KEYS,
    MATMUL_PACK_BLOCK,
    MATMUL_PROD_KEYS,
    MATMUL_ROW_MOP,
    MATMUL_STREAM_NARROW,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from perf_matmul import matmul_combos

FIDELITIES = [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]
FORMS = [(kind, hoist) for kind in (0, 1, 2) for hoist in (False, True)]


def _cases():
    combos = matmul_combos(
        formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b]),
        dest_acc=lambda formats: (
            [DestAccumulation.Yes] if is_dest_acc_needed(formats) else [DestAccumulation.No, DestAccumulation.Yes]
        ),
    )
    return [(c, f, k, h) for c in combos if c[2] == DestSync.Half for f in FIDELITIES for k, h in FORMS]


_CASES = _cases()


@pytest.mark.perf
@skip_for_wormhole
@pytest.mark.parametrize(
    "combos,math_fidelity,kind,hoist",
    _CASES,
    ids=[f"{i}-{c[1].name}-{('tile', 'rows', 'row')[c[2]]}-{'h' if c[3] else 'p'}" for i, c in enumerate(_CASES)],
)
def test_perf_matmul_forms(perf_report, combos, math_fidelity, kind, hoist):
    formats, dest_acc, dest_sync, (matrix_a, matrix_b) = combos

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]
    dims = generate_tile_dims((matrix_a, matrix_b))
    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)

    configuration = PerfConfig(
        "sources/matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            MATMUL_PROD_KEYS(),
            MATMUL_FORCE_FORM(kind, hoist),
            *([MATMUL_ROW_MOP()] if kind == 2 else []),
            *MATMUL_STREAM_NARROW.for_block(
                formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_FORM_KEYS.for_block(
                formats, dims.kt_dim, dims.ct_dim, dims.rt_dim,
                math_fidelity=math_fidelity, dest_sync=dest_sync, dest_acc=dest_acc,
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
