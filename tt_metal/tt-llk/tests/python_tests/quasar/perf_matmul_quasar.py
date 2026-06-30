# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Representative Quasar matmul L1-to-L1 perf tests."""

from dataclasses import dataclass

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    ENABLE_2X_FORMAT,
    ENABLE_DIRECT_INDEXING,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block


@dataclass
class MatmulPerfCase:
    name: str
    formats: InputOutputFormat
    math_fidelity: MathFidelity
    input_A_dimensions: list[int]
    input_B_dimensions: list[int]
    dest_acc: DestAccumulation
    dest_sync: DestSync
    implied_math_format: ImpliedMathFormat
    register_format_hint: DataFormat = None
    enable_direct_indexing: bool = False


_ARCH = get_chip_architecture()
_MXFP4_2X_HINT = DataFormat.MxFp4_2x_A if _ARCH == ChipArchitecture.QUASAR else None
LOOP_FACTORS = [1, 2, 4, 8, 16, 32, 64]
MIN_MATMUL_TILE_COUNT = 8
MIN_MATMUL_TILE_COUNT_HALF = MIN_MATMUL_TILE_COUNT // 2


MATMUL_PERF_CASES = [
    MatmulPerfCase(
        name="float16_b_1x8x1_hifi4",
        formats=InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        math_fidelity=MathFidelity.HiFi4,
        input_A_dimensions=[TILE_DIM, MIN_MATMUL_TILE_COUNT * TILE_DIM],
        input_B_dimensions=[MIN_MATMUL_TILE_COUNT * TILE_DIM, TILE_DIM],
        dest_acc=DestAccumulation.No,
        dest_sync=DestSync.Half,
        implied_math_format=ImpliedMathFormat.No,
    ),
    MatmulPerfCase(
        name="float16_2x1x4_dest_acc_full_sync",
        formats=InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
        math_fidelity=MathFidelity.HiFi2,
        input_A_dimensions=[2 * TILE_DIM, MIN_MATMUL_TILE_COUNT_HALF * TILE_DIM],
        input_B_dimensions=[MIN_MATMUL_TILE_COUNT_HALF * TILE_DIM, TILE_DIM],
        dest_acc=DestAccumulation.Yes,
        dest_sync=DestSync.Half,
        implied_math_format=ImpliedMathFormat.Yes,
    ),
    MatmulPerfCase(
        name="mxfp4_1x8x1_direct_indexing",
        formats=InputOutputFormat(DataFormat.MxFp4, DataFormat.Float16_b),
        math_fidelity=MathFidelity.LoFi,
        input_A_dimensions=[TILE_DIM, MIN_MATMUL_TILE_COUNT * TILE_DIM],
        input_B_dimensions=[MIN_MATMUL_TILE_COUNT * TILE_DIM, TILE_DIM],
        dest_acc=DestAccumulation.No,
        dest_sync=DestSync.Half,
        implied_math_format=ImpliedMathFormat.Yes,
        register_format_hint=_MXFP4_2X_HINT,
        enable_direct_indexing=True,
    ),
    MatmulPerfCase(
        name="int8_int32_1x8x1_lofi",
        formats=InputOutputFormat(DataFormat.Int8, DataFormat.Int32),
        math_fidelity=MathFidelity.LoFi,
        input_A_dimensions=[TILE_DIM, MIN_MATMUL_TILE_COUNT * TILE_DIM],
        input_B_dimensions=[MIN_MATMUL_TILE_COUNT * TILE_DIM, TILE_DIM],
        dest_acc=DestAccumulation.Yes,
        dest_sync=DestSync.Half,
        implied_math_format=ImpliedMathFormat.No,
    ),
]


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.name) for case in MATMUL_PERF_CASES],
)
@pytest.mark.parametrize(
    "loop_factor",
    [
        pytest.param(loop_factor, id=f"loop_factor:{loop_factor}")
        for loop_factor in LOOP_FACTORS
    ],
)
def test_perf_matmul_quasar(case, loop_factor, perf_report):
    formats = InputOutputFormat(
        case.formats.input_format,
        case.formats.output_format,
        input_format_B=case.formats.input_format_B,
        register_format_hint=case.register_format_hint,
    )

    if formats.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=case.input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=case.input_B_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
        output_format=formats.output_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=case.input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=case.input_B_dimensions, stimuli_format=formats.input_format
    )
    matmul_dims = generate_tile_dims((case.input_A_dimensions, case.input_B_dimensions))
    matmul_tile_count = matmul_dims.rt_dim * matmul_dims.ct_dim * matmul_dims.kt_dim
    assert matmul_tile_count >= MIN_MATMUL_TILE_COUNT

    num_faces = 4
    configuration = PerfConfig(
        "sources/quasar/matmul_quasar_test.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
        ],
        templates=[
            MATH_FIDELITY(case.math_fidelity),
            IMPLIED_MATH_FORMAT(case.implied_math_format),
            ENABLE_2X_FORMAT(
                formats.register_format_hint
                in (DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B)
            ),
            ENABLE_DIRECT_INDEXING(case.enable_direct_indexing),
            DEST_SYNC(case.dest_sync),
            UNPACK_TRANS_FACES(Transpose.No),
            LOOP_FACTOR(loop_factor),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            TILE_COUNT(matmul_tile_count),
            NUM_FACES(num_faces, num_faces, num_faces),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=case.dest_acc,
        disable_format_inference=(
            formats.input_format.is_mx_format() and formats.register_format_hint is None
        ),
    )

    configuration.run(perf_report)
