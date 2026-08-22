# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lane EU LLK-coverage-expansion perf vehicle (MATH_ISOLATE cycles/tile).

Perf twin of test_sfpu_coverage.py through sources/sfpu_coverage_perf.cpp:
per raced arm (production hand kernel vs fresh semantic body) the TILE_LOOP
row of the .post.csv carries mean(MATH_ISOLATE) and the math ELF size.  Op
ids and fixed parameters mirror sources/sfpu_coverage_test.cpp exactly.
"""

import struct

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    COVERAGE_OP,
    COVERAGE_SUBOP,
    FRESH_CPP_IMPL,
    ITERATIONS,
    LOOP_FACTOR,
    NUM_FACES,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


_BF16 = input_output_formats([DataFormat.Float16_b], same=True)[0]
_INT32 = input_output_formats([DataFormat.Int32], same=True)[0]

# op name -> (COVERAGE_OP id, COVERAGE_SUBOP, formats, dest_acc, scalar_bits)
# Sub-op vehicles follow the row convention: unarybitwise races XOR, intsum
# races COL (the corr suite covers every sub-op).
#
# scalar_bits is a REAL operand only for unarybitwise/addrsqrt; every other
# kernel ignores the SFPU_UNARY_SCALAR constant.  It is still passed for
# EVERY op (0 for the ignoring ones) so all ops emit the identical perf-CSV
# column set — one homogeneous schema per file, the PerfSchemaError
# contract (FM-F1 repair; validated by selftest_perf_schema_columns.py).
_COVERAGE_PERF_OPS = {
    "rotate90": (1, 0, _BF16, DestAccumulation.No, None),
    "unarybitwise": (2, 2, _INT32, DestAccumulation.Yes, 0x5A5A0FF0),
    "addrsqrt": (3, 0, _BF16, DestAccumulation.No, _bits(0.5)),
    "smoothstep": (4, 0, _BF16, DestAccumulation.No, None),
    "tiledprod": (5, 0, _BF16, DestAccumulation.No, None),
    "zeropad": (6, 0, _BF16, DestAccumulation.No, None),
    "sparsekfilter": (7, 0, _INT32, DestAccumulation.Yes, None),
    "customadd": (8, 0, _BF16, DestAccumulation.No, None),
    "copydest": (9, 0, _BF16, DestAccumulation.No, None),
    "intsum": (10, 0, _INT32, DestAccumulation.Yes, None),
}


@pytest.mark.perf
@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize("op", list(_COVERAGE_PERF_OPS), ids=lambda o: o)
def test_perf_sfpu_coverage(perf_report, op, fresh_cpp_impl):
    cov_op, subop, formats, dest_acc, scalar_bits = _COVERAGE_PERF_OPS[op]

    input_dimensions = [128, 64]  # tile_cnt: 8
    unpack_to_dest = formats.input_format.is_32_bit()

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    # One literal list, no conditional appends: every (op, impl) node emits
    # the same columns, so the module CSV always carries a single schema.
    templates = [
        COVERAGE_OP(cov_op),
        COVERAGE_SUBOP(subop),
        FRESH_CPP_IMPL(fresh_cpp_impl),
        APPROX_MODE(ApproximationMode.No),
        ITERATIONS(32),
        TILE_COUNT(tile_count),
        LOOP_FACTOR(16),
        NUM_FACES(num_faces=faces_to_generate),
        UNPACK_TRANS_FACES(Transpose.No),
        UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        SFPU_UNARY_SCALAR(scalar_bits if scalar_bits is not None else 0),
    ]

    configuration = PerfConfig(
        "sources/sfpu_coverage_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=templates,
        runtimes=[],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
