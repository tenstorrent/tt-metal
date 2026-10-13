# SPDX-License-Identifier: Apache-2.0
#
# llk_analysis coverage gap round (ops/blackhole/sfpu_gaps). Shadow perf driver for the Blackhole SFPU compute API entry
# points that no checked-in perf module times. It drives tests/sources/sfpu_gaps_perf_x.cpp (the unary SFPU perf kernel
# with the dispatch of tests/helpers/include/sfpu_gaps_x.h) over every gap op in its <op>_tile form: VectorMode::RC with
# 8 rows per call, the formats the entry point is written for (Float16_b with a 16-bit and a 32-bit DEST, Float32 with a
# 32-bit DEST, Int32, UInt32, UInt16) and both approximation flags where the body is templated on one. loop_factor 16,
# [128, 64] = 8 tiles, run types MATH_ISOLATE and L1_TO_L1. Shadow tree only.

from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, PerfRunType, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    TemplateParameter,
)


@dataclass
class GAP_OP(TemplateParameter):
    gap_op: str = "isinf"
    gap_fmt: str = "Float16_b"
    gap_all32: int = 0

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [f"#define GAP_OP_NAME {self.gap_op}", f"#define GAP_FMT_NAME {self.gap_fmt}", f"#define GAP_ALL32 {self.gap_all32}"]
        )


CLASS_FORMATS = {"I32": InputOutputFormat(DataFormat.Int32, DataFormat.Int32)}
CLASS_DEST = {"I32": [DestAccumulation.No]}
CLASS_GAP_FMT = {"I32": "Int32"}


# The integer binary entry points of #58825's one-call change, four 8-row calls (all32 0) against one 32-row call (1), and
# the ordered int32 scalar compares (scalar 5 and -5) in the compute API form.
OC2 = []
for _op in ("unary_lt_int32", "unary_gt_int32", "unary_le_int32", "unary_ge_int32", "unary_lt_int32_n", "unary_gt_int32_n", "unary_le_int32_n", "unary_ge_int32_n"):
    OC2.append((_op, "I32", 0))


@pytest.mark.perf
@parametrize(variant=OC2)
def test_perf_sfpu_gaps_oc2_x(perf_report, variant):
    if len(variant) == 1:
        (variant,) = variant
    op, cls, all32 = variant
    formats = CLASS_FORMATS[cls]
    input_dimensions = [128, 64]
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    PerfConfig(
        "sources/sfpu_gaps_perf_x.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE, PerfRunType.UNPACK_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[
            GAP_OP(gap_op=op, gap_fmt=CLASS_GAP_FMT[cls], gap_all32=all32),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(16),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=CLASS_DEST[cls][0],
    ).run(perf_report)
