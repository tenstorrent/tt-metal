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

    def convert_to_cpp(self) -> str:
        return "\n".join([f"#define GAP_OP_NAME {self.gap_op}", f"#define GAP_FMT_NAME {self.gap_fmt}"])


# Data classes: F = Float16_b tiles with a 16-bit and a 32-bit DEST; F32 = Float32 tiles unpacked to a 32-bit DEST;
# I32 = Int32 tiles unpacked to DEST (dest_acc No, the harness rule for Int32); U32 = UInt32 tiles unpacked to DEST
# (dest_acc Yes, the harness rule for UInt32); U16 = UInt16 tiles through the source registers.
CLASS_FORMATS = {
    "F": InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    "F32": InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    "I32": InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
    "U32": InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32),
    "U16": InputOutputFormat(DataFormat.UInt16, DataFormat.UInt16),
}
CLASS_DEST = {
    "F": [DestAccumulation.No, DestAccumulation.Yes],
    "F32": [DestAccumulation.Yes],
    "I32": [DestAccumulation.No],
    "U32": [DestAccumulation.Yes],
    "U16": [DestAccumulation.No],
}
# The DataFormat template argument of the entry points that take one (bitwise, shifts, fill_int, logical_not, the
# integer binary ops, mac); for the others it is the tile format and unused.
CLASS_GAP_FMT = {"F": "Float16_b", "F32": "Float32", "I32": "Int32", "U32": "UInt32", "U16": "UInt16"}

# op -> (data classes, approximation flags swept)
BOTH = [ApproximationMode.Yes, ApproximationMode.No]
ONE = [ApproximationMode.No]
OPS = {
    # one-input float entry points
    "isinf": (["F"], BOTH),
    "isposinf": (["F"], BOTH),
    "isneginf": (["F"], BOTH),
    "isnan": (["F"], BOTH),
    "isfinite": (["F"], BOTH),
    "unary_ne": (["F"], BOTH),
    "unary_eq": (["F"], BOTH),
    "stochastic_round": (["F"], BOTH),
    "tiled_prod": (["F"], BOTH),
    "power_iterative": (["F"], BOTH),
    "alt_complex_rotate90": (["F"], BOTH),
    "fill_bitcast": (["F"], BOTH),
    "logical_not": (["F", "U16"], BOTH),
    "mask_posinf": (["F"], ONE),
    # one-input integer entry points
    "unary_ne_int32": (["I32"], BOTH),
    "unary_eq_int32": (["I32"], BOTH),
    "unary_gt_int32": (["I32"], BOTH),
    "unary_ge_int32": (["I32"], BOTH),
    "unary_lt_int32": (["I32"], BOTH),
    "unary_le_int32": (["I32"], BOTH),
    "gtz_int32": (["I32"], BOTH),
    "nez_int32": (["I32"], BOTH),
    "gez_int32": (["I32"], BOTH),
    "ltz_int32": (["I32"], BOTH),
    "eqz_int32": (["I32"], BOTH),
    "lez_int32": (["I32"], BOTH),
    "bitwise_and": (["I32", "U32", "U16"], BOTH),
    "bitwise_or": (["I32", "U32", "U16"], BOTH),
    "bitwise_xor": (["I32", "U32", "U16"], BOTH),
    "left_shift": (["I32", "U32", "U16"], BOTH),
    "right_shift": (["I32", "U32", "U16"], BOTH),
    "relu_max_int32": (["I32"], BOTH),
    "relu_min_int32": (["I32"], BOTH),
    "relu_int32": (["I32"], BOTH),
    "relu_max_uint32": (["U32"], BOTH),
    "relu_min_uint32": (["U32"], BOTH),
    "relu_max_uint16": (["U16"], BOTH),
    "relu_min_uint16": (["U16"], BOTH),
    "identity_uint32": (["U32"], BOTH),
    "clamp_int32": (["I32"], BOTH),
    "signbit_int32": (["I32"], BOTH),
    "unary_max_int32": (["I32"], BOTH),
    "unary_min_int32": (["I32"], BOTH),
    "unary_max_uint32": (["U32"], BOTH),
    "unary_min_uint32": (["U32"], BOTH),
    "sum_int_col": (["I32"], BOTH),
    "sum_int_row": (["I32"], BOTH),
    "add_int_unary": (["I32"], BOTH),
    "rsub_unary_int32": (["I32"], BOTH),
    "add_unary_int32": (["I32"], BOTH),
    "sub_unary_int32": (["I32"], BOTH),
    "fill_int": (["I32", "U32", "U16"], BOTH),
    "negative_int32": (["I32"], BOTH),
    "remainder_uint32": (["U32"], BOTH),
    # two-input entry points
    "lgamma_stirling_float": (["F", "F32"], BOTH),
    "gcd": (["I32"], ONE),
    "gcd_one": (["I32"], ONE),
    "lcm": (["I32"], ONE),
    "lcm_one": (["I32"], ONE),
    "isclose": (["F"], BOTH),
    "logsigmoid": (["F"], BOTH),
    "mul_int_uint16": (["U16"], BOTH),
    "add_int_tile": (["U16"], BOTH),
    "sub_int_tile": (["U16"], BOTH),
    "rsub_int_tile": (["U16"], BOTH),
    "binary_left_shift": (["U16"], BOTH),
    "binary_right_shift": (["U16", "U32"], BOTH),
    "binary_logical_right_shift": (["U16"], BOTH),
    "div_int32": (["I32"], BOTH),
    "div_int32_one": (["I32"], BOTH),
    "div_int32_floor": (["I32"], BOTH),
    "div_int32_floor_one": (["I32"], BOTH),
    "div_int32_trunc": (["I32"], BOTH),
    "div_int32_trunc_one": (["I32"], BOTH),
    "remainder_int32_bin": (["I32"], ONE),
    "remainder_int32_bin_one": (["I32"], ONE),
    "remainder_uint32_bin": (["U32"], ONE),
    "remainder_uint32_bin_one": (["U32"], ONE),
    "fmod_int32_bin": (["I32"], ONE),
    "fmod_int32_bin_one": (["I32"], ONE),
    "nextafter": (["F32"], BOTH),
    "nextafter_bf16": (["F"], BOTH),
    # max_reduce_with_indices: its body header does not compile under the harness flags (-Werror=unused-parameter)
    # three-input entry points
    "lgamma_adjusted": (["F"], BOTH),
    "mac": (["F"], BOTH),
}

GAP_OPS = sorted(OPS)


def _formats(op):
    return [CLASS_FORMATS[c] for c in OPS[op][0]]


def _class_of(formats):
    for c, f in CLASS_FORMATS.items():
        if f == formats:
            return c
    raise ValueError(formats)


def _dest_modes(op, formats):
    return CLASS_DEST[_class_of(formats)]


def _approx_modes(op):
    return OPS[op][1]


@pytest.mark.perf
@parametrize(
    op=GAP_OPS,
    formats=lambda op: _formats(op),
    dest_acc=lambda op, formats: _dest_modes(op, formats),
    approx_mode=lambda op: _approx_modes(op),
)
def test_perf_sfpu_gaps_x(perf_report, op, formats, dest_acc, approx_mode):
    input_dimensions = [128, 64]
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    # every 32-bit input of this sweep is unpacked straight into DEST (Float32 with a 32-bit DEST, Int32, UInt32)
    unpack_to_dest = formats.input_format.is_32_bit()

    configuration = PerfConfig(
        "sources/sfpu_gaps_perf_x.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE, PerfRunType.L1_TO_L1],
        templates=[
            GAP_OP(gap_op=op, gap_fmt=CLASS_GAP_FMT[_class_of(formats)]),
            APPROX_MODE(approx_mode),
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
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )
    configuration.run(perf_report)
