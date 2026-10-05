# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf rows for the Blackhole compute API SFPU entry points that the registry sweeps do not time.

Each row runs one entry point the way its <op>_tile does (its vector mode, init and template and scalar arguments, as
ttnn calls it), through helpers/include/sfpu_compute_api.h, over the tile formats the entry point is written for.
"""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    NUM_FACES,
    SFPU_API_OP,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

# Tile format and DEST modes of each data class. Int32 runs with a 16-bit DEST flag and UInt32 with a 32-bit one, as
# the harness requires; both are unpacked straight into DEST.
_CLASSES = {
    "F": (DataFormat.Float16_b, [DestAccumulation.No, DestAccumulation.Yes]),
    "F32": (DataFormat.Float32, [DestAccumulation.Yes]),
    "I32": (DataFormat.Int32, [DestAccumulation.No]),
    "U32": (DataFormat.UInt32, [DestAccumulation.Yes]),
    "U16": (DataFormat.UInt16, [DestAccumulation.No]),
}

_BOTH = [ApproximationMode.Yes, ApproximationMode.No]
# Bodies with no approximation template argument, or one the entry point fixes.
_ONE = [ApproximationMode.No]

# entry point -> (data classes, approximation flags)
_OPS = {
    "isinf": (["F"], _BOTH),
    "isposinf": (["F"], _BOTH),
    "isneginf": (["F"], _BOTH),
    "isnan": (["F"], _BOTH),
    "isfinite": (["F"], _BOTH),
    "unary_ne": (["F"], _BOTH),
    "unary_eq": (["F"], _BOTH),
    "stochastic_round": (["F"], _BOTH),
    "tiled_prod": (["F"], _BOTH),
    "power_iterative": (["F"], _BOTH),
    "alt_complex_rotate90": (["F"], _BOTH),
    "fill_bitcast": (["F"], _BOTH),
    "logical_not": (["F", "U16"], _BOTH),
    "mask_posinf": (["F"], _ONE),
    "unary_ne_int32": (["I32"], _BOTH),
    "unary_eq_int32": (["I32"], _BOTH),
    "unary_gt_int32": (["I32"], _BOTH),
    "unary_ge_int32": (["I32"], _BOTH),
    "unary_lt_int32": (["I32"], _BOTH),
    "unary_le_int32": (["I32"], _BOTH),
    "gtz_int32": (["I32"], _BOTH),
    "nez_int32": (["I32"], _BOTH),
    "gez_int32": (["I32"], _BOTH),
    "ltz_int32": (["I32"], _BOTH),
    "eqz_int32": (["I32"], _BOTH),
    "lez_int32": (["I32"], _BOTH),
    "bitwise_and": (["I32", "U32", "U16"], _BOTH),
    "bitwise_or": (["I32", "U32", "U16"], _BOTH),
    "bitwise_xor": (["I32", "U32", "U16"], _BOTH),
    "left_shift": (["I32", "U32", "U16"], _BOTH),
    "right_shift": (["I32", "U32", "U16"], _BOTH),
    "relu_max_int32": (["I32"], _BOTH),
    "relu_min_int32": (["I32"], _BOTH),
    "relu_int32": (["I32"], _BOTH),
    "relu_max_uint32": (["U32"], _BOTH),
    "relu_min_uint32": (["U32"], _BOTH),
    "relu_max_uint16": (["U16"], _BOTH),
    "relu_min_uint16": (["U16"], _BOTH),
    "identity_uint32": (["U32"], _BOTH),
    "clamp_int32": (["I32"], _BOTH),
    "signbit_int32": (["I32"], _BOTH),
    "unary_max_int32": (["I32"], _BOTH),
    "unary_min_int32": (["I32"], _BOTH),
    "unary_max_uint32": (["U32"], _BOTH),
    "unary_min_uint32": (["U32"], _BOTH),
    "sum_int_col": (["I32"], _BOTH),
    "sum_int_row": (["I32"], _BOTH),
    "add_int_unary": (["I32"], _BOTH),
    "rsub_unary_int32": (["I32"], _BOTH),
    "add_unary_int32": (["I32"], _BOTH),
    "sub_unary_int32": (["I32"], _BOTH),
    "fill_int": (["I32", "U32", "U16"], _BOTH),
    "negative_int32": (["I32"], _BOTH),
    "remainder_uint32": (["U32"], _BOTH),
    "lgamma_stirling_float": (["F", "F32"], _BOTH),
    "gcd": (["I32"], _ONE),
    "lcm": (["I32"], _ONE),
    "isclose": (["F"], _BOTH),
    "logsigmoid": (["F"], _BOTH),
    "mul_int_uint16": (["U16"], _BOTH),
    "add_int_tile": (["U16"], _BOTH),
    "sub_int_tile": (["U16"], _BOTH),
    "rsub_int_tile": (["U16"], _BOTH),
    "binary_left_shift": (["U16"], _BOTH),
    "binary_right_shift": (["U16", "U32"], _BOTH),
    "binary_logical_right_shift": (["U16"], _BOTH),
    "div_int32": (["I32"], _BOTH),
    "div_int32_floor": (["I32"], _BOTH),
    "div_int32_trunc": (["I32"], _BOTH),
    "nextafter": (["F32"], _BOTH),
    "nextafter_bf16": (["F"], _BOTH),
    "max_reduce_with_indices": (["F"], _ONE),
    "lgamma_adjusted": (["F"], _BOTH),
    "mac": (["F"], _BOTH),
}


def _class_of(formats):
    for name, (fmt, _) in _CLASSES.items():
        if formats.input_format == fmt:
            return name
    raise ValueError(f"no data class for {formats}")


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    api_op=sorted(_OPS),
    formats=lambda api_op: [
        InputOutputFormat(_CLASSES[c][0], _CLASSES[c][0]) for c in _OPS[api_op][0]
    ],
    dest_acc=lambda formats: _CLASSES[_class_of(formats)][1],
    approx_mode=lambda api_op: _OPS[api_op][1],
)
def test_perf_sfpu_compute_api(perf_report, api_op, formats, dest_acc, approx_mode):
    input_dimensions = [128, 64]
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_compute_api_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            SFPU_API_OP(api_op=api_op, api_fmt=formats.input_format.name),
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
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
    )
    configuration.run(perf_report)
