# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Perf rows for the SFPU kernels that fit neither the unary registry sweep nor the binary one:
rand, dropout, mask (float and Int32), copy_dest_values, reshuffle_rows, softcap, situ_glu and
clamped_silu_glu. None of them had a perf row before; the first cycle numbers came from an
out-of-tree copy of this kernel (tenstorrent/tt-metal#58510).

sources/sfpu_misc_perf.cpp keeps the unpack and pack threads and the zones of
eltwise_unary_sfpu_perf.cpp and runs one body per tile on the math thread, after the datacopy
carrier, in the compute API form (four faces of eight rows). The two-operand bodies (mask,
copy_dest_values, situ_glu, clamped_silu_glu) copy the tile into DEST tiles 0 and 1 and write
tile 0; reshuffle_rows accumulates into DEST tile 1 with a 32-byte index array the math thread
writes into an unused input ring once, in the INIT zone. cycles/tile lands in the TILE_LOOP row of
the .post.csv as mean(MATH_ISOLATE), with the datacopy carrier included as in every SFPU perf test.

misc_param selects the rand scale form (0: the normalisation folded into the scale, the
16-instruction row the randn and uniform kernels run; 1: a scale below 2^-95, the 17-instruction
row) and the reshuffle_rows index pattern (0 identity, 1 reversed, 2 every second row skipped).
misc_init_per_tile re-runs the op's init before every tile, which for rand and dropout is the
PRNG seed write and its settle wait.
"""

import pytest
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
    SFPU_MISC_OP,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

_BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
_INT32 = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

# (op, formats, dest_acc, misc_param, misc_init_per_tile)
_ROWS = [
    ("rand", _BF16, DestAccumulation.No, 0, False),
    ("rand", _BF16, DestAccumulation.No, 1, False),
    ("rand", _BF16, DestAccumulation.No, 0, True),
    ("dropout", _BF16, DestAccumulation.No, 0, False),
    ("dropout", _BF16, DestAccumulation.No, 0, True),
    ("mask", _BF16, DestAccumulation.No, 0, False),
    ("mask", _BF16, DestAccumulation.Yes, 0, False),
    ("mask_int", _INT32, DestAccumulation.Yes, 0, False),
    ("copy_dest_values", _BF16, DestAccumulation.No, 0, False),
    ("copy_dest_values", _INT32, DestAccumulation.Yes, 0, False),
    ("reshuffle_rows", _BF16, DestAccumulation.No, 0, False),
    ("reshuffle_rows", _BF16, DestAccumulation.No, 1, False),
    ("reshuffle_rows", _BF16, DestAccumulation.No, 2, False),
    ("softcap", _BF16, DestAccumulation.No, 0, False),
    ("softcap", _BF16, DestAccumulation.Yes, 0, False),
    ("softcap", _BF16, DestAccumulation.No, 0, True),
    ("situ_glu", _BF16, DestAccumulation.No, 0, False),
    ("situ_glu", _BF16, DestAccumulation.Yes, 0, False),
    ("clamped_silu_glu", _BF16, DestAccumulation.No, 0, False),
    ("clamped_silu_glu", _BF16, DestAccumulation.Yes, 0, False),
]


@pytest.mark.perf
@parametrize(
    row=_ROWS,
    loop_factor=[16],  # amortise profiler overhead
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_sfpu_misc(perf_report, row, loop_factor, input_dimensions):
    op, formats, dest_acc, misc_param, init_per_tile = row

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )

    configuration = PerfConfig(
        "sources/sfpu_misc_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            SFPU_MISC_OP(misc_mathop=op, misc_param=misc_param, misc_init_per_tile=init_per_tile),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
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
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        # 32-bit inputs unpack straight into the 32-bit DEST, as in the unary sweep.
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
