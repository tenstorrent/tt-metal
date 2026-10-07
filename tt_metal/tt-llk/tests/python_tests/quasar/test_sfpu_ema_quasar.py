# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Production-path coverage for the Quasar EMA entry (llk_math_ema_sfpu_entry.h).

The unary sweep in test_eltwise_unary_sfpu_quasar.py runs EMA in place with a fresh chain per
tile. This test drives the entry the way the ema compute kernel does instead: the carry is cleared
once, TILE_CNT time tiles are fed top to bottom so the carry chains across tile boundaries, and
each tile's EMA is stored to the Dest tile after its input (EMA_OUTPUT_TILE_DELTA = 1). The golden
is one continuous recurrence down all TILE_CNT * 32 rows.

Three schedules (see sfpu_ema_quasar_test.cpp):
  * DestSync.Full: the whole chain in one Dest section.
  * DestSync.Half: one Dest section per time tile, released after each pack, as ema_compute.cpp
    runs it, so the carry crosses a section release / bank flip at every tile boundary.
  * DestSync.Half + MATH_TRANSPOSE_FACES: each input tile is transposed in Dest by transpose_dest,
    a replay-bank-0 FPU op, between EMA tiles. 16-bit Dest only: 32-bit transpose_dest needs
    unpack-to-Dest, which this datacopy-based source does not use.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIM,
    TilizeGolden,
    UntilizeGolden,
    ema_down_columns,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    Transpose,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.sfpu_dispatch_constants import EMA_ALPHA_BITS, EMA_BETA_BITS
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_SYNC,
    EMA_ALPHA_BETA,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    MATH_TRANSPOSE_FACES,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

# Float32 is left out: through the SrcA datacopy this source uses it is narrowed to TF32, which a
# float32 golden does not model. The unary sweep covers Float32 unpacked straight to Dest.
EMA_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
]


def _ema_schedules():
    """(dest_sync, dest_acc, transpose, time_tiles) per schedule.

    Full sync keeps input tile t at Dest tile 2t and its EMA at 2t + 1, so 4 time tiles fill the
    8 tiles of a full 32-bit Dest. Half sync reuses tiles 0 / 1 of each section, so any count
    crosses a release; 4 tiles cycle both Dest halves twice.
    """
    schedules = []
    for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
        schedules += [(DestSync.Full, dest_acc, Transpose.No, n) for n in (1, 2, 4)]
        schedules += [(DestSync.Half, dest_acc, Transpose.No, n) for n in (2, 4)]
    schedules += [
        (DestSync.Half, DestAccumulation.No, Transpose.Yes, n) for n in (2, 4)
    ]
    return schedules


@pytest.mark.quasar
@parametrize(
    formats=EMA_FORMATS,
    schedule=_ema_schedules(),
)
def test_sfpu_ema_quasar(formats, schedule):
    dest_sync, dest_acc, transpose, time_tiles = schedule
    torch.manual_seed(0)

    input_dimensions = [time_tiles * TILE_DIM, TILE_DIM]
    torch_format = format_dict[formats.input_format]

    # alpha + beta = 1 keeps every output a convex mix of its column's inputs; both signs make the
    # carry and the input cancel as well as reinforce.
    src_A = torch.empty(input_dimensions, dtype=torch.float32).uniform_(-4.0, 4.0)
    src_A = src_A.to(torch_format)
    src_B = torch.zeros(time_tiles * TILE_DIM * TILE_DIM, dtype=torch_format)

    # transpose_dest transposes each 32x32 input tile in Dest before its EMA.
    ema_input = src_A
    if transpose == Transpose.Yes:
        ema_input = src_A.reshape(time_tiles, TILE_DIM, TILE_DIM).transpose(1, 2)
        ema_input = ema_input.reshape(input_dimensions)
    golden = ema_down_columns(ema_input).to(format_dict[formats.output_format])

    device_src_A = get_golden_generator(TilizeGolden)(
        src_A.flatten(), input_dimensions, formats.input_format
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_ema_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Ema),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(dest_sync),
            MATH_TRANSPOSE_FACES(transpose),
            EMA_ALPHA_BETA(alpha_bits=EMA_ALPHA_BITS, beta_bits=EMA_BETA_BITS),
        ],
        runtimes=[
            TILE_COUNT(time_tiles),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
        ],
        variant_stimuli=StimuliConfig(
            device_src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=time_tiles,
            tile_count_B=time_tiles,
            tile_count_res=time_tiles,
            num_faces=MAX_NUM_FACES,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = get_golden_generator(UntilizeGolden)(
        res_tensor, formats.output_format, input_dimensions
    )

    assert passed_test(
        golden.flatten(), res_tensor.flatten(), formats.output_format
    ), "Assert against golden failed"
