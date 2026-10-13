# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar EMA entry as ema_compute.cpp drives it: one chain carried across tiles, output at dst + 1.

Schedules are described in sfpu_ema_quasar_test.cpp."""

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

# No Float32: the SrcA datacopy narrows it to TF32 (the unary sweep covers it via unpack-to-Dest).
EMA_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
]


def _ema_schedules():
    """(dest_sync, dest_acc, transpose, time_tiles). Full sync needs 2 Dest tiles per time tile;
    transpose_dest runs on 16-bit Dest only (32-bit needs unpack-to-Dest)."""
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

    src_A = torch.empty(input_dimensions, dtype=torch.float32).uniform_(-4.0, 4.0)
    src_A = src_A.to(torch_format)
    src_B = torch.zeros(time_tiles * TILE_DIM * TILE_DIM, dtype=torch_format)

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
