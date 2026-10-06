# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Production-path coverage for the Quasar EMA entry (llk_math_ema_sfpu_entry.h).

The unary sweep in test_eltwise_unary_sfpu_quasar.py runs EMA in place with a fresh chain per
tile. This test drives the entry the way the ema compute kernel does instead: the carry is cleared
once, TILE_CNT time tiles are fed top to bottom so the carry chains across tile boundaries, and
each tile's EMA is stored to the Dest tile after its input (OUT_TILE_DELTA = 1). The golden is
one continuous recurrence down all TILE_CNT * 32 rows.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIM,
    TilizeGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.sfpu_dispatch_constants import EMA_ALPHA_BITS, EMA_BETA_BITS
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    EMA_ALPHA_BETA,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

# Float32 is left out: through the SrcA datacopy this source uses it is narrowed to TF32, which a
# float32 golden does not model. The unary sweep covers Float32 unpacked straight to Dest.
EMA_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
]

# Input tile t is copied to Dest tile 2t and its EMA stored to 2t + 1, so 4 time tiles fill the
# 8 tiles of a full 32-bit Dest.
EMA_TIME_TILES = [1, 2, 4]


def _fp32(bits: int) -> float:
    return torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()


def _continuous_ema_golden(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """EMA down all rows of a row-major [rows, 32] tensor, carry starting at 0 and never reset.

    The kernel keeps the carry in an fp32 LREG across tiles, so the recurrence runs in float32 and
    only the stored output is rounded to the output format.
    """
    x = x.to(torch.float32)
    out = torch.empty_like(x)
    prev = torch.zeros(x.shape[1], dtype=torch.float32)
    for row in range(x.shape[0]):
        prev = alpha * prev + beta * x[row]
        out[row] = prev
    return out


@pytest.mark.quasar
@parametrize(
    formats=EMA_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    time_tiles=EMA_TIME_TILES,
)
def test_sfpu_ema_quasar(formats, dest_acc, time_tiles):
    torch.manual_seed(0)

    input_dimensions = [time_tiles * TILE_DIM, TILE_DIM]
    torch_format = format_dict[formats.input_format]

    # alpha + beta = 1 keeps every output a convex mix of its column's inputs; both signs make the
    # carry and the input cancel as well as reinforce.
    src_A = torch.empty(input_dimensions, dtype=torch.float32).uniform_(-4.0, 4.0)
    src_A = src_A.to(torch_format)
    src_B = torch.zeros(time_tiles * TILE_DIM * TILE_DIM, dtype=torch_format)

    golden = _continuous_ema_golden(
        src_A, _fp32(EMA_ALPHA_BITS), _fp32(EMA_BETA_BITS)
    ).to(format_dict[formats.output_format])

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
            DEST_SYNC(DestSync.Full),
            EMA_ALPHA_BETA(alpha_bits=EMA_ALPHA_BITS, beta_bits=EMA_BETA_BITS),
        ],
        runtimes=[
            TILE_COUNT(time_tiles),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
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
            num_faces=4,
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
