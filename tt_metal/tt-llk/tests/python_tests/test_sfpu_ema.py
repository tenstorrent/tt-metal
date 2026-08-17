# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    EMA_ALPHA_BETA,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# EMA smoothing factor chosen for this test (ttnn.ema has no default alpha; it is a
# required argument). beta is derived as 1 - alpha.
EMA_ALPHA = 0.25
EMA_BETA = 1.0 - EMA_ALPHA


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _ema_golden(input_2d: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Continuous per-column EMA down the rows (time axis), carry starting at 0.

    EMA_new = alpha * EMA_old + beta * input. The device carries EMA_old in LREG4
    at full fp32 precision across tiles, so the recurrence runs in fp32 over the
    whole [rows, cols] input; only the stored output is later rounded to the
    output format.
    """
    rows, cols = input_2d.shape
    out = torch.empty((rows, cols), dtype=torch.float32)
    prev = torch.zeros(cols, dtype=torch.float32)
    x = input_2d.to(torch.float32)
    for t in range(rows):
        cur = alpha * prev + beta * x[t]
        out[t] = cur
        prev = cur
    return out


# dest_acc=Yes (32-bit DEST) is intentionally not swept: the EMA kernel is hardcoded for
# 16-bit bf16 DEST (_ema_load/store_current_input_ use SFPLOADI_MOD0_FLOATB at fixed fp16
# offsets with no is_fp32_dest_acc_en branch), so a 32-bit DEST is not supported.
@parametrize(
    dest_acc=[DestAccumulation.No],
    num_time_tiles=[1, 2, 4],
)
def test_sfpu_ema(dest_acc, num_time_tiles):
    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    # [rows, cols] = [num_time_tiles*32 time steps, 32 parallel channels].
    input_dimensions = [num_time_tiles * TILE_DIM, TILE_DIM]
    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(-4.0, 4.0)
    src_B = torch.zeros_like(src_A)

    # Untilized 2D view for the golden (row = time step, col = channel).
    golden_input = src_A.view(input_dimensions[0], input_dimensions[1])
    golden_tensor = _ema_golden(golden_input, EMA_ALPHA, EMA_BETA)

    # Tilize the input the same way the device consumes it (row-major tile order,
    # tile k = rows [32k, 32k+32)).
    src_A_tilized = tilize_block(
        src_A, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_ema_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            EMA_ALPHA_BETA(
                alpha_bits=_f32_bits(EMA_ALPHA), beta_bits=_f32_bits(EMA_BETA)
            ),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
        ],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "EMA result does not match golden"
