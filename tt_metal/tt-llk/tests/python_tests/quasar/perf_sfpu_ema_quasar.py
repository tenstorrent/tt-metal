# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar EMA perf, laid out as ema_compute.cpp's EMA step (sources/quasar/sfpu_ema_perf_quasar_test.cpp)."""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM, TilizeGolden, get_golden_generator
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DataCopyType,
    DestAccumulation,
    DestSync,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.sfpu_dispatch_constants import EMA_ALPHA_BITS, EMA_BETA_BITS
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    EMA_ALPHA_BETA,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES

# ttnn's EMA takes and returns BFLOAT16 only (ema_device_operation.cpp).
EMA_PERF_FORMATS = [InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)]
# One chain of this many time tiles per measurement pass.
EMA_PERF_TIME_TILES = 4


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=EMA_PERF_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
)
def test_perf_sfpu_ema_quasar(
    perf_report, formats, dest_acc, dest_sync, run_types, loop_factor
):
    torch.manual_seed(0)
    time_tiles = EMA_PERF_TIME_TILES
    input_dimensions = [time_tiles * TILE_DIM, TILE_DIM]
    torch_format = format_dict[formats.input_format]

    src_A = torch.empty(input_dimensions, dtype=torch.float32).uniform_(-4.0, 4.0)
    src_A = src_A.to(torch_format)
    src_B = torch.zeros(time_tiles * TILE_DIM * TILE_DIM, dtype=torch_format)
    device_src_A = get_golden_generator(TilizeGolden)(
        src_A.flatten(), input_dimensions, formats.input_format
    )

    configuration = create_test_or_perf_config(
        is_perf=True,
        run_types=run_types,
        test_config_kwargs={
            "test_name": "sources/quasar/sfpu_ema_perf_quasar_test.cpp",
            "formats": formats,
            "templates": [
                MATH_OP(mathop=MathOperation.Ema),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
                DEST_SYNC(dest_sync),
                EMA_ALPHA_BETA(alpha_bits=EMA_ALPHA_BITS, beta_bits=EMA_BETA_BITS),
            ],
            "runtimes": [
                TILE_COUNT(time_tiles),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(loop_factor),
            ],
            "variant_stimuli": StimuliConfig(
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
            "unpack_to_dest": False,
            "dest_acc": dest_acc,
        },
    )
    configuration.run(perf_report)
