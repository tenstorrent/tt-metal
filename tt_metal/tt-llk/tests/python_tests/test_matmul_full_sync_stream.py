# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    PERF_RUN_TYPE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

FULL_SYNC_STREAM_DIMENSIONS = [
    ([32, 64], [64, 512]),
    ([512, 64], [64, 32]),
    ([32, 64], [64, 320]),
    ([320, 64], [64, 32]),
    ([64, 64], [64, 256]),
    ([256, 64], [64, 64]),
    ([128, 64], [64, 128]),
]


@parametrize(
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2],
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Bfp8_b]),
    dimensions=FULL_SYNC_STREAM_DIMENSIONS,
)
def test_matmul_full_sync_stream(math_fidelity, formats, dimensions):
    input_A_dimensions, input_B_dimensions = dimensions

    torch_format = format_dict[formats.output_format]
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    golden_tensor = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    if formats.input_format != DataFormat.Bfp8_b:
        src_A = tilize_block(
            src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
        )
        src_B = tilize_block(
            src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
        )

    configuration = TestConfig(
        "sources/matmul_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
            DEST_SYNC(DestSync.Full),
            THROTTLE_LEVEL(),
        ],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            LOOP_FACTOR(1),
            UNPACK_TRANS_FACES(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=DestAccumulation.No,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
