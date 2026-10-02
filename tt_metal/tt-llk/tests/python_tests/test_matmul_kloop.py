# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Golden test for the K-looping matmul entry point (_llk_unpack_AB_matmul_kloop_ / _llk_math_matmul_kloop_).

On Wormhole a 2x2 block alternates the MVMUL order between K tiles; the result must be bit-identical to the
per-K-tile matmul, so this reuses the matmul golden. Blocks: 2x2 (the alternating path) at kt 4 and 32, plus
2x1 and 1x2 (the fall-through to the standard loop) at kt 4. bf16 and fp32 inputs, dest accumulation on and off,
all four fidelities (kt 32 with fp32 accumulation only; see kloop_combinations).
"""

from typing import List

import torch
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
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

TILE = 32
# (rt, ct, kt): output block rows x columns in tiles, K tiles
BLOCKS = [(2, 2, 4), (2, 2, 32), (2, 1, 4), (1, 2, 4)]


def kloop_combinations(
    formats_list: List[FormatConfig], dest_acc_modes: List[DestAccumulation]
):
    combos = []
    for fmt in formats_list:
        for dest_acc in dest_acc_modes:
            if is_dest_acc_needed(fmt) and dest_acc == DestAccumulation.No:
                continue
            for rt, ct, kt in BLOCKS:
                # The matmul golden's tolerance does not cover 16-bit dest accumulation over K = 1024 (the standard
                # matmul_test.cpp fails those cases the same way), so kt 32 runs with fp32 accumulation only.
                if kt > 4 and dest_acc == DestAccumulation.No:
                    continue
                combos.append(
                    (fmt, dest_acc, ([rt * TILE, kt * TILE], [kt * TILE, ct * TILE]))
                )
    return combos


KLOOP_FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
KLOOP_COMBINATIONS = kloop_combinations(
    KLOOP_FORMATS, [DestAccumulation.No, DestAccumulation.Yes]
)


@parametrize(
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    format_dest_acc_and_dims=KLOOP_COMBINATIONS,
)
def test_matmul_kloop(math_fidelity, format_dest_acc_and_dims):
    torch_format = format_dict[format_dest_acc_and_dims[0].output_format]

    formats = format_dest_acc_and_dims[0]
    dest_acc = format_dest_acc_and_dims[1]
    input_A_dimensions = format_dest_acc_and_dims[2][0]
    input_B_dimensions = format_dest_acc_and_dims[2][1]

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

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
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

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/matmul_kloop_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
            DEST_SYNC(),
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
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
