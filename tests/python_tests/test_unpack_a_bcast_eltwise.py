# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    SRCA_REUSE_COUNT,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


@skip_for_blackhole
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    mathop=[MathOperation.Elwsub, MathOperation.Elwadd, MathOperation.Elwmul],
    dest_acc=[DestAccumulation.No],
    srca_reuse_count=[2, 4, 8],
    math_fidelity=[
        MathFidelity.LoFi,
    ],
    input_dimensions=[
        [128, 32],
        [32, 128],
        [64, 128],
    ],
)
def test_unp_bcast_sub_sdpa(
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    input_dimensions,
    srca_reuse_count,
    workers_tensix_coordinates,
):

    # Precompute constants
    input_tiles = input_dimensions[0] * input_dimensions[1] // 1024
    reuse_factor = input_tiles // srca_reuse_count

    if input_tiles % srca_reuse_count != 0:
        pytest.skip("Input tiles must be divisible by reuse factor")

    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    src_A = src_A[: 1024 * reuse_factor]

    reshaped_a = src_A.reshape(64 * reuse_factor, 16)

    b_tiles = [src_B[i : i + 1024].tolist() for i in range(0, len(src_B), 1024)]
    b_tiles = b_tiles[:]

    take = []
    for i in range(0, reshaped_a.shape[0], 64):
        take.append(reshaped_a[i])
        take.append(reshaped_a[i + 16])

    # Reconstruct tiles with broadcasted data

    reconstructed_tiles = []
    for i in range(0, len(take), 2):
        if i + 1 < len(take):
            # Combine pair into 1x32 element
            combined = torch.cat([take[i], take[i + 1]], dim=0)
            # Replicate to create 32x32 tile
            tile_32x32 = combined.repeat(32, 1)
            reconstructed_tiles.append(tile_32x32.flatten())

    tilized_reconstructed_tiles = [tilize(tile) for tile in reconstructed_tiles]

    golden = []

    for tile_idx, reconstructed_tile in enumerate(tilized_reconstructed_tiles):
        start_b_idx = tile_idx * srca_reuse_count
        for reuse_idx in range(srca_reuse_count):
            b_tile_idx = start_b_idx + reuse_idx
            if b_tile_idx < len(b_tiles):
                b_tile = torch.tensor(b_tiles[b_tile_idx])

                if mathop == MathOperation.Elwadd:
                    result = reconstructed_tile + b_tile
                elif mathop == MathOperation.Elwsub:
                    result = reconstructed_tile - b_tile
                elif mathop == MathOperation.Elwmul:
                    result = reconstructed_tile * b_tile

                golden.append(result)

    golden_tensor = torch.cat(golden).to(dtype=format_dict[formats.output_format])[
        : 1024 * reuse_factor
    ]

    configuration = TestConfig(
        "sources/unpack_a_bcast_eltwise_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            DEST_SYNC(),
            TILE_COUNT(tile_cnt_A),
            SRCA_REUSE_COUNT(srca_reuse_count),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            SRCA_REUSE_COUNT(srca_reuse_count),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=reuse_factor,
            tile_count_B=reuse_factor,
            tile_count_res=reuse_factor,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
