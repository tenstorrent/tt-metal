# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_blackhole
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


@skip_for_blackhole
@parametrize(
    test_name="unpack_a_bcast_eltwise_test",
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
    test_name,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    input_dimensions,
    srca_reuse_count,
):

    # Precompute constants
    input_tiles = input_dimensions[0] * input_dimensions[1] // 1024
    reuse_factor = input_tiles // srca_reuse_count

    if input_tiles % srca_reuse_count != 0:
        pytest.skip("Input tiles must be divisible by reuse factor")

    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    src_A = src_A[: 1024 * reuse_factor]

    reshaped_a = src_A.reshape(64 * reuse_factor, 16)

    b_tiles = [src_B[i : i + 1024].tolist() for i in range(0, len(src_B), 1024)]
    b_tiles = b_tiles[:]

    take = []
    for i in range(0, reshaped_a.shape[0], 64):
        take.append(reshaped_a[i])
        take.append(reshaped_a[i + 16])

    # Reconstruct tiles with boradcasted data

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

    golden_tensor = torch.cat(golden).to(dtype=format_dict[formats.output_format])

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "mathop": mathop,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_cnt,
        "srca_reuse_count": srca_reuse_count,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=reuse_factor,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
