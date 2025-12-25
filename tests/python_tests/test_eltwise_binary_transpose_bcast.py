# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_blackhole
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    Transpose,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize, tilize_block
from helpers.utils import passed_test


@skip_for_blackhole
@parametrize(
    test_name="eltwise_binary_transpose_bcast_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    broadcast_type=[BroadcastType.Column, BroadcastType.Row],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.LoFi],
    transpose_srca=[Transpose.Yes],
    input_dimensions=[[32, 32]],
)
def test_eltwise_binary_transpose_bcast(
    test_name,
    formats,
    broadcast_type,
    dest_acc,
    math_fidelity,
    transpose_srca,
    input_dimensions,
):

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    # Tilize the input data for hardware
    src_A_tilized = tilize_block(src_A, input_dimensions, formats.input_format)
    src_B_tilized = tilize_block(src_B, input_dimensions, formats.input_format)

    # Compute golden using proper transpose generator that understands tilized data
    transpose_golden = get_golden_generator(TransposeGolden)

    # Apply transpose to srcA: hardware does transpose_faces then transpose_within_faces
    src_A_transposed = transpose_golden.transpose_faces_multi_tile(
        src_A,
        formats.input_format,
        num_tiles=tile_cnt,
        tilize=True,  # Tilize before transpose (models hardware behavior)
        input_dimensions=input_dimensions,
    )
    src_A_transposed = transpose_golden.transpose_within_faces_multi_tile(
        src_A_transposed,
        formats.input_format,
        num_tiles=tile_cnt,
        untilize=True,  # Untilize after transpose for golden comparison
        input_dimensions=input_dimensions,
    )

    src_B_tilized_for_bcast = tilize(
        src_B, stimuli_format=formats.input_format, num_faces=4
    )
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted_tilized = broadcast_golden(
        broadcast_type,
        src_B_tilized_for_bcast,  # Tilized data
        formats.input_format,
        num_faces=4,
        tile_cnt=tile_cnt,
        face_r_dim=16,
    )

    src_A_transposed_tilized = tilize(
        src_A_transposed, stimuli_format=formats.output_format, num_faces=4
    )

    # Compute element-wise subtraction in tilized format
    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = binary_golden(
        MathOperation.Elwsub,
        src_A_transposed_tilized,  # Tilized
        src_B_broadcasted_tilized,  # Tilized
        formats.output_format,
        math_fidelity,
    )

    # Build test configuration
    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "mathop": MathOperation.Elwsub,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_cnt,
        "broadcast_type": broadcast_type,
        "unpack_transpose_faces": transpose_srca,
        "unpack_transpose_within_face": transpose_srca,
        "num_faces": 4,
        "unpack_to_dest": False,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A_tilized,
        src_B_tilized,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Compare in tilized format
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
