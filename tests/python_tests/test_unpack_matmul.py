# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import (
    DestAccumulation,
    MathFidelity,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


@parametrize(
    test_name="unpack_matmul_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    stochastic_rnd=[
        StochasticRounding.Fpu,
        StochasticRounding.Pack,
        StochasticRounding.All,
        StochasticRounding.No,
    ],
    transpose=[Transpose.Yes, Transpose.No],
)
def test_matmul(test_name, formats, dest_acc, math_fidelity, stochastic_rnd, transpose):

    torch_format = format_dict[formats.output_format]

    input_dimensions = [32, 32]  # Will be sweeping over dimensions

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )
    src_B_golden = src_B
    if transpose == Transpose.Yes:
        # In hw we tilize inputs for matmul and then transpose on src_B
        # We must first tilize src_B before transpose + haloize
        # However, torch works with row major data so we untilize this for now in order to properly compute golden
        t_matrix = get_golden_generator(TransposeGolden)
        src_B_golden = t_matrix.transpose_faces(
            src_B, formats.input_format, tilize=True, input_dimensions=input_dimensions
        )
        src_B_golden = t_matrix.transpose_within_faces(
            src_B_golden,
            formats.input_format,
            untilize=True,
            input_dimensions=input_dimensions,
        )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,  # not tilized
        src_B_golden,  # needs to be transposed and tilized
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_dimensions,
        input_B_dimensions=input_dimensions,
    )
    # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
    golden_tensor = (
        tilize_block(
            golden_tensor,
            dimensions=input_dimensions,
            stimuli_format=formats.output_format,
        )
        .to(torch_format)
        .flatten()
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_dimensions, stimuli_format=formats.input_format
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_cnt,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "stochastic_rnd": stochastic_rnd,
        "unpack_transpose_faces": transpose.value,
        "unpack_transpose_within_face": transpose.value,  # matmul transposes both faces and within faces, there is no option for one or the other
    }

    res_address = write_stimuli_to_l1(
        test_config,
        tilized_A.flatten(),
        tilized_B.flatten(),
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
