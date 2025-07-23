# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import DestAccumulation, MathFidelity, format_dict
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


@parametrize(
    test_name="matmul_unpack_tilize_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ],  #  Add DataFormat.Bfp8_b only as input when Data format Inference Model 2.0 supports format conversions for > 1 pipeline run with different inputs and outputs.
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_matmul_unpack_tilize(test_name, formats, dest_acc, math_fidelity):

    torch_format = format_dict[formats.output_format]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = tilize(
        generate_golden(src_A, src_B, formats.output_format, math_fidelity)
    )
    golden_tensor = golden_tensor.to(torch_format)

    res_address = write_stimuli_to_l1(
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
    )
    buffer_dest_address = 0x1E000
    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "L1_to_L1_iterations": 2,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        test_config.get(
            "L1_to_L1_iterations"  # Needed to calculate accumulated precision loss for fused tests that copy result tensor as input for next runs
        ),
    )
