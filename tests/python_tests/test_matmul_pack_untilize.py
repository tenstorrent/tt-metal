# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
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
    test_name="matmul_pack_untilize_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Bfp8_b,  # Pack Untilize doesn't work for block float formats (Bfp8_b); we only include as input format in our test
            DataFormat.Float32,
        ]
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_matmul_pack_untilize(test_name, formats, dest_acc, math_fidelity):
    if formats.output == DataFormat.Bfp8_b:
        pytest.skip("Pack untilize does not support Bfp8_b")

    torch_format = format_dict[formats.output_format]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(src_A, src_B, formats.output_format, math_fidelity)

    res_address = write_stimuli_to_l1(
        tilize(src_A, formats.input_format),
        tilize(src_B, formats.input_format),
        formats.input_format,
        formats.input_format,
        tile_count=tile_cnt,
    )

    test_config = {
        "testname": test_name,
        "formats": formats,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=(torch_format))

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
