# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Isolated SFPU where (ternary): UNPACK2 (UNP_S) x3 -> SrcS -> SFPU -> PACK1 -> L1.
No MATH kernel. condition (buffer_A), true value (buffer_B) and false value
(buffer_C) are unpacked to SrcS slices 0, 1 and 2; the SFPU selects into slice 2
and PACK1 writes it to L1.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import WhereGolden, get_golden_generator
from helpers.llk_params import PerfRunType, format_dict
from helpers.param_config import (
    generate_quasar_srcs_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_INDEX,
    LOOP_FACTOR,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test, result_tensor
from quasar.test_sfpu_where_quasar import _build_condition_for_test_case

# Float formats with a SrcS register layout.
SFPU_WHERE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
    ]
)

INPUT_DIMENSIONS = ([32, 32], [64, 64])
# Perf uses the multi-tile profile; one tile is too short to time.
PERF_INPUT_DIMENSIONS = [64, 64]


def generate_isolate_where_combinations(*, is_perf: bool = False):
    dimension_profiles = (PERF_INPUT_DIMENSIONS,) if is_perf else INPUT_DIMENSIONS
    return [
        (fmt, dest_acc, runtime(input_dimensions))
        for fmt, dest_acc in generate_quasar_srcs_format_dest_acc_combinations(
            SFPU_WHERE_FORMATS
        )
        for input_dimensions in dimension_profiles
    ]


SFPU_WHERE_COMBINATIONS = generate_isolate_where_combinations()


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_input_dims=SFPU_WHERE_COMBINATIONS,
    # all_zeros selects in2 (already in slice 2), all_ones selects in1, mixed both.
    test_case=runtime(["mixed", "all_ones", "all_zeros"]),
)
def test_isolate_sfpu_where_quasar(
    formats_dest_acc_input_dims,
    test_case,
    *,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor=1,
    is_perf=False,
    perf_report=None,
):
    """
    Test isolated SFPU where (ternary): UNPACK2 (UNP_S) x3 -> SrcS -> SFPU -> PACK1 -> L1.
    No MATH kernel (stub only). Three input operands unpacked to SrcS, selected, packed.
    """
    formats, dest_acc, input_dimensions = formats_dest_acc_input_dims
    torch_format_in = format_dict[formats.input_format]
    torch_format_out = format_dict[formats.output_format]

    torch.manual_seed(42)
    src_cond_raw, tile_cnt, src_true_raw, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(43)
    src_false_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    condition = _build_condition_for_test_case(
        src_cond_raw, formats.input_format, test_case
    )
    true_val = src_true_raw.to(torch_format_in)
    false_val = src_false_raw.to(torch_format_in)

    num_faces = 4

    if not is_perf:
        # SrcS path is untilized and where is element-wise, so the row-major golden
        # matches the packed result directly.
        generate_golden = get_golden_generator(WhereGolden)
        golden_tensor = generate_golden(condition, true_val, false_val).to(
            torch_format_out
        )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/isolate_sfpu_where_quasar_test.cpp",
        "formats": formats,
        "templates": [],
        "runtimes": [
            TILE_COUNT(tile_cnt),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            condition,
            formats.input_format,
            true_val,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            buffer_C=false_val,
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
        ),
        "unpack_to_srcs": True,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )

    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)
    assert passed_test(golden_tensor, res_tensor, formats.output_format, max_ulp=0)
