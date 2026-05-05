# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Isolated SFPU add (binary): UNPACK2 (UNP_S) x2 -> SrcS -> SFPU -> PACK1 -> L1.
No MATH kernel. Two operands unpacked to SrcS slices 0 and 1, added by SFPU,
result packed from SrcS slice 2 to L1.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import ImpliedMathFormat, MathOperation, format_dict
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    apply_log_uniform_magnitudes,
    compute_safe_input_magnitude_range,
    format_elem_max,
    generate_stimuli,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test

# Safety factor applied to format maxima when clamping add operands.
# Ensures |a| + |b| stays within representable range with headroom for rounding /
# quantization (two operands each capped at 45% of max -> sum <= 90% of max).
ADD_RANGE_SAFETY_FACTOR = 0.45

SFPU_ADD_FORMATS = input_output_formats(
    [
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)

SFPU_ADD_COMBINATIONS = [
    (fmt, dest_acc, implied_math_format, input_dimensions)
    for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(SFPU_ADD_FORMATS)
    for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
    for input_dimensions in [[32, 32], [64, 64]]
]


@pytest.mark.quasar
@parametrize(formats_dest_acc_implied_math_input_dims=SFPU_ADD_COMBINATIONS)
def test_isolate_sfpu_add_quasar(formats_dest_acc_implied_math_input_dims):
    """
    Test isolated SFPU add (binary): UNPACK2 (UNP_S) x2 -> SrcS -> SFPU -> PACK1 -> L1.
    No MATH kernel (stub only). Two input operands unpacked to SrcS, added, packed.
    """
    (formats, dest_acc, implied_math_format, input_dimensions) = (
        formats_dest_acc_implied_math_input_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    min_magnitude, max_magnitude = compute_safe_input_magnitude_range(
        formats.input_format,
        formats.output_format,
        input_magnitude_cap=format_elem_max(formats.input_format)
        * ADD_RANGE_SAFETY_FACTOR,
        output_magnitude_cap=format_elem_max(formats.output_format)
        * ADD_RANGE_SAFETY_FACTOR,
    )
    src_A = apply_log_uniform_magnitudes(
        src_A,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        cast_to_format=formats.input_format,
        alternate_sign_every_n=3,
    )
    src_B = apply_log_uniform_magnitudes(
        src_B,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        cast_to_format=formats.input_format,
        alternate_sign_every_n=3,
    )

    num_faces = 4

    # Golden: use BinarySFPUGolden so we can swap ops for future binary kernels.
    # SrcS path is untilized, so skip_tilize=True. Concatenate full tensors:
    # [all A tiles | all B tiles], then index by tile count offset.
    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.SfpuElwadd,
        torch.cat([src_A, src_B]),
        0,  # src1_idx: first tile of A
        tile_cnt_A,  # src2_idx: first tile of B
        0,  # dst_idx: write result starting at tile 0
        tile_cnt_A * 32,  # num_iterations: 32 rows per tile
        [input_dimensions[0] * 2, input_dimensions[1]],
        formats.output_format,
        skip_tilize=True,
        input_format=formats.input_format,
    )[
        : src_A.numel()
    ]  # Extract only the result region (A's tiles)

    configuration = TestConfig(
        "sources/quasar/isolate_sfpu_add_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(implied_math_format),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_srcs=True,
        dest_acc=dest_acc,
        disable_format_inference=formats.input_format.is_mx_format(),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
