# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import WhereGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


def _get_valid_formats_dest_acc():
    formats = input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Float32,
        ]
    )
    return [
        (fmt, dest_acc)
        for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats)
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        )
    ]


def _get_valid_implied_math_formats(fmt: FormatConfig):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _build_condition_for_test_case(
    base: torch.Tensor, input_format: DataFormat, test_case: str
) -> torch.Tensor:
    """Apply the condition regime (mixed / all_ones / all_zeros) for a variant."""
    torch_format = format_dict[input_format]
    if test_case == "all_ones":
        return torch.ones_like(base, dtype=torch_format)
    if test_case == "all_zeros":
        return torch.zeros_like(base, dtype=torch_format)
    # "mixed" — raw stimuli as condition (mostly non-zero, exercises true branch).
    return base.to(torch_format)


def _is_unpack_to_dest(fmt: FormatConfig, dest_acc: DestAccumulation) -> bool:
    """UNPACK→DEST is selected only for 32-bit inputs with dest_acc=Yes."""
    return fmt.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    test_case=["mixed", "all_ones", "all_zeros"],
)
def test_sfpu_where_quasar(formats_dest_acc, implied_math_format, test_case):
    """
    Test ternary `where(condition, true_val, false_val) -> output` on Quasar.

    The C++ test source packs 3 input tiles (condition, true_val, false_val)
    into `buffer_A`, datacopies them into DEST at tile indices 0, 1, 2, then
    runs the SFPU `where` kernel face-by-face writing output to DEST tile 0.
    PACK writes DEST tile 0 out to `buffer_Res`.

    Variants cover `mixed` / `all_ones` / `all_zeros` condition regimes — the
    last two pin the selector to a single branch so format issues on either
    side show up in isolation.
    """
    formats, dest_acc = formats_dest_acc
    input_dimensions = [32, 32]
    torch_format_in = format_dict[formats.input_format]

    torch.manual_seed(42)
    src_cond_raw, tile_cnt_single, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(43)
    src_true_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(44)
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

    src_A = torch.cat([condition, true_val, false_val])
    tile_cnt_A = tile_cnt_single * 3
    num_faces = 4

    generate_golden = get_golden_generator(WhereGolden)
    golden_tensor = generate_golden(condition, true_val, false_val)
    torch_format_out = format_dict[formats.output_format]
    golden_tensor = golden_tensor.to(torch_format_out)

    unpack_to_dest = _is_unpack_to_dest(formats, dest_acc)
    src_B_dummy = torch.zeros_like(condition)

    configuration = TestConfig(
        "sources/quasar/sfpu_where_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuWhere),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B_dummy,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_single,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_formats_dest_acc()[:3],
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
)
def test_sfpu_where_mcw_quasar(formats_dest_acc, implied_math_format):
    """
    Deterministic where test — alternating 0/1 condition pattern with
    known true/false scalars (2 and 11) for easy debugging.

    Runs through the same C++ harness as `test_sfpu_where_quasar`, so if
    this fails but the stimulus-driven test passes, the problem is in
    stimulus generation rather than the kernel.
    """
    formats, dest_acc = formats_dest_acc
    torch_format_in = format_dict[formats.input_format]
    input_dimensions = [32, 32]
    height, width = input_dimensions

    pattern = torch.arange(height * width) % 2
    condition = pattern.view(height, width).to(torch_format_in).flatten()
    true_val = (torch.ones(height, width, dtype=torch_format_in) * 2).flatten()
    false_val = (torch.ones(height, width, dtype=torch_format_in) * 11).flatten()

    tile_cnt_single = 1
    src_A = torch.cat([condition, true_val, false_val])
    tile_cnt_A = tile_cnt_single * 3
    num_faces = 4

    generate_golden = get_golden_generator(WhereGolden)
    golden_tensor = generate_golden(condition, true_val, false_val)
    torch_format_out = format_dict[formats.output_format]
    golden_tensor = golden_tensor.to(torch_format_out)

    unpack_to_dest = _is_unpack_to_dest(formats, dest_acc)
    src_B_dummy = torch.zeros_like(condition)

    configuration = TestConfig(
        "sources/quasar/sfpu_where_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuWhere),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B_dummy,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_single,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
