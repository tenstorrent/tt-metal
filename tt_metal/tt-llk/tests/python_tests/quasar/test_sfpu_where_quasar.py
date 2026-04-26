# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — ternary SFPU where test for Quasar.

from typing import List

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
from helpers.param_config import input_output_formats, parametrize
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


def _is_invalid_quasar_combination(
    fmt: FormatConfig, dest_acc: DestAccumulation
) -> bool:
    """
    Check if format combination is invalid for Quasar.
    """
    in_fmt = fmt.input_format
    out_fmt = fmt.output_format

    # Quasar packer does not support non-Float32 to Float32 conversion when dest_acc=No
    if (
        in_fmt != DataFormat.Float32
        and out_fmt == DataFormat.Float32
        and dest_acc == DestAccumulation.No
    ):
        return True

    # Quasar SFPU with Float32 input and Float16 output requires dest_acc=Yes
    if (
        in_fmt == DataFormat.Float32
        and out_fmt == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        return True

    # Integer and float formats cannot be mixed in input/output
    if in_fmt.is_integer() != out_fmt.is_integer():
        return True

    return False


def generate_sfpu_where_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU where test combinations.

    Includes three condition regimes (`mixed`, `all_ones`, `all_zeros`) so every
    format exercises both the true-branch-only path, the false-branch-only path,
    and the mixed select.
    """
    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        for dest_acc in dest_acc_modes:
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                # MX formats require implied_math_format=Yes
                if (
                    in_fmt.is_mx_format()
                    and implied_math_format == ImpliedMathFormat.No
                ):
                    continue

                for test_case in ["mixed", "all_ones", "all_zeros"]:
                    # Golden generator hardcodes 32x32 shape; restrict inputs accordingly.
                    for input_dimensions in [[32, 32]]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                implied_math_format,
                                test_case,
                                input_dimensions,
                            )
                        )

    return combinations


# Start with the canonical float float-matrix that every Quasar SFPU test uses.
# Integer / MX coverage can be widened once the baseline passes.
SFPU_WHERE_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Float32,
    ]
)


def _prepare_where_condition(
    src: torch.Tensor, input_format: DataFormat
) -> torch.Tensor:
    """
    Prepare a condition tensor with a mix of zero and non-zero values.

    The where kernel tests `condition == 0` per lane. We want a healthy mix of
    zero / non-zero so every variant exercises both code paths.
    """
    torch_format = format_dict[input_format]
    src_float = src.to(torch.float32)
    # Normalise to [0, 1) and threshold at 0.5 — ~50% zeros, ~50% ones.
    s_min = src_float.min()
    s_max = src_float.max()
    if s_max > s_min:
        normalized = (src_float - s_min) / (s_max - s_min)
    else:
        normalized = torch.zeros_like(src_float)
    cond = torch.where(
        normalized < 0.5,
        torch.zeros_like(src_float),
        torch.ones_like(src_float),
    )
    return cond.to(torch_format)


def _prepare_where_value(
    src: torch.Tensor, input_format: DataFormat, scale: float
) -> torch.Tensor:
    """
    Prepare a true_val/false_val tensor with moderate magnitudes.

    Keep values well within the format range so the select (bit-preserving) is
    an identity — format overflow would make comparisons non-informative.
    """
    torch_format = format_dict[input_format]
    src_float = src.to(torch.float32)
    s_min = src_float.min()
    s_max = src_float.max()
    if s_max > s_min:
        normalized = (src_float - s_min) / (s_max - s_min)
    else:
        normalized = torch.zeros_like(src_float)
    # Values in [-scale, scale]
    vals = (normalized * 2.0 - 1.0) * scale
    return vals.to(torch_format)


def _build_condition_for_test_case(
    base: torch.Tensor, input_format: DataFormat, test_case: str
) -> torch.Tensor:
    """Apply the condition regime (mixed / all_ones / all_zeros) for a variant."""
    torch_format = format_dict[input_format]
    if test_case == "all_ones":
        return torch.ones_like(base, dtype=torch_format)
    if test_case == "all_zeros":
        return torch.zeros_like(base, dtype=torch_format)
    # "mixed" — normalised 50/50 zero / one condition derived from `base`.
    return _prepare_where_condition(base, input_format)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_test_case_input_dims=generate_sfpu_where_combinations(
        SFPU_WHERE_FORMATS
    ),
)
def test_sfpu_where_quasar(formats_dest_acc_implied_test_case_input_dims):
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
    (
        formats,
        dest_acc,
        implied_math_format,
        test_case,
        input_dimensions,
    ) = formats_dest_acc_implied_test_case_input_dims[0]

    torch.manual_seed(42)

    # Build 3 tile-shaped tensors (condition, true_val, false_val) and
    # concatenate them into buffer_A. generate_stimuli returns one 32x32
    # tensor per call; we call it three times with different seeds so the
    # three buffers look different.
    src_cond_raw, tile_cnt_single, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )
    torch.manual_seed(43)
    src_true_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )
    torch.manual_seed(44)
    src_false_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    condition = _build_condition_for_test_case(
        src_cond_raw, formats.input_format, test_case
    )
    true_val = _prepare_where_value(src_true_raw, formats.input_format, scale=10.0)
    false_val = _prepare_where_value(src_false_raw, formats.input_format, scale=10.0)

    # buffer_A = concat([condition, true_val, false_val]) — 3 tiles.
    src_A = torch.cat([condition, true_val, false_val])
    tile_cnt_A = tile_cnt_single * 3

    num_faces = 4

    # Golden: torch.where(condition != 0, true_val, false_val).
    generate_golden = get_golden_generator(WhereGolden)
    golden_tensor = generate_golden(condition, true_val, false_val)
    # Convert golden to output format for comparison.
    torch_format_out = format_dict[formats.output_format]
    golden_tensor = golden_tensor.to(torch_format_out)

    unpack_to_dest = formats.input_format.is_32_bit() == (
        dest_acc == DestAccumulation.Yes
    )

    # src_B is unused by the where kernel but StimuliConfig requires a non-None
    # buffer_B tensor. Supply a dummy tensor of matching shape.
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


# Subset of format combinations for the deterministic debug test — keep the
# matrix small since the pattern itself is the value here.
_MCW_COMBINATIONS = generate_sfpu_where_combinations(SFPU_WHERE_FORMATS)[:6]


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_test_case_input_dims=_MCW_COMBINATIONS,
)
def test_sfpu_where_mcw_quasar(formats_dest_acc_implied_test_case_input_dims):
    """
    Deterministic where test — alternating 0/1 condition pattern with
    known true/false scalars (2 and 11) for easy debugging.

    This is the Quasar port of the old run's `test_where_mcw_quasar`.
    Runs through the same C++ harness as `test_sfpu_where_quasar`, so if
    this fails but the stimulus-driven test passes, the problem is in
    stimulus generation rather than the kernel.
    """
    (
        formats,
        dest_acc,
        implied_math_format,
        _,
        input_dimensions,
    ) = formats_dest_acc_implied_test_case_input_dims[0]

    torch_format_in = format_dict[formats.input_format]
    height, width = input_dimensions

    # Alternating 0/1 condition (0, 1, 0, 1, ...).
    pattern = torch.arange(height * width) % 2
    condition = pattern.view(height, width).to(torch_format_in).flatten()

    # Deterministic constant-value tensors — large enough gap to see errors,
    # small enough to roundtrip through every float16 variant cleanly.
    true_val = (torch.ones(height, width, dtype=torch_format_in) * 2).flatten()
    false_val = (torch.ones(height, width, dtype=torch_format_in) * 11).flatten()

    src_A = torch.cat([condition, true_val, false_val])
    tile_cnt_single = 1
    tile_cnt_A = tile_cnt_single * 3

    num_faces = 4

    generate_golden = get_golden_generator(WhereGolden)
    golden_tensor = generate_golden(condition, true_val, false_val)
    torch_format_out = format_dict[formats.output_format]
    golden_tensor = golden_tensor.to(torch_format_out)

    unpack_to_dest = formats.input_format.is_32_bit() == (
        dest_acc == DestAccumulation.Yes
    )

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
