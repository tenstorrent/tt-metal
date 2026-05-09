# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import (
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
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    SFPU_TILE_INDICES,
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

    # The div C++ test source has no datacopy stage — operands are streamed
    # straight into DEST via unpack-to-dest. Quasar unpack-to-dest only works
    # when the dest cell width matches the input width:
    #   16-bit input  ↔ dest_acc=No  (16-bit dest cells)
    #   32-bit input  ↔ dest_acc=Yes (32-bit dest cells)
    # Mismatched widths hang because no valid data ever lands in DEST and the
    # math thread blocks forever on dvalid.
    if in_fmt.is_32_bit() != (dest_acc == DestAccumulation.Yes):
        return True

    return False


# Tile-index permutations exercised per format. Picked to cover:
#   - same vs distinct dst (in-place vs out-of-place writeback),
#   - src0/src1 ordering (verifies operand-swap path is symmetric),
#   - non-zero src/dst indices (catches DEST-tile addressing bugs).
_TILE_INDEX_VARIANTS = [
    (0, 1, 0),
    (1, 0, 0),
    (0, 1, 1),
    (0, 2, 1),
    (2, 3, 0),
]


def generate_sfpu_binary_div_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU binary-div test combinations.

    Mirrors the structure used by `test_sfpu_where_quasar` so format /
    dest_acc / implied_math_format coverage is enumerated identically. The
    div-specific axis is the (src0_idx, src1_idx, dst_idx) tile-index
    permutation rather than a condition regime.
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

                for src0_idx, src1_idx, dst_idx in _TILE_INDEX_VARIANTS:
                    combinations.append(
                        (
                            fmt,
                            dest_acc,
                            implied_math_format,
                            src0_idx,
                            src1_idx,
                            dst_idx,
                        )
                    )

    return combinations


# Start with the canonical float matrix that every Quasar SFPU test uses.
# Integer / MX coverage can be widened once the baseline passes.
SFPU_BINARY_DIV_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Float32,
    ]
)


def _prepare_div_inputs(src_A: torch.Tensor, data_format: DataFormat) -> torch.Tensor:
    """
    Map the [0, 1) uniform stimuli into a numerically friendly range for the
    SFPU divide kernel.

    The kernel computes `in0 / in1` as `in0 * reciprocal(in1)` with a
    Newton-Raphson refinement (BH-port). To avoid:

      * spurious 0/0 -> NaN, x/0 -> ±inf in non-special-case tiles, and
      * catastrophic loss of precision when the divisor is sub-normal,

    every element is mapped to a value in [-4.0, -0.25] union [0.25, 4.0].
    Both halves of the range are exercised so the sign-handling in the
    helper (`sfpi::setsgn`) is covered for negative dividends.

    Each element of `src_A` is used both as a potential dividend (when its
    tile is selected as `src0_idx`) and as a divisor (when selected as
    `src1_idx`), so the same range constraint applies to the whole tensor.
    """
    torch_format = format_dict[data_format]

    # Uniform [0, 1) -> [-4, 4]
    scaled = (src_A.to(torch.float32) - 0.5) * 8.0

    # Push values out of (-0.25, 0.25): if abs(x) < 0.25, snap to ±0.25 keeping sign.
    sign = torch.where(scaled >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    abs_scaled = torch.maximum(scaled.abs(), torch.tensor(0.25))
    scaled = sign * abs_scaled

    return scaled.to(torch_format)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_tile_indices=generate_sfpu_binary_div_combinations(
        SFPU_BINARY_DIV_FORMATS
    ),
)
def test_sfpu_binary_div_quasar(formats_dest_acc_implied_tile_indices):
    """
    Test binary SFPU DIV on Quasar architecture.

    Loads tiles directly into DEST via unpack-to-dest, then runs the BH-style
    sfpi-vFloat divide helper (`_calculate_sfpu_binary_div_`) once per face
    via the binary SFPU harness, and verifies the result against a golden
    reference computed in fp32 by `BinarySFPUGolden._div`.

    Stimuli are mapped to ±[0.25, 4.0] (see `_prepare_div_inputs`) so the
    divisor tile never contains 0; the special-case branches (0/0 -> NaN,
    x/0 -> ±inf) are not exercised by this test, only the main reciprocal
    path is.
    """
    (
        formats,
        dest_acc,
        implied_math_format,
        src0_idx,
        src1_idx,
        dst_idx,
    ) = formats_dest_acc_implied_tile_indices[0]

    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    input_dimensions = [num_tiles_needed * 32, 32]

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    src_A = _prepare_div_inputs(src_A, formats.input_format)

    num_faces = 4
    mathop = MathOperation.SfpuElwdiv

    elements_per_tile = 1024  # 4 faces * 16 rows * 16 cols
    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop,
        src_A,
        src0_idx,
        src1_idx,
        dst_idx,
        32,  # num_iterations: 32 rows = 1 full tile
        input_dimensions,
        formats.input_format,
    ).flatten()
    dst_start = dst_idx * elements_per_tile
    golden_tensor = golden_full[dst_start : dst_start + elements_per_tile]

    # Convert golden to output format for comparison.
    torch_format_out = format_dict[formats.output_format]
    golden_tensor = golden_tensor.to(torch_format_out)

    tile_count_res = 1  # we only pack the single output tile

    # The C++ source streams operands straight into DEST — there's no datacopy
    # stage — so the unpack engine must always be UnpDest. Width-mismatched
    # configurations are pre-filtered by `_is_invalid_quasar_combination`.
    configuration = TestConfig(
        "sources/quasar/sfpu_binary_div_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_count_res,
            num_faces=num_faces,
        ),
        unpack_to_dest=True,
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
