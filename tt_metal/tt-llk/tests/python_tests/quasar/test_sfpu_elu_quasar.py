# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
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


def prepare_elu_inputs(
    src_A: torch.Tensor,
    input_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for ELU operation with safe value ranges.

    ELU(x) = x                  if x >= 0
    ELU(x) = alpha*(exp(x) - 1) if x <  0  (saturates near -alpha for very negative x)

    The negative tail goes through exp(x), so we must avoid catastrophic
    underflow / loss of precision. A symmetric range of [-10, 10] mirrors what
    the existing exp/sigmoid/silu Quasar tests use and gives good coverage of
    both the linear (x >= 0) and saturating (x << 0) regimes.

    Args:
        src_A: Source tensor A (uniform in [0, 1) from the stimuli generator)
        input_format: Input data format

    Returns:
        Prepared tensor with safe values for ELU
    """
    torch_format = format_dict[input_format]
    min_val = -10.0
    max_val = 10.0
    src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
    return src_A.to(torch_format)


def _is_invalid_quasar_combination(
    fmt: FormatConfig, dest_acc: DestAccumulation
) -> bool:
    """
    Check if format combination is invalid for Quasar.

    Args:
        fmt: Format configuration with input and output formats
        dest_acc: Destination accumulation mode

    Returns:
        True if the combination is invalid, False otherwise
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

    return False


def generate_sfpu_elu_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU ELU test combinations.

    Args: Input-output format pairs

    Returns: List of (format, dest_acc, dest_sync, implied_math_format, input_dimensions) tuples
    """
    combinations = []

    dest_sync_modes = (DestSync.Half, DestSync.Full)
    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        for dest_acc in dest_acc_modes:
            # Skip invalid format combinations for Quasar
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for dest_sync in dest_sync_modes:
                for implied_math_format in [
                    ImpliedMathFormat.No,
                    ImpliedMathFormat.Yes,
                ]:
                    for input_dimensions in [[32, 32], [64, 64], [32, 64]]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                dest_sync,
                                implied_math_format,
                                input_dimensions,
                            )
                        )

    return combinations


SFPU_ELU_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_implied_math_input_dims=generate_sfpu_elu_combinations(
        SFPU_ELU_FORMATS
    ),
)
def test_sfpu_elu_quasar(formats_dest_acc_sync_implied_math_input_dims):
    """
    Test ELU operation on Quasar architecture.

    Uses PyTorch's torch.nn.functional.elu(x, alpha=1.0) as the golden
    reference. The kernel hard-codes the ELU slope to 1.0 to match the
    golden generator's fixed alpha.
    """
    (formats, dest_acc, dest_sync, implied_math_format, input_dimensions) = (
        formats_dest_acc_sync_implied_math_input_dims[0]
    )

    # Set seed for reproducibility
    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    # Prepare inputs with a symmetric range that exercises both ELU branches
    src_A = prepare_elu_inputs(src_A, formats.input_format)

    num_faces = 4

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Elu,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        "sources/quasar/sfpu_elu_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Elu),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
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
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
