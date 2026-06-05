# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
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

# DEST-tile offsets (dividend, divisor, result) for the divide helper, chosen
# to cover writing the result back over the dividend tile, over the divisor
# tile, and to a separate tile with both operand offsets non-zero.
_TILE_INDEX_VARIANTS = [
    (0, 1, 0),
    (0, 1, 1),
    (2, 3, 0),
]

# Crafted lanes within face 0 of the dividend and divisor tiles that exercise
# the special-case branches in `_calculate_sfpu_binary_div_`. Entries are
# (lane_in_tile, dividend_value, divisor_value, expected_result_kind).
_SPECIAL_CASE_LANES = [
    (0, 0.0, 0.0, "nan"),
    (1, 1.5, 0.0, "pos_inf"),
    (2, -1.5, 0.0, "neg_inf"),
    (3, 2.7, 2.7, "one"),
    (4, -3.3, -3.3, "one"),
]
_ELEMENTS_PER_TILE = 1024


def _get_valid_formats_dest_acc():
    """Float16 + DestAccumulation.Yes is not supported."""
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
    """MX formats require implied math format enabled."""
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _prepare_inputs(
    src_A: torch.Tensor,
    data_format: DataFormat,
    src0_idx: int,
    src1_idx: int,
    mathop: MathOperation,
) -> torch.Tensor:
    """
    Map the [0, 1) uniform stimuli into a numerically friendly range whose
    bounds depend on the operation being tested.

    For DIV, the bulk is mapped to ±[0.25, 4.0] so the reciprocal +
    Newton-Raphson path is not contaminated by accidental zeros or subnormals.
    A handful of lanes in face 0 of the operand tiles are then overwritten
    with crafted values (see `_SPECIAL_CASE_LANES`) to exercise the
    special-case branches (0/0 -> NaN, x/0 -> ±inf, x/x -> 1.0).

    For MUL, the bulk is mapped to ±[-250, 250] to exercise a wide dynamic
    range without special-case lane overrides.
    """
    torch_format = format_dict[data_format]

    if mathop == MathOperation.SfpuElwdiv:
        scaled = (src_A.to(torch.float32) - 0.5) * 8.0

        sign = torch.where(scaled >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        abs_scaled = torch.maximum(scaled.abs(), torch.tensor(0.25))
        scaled = sign * abs_scaled

        scaled = scaled.to(torch_format)

        flat = scaled.flatten()
        for lane, dividend, divisor, _ in _SPECIAL_CASE_LANES:
            flat[src0_idx * _ELEMENTS_PER_TILE + lane] = dividend
            flat[src1_idx * _ELEMENTS_PER_TILE + lane] = divisor
    elif mathop == MathOperation.SfpuElwmul:
        scaled = (src_A.to(torch.float32) - 0.5) * 500.0
        scaled = scaled.to(torch_format)
        flat = scaled.flatten()

    return flat.reshape(scaled.shape)


def _run_sfpu_binary_test(
    formats_dest_acc,
    implied_math_format,
    tile_indices,
    mathop: MathOperation,
):
    """
    Shared test body for binary SFPU operations on Quasar.

    Loads tiles directly into DEST via unpack-to-dest, then runs the
    binary SFPU helper (`_calculate_sfpu_binary_`) once per face via the
    binary SFPU harness, and verifies the result against a golden reference
    computed in fp32 by `BinarySFPUGolden`.

    For DIV, a handful of lanes in face 0 are overwritten with crafted
    values (see `_SPECIAL_CASE_LANES`) so that 0/0 -> NaN, x/0 -> ±inf,
    and x/x -> 1.0 branches are all hit. The forced-exact x/x branch is
    additionally checked bit-exact against 1.0 after the main
    tolerance-based comparison.
    """
    formats, dest_acc = formats_dest_acc
    src0_idx, src1_idx, dst_idx = tile_indices

    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    input_dimensions = [num_tiles_needed * 32, 32]

    torch.manual_seed(42)

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    src_A = _prepare_inputs(src_A, formats.input_format, src0_idx, src1_idx, mathop)

    num_faces = 4

    # Convert golden to output format for comparison.
    torch_format_out = format_dict[formats.output_format]

    # Defer golden generation to a closure so run() can compute it while the
    # tensixes execute, overlapping the host work with the device wait.
    generate_golden = get_golden_generator(BinarySFPUGolden)

    def _golden():
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
        dst_start = dst_idx * _ELEMENTS_PER_TILE
        golden_tensor = golden_full[dst_start : dst_start + _ELEMENTS_PER_TILE]
        return golden_tensor.to(torch_format_out)

    tile_count_res = 1  # we only pack the single output tile

    # SFPU reads from DEST directly, so we use UnpDest to load operands there —
    # no need to route through SRC registers and the FPU.
    configuration = TestConfig(
        "sources/quasar/sfpu_binary_float_quasar_test.cpp",
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

    outcome = configuration.run(golden_fn=_golden)
    res_from_L1 = outcome.result
    golden_tensor = outcome.golden

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format_out)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"

    # The kernel's x/x branch forces an exact 1.0 regardless of reciprocal
    # rounding, so check bit-exact rather than relying on isclose tolerance.
    if mathop == MathOperation.SfpuElwdiv:
        for lane, _, _, kind in _SPECIAL_CASE_LANES:
            if kind != "one":
                continue
            actual = res_tensor[lane].item()
            assert (
                actual == 1.0
            ), f"x/x special case at lane {lane}: expected exact 1.0, got {actual}"


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    tile_indices=_TILE_INDEX_VARIANTS,
)
def test_sfpu_binary_div_quasar(formats_dest_acc, implied_math_format, tile_indices):
    _run_sfpu_binary_test(
        formats_dest_acc, implied_math_format, tile_indices, MathOperation.SfpuElwdiv
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_valid_implied_math_formats(
        formats_dest_acc[0]
    ),
    tile_indices=_TILE_INDEX_VARIANTS,
)
def test_sfpu_binary_mul_quasar(formats_dest_acc, implied_math_format, tile_indices):
    _run_sfpu_binary_test(
        formats_dest_acc, implied_math_format, tile_indices, MathOperation.SfpuElwmul
    )
