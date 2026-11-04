# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every finite BF16 value through a generated BF16 SFPU kernel, with FP32 DEST off.

Each test_eltwise_unary_sfpu_bf16_<op>.py lists its op's rows and runs them here. Subnormal inputs
and NaN lanes are outside a step count (see ulp_sweep.py); finite/non-finite disagreements on normal
inputs are failures in their own right. The step metric ranks -0 with +0, so the specials are judged
apart, by output class and sign. max_ulp is 0 for an exact fit.
"""

import struct
from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    format_dict,
)
from helpers.sfpu_domains import (
    SPECIALS_READY_OPS,
    negative_zero_delivered,
    specials_after_nan_sign_gate,
    specials_safe,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import RuntimeParameter, TemplateParameter
from helpers.utils import passed_test
from test_eltwise_unary_sfpu import build_eltwise_unary_sfpu_config

_NEGATIVE_NAN = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]

# Consecutive rows of a file's list that share one ELF of sources/eltwise_unary_sfpu_bf16_sweep_test.cpp.
# Building it is most of a row's time; the first row to need it builds it and the others reuse it.
_SWEEP_GROUP = 4


@dataclass
class _SWEEP_OPERATIONS(TemplateParameter):
    """The (operation, approximation mode) pairs one sweep ELF compiles into its math thread."""

    rows: tuple = ()

    def convert_to_cpp(self) -> str:
        operations = ", ".join(
            f"SfpuType::{mathop.cpp_enum_value}" for mathop, _ in self.rows
        )
        modes = ", ".join(mode.cpp_enum_value for _, mode in self.rows)
        return (
            f"constexpr SfpuType SWEEP_OPERATIONS[] = {{{operations}}};\n"
            f"constexpr bool SWEEP_APPROXIMATION[] = {{{modes}}};"
        )


@dataclass
class _SWEEP_INDEX(RuntimeParameter):
    """Which of its ELF's operations a run executes."""

    index: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SWEEP_INDEX = {self.index};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t SWEEP_INDEX;", "I"


def _special_class(value):
    sign = "neg" if struct.unpack("<I", struct.pack("<f", value))[0] >> 31 else "pos"
    kind = (
        "nan" if value != value else ("inf" if abs(value) == float("inf") else "zero")
    )
    return f"{sign}_{kind}"


def replaced_instance_params(unary, replaced):
    """The stock sweep's parameters, from ``unary`` (test_eltwise_unary_sfpu), for every instance
    ``replaced`` maps an op to, on BF16 and FP32 data with FP32 DEST off."""
    return [
        params
        for params in unary._sweep_params(
            [
                InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
                InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
            ],
            list(replaced),
            [ApproximationMode.No, ApproximationMode.Yes],
            unary.STANDARD_DIMENSIONS,
        )
        if params[4] == DestAccumulation.No
        and (params[1], params[3]) in replaced[params[2]]
    ]


def bf16_exhaustive_sweep(
    mathop, approx_mode, max_ulp, rows, stock_boards, stock_specials
):
    """Row (mathop, approx_mode, max_ulp) of ``rows``; ``stock_boards`` and ``stock_specials`` map an op to
    the boards that keep its stock kernel and to the stock words it stores at special inputs per board.
    """
    if TestConfig.CHIP_ARCH in stock_boards.get(mathop, ()):
        pytest.skip(f"{mathop.name} keeps the stock kernel on {TestConfig.CHIP_ARCH}")
    from helpers.sfpu_accuracy_budget import accuracy_contract
    from helpers.ulp import nonfinite_mismatches, ulp_distance
    from helpers.ulp_sweep import _normal_input, measurable_mask, sweep_spec
    from helpers.ulp_table import Metric

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    dest_acc = DestAccumulation.No
    dimensions = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * 64]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dimensions,
        spec_A=sweep_spec(formats.input_format),
    )
    # The sweep pads with +0. The specials go in that padding, under the edge sweep's
    # gates for what the golden defines and the pipeline delivers.
    specials = [0.0]
    if negative_zero_delivered(formats.input_format, dest_acc):
        specials.append(-0.0)
    else:
        # The kernel reads an undelivered -0 as +0, so the golden must too: rsqrt(-0)
        # is -inf, which the +0 the pipeline delivers can never return.
        src_A[(src_A == 0) & torch.signbit(src_A.float())] = 0.0
    nonfinite = mathop in SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )
    nan_sign = specials_after_nan_sign_gate(
        mathop,
        formats.input_format,
        formats.output_format,
        dest_acc,
        nonfinite,
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE,
    )
    if nonfinite:
        specials += [float("inf"), float("-inf"), float("nan"), _NEGATIVE_NAN]
    stock = stock_specials.get(mathop, {}).get(TestConfig.CHIP_ARCH, {})
    # Through the bit pattern: a float-to-bfloat16 cast drops a NaN's sign.
    bits = torch.tensor(specials, dtype=torch.float32).view(torch.int32) >> 16
    tail = slice(len(src_A) - len(specials), None)
    src_A[tail] = bits.to(torch.int16).view(torch.bfloat16).to(src_A.dtype)
    golden = get_golden_generator(UnarySFPUGolden)(
        mathop, src_A, formats.output_format, dest_acc, formats.input_format, dimensions
    )
    row = rows.index((mathop, approx_mode, max_ulp))
    first = row - row % _SWEEP_GROUP
    group = tuple((op, mode) for op, mode, _ in rows[first : first + _SWEEP_GROUP])
    # The driver is the stock eltwise_unary_sfpu_test.cpp with a sweep's math thread, so the stock
    # test's own builder gives every parameter that driver reads; the sweep adds its rows. A config
    # lays out its runtime struct when built, so the sweep's is a new one from the stock's parts.
    stock_config = build_eltwise_unary_sfpu_config(
        formats,
        dest_acc,
        approx_mode,
        mathop,
        dimensions,
        src_A,
        src_B,
        tile_cnt_A,
        tile_cnt_B,
        unpack_to_dest=False,
    )
    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_bf16_sweep_test.cpp",
        formats,
        templates=stock_config.templates + [_SWEEP_OPERATIONS(group)],
        runtimes=stock_config.runtimes + [_SWEEP_INDEX(row - first)],
        variant_stimuli=stock_config.variant_stimuli,
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )
    result = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    # Every lane whose input the unpack does not flush must agree with the golden about
    # being NaN or infinite. Stricter than ulp_sweep.nonfinite_failures, which also excuses
    # a NaN golden, a store saturating past the output's range and inputs outside the op's
    # claim: a kernel replacing stock's on every BF16 input is held to the golden there too.
    # NaN inputs are left to the output-class check below, which is stricter. A special input
    # where the kernel returns its stock word is held to that word in both checks.
    expected = golden[tail].clone()
    held = torch.zeros(len(specials), dtype=torch.bool)
    for lane, value in enumerate(specials):
        if stock.get(_special_class(value)) is not None:
            held[lane] = True
            word = torch.tensor([stock[_special_class(value)]], dtype=torch.int32).to(
                torch.int16
            )
            expected[lane] = word.view(torch.bfloat16).to(expected.dtype)[0]
    judged = golden.clone()
    judged[tail] = expected
    # Where Wormhole leaves the sign of a NaN the op makes open (nan_sign off), the
    # pack stores that NaN as an infinity of either sign and the golden as +inf: that
    # lane's sign is spared in both checks. Those lanes are the specials whose golden
    # is NaN, from a non-NaN input, in a pipeline that keeps NaN, and not held to stock's word.
    made_nan = torch.zeros(len(specials), dtype=torch.bool)
    if nonfinite and not nan_sign:
        probe = torch.zeros(TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1], dtype=src_A.dtype)
        probe[: len(specials)] = src_A[tail]
        kept = get_golden_generator(UnarySFPUGolden)(
            mathop,
            probe,
            DataFormat.Float32,
            DestAccumulation.Yes,
            formats.input_format,
            list(TILE_DIMENSIONS),
        )
        made_nan = kept[: len(specials)].isnan() & ~src_A[tail].float().isnan()
    spared = made_nan & result[tail].float().isinf() & ~held
    failures = nonfinite_mismatches(judged, result) & _normal_input(
        src_A, formats.input_format, formats.output_format, dest_acc
    )
    failures[tail] &= ~spared
    assert (
        not failures.any()
    ), f"{mathop.name}: {int(failures.sum())} lanes disagree on finiteness"

    def output_class(values):
        values = values.float()
        return torch.stack(
            [
                values.isnan(),
                values.isinf(),
                values == 0,
                values.signbit() & ~values.isnan(),
            ]
        )

    wrong = output_class(expected) != output_class(result[tail])
    wrong[3] &= ~spared
    wrong = wrong.any(0)
    assert not wrong.any(), (
        f"{mathop.name}: {int(wrong.sum())} of {len(specials)} special inputs "
        f"{specials} change output class or sign"
    )
    # The specials sit in the padding measurable_mask leaves out: a finite result there
    # is held to the same budget here.
    assert passed_test(
        expected,
        result[tail],
        formats.output_format,
        max_ulp=max_ulp,
        mask=torch.isfinite(expected.float()),
    ), f"{mathop.name}: special-input outputs exceed the {max_ulp}-ULP budget"
    mask = measurable_mask(
        src_A, golden, result, formats.input_format, formats.output_format, dest_acc
    )
    over = int(((ulp_distance(golden.to(result.dtype), result) > max_ulp) & mask).sum())
    assert passed_test(
        golden, result, formats.output_format, max_ulp=max_ulp, mask=mask
    ), f"{mathop.name}: {over} lanes over the {max_ulp}-ULP budget"
    # The step budget helpers/sfpu_accuracy_budget.yaml declares for this cell, where it
    # declares one for this board, judged as test_unary_sfpu_ulp.py judges it.
    contract = accuracy_contract(
        mathop,
        output_format=formats.output_format,
        input_format=formats.input_format,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=TestConfig.CHIP_ARCH,
    )
    if contract.metric == Metric.ULP:
        assert passed_test(
            golden,
            result,
            formats.output_format,
            mask=mask,
            flush_subnormals=True,
            **contract.passed_test_kwargs(),
        ), f"{mathop.name}: over the {contract.max_ulp}-step budget of helpers/sfpu_accuracy_budget.yaml"
