# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""calculate_unary_power_iterative (``power_iterative_tile``) on Wormhole, bit-exact.

The kernel walks the exponent's bits once per pair of Dest rows. With a runtime exponent (moreh's
``power_tile_to_cb`` passes one) the row loop stays rolled and the walk is a runtime branch sequence; with a
compile-time exponent (ttnn's ``power_iterative_tile(i, Nu)``) the walk folds and the rows are unrolled.
``test_sfpu_power_iterative_runtime_exponent`` reads the exponent from the runtime parameter block, so every
exponent shares one ELF and reaches the runtime branch; ``test_sfpu_power_iterative_constant_exponent`` bakes
it in.

Every Dest row gets distinct random bases, so a row-pair addressing slip (a row skipped, written twice or
taking its neighbour's value) changes the output. Bases are +-[0.5, 2), so x**31 stays within [2**-31, 2**31]:
no overflow and no denormals, and the SFPU's fp32 multiplies match the host's IEEE ones. The golden repeats
the kernel's multiply order (binary exponentiation of x**(e - 1) times x); a 16-bit Dest store truncates the
fp32 result to bf16.
"""

from dataclasses import dataclass

import torch
from conftest import wormhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    RuntimeParameter,
    TemplateParameter,
    generate_input_dim,
)

# The Wormhole kernel only: the Blackhole twin keeps the single-row walk, and the golden's truncating 16-bit
# store is measured on Wormhole silicon only.
pytestmark = wormhole_only

ELEMENTS_PER_TILE = TILE_DIM * TILE_DIM

# 0 is the x**0 early return, 1 skips the walk, the rest exercise single- and multi-bit walks (8 is a lone
# high bit, 31 is all ones).
EXPONENTS = [0, 1, 2, 3, 5, 8, 17, 31]

# Float16_b runs a 16-bit Dest fed through SrcA; Float32 runs a 32-bit Dest (dest_acc) unpacked to Dest.
DATA_FORMATS = [DataFormat.Float16_b, DataFormat.Float32]

# Keeps the bf16 half of an fp32 bit pattern (0xFFFF0000 as an int32).
BF16_HIGH_HALF_MASK = -0x10000


@dataclass
class POWER_EXPONENT(RuntimeParameter):
    """The exponent as read from the runtime parameter block: opaque to the compiler."""

    power_exponent: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t POWER_EXPONENT = {self.power_exponent}u;"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t POWER_EXPONENT;", "I"


@dataclass
class POWER_CONST_EXPONENT(TemplateParameter):
    """A compile-time exponent; None selects the runtime POWER_EXPONENT instead."""

    power_const_exponent: int | None = None

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr bool POWER_EXPONENT_IS_CONSTANT = {str(self.power_const_exponent is not None).lower()};\n"
            f"constexpr std::uint32_t POWER_CONST_EXPONENT = {self.power_const_exponent or 0}u;"
        )


def _bases(torch_format: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(62801)
    magnitude = 0.5 + 1.5 * torch.rand(ELEMENTS_PER_TILE, generator=generator)
    sign = torch.where(
        torch.rand(ELEMENTS_PER_TILE, generator=generator) < 0.5, -1.0, 1.0
    )
    return (sign * magnitude).to(torch_format).to(torch.float32)


def _golden(bases: torch.Tensor, exponent: int) -> torch.Tensor:
    if exponent == 0:
        return torch.ones_like(bases)
    power = bases.clone()
    result = bases.clone()
    exp = exponent - 1
    while exp:
        if exp & 1:
            result = result * power
        exp >>= 1
        if exp:
            power = power * power
    return result


def _run(data_format, exponent, const_exponent):
    dest_acc = DestAccumulation.Yes if data_format.is_32_bit() else DestAccumulation.No
    formats = InputOutputFormat(data_format, data_format)
    torch_format = format_dict[data_format]
    src_A = _bases(torch_format)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_power_iterative_test.cpp",
        formats,
        templates=[
            generate_input_dim([TILE_DIM, TILE_DIM], [TILE_DIM, TILE_DIM]),
            APPROX_MODE(ApproximationMode.No),
            POWER_CONST_EXPONENT(const_exponent),
            DEST_SYNC(),
        ],
        runtimes=[POWER_EXPONENT(exponent)],
        variant_stimuli=StimuliConfig(
            src_A.to(torch_format),
            formats.input_format,
            src_B.to(torch_format),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=data_format.is_32_bit(),
        compile_time_formats=True,
    )

    res = (
        torch.tensor(configuration.run().result[:ELEMENTS_PER_TILE], dtype=torch_format)
        .to(torch.float32)
        .view(torch.int32)
    )

    golden = _golden(src_A, exponent).view(torch.int32)
    if not data_format.is_32_bit():
        golden = golden & BF16_HIGH_HALF_MASK

    wrong = (res != golden).nonzero().flatten().tolist()
    assert not wrong, (
        f"x**{exponent} ({data_format.name}, dest_acc={dest_acc.name}) differs from the golden on "
        f"{len(wrong)} of {ELEMENTS_PER_TILE} lanes, first:\n"
        + "\n".join(
            f"  lane {i}: x = {src_A[i].item():.9g}: got 0x{res[i].item() & 0xFFFFFFFF:08X}, "
            f"expected 0x{golden[i].item() & 0xFFFFFFFF:08X}"
            for i in wrong[:16]
        )
    )


@parametrize(data_format=DATA_FORMATS, exponent=runtime(EXPONENTS))
def test_sfpu_power_iterative_runtime_exponent(data_format, exponent):
    _run(data_format, exponent, const_exponent=None)


@parametrize(data_format=DATA_FORMATS, exponent=EXPONENTS)
def test_sfpu_power_iterative_constant_exponent(data_format, exponent):
    _run(data_format, exponent, const_exponent=exponent)
