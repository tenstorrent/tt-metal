# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""IEEE-754 semantics of the tt-llk ordered scalar compares (tt-llk#1701 item 3).

Drives ``_calculate_comp_unary_<APPROX, unary_gt|lt|ge|le>`` from
``tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h``. Its vFloat compares lower to SFPGT/SFPLE,
which order floats by sign and magnitude (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN). IEEE-754
instead has -0 == +0, and every ordered compare with a NaN is false. Before the fix the kernel returned,
for example, gt(+NaN, 0) = 1, lt(-0, +0) = 1, ge(-0, +0) = 0, and gt(x, -NaN) = 1 for every x.

The golden is torch's IEEE compare and the check is exact: every lane is 0.0 or 1.0.

Float32 -> Float32 at dest_acc=Yes only. It is the one tt-llk pipeline that unpacks straight to Dest, so
it is the only one that hands the SFPU -0.0 and NaN unchanged (see ``negative_zero_delivered`` in
``helpers/sfpu_domains.py``); the 16-bit paths turn them into +0.0 and +-inf before the kernel runs.

Blackhole only: the Wormhole copy of this header is not changed by the fix.
"""

import struct

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
    generate_input_dim,
)

pytestmark = pytest.mark.skipif(
    get_chip_architecture() != ChipArchitecture.BLACKHOLE,
    reason="tt-llk#1701 item 3: only the Blackhole ckernel_sfpu_comp.h compares are IEEE-754 so far",
)

ELEMENTS_PER_TILE = 1024

# Inputs, as raw fp32 bits so the NaN signs and -0.0 are exact.
INPUT_BITS = (
    0x00000000,  # +0
    0x80000000,  # -0
    0x7FC00000,  # +NaN
    0xFFC00000,  # -NaN
    0x7FFFFFFF,  # +NaN, largest payload: the top of the sign-magnitude order
    0xFFFFFFFF,  # -NaN, largest payload: the bottom of it
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x3F800000,  # 1.0
    0xBF800000,  # -1.0
    0x3F000000,  # 0.5
    0xBF000000,  # -0.5
    0x3F000001,  # 0.5 + 1 ulp
    0x3EFFFFFF,  # 0.5 - 1 ulp
    0xC0200000,  # -2.5
    0x00000001,  # smallest +denormal
    0x80000001,  # smallest -denormal
    0x7F7FFFFF,  # +max
    0xFF7FFFFF,  # -max
)

# Thresholds: both zeros, both infinities, both NaN signs, and finite values on either side.
THRESHOLD_BITS = [
    0x00000000,  # +0
    0x80000000,  # -0
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7FC00000,  # +NaN
    0xFFC00000,  # -NaN
    0x3F000000,  # 0.5
    0xC0200000,  # -2.5
]

_TORCH_COMPARE = {
    MathOperation.UnaryGt: torch.gt,
    MathOperation.UnaryLt: torch.lt,
    MathOperation.UnaryGe: torch.ge,
    MathOperation.UnaryLe: torch.le,
}


def _floats(bits) -> torch.Tensor:
    return (
        torch.tensor(list(bits), dtype=torch.int64).to(torch.int32).view(torch.float32)
    )


def _fmt(bits: int) -> str:
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    sign = "-" if bits >> 31 else "+"
    if value != value:
        return f"{sign}nan(0x{bits:08x})"
    if value == 0.0:
        return f"{sign}0.0"
    return f"{value:g}"


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    mathop=list(_TORCH_COMPARE),
    threshold=THRESHOLD_BITS,
)
def test_sfpu_comp_unary_ieee(formats, mathop, threshold):
    reps = ELEMENTS_PER_TILE // len(INPUT_BITS) + 1
    tile_bits = (list(INPUT_BITS) * reps)[:ELEMENTS_PER_TILE]
    src_A = _floats(tile_bits)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_comp_unary_ieee_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            MATH_OP(mathop=mathop),
            SFPU_UNARY_SCALAR(threshold),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        compile_time_formats=True,
    )

    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE],
        dtype=format_dict[formats.output_format],
    ).to(torch.float32)

    golden = _TORCH_COMPARE[mathop](src_A, _floats([threshold])[0]).to(torch.float32)

    wrong = sorted(
        {
            (_fmt(tile_bits[i]), res[i].item(), golden[i].item())
            for i in range(ELEMENTS_PER_TILE)
            if res[i].item() != golden[i].item()
        }
    )
    assert not wrong, (
        f"{mathop.name}(x, {_fmt(threshold)}) disagrees with IEEE-754 on {len(wrong)} input classes:\n"
        + "\n".join(f"  x = {x}: got {got}, expected {want}" for x, got, want in wrong)
    )
