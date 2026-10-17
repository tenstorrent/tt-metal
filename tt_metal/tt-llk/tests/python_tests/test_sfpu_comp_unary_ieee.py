# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""IEEE-754 semantics of the tt-llk ordered scalar compares (tt-llk#1701 item 3).

Drives ``_calculate_comp_unary_<APPROX, unary_gt|lt|ge|le>`` from
``tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h``. Its vFloat compares lower to SFPGT/SFPLE,
which order floats by sign and magnitude (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN). IEEE-754
instead has -0 == +0, and every ordered compare with a NaN is false. Before the fix the kernel returned,
for example, gt(+NaN, 0) = 1, lt(-0, +0) = 1, ge(-0, +0) = 0, and gt(x, -NaN) = 1 for every x.

The golden is torch's IEEE compare and the check is exact: every lane is 0.0 or 1.0.

Two entry points (``entry``):
- ``vfloat``: the vFloat overload the fix adds, with the threshold decoded from its fp32 bits, so -0.0, +-inf
  and +-NaN thresholds are pinned independently of the uint32 overload.
- ``uint32``: the shipped ``_calculate_comp_unary_(std::uint32_t)`` entry point, threshold +0.0 only. Its
  numeric decode of the bits (tt-llk#1701 item 16) agrees with the float value for 0x00000000 alone, so these
  variants compile and run against the unfixed header too: there they fail with gt(+NaN, 0) = 1,
  ge(+NaN, 0) = 1, ge(-0.0, +0.0) = 0, lt(-NaN, 0) = 1, lt(-0.0, +0.0) = 1 and le(-NaN, 0) = 1.

Two pipelines:
- Float32 -> Float32 at dest_acc=Yes unpacks straight to Dest, the only path that delivers both -0.0
  and NaN to the SFPU (see ``negative_zero_delivered`` in ``helpers/sfpu_domains.py``).
- Float16_b -> Float16_b at dest_acc=No, the common bf16 path. It carries NaN and +-inf but flattens
  -0.0 to +0.0, which does not change an IEEE compare. Its inputs are the bf16-exact subset.

Blackhole only: the Wormhole copy of this header is not changed by the fix.
"""

import struct
from typing import NamedTuple

import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    VectorMode,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.sfpu_domains import negative_zero_delivered
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    SFPU_COMP_SCALAR_ENTRY,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
    generate_input_dim,
)

# https://github.com/tenstorrent/tt-llk/issues/1701 item 3: only the Blackhole compares are fixed.
pytestmark = blackhole_only

ELEMENTS_PER_TILE = 1024


class Bits(NamedTuple):
    """An fp32 bit pattern with a readable name, so test ids read `threshold:-0.0`, not an integer."""

    name: str
    bits: int


class Pipeline(NamedTuple):
    name: str
    formats: InputOutputFormat
    dest_acc: DestAccumulation


# Inputs, as raw fp32 bits so the NaN signs and -0.0 are exact. bf16_exact marks the ones a Float16_b
# tile holds unchanged.
INPUTS = (
    # (bits, bf16_exact)
    (0x00000000, True),  # +0
    (0x80000000, True),  # -0
    (0x7FC00000, True),  # +NaN
    (0xFFC00000, True),  # -NaN
    (0x7FFFFFFF, False),  # +NaN, largest payload: the top of the sign-magnitude order
    (0xFFFFFFFF, False),  # -NaN, largest payload: the bottom of it
    (0x7F800000, True),  # +inf
    (0xFF800000, True),  # -inf
    (0x3F800000, True),  # 1.0
    (0xBF800000, True),  # -1.0
    (0x3F000000, True),  # 0.5
    (0xBF000000, True),  # -0.5
    (0x3F000001, False),  # 0.5 + 1 ulp
    (0x3EFFFFFF, False),  # 0.5 - 1 ulp
    (0xC0200000, True),  # -2.5
    (0x00000001, False),  # smallest +denormal
    (0x80000001, False),  # smallest -denormal
    (0x7F7FFFFF, False),  # +FLT_MAX
    (0xFF7FFFFF, False),  # -FLT_MAX
    (0x7F7F0000, True),  # +bf16 max
    (0xFF7F0000, True),  # -bf16 max
)

# Thresholds: both zeros, both infinities, both NaN signs, and finite values on either side.
# "+0.0" must stay first: it is the only threshold the uint32 entry point can take (see the module docstring).
THRESHOLDS = [
    Bits("+0.0", 0x00000000),
    Bits("-0.0", 0x80000000),
    Bits("+inf", 0x7F800000),
    Bits("-inf", 0xFF800000),
    Bits("+nan", 0x7FC00000),
    Bits("-nan", 0xFFC00000),
    Bits("0.5", 0x3F000000),
    Bits("-2.5", 0xC0200000),
]

PIPELINES = [
    Pipeline(
        "Float32-dest_acc_Yes",
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.Yes,
    ),
    Pipeline(
        "Float16_b-dest_acc_No",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
    ),
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
    return f"{value:.9g}"


@parametrize(
    pipeline=PIPELINES,
    mathop=list(_TORCH_COMPARE),
    entry=["vfloat", "uint32"],
    threshold=lambda entry: THRESHOLDS if entry == "vfloat" else THRESHOLDS[:1],
)
def test_sfpu_comp_unary_ieee(pipeline, mathop, entry, threshold):
    assert (
        entry == "vfloat" or threshold.bits == 0
    ), "the uint32 entry point decodes only +0.0 correctly"
    formats, dest_acc = pipeline.formats, pipeline.dest_acc
    unpack_to_dest = formats.input_format.is_32_bit()
    if unpack_to_dest:
        # Without a real -0.0 in Dest the -0.0 lanes would pass on the unfixed kernel too.
        assert negative_zero_delivered(formats.input_format, dest_acc)

    input_bits = [bits for bits, bf16_exact in INPUTS if unpack_to_dest or bf16_exact]
    reps = ELEMENTS_PER_TILE // len(input_bits) + 1
    tile_bits = (input_bits * reps)[:ELEMENTS_PER_TILE]
    src_A = _floats(tile_bits)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_comp_unary_ieee_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            MATH_OP(mathop=mathop),
            SFPU_UNARY_SCALAR(threshold.bits),
            SFPU_COMP_SCALAR_ENTRY(entry),
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
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
        compile_time_formats=True,
    )

    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE],
        dtype=format_dict[formats.output_format],
    ).to(torch.float32)

    golden = _TORCH_COMPARE[mathop](src_A, _floats([threshold.bits])[0]).to(
        torch.float32
    )

    wrong = sorted(
        {
            (_fmt(tile_bits[i]), res[i].item(), golden[i].item())
            for i in range(ELEMENTS_PER_TILE)
            if res[i].item() != golden[i].item()
        }
    )
    assert not wrong, (
        f"{mathop.name}(x, {threshold.name}) via the {entry} entry point disagrees with IEEE-754 on "
        f"{len(wrong)} input classes:\n"
        + "\n".join(f"  x = {x}: got {got}, expected {want}" for x, got, want in wrong)
    )
