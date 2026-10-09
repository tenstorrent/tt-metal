# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Threshold decoding of the tt-llk scalar compares (tt-llk#1701 item 16).

``_calculate_comp_unary_<APPROX, unary_*>(std::uint32_t value)`` in
``tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h`` takes the threshold as fp32 bits, like the metal
``calculate_unary_*`` kernels and the other tt-llk ops with a uint32 scalar (relu, threshold, fill). It
used to build the vFloat with ``vFloat s = value``, a numeric uint32 -> float conversion: 0x3F000000
(0.5f) became 1056964608.0f, and no negative, infinite or NaN threshold could be expressed at all.

Inputs and thresholds stay away from NaN and from a zero threshold, whose IEEE handling is a separate
defect (item 3), so this test checks only how the threshold is decoded. The golden is torch's compare
and the check is exact: every lane is 0.0 or 1.0.

Wormhole's ordered compares are an SFPMAD subtract plus a sign test. When x and the threshold are the
same infinity, inf - inf is a NaN that keeps the product's sign, so gt/lt/ge/le(+inf, +inf) come out
wrong (-inf happens to route right). That is a compare defect, not a decode one, and those lanes are
left out of the check on Wormhole; the infinite thresholds are still checked against every other input.

Wormhole only: the Blackhole copy of this header is fixed separately (tt-metal#59870).
"""

import struct

import torch
from conftest import wormhole_only
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
    MathOperation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    MATH_OP,
    SFPU_UNARY_SCALAR,
    VECTOR_MODE,
    generate_input_dim,
)

# https://github.com/tenstorrent/tt-llk/issues/1701 item 16: the Blackhole _calculate_comp_unary_ is fixed in
# tt-metal#59870.
pytestmark = wormhole_only

ELEMENTS_PER_TILE = 1024


def _from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# Inputs: each finite threshold below with its fp32 neighbours, the largest finite values next to the
# infinities, and values on both sides of every threshold.
INPUTS = (
    0.0,
    -0.0,
    0.5,
    _from_bits(0x3EFFFFFF),  # 0.5 - 1 ulp
    _from_bits(0x3F000001),  # 0.5 + 1 ulp
    1.0,
    _from_bits(0x3F7FFFFF),  # 1.0 - 1 ulp
    _from_bits(0x3F800001),  # 1.0 + 1 ulp
    -1.0,
    -2.5,
    _from_bits(0xC01FFFFF),  # -2.5 + 1 ulp
    _from_bits(0xC0200001),  # -2.5 - 1 ulp
    _from_bits(
        0x3DCCCCCD
    ),  # 0.1f: low mantissa bits set, so a decode that kept only the upper half fails
    _from_bits(0x3DCCCCCC),  # 0.1f - 1 ulp
    _from_bits(0x3DCCCCCE),  # 0.1f + 1 ulp
    0.099609375,  # 0.1f truncated to bf16 (0x3DCC0000)
    3.0,
    1.0e9,
    1.1e9,  # straddle 1056964608.0, the value 0x3F000000 used to decode to
    _from_bits(0x7F7FFFFF),  # FLT_MAX
    _from_bits(0xFF7FFFFF),  # -FLT_MAX
    float("inf"),
    float("-inf"),
)

THRESHOLDS = [0.5, 1.0, -2.5, _from_bits(0x3DCCCCCD), float("inf"), float("-inf")]

_ORDERED_COMPARES = {
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
}

_TORCH_COMPARE = {
    MathOperation.UnaryGt: torch.gt,
    MathOperation.UnaryLt: torch.lt,
    MathOperation.UnaryGe: torch.ge,
    MathOperation.UnaryLe: torch.le,
    MathOperation.UnaryEq: torch.eq,
    MathOperation.UnaryNe: torch.ne,
}


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    mathop=list(_TORCH_COMPARE),
    threshold=THRESHOLDS,
    dest_sync=[DestSync.Half],
)
def test_sfpu_comp_unary_threshold(formats, mathop, threshold, dest_sync):
    reps = ELEMENTS_PER_TILE // len(INPUTS) + 1
    src_A = torch.tensor((list(INPUTS) * reps)[:ELEMENTS_PER_TILE], dtype=torch.float32)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_comp_unary_threshold_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            MATH_OP(mathop=mathop),
            SFPU_UNARY_SCALAR(_bits(threshold)),
            VECTOR_MODE(VectorMode.RC),
            DEST_SYNC(dest_sync),
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

    golden = _TORCH_COMPARE[mathop](
        src_A, torch.tensor(threshold, dtype=torch.float32)
    ).to(torch.float32)

    checked = torch.ones(ELEMENTS_PER_TILE, dtype=torch.bool)
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and mathop in _ORDERED_COMPARES
    ):
        # WH compares by subtracting: inf - inf is a NaN with the product's sign (see the docstring).
        checked = ~(torch.isinf(src_A) & (src_A == threshold))

    wrong = sorted(
        {
            (src_A[i].item(), res[i].item(), golden[i].item())
            for i in range(ELEMENTS_PER_TILE)
            if checked[i] and res[i].item() != golden[i].item()
        }
    )
    assert not wrong, (
        f"{mathop.name}(x, {threshold:g}) with value=0x{_bits(threshold):08X} disagrees with torch on "
        f"{len(wrong)} inputs:\n"
        + "\n".join(
            f"  x = {x:.9g}: got {got}, expected {want}" for x, got, want in wrong
        )
    )
