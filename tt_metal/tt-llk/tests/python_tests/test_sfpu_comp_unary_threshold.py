# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Threshold decoding of the tt-llk scalar compares (tt-llk#1701 item 16).

``_calculate_comp_unary_<APPROX, unary_*>(std::uint32_t value)`` in
``tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h`` takes the threshold as fp32 bits, like the metal
``calculate_unary_*`` kernels and the other tt-llk ops with a uint32 scalar (relu, threshold, fill). It
used to build the vFloat with ``vFloat s = value``, a numeric uint32 -> float conversion: 0x3F000000
(0.5f) became 1056964608.0f, and no negative, infinite or NaN threshold could be expressed at all.

Inputs and thresholds stay away from NaN and from a zero threshold, whose IEEE handling is a separate
defect (item 3), so this test checks only how the threshold is decoded. The golden is torch's compare
and the check is exact: every lane is 0.0 or 1.0.

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
    reason="tt-llk#1701 item 16: only the Blackhole _calculate_comp_unary_ decodes its threshold bits so far",
)

ELEMENTS_PER_TILE = 1024

# Inputs: each threshold below, its fp32 neighbours, and values on both sides of it.
INPUTS = (
    0.0,
    -0.0,
    0.5,
    0.49999997,  # 0.5 - 1 ulp
    0.50000006,  # 0.5 + 1 ulp
    1.0,
    -1.0,
    -2.5,
    -2.4999998,  # -2.5 + 1 ulp
    -2.5000002,  # -2.5 - 1 ulp
    3.0,
    1.0e9,
    1.1e9,  # straddle 1056964608.0, the value 0x3F000000 used to decode to
    -3.0e38,
    3.0e38,
    float("inf"),
    float("-inf"),
)

THRESHOLDS = [0.5, 1.0, -2.5, float("inf"), float("-inf")]

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
)
def test_sfpu_comp_unary_threshold(formats, mathop, threshold):
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

    wrong = sorted(
        {
            (src_A[i].item(), res[i].item(), golden[i].item())
            for i in range(ELEMENTS_PER_TILE)
            if res[i].item() != golden[i].item()
        }
    )
    assert not wrong, (
        f"{mathop.name}(x, {threshold:g}) with value=0x{_bits(threshold):08X} disagrees with torch on "
        f"{len(wrong)} inputs:\n"
        + "\n".join(
            f"  x = {x:.9g}: got {got}, expected {want}" for x, got, want in wrong
        )
    )
