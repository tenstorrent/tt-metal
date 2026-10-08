# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""-0.0 in the tt-llk ordered compares against zero (tt-llk#1701 item 4).

Drives ``_calculate_zero_comp_<APPROX, less_than_zero | greater_than_equal_zero | greater_than_zero |
less_than_equal_zero>`` from ``tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h``. ltz and gez
decided with ``v >= 0.0f``, which lowers to SFPLE, and SFPLE orders -0.0 below +0.0. So the kernel
returned ltz(-0.0) = 1 and gez(-0.0) = 0, where IEEE-754 and torch give 0 and 1. gtz and lez are
included as controls (they were already right for -0.0).

NaN inputs are left out on purpose: these four kernels also rank NaN by its sign bit (gtz(+NaN) = 1,
ltz(-NaN) = 1), which is a separate deviation that the fix does not change.

Float32 -> Float32 at dest_acc=Yes only. It is the one tt-llk pipeline that unpacks straight to Dest,
so it is the only one that hands the SFPU a -0.0 (see ``negative_zero_delivered`` in
``helpers/sfpu_domains.py``); every 16-bit path delivers +0.0, so -0.0 lanes cannot fail there.

The golden is torch's compare and the check is exact: every lane is 0.0 or 1.0.

Blackhole only: the Wormhole copy of this header is not changed by the fix.
"""

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
    VECTOR_MODE,
    generate_input_dim,
)

pytestmark = pytest.mark.skipif(
    get_chip_architecture() != ChipArchitecture.BLACKHOLE,
    reason="tt-llk#1701 item 4: only the Blackhole _calculate_zero_comp_ handles -0.0 so far",
)

ELEMENTS_PER_TILE = 1024

INPUTS = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.0e-45,  # smallest +denormal (Float32 unpack-to-dest delivers it intact)
    -1.0e-45,  # smallest -denormal
    1.1754944e-38,  # smallest +normal
    -1.1754944e-38,
    3.0e38,
    -3.0e38,
    float("inf"),
    float("-inf"),
)

_TORCH_COMPARE = {
    MathOperation.LessThanZero: torch.lt,
    MathOperation.GreaterThanEqualZero: torch.ge,
    MathOperation.GreaterThanZero: torch.gt,
    MathOperation.LessThanEqualZero: torch.le,
}


def _fmt(value: float) -> str:
    if value == 0.0:
        return "-0.0" if torch.tensor(value).signbit().item() else "+0.0"
    return f"{value:g}"


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    mathop=list(_TORCH_COMPARE),
)
def test_sfpu_zero_comp_signed_zero(formats, mathop):
    reps = ELEMENTS_PER_TILE // len(INPUTS) + 1
    src_A = torch.tensor((list(INPUTS) * reps)[:ELEMENTS_PER_TILE], dtype=torch.float32)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_zero_comp_signed_zero_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            MATH_OP(mathop=mathop),
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

    golden = _TORCH_COMPARE[mathop](src_A, 0.0).to(torch.float32)

    wrong = sorted(
        {
            (_fmt(src_A[i].item()), res[i].item(), golden[i].item())
            for i in range(ELEMENTS_PER_TILE)
            if res[i].item() != golden[i].item()
        }
    )
    assert not wrong, (
        f"{mathop.name} disagrees with torch on {len(wrong)} input classes:\n"
        + "\n".join(f"  x = {x}: got {got}, expected {want}" for x, got, want in wrong)
    )
