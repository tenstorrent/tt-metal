# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""-0.0 in ``_sfpu_is_fp16_zero_`` (tt-llk#1701 item 6), through the tt-llk eqz/nez kernels.

``_sfpu_is_fp16_zero_`` in ``tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h`` was
``v == 0.0F``, which lowers to SFPSETCC LREG_EQ0, an all-32-bits-zero test. -0.0 (0x80000000) failed it,
so ``_calculate_zero_comp_<equal_zero>`` returned 0 and ``<not_equal_zero>`` returned 1 for -0.0, where
IEEE-754 and torch give 1 and 0. The metal ``calculate_mask`` / ``calculate_mask_posinf`` use the same
helper; ``test_eltwise_binary_sfpu_mask_negative_zero`` covers that caller.

Float32 -> Float32 at dest_acc=Yes only. It is the one tt-llk pipeline that unpacks straight to Dest,
so it is the only one that hands the SFPU a -0.0 (see ``negative_zero_delivered`` in
``helpers/sfpu_domains.py``); every 16-bit path delivers +0.0, so -0.0 lanes cannot fail there.

The golden is torch's compare and the check is exact: every lane is 0.0 or 1.0.

Blackhole only: the Wormhole copy of the helper is not changed by the fix.
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
    reason="tt-llk#1701 item 6: only the Blackhole _sfpu_is_fp16_zero_ treats -0.0 as zero so far",
)

ELEMENTS_PER_TILE = 1024

INPUTS = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.0e-45,  # smallest +denormal (Float32 unpack-to-dest delivers it intact): not zero
    -1.0e-45,  # smallest -denormal
    3.0e38,
    -3.0e38,
    float("inf"),
    float("-inf"),
    float("nan"),
    -float("nan"),  # torch keeps the sign of an fp32 NaN: 0xFFC00000
)

_TORCH_COMPARE = {
    MathOperation.EqualZero: torch.eq,
    MathOperation.NotEqualZero: torch.ne,
}


def _fmt(value: float) -> str:
    sign = "-" if torch.tensor(value).signbit().item() else "+"
    if value != value:
        return f"{sign}nan"
    if value == 0.0:
        return f"{sign}0.0"
    return f"{value:g}"


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    mathop=list(_TORCH_COMPARE),
)
def test_sfpu_is_fp16_zero(formats, mathop):
    reps = ELEMENTS_PER_TILE // len(INPUTS) + 1
    src_A = torch.tensor((list(INPUTS) * reps)[:ELEMENTS_PER_TILE], dtype=torch.float32)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sfpu_is_fp16_zero_test.cpp",
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
