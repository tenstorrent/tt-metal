# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The generated BF16 selu kernel over every finite BF16 value (helpers/bf16_sweep.py)."""

import pytest
from helpers.bf16_sweep import bf16_exhaustive_sweep
from helpers.chip_architecture import ChipArchitecture
from helpers.llk_params import ApproximationMode, MathOperation

# (operation, approximation mode of the replaced instance, max ULP); max ULP 0 is an exact fit.
_BF16_EXHAUSTIVE_OPS = [
    (MathOperation.Selu, ApproximationMode.No, 0),
]
# Boards where the op keeps its stock kernel, which this sweep does not test.
_BF16_STOCK_BOARDS = {}
# Per board, the word stored at each special input class where the kernel returns stock's result.
_BF16_STOCK_SPECIALS = {
    MathOperation.Selu: {
        ChipArchitecture.BLACKHOLE: {"neg_nan": 0xBFE1},
        ChipArchitecture.WORMHOLE: {"neg_nan": 0xBFE1},
    },
}


@pytest.mark.parametrize("mathop,approx_mode,max_ulp", _BF16_EXHAUSTIVE_OPS)
def test_eltwise_unary_sfpu_bf16_exhaustive(mathop, approx_mode, max_ulp):
    bf16_exhaustive_sweep(
        mathop,
        approx_mode,
        max_ulp,
        _BF16_EXHAUSTIVE_OPS,
        _BF16_STOCK_BOARDS,
        _BF16_STOCK_SPECIALS,
    )
