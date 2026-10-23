# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The generated BF16 log10 kernel over every finite BF16 value (helpers/bf16_sweep.py)."""

import pytest
import test_eltwise_unary_sfpu as _unary
from helpers.bf16_sweep import bf16_exhaustive_sweep, replaced_instance_params
from helpers.chip_architecture import ChipArchitecture
from helpers.llk_params import ApproximationMode, FastMode, MathOperation
from helpers.param_config import build_param_id

# (operation, approximation mode of the replaced instance, max ULP); max ULP 0 is an exact fit.
_BF16_EXHAUSTIVE_OPS = [
    (MathOperation.Log10, ApproximationMode.No, 1),
]
# Boards where the op keeps its stock kernel, which this sweep does not test.
_BF16_STOCK_BOARDS = {}
# The stored word, per board, at each special input class where the kernel returns its stock
# kernel's result instead of torch's; the sweep expects that word's class there.
_BF16_STOCK_SPECIALS = {
    MathOperation.Log10: {
        ChipArchitecture.BLACKHOLE: {"neg_subnormal": 0xFF80, "pos_subnormal": 0xFF80},
        ChipArchitecture.WORMHOLE: {"neg_subnormal": 0xFF80, "pos_subnormal": 0xFF80},
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


# The (approximation, fast mode) instances the op's BF16 kernel replaces with FP32 DEST off, read from
# its kernel's gate. Its setup runs from an init that stock instances share; export compiles every other
# instance's init and tile from this tree and from stock and refuses the PR unless they are identical,
# so only these instances run here.
_BF16_REPLACED_INSTANCES = {
    MathOperation.Log10: (
        (ApproximationMode.No, FastMode.No),
        (ApproximationMode.No, FastMode.Yes),
        (ApproximationMode.Yes, FastMode.No),
        (ApproximationMode.Yes, FastMode.Yes),
    ),
}
_BF16_SETUP_PARAMS = replaced_instance_params(_unary, _BF16_REPLACED_INSTANCES)


@pytest.mark.parametrize(
    ",".join(_unary._UNARY_SWEEP_ARGNAMES),
    _BF16_SETUP_PARAMS,
    ids=[build_param_id(_unary._UNARY_SWEEP_ARGNAMES, p) for p in _BF16_SETUP_PARAMS],
)
def test_eltwise_unary_sfpu_bf16_replaced_instances(
    formats, approx_mode, mathop, fast_mode, dest_acc, input_dimensions
):
    """Every instance the BF16 kernel replaces, on BF16 and FP32 data, against the sweep's golden."""
    _unary.test_eltwise_unary_sfpu(
        formats, approx_mode, mathop, fast_mode, dest_acc, input_dimensions
    )
