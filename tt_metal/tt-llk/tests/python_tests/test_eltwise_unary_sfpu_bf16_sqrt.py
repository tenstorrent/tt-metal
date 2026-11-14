# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The generated BF16 sqrt kernel over every finite BF16 value (helpers/bf16_sweep.py)."""

import pytest
import test_eltwise_unary_sfpu as _unary
from helpers.bf16_sweep import bf16_exhaustive_sweep, replaced_instance_params
from helpers.chip_architecture import ChipArchitecture
from helpers.llk_params import ApproximationMode, DestSync, FastMode, MathOperation
from helpers.param_config import build_param_id

# (operation, approximation mode of the replaced instance, max ULP); max ULP 0 is an exact fit.
_BF16_EXHAUSTIVE_OPS = [
    (MathOperation.Sqrt, ApproximationMode.No, 1),
]
# Boards where the op keeps its stock kernel, which this sweep does not test.
_BF16_STOCK_BOARDS = {
    MathOperation.Sqrt: (ChipArchitecture.BLACKHOLE,),
}
# Per board, the word stored at each special input class where the kernel returns stock's result.
_BF16_STOCK_SPECIALS = {}


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


# The (approximation, fast mode) instances the BF16 kernel replaces with FP32 DEST off.
_BF16_REPLACED_INSTANCES = {
    MathOperation.Sqrt: ((ApproximationMode.No, FastMode.No),),
}
_BF16_SETUP_PARAMS = replaced_instance_params(_unary, _BF16_REPLACED_INSTANCES)


# The stock test's own DEST sync axis (test_eltwise_unary_sfpu's dest_sync parametrize).
@pytest.mark.parametrize("dest_sync", [DestSync.Half])
@pytest.mark.parametrize(
    ",".join(_unary._UNARY_SWEEP_ARGNAMES),
    _BF16_SETUP_PARAMS,
    ids=[build_param_id(_unary._UNARY_SWEEP_ARGNAMES, p) for p in _BF16_SETUP_PARAMS],
)
def test_eltwise_unary_sfpu_bf16_replaced_instances(
    formats, approx_mode, mathop, fast_mode, dest_acc, input_dimensions, dest_sync
):
    """Every instance the BF16 kernel replaces, on BF16 and FP32 data, against the sweep's golden."""
    _unary.test_eltwise_unary_sfpu(
        formats, approx_mode, mathop, fast_mode, dest_acc, input_dimensions, dest_sync
    )
