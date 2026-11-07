# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The generated BF16 leaky_relu kernel over every finite BF16 value (helpers/bf16_sweep.py)."""

import functools
import struct
from dataclasses import dataclass

import pytest
from helpers import bf16_sweep
from helpers.bf16_sweep import bf16_exhaustive_sweep
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import ApproximationMode, MathOperation
from helpers.test_variant_parameters import TemplateParameter

# (operation, approximation mode of the replaced instance, max ULP); max ULP 0 is an exact fit.
_BF16_EXHAUSTIVE_OPS = [
    (MathOperation.Lrelu, ApproximationMode.No, 0),
]
# Boards where the op keeps its stock kernel, which this sweep does not test.
_BF16_STOCK_BOARDS = {}
# Per board, the word stored at each special input class where the kernel returns stock's result.
_BF16_STOCK_SPECIALS = {}

# The negative_slope the BF16 kernel serves; the shared rows keep upstream's, which run the stock kernel.
_FITTED_BITS = 0x3C23D70A


@dataclass
class _HarnessScalar(TemplateParameter):
    """Overrides LRELU_SLOPE_BITS in helpers/include/sfpu_operations.h."""

    def convert_to_cpp(self) -> str:
        return f"#define LRELU_SLOPE_BITS 0x{_FITTED_BITS:08x}u"


@pytest.fixture
def fitted_scalar(monkeypatch):
    """Runs the sweep's kernel and golden at the fitted negative_slope."""
    stock_builder = bf16_sweep.build_eltwise_unary_sfpu_config

    def builder(*args, **kwargs):
        config = stock_builder(*args, **kwargs)
        config.templates.append(_HarnessScalar())
        return config

    monkeypatch.setattr(bf16_sweep, "build_eltwise_unary_sfpu_config", builder)
    golden = get_golden_generator(UnarySFPUGolden)
    monkeypatch.setitem(
        golden.ops,
        MathOperation.Lrelu,
        functools.partial(
            golden._lrelu,
            negative_slope=struct.unpack("<f", struct.pack("<I", _FITTED_BITS))[0],
        ),
    )


@pytest.mark.parametrize("mathop,approx_mode,max_ulp", _BF16_EXHAUSTIVE_OPS)
def test_eltwise_unary_sfpu_bf16_exhaustive(
    fitted_scalar, mathop, approx_mode, max_ulp
):
    bf16_exhaustive_sweep(
        mathop,
        approx_mode,
        max_ulp,
        _BF16_EXHAUSTIVE_OPS,
        _BF16_STOCK_BOARDS,
        _BF16_STOCK_SPECIALS,
    )
