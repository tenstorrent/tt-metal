#!/usr/bin/env python3
"""Regression check for the licensed GELU asymptotic tail slot."""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path

from formal_equiv import Concrete, fma_model_bh

HERE = Path(__file__).resolve().parent
HEADER = HERE.parents[1] / "helpers/include/fresh_cpp/gelu_appx_licensed.h"


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bf16_value(bits: int) -> float:
    return f32_value(bits << 16)


def rounded_bf16(value: float) -> float:
    bits = f32_bits(value)
    bias = 0x7FFF + ((bits >> 16) & 1)
    return bf16_value(((bits + bias) >> 16) & 0xFFFF)


def parse_asymptote() -> tuple[float, float]:
    text = HEADER.read_text()
    match = re.search(
        r"sfpi::vFloat g = a \* (?P<slope>[^; ]+) \+ (?P<intercept>[^; ]+);",
        text,
    )
    assert match is not None

    def c_float(token: str) -> float:
        token = token.removesuffix("f")
        return float.fromhex(token) if "p" in token.lower() else float(token)

    return c_float(match.group("slope")), c_float(match.group("intercept"))


def stored_asymptote(input_bf16: int, slope: float, intercept: float) -> int:
    x = input_bf16 << 16
    magnitude = x & 0x7FFFFFFF
    g = fma_model_bh(Concrete, magnitude, f32_bits(slope), f32_bits(intercept))
    result = fma_model_bh(Concrete, x, f32_bits(0.5), g)
    return result >> 16


def main() -> None:
    slope, intercept = parse_asymptote()
    assert slope == 0.5
    assert intercept == 0.0

    # These positive-tail values previously lost one bf16 step when a small
    # negative intercept was truncated by the Float16_b store.
    assert stored_asymptote(0x4381, slope, intercept) == 0x4381  # 258
    assert stored_asymptote(0x4081, slope, intercept) == 0x4081  # 4.03125

    old_intercept = -float.fromhex("0x1.2fcp-15")
    assert stored_asymptote(0x4381, slope, old_intercept) == 0x4380
    assert stored_asymptote(0x4081, slope, old_intercept) == 0x4080

    # The zero-intercept asymptotic line produces the negative-tail limit.
    assert stored_asymptote(0xC0A0, slope, intercept) == 0x0000  # -5 -> +0

    changed = 0
    rounded_regressions = 0
    exact_regressions = 0
    max_regression = 0.0
    for input_bits in range(1 << 16):
        x = bf16_value(input_bits)
        if not math.isfinite(x) or abs(x) < 3.0:
            continue
        old = bf16_value(stored_asymptote(input_bits, slope, old_intercept))
        new = bf16_value(stored_asymptote(input_bits, slope, intercept))
        if old == new:
            continue
        changed += 1
        exact = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
        rounded = rounded_bf16(exact)
        rounded_delta = abs(new - rounded) - abs(old - rounded)
        exact_delta = abs(new - exact) - abs(old - exact)
        rounded_regressions += rounded_delta > 0.0
        exact_regressions += exact_delta > 0.0
        max_regression = max(max_regression, rounded_delta, exact_delta)

    # Host SFPMAD/store modeling, not a device certificate. The correction
    # improves the global max-error contract, but the removed negative bias was
    # marginally closer at 79 negative-tail inputs; keep that tradeoff explicit.
    assert changed == 2178
    assert rounded_regressions == 79
    assert exact_regressions == 79
    assert max_regression == 3.62396240234375e-05


if __name__ == "__main__":
    main()
