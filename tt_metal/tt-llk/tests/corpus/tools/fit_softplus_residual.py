#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the independently derived BF16 Softplus residual coefficients.

This host calculation is not a silicon accuracy or performance test. It does
not model fresh_exp_21f error, Tensix rounding/FTZ, or special-value handling.
Use the pinned NumPy in requirements-formal.txt for coefficient reproduction.
"""

import json

import numpy as np
from numpy.polynomial import Polynomial


def main():
    if np.__version__ != "1.26.4":
        raise SystemExit("coefficient reproduction requires numpy==1.26.4")

    # log1p(u) = u*q(u). Fix the continuous endpoint q(0)=1 and
    # q(1)=log(2); least squares determines only the interior correction.
    u = np.linspace(0.0, 1.0, 65537, dtype=np.float64)
    q = np.divide(np.log1p(u), u, out=np.ones_like(u), where=u != 0)
    base = 1.0 + (np.log(2.0) - 1.0) * u
    basis = np.stack([u * (1.0 - u), u * u * (1.0 - u)], axis=1)
    interior = np.linalg.lstsq(basis, q - base, rcond=None)[0]
    poly = Polynomial([1.0, np.log(2.0) - 1.0]) + Polynomial(
        [0.0, 1.0, -1.0]
    ) * Polynomial(interior)

    # The folded-normalization reference puts the existing exponential's
    # non-unit origin into P. Reproduce it before deriving the unit-origin form.
    origin = np.float32(1.0017248)
    coefficients = (poly.coef / float(origin) ** np.arange(1, 5)).astype(np.float32)
    bits = coefficients.view(np.uint32)
    expected = np.array(
        [0x3F7F8F28, 0xBEF94255, 0x3E82DC65, 0xBD990DCB], dtype=np.uint32
    )
    if not np.array_equal(bits, expected):
        raise SystemExit(f"coefficient reproduction differs: {bits.tolist()}")

    # Dividing all exp mantissa-polynomial coefficients by C0 makes exp(0)=1.
    # setexp commutes with this normalization while the fraction remains in
    # [1, 2), so the residual uses the direct endpoint-constrained q(u).
    exp_coefficients = np.array(
        [
            np.float32(1.0),
            np.float32(7.839635491371155e-08) / origin,
            np.float32(4.791750143340323e-15) / origin,
        ],
        dtype=np.float32,
    )
    exp_bits = exp_coefficients.view(np.uint32)
    expected_exp = np.array([0x3F800000, 0x33A810A4, 0x27AC57FE], dtype=np.uint32)
    if not np.array_equal(exp_bits, expected_exp):
        raise SystemExit(f"unit-origin exp coefficients differ: {exp_bits.tolist()}")

    direct = poly.coef.astype(np.float32)
    direct_bits = direct.view(np.uint32)
    expected_direct = np.array(
        [0x3F800000, 0xBEFA1EA6, 0x3E838A0B, 0xBD9A1CD2], dtype=np.uint32
    )
    if not np.array_equal(direct_bits, expected_direct):
        raise SystemExit(f"direct q coefficients differ: {direct_bits.tolist()}")

    rounded = poly.coef.astype(np.float32)
    uf = u.astype(np.float32)
    estimate = np.full_like(uf, rounded[-1])
    for c in rounded[-2::-1]:
        estimate = estimate * uf + c

    origin_estimate = coefficients[-1]
    for c in coefficients[-2::-1]:
        origin_estimate = np.float32(origin_estimate * origin + c)
    origin_residual = np.float32(origin * origin_estimate)

    print(
        json.dumps(
            {
                "numpy": np.__version__,
                "samples": len(u),
                "fit": "endpoint-constrained degree-3 q(u)=log1p(u)/u on [0,1]",
                "normalized_coefficients": [float(c) for c in coefficients],
                "normalized_coefficient_bits": [f"0x{int(b):08x}" for b in bits],
                "unit_origin_exp_coefficients": [float(c) for c in exp_coefficients],
                "unit_origin_exp_coefficient_bits": [
                    f"0x{int(b):08x}" for b in exp_bits
                ],
                "unit_origin_q_coefficients": [float(c) for c in direct],
                "unit_origin_q_coefficient_bits": [
                    f"0x{int(b):08x}" for b in direct_bits
                ],
                "q_max_relative_error_f32_nonfused_horner": float(
                    np.max(np.abs(estimate - q) / q)
                ),
                "origin_residual_f32_nonfused_horner": float(origin_residual),
                "scope": "host coefficient reproduction only; device admission is separate",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
