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

    # The existing exponential approximation has e(0)=C0, not 1. Express
    # log1p(e/C0) directly as e*P(e), folding the normalization into P.
    origin = np.float32(1.0017248)
    coefficients = (poly.coef / float(origin) ** np.arange(1, 5)).astype(np.float32)
    bits = coefficients.view(np.uint32)
    expected = np.array(
        [0x3F7F8F28, 0xBEF94255, 0x3E82DC65, 0xBD990DCB], dtype=np.uint32
    )
    if not np.array_equal(bits, expected):
        raise SystemExit(f"coefficient reproduction differs: {bits.tolist()}")

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
