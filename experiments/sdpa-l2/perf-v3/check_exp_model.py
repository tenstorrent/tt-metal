# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the input-independent polynomial fit; not a hardware accuracy test.

Run with the repository Python environment (NumPy). The SDPA device regression
is validate.sh. This model does not emulate SFPU instruction rounding.
"""

import numpy as np


def main():
    m = np.linspace(1.0, 2.0, 65536)
    target = 0.995 * np.exp2(m - 1) + 1 / 128
    # Fit m*q(m) relative to the biased exponential, q a cubic. Factoring out
    # m allows the kernel to reuse the fast-exp result's original exponent.
    design = np.vander(m, 4) * (m / target)[:, None]
    coefficients = np.linalg.lstsq(design, np.ones_like(m), rcond=None)[0]
    implemented = np.array([-8.106702647031e27, 5.446591617222e28, -1.069245718291e29, 1.399030410717e29])
    np.testing.assert_allclose(coefficients * 2.0**96, implemented, rtol=1e-10)
    approximation = m * np.polyval(implemented / 2.0**96, m)
    relative = approximation / target - 1
    assert np.sqrt(np.mean(relative**2)) < 0.0005
    # Sign-preserving underflow: the fast macro encodes sufficiently negative
    # logits as negative floats; this polynomial cannot turn those positive.
    negative_m = np.linspace(-2.0, -1.0, 65536)
    assert (np.polyval(coefficients, negative_m) > 0).all()
    # P is loaded into SrcB as TF32, then HiFi2 discards all but its six high
    # fraction bits. LoFi with a SrcA operand of exactly 1 gives the same P.
    # These are the effective weights, not the untruncated FP32 values in L1.
    assert approximation.min() >= 1 and approximation.max() < 2
    effective = np.floor(approximation * 64) / 64
    print("scaled coefficients:", coefficients * 2.0**96)
    print(
        "biased-target relative error %: min/max/RMS",
        100 * relative.min(),
        100 * relative.max(),
        100 * np.sqrt(np.mean(relative**2)),
    )
    error = effective / (0.995 * np.exp2(m - 1)) - 1
    print(
        "six-bit effective-weight relative error %: min/max/RMS",
        100 * error.min(),
        100 * error.max(),
        100 * np.sqrt(np.mean(error**2)),
    )
    print("PASS: coefficient reproduction, error bound, underflow sign, mantissa range")


if __name__ == "__main__":
    main()
