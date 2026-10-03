# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""div where an operand is infinite.

With fp32 destination accumulation the quotient is refined by a residual step,
and that step cannot be formed when the divisor is infinite: the quotient is
already zero, so the residual is 0 * inf. These pin the cases where the
refinement used to turn a correct zero into a NaN.
"""

import pytest
import torch
import ttnn

INF = float("inf")

# finite / infinite, which is a signed zero, and the cases either side of it.
CASES = [
    (1.0, INF),
    (2.0, INF),
    (-3.0, INF),
    (1.0, -INF),
    (-1.0, -INF),
    (0.0, INF),
    (-0.0, INF),
    (1e30, INF),
    (1e-30, INF),
    (INF, INF),
    (-INF, INF),
    (INF, 2.0),
    (-INF, 2.0),
]


@pytest.mark.parametrize("a, b", CASES)
def test_div_by_infinity_fp32(device, a, b):
    ta = torch.full((1, 1, 32, 32), a, dtype=torch.float32)
    tb = torch.full((1, 1, 32, 32), b, dtype=torch.float32)
    want = (ta / tb).flatten()[0]

    ia = ttnn.from_torch(ta, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(tb, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.divide(ia, ib)).flatten()[0]

    if torch.isnan(want):
        assert torch.isnan(got), f"div({a}, {b}) returned {got}, want nan"
    else:
        assert got == want, f"div({a}, {b}) returned {got}, want {want}"


# The sign of the zero is not asserted above. It is wrong for a negative
# dividend, and that is not this code path: an ordinary multiply loses it too,
# ttnn.multiply(-3.0, 0.0) returns +0 where torch returns -0, while a plain
# round trip through from_torch/to_torch preserves -0. Asserting it here would
# be testing that separate defect.
