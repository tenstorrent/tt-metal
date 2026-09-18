# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""nextafter returns the adjacent representable value.

The gate is bit-exactness against torch.nextafter, which is the op's own golden.
Nothing weaker can test this: the answer differs from the input by one unit in
the last place, so any tolerance that admits a near miss admits every miss.
"""

import pytest
import torch
import ttnn

# Magnitudes spanning six decades either side of 1. The composite this replaced
# stepped by FLT_EPSILON, which is one ULP only inside [1, 2), so it failed
# everywhere except that band, in both directions.
MAGNITUDES = [1e-6, 1e-3, 0.5, 1.0, 1.5, 3.0, 100.0, 1e4, 1e6]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_nextafter_steps_one_ulp(device, dtype, magnitude, sign, direction):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    a_val = sign * magnitude
    b_val = a_val + direction * (magnitude * 10 + 1)

    a = torch.full((1, 1, 32, 32), a_val, dtype=torch_dtype)
    b = torch.full((1, 1, 32, 32), b_val, dtype=torch_dtype)
    want = torch.nextafter(a, b).flatten()[0]

    ia = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.nextafter(ia, ib)).flatten()[0]

    assert got == want, f"nextafter({a_val}, {b_val}) returned {got}, want {want}"
    # Stated separately because a result that fails to move and one that moves
    # the wrong way are different defects, and the composite did both.
    assert (got - a_val) * (b_val - a_val) > 0, f"nextafter({a_val}, {b_val}) did not step toward b"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_nextafter_equal_inputs_are_unchanged(device, dtype):
    """IEEE-754: nextafter(x, x) is x, for every x."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    vals = torch.tensor(MAGNITUDES + [-m for m in MAGNITUDES], dtype=torch_dtype)
    a = vals.repeat(32 * 32 // vals.numel() + 1)[: 32 * 32].reshape(1, 1, 32, 32)

    ia = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.nextafter(ia, ia))

    assert torch.equal(got, a), f"{int((got != a).sum())} of {got.numel()} changed"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_nextafter_from_infinity_is_the_largest_finite(device, dtype):
    """One step down from infinity is the largest finite value, not infinity."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    for sign in (1.0, -1.0):
        a = torch.full((1, 1, 32, 32), sign * float("inf"), dtype=torch_dtype)
        b = torch.zeros((1, 1, 32, 32), dtype=torch_dtype)
        want = torch.nextafter(a, b).flatten()[0]

        ia = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        ib = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(ttnn.nextafter(ia, ib)).flatten()[0]

        assert torch.isfinite(got), f"nextafter({sign} inf, 0) returned {got}"
        assert got == want, f"nextafter({sign} inf, 0) returned {got}, want {want}"
