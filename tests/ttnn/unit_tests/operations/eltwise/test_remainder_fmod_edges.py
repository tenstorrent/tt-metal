# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_wormhole_b0

pytestmark = pytest.mark.use_module_device

INF = float("inf")


def _cases(with_nan):
    a, b = [], []
    for divisor in (3.0, 5.0, 7.0, 11.0, 13.0):
        for k in range(1, 20):
            for sa, sb in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
                a.append(sa * k * divisor)
                b.append(sb * divisor)
    for x, d in ((5.625, -1.875), (-5.625, 1.875), (5.625, 1.875), (-5.625, -1.875)):
        a.append(x)
        b.append(d)
    for x in (0.0, -0.0, 0.5, -0.5, 5.0, -5.0, 100.0, -100.0):
        for d in (INF, -INF):
            a.append(x)
            b.append(d)
    if with_nan:
        nan = float("nan")
        for x, d in (
            (INF, INF),
            (-INF, -INF),
            (INF, -INF),
            (-INF, INF),
            (5.0, nan),
            (-5.0, nan),
            (nan, 5.0),
            (nan, -INF),
        ):
            a.append(x)
            b.append(d)
    while len(a) % 1024:
        a.append(10.0)
        b.append(3.0)
    return torch.tensor(a).reshape(1, 1, -1, 32), torch.tensor(b).reshape(1, 1, -1, 32)


@pytest.mark.xfail(is_wormhole_b0(), reason="Wormhole fix in a separate PR")
@pytest.mark.parametrize("op", ["remainder", "fmod"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_binary_remainder_fmod_exact_multiples_and_infinite_divisor(op, dtype, device):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_a, torch_b = _cases(with_nan=dtype == ttnn.float32)
    torch_a, torch_b = torch_a.to(torch_dtype), torch_b.to(torch_dtype)
    golden = getattr(torch, op)(torch_a, torch_b)

    tt_a = ttnn.from_torch(torch_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.to_torch(getattr(ttnn, op)(tt_a, tt_b))

    torch.testing.assert_close(output, golden, rtol=0, atol=0, equal_nan=True)
