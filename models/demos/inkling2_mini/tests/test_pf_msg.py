# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill cross-pass messages (tt/fused/pf_msg.py) vs prefill_model._messages' op chain: the 4 messages of a
loop layer bit-equal at T = 128, 1024 and 8448 (the row split differs per T)."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_msg import PfMessages


@pytest.mark.parametrize("t", [128, 1024, 8448])
def test_pf_msg(device, t):
    g = torch.Generator().manual_seed(t)
    put = lambda a: ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731
    n1 = put(torch.randn(1, 1, t, 2304, generator=g))
    diff = put(torch.randn(1, 1, t, 2304, generator=g) * 0.1)
    w = put(torch.rand(1, 1, t, 64, generator=g))
    mod = PfMessages(device)
    for l in (0, 5, 11):
        new = mod(n1, diff, w, 4 * l)
        for k in range(4):
            col = ttnn.slice(w, [0, 0, 0, 4 * l + k], [1, 1, t, 4 * l + k + 1])
            a = ttnn.to_torch(ttnn.add(n1, ttnn.multiply(diff, col)))
            b = ttnn.to_torch(new[k])
            assert torch.equal(a, b), (l, k, float((a - b).abs().max()))
