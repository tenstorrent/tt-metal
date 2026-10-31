# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

SHAPE = (1, 1, 64, 128)

CASES = {
    "where(m, s, -s)": lambda t, s, st, sc: (
        ttnn.where(t["m"], sc, -sc),
        torch.where(t["M"] != 0, s, -s).to(torch.bfloat16),
    ),
    "clamp(x, -|s|, |s|)": lambda t, s, st, sc: (
        ttnn.clamp(t["a"], min=-abs(sc), max=abs(sc)),
        torch.clamp(t["A"], -abs(s), abs(s)),
    ),
    "hardtanh(x, -|s|, |s|)": lambda t, s, st, sc: (
        ttnn.hardtanh(t["a"], min_val=-abs(sc), max_val=abs(sc)),
        torch.clamp(t["A"], -abs(s), abs(s)),
    ),
    "relu_max(x, |s|)": lambda t, s, st, sc: (
        ttnn.relu_max(t["a"], abs(sc)),
        torch.relu(torch.clamp_max(t["A"], abs(s))),
    ),
    "relu_min(x, s)": lambda t, s, st, sc: (ttnn.relu_min(t["a"], sc), torch.clamp_min(t["A"], s)),
    "heaviside(x, s)": lambda t, s, st, sc: (
        ttnn.heaviside(t["z"], sc),
        torch.heaviside(t["Z"], torch.tensor(s, dtype=torch.bfloat16)),
    ),
    "threshold(x, 0.5, s)": lambda t, s, st, sc: (ttnn.threshold(t["a"], 0.5, sc), torch.threshold(t["A"], 0.5, s)),
    "maximum(x, s)": lambda t, s, st, sc: (ttnn.maximum(t["a"], sc), torch.clamp_min(t["A"], s)),
    "minimum(x, s)": lambda t, s, st, sc: (ttnn.minimum(t["a"], sc), torch.clamp_max(t["A"], s)),
    "lerp(a, b, s)": lambda t, s, st, sc: (ttnn.lerp(t["a"], t["b"], sc), ttnn.to_torch(ttnn.lerp(t["a"], t["b"], st))),
    "mac(a, b, s)": lambda t, s, st, sc: (ttnn.mac(t["a"], t["b"], sc), ttnn.to_torch(ttnn.mac(t["a"], t["b"], st))),
    "mac(a, s, b)": lambda t, s, st, sc: (ttnn.mac(t["a"], sc, t["b"]), ttnn.to_torch(ttnn.mac(t["a"], st, t["b"]))),
    "mac(a, s, s)": lambda t, s, st, sc: (ttnn.mac(t["a"], sc, sc), ttnn.to_torch(ttnn.mac(t["a"], st, st))),
}


@pytest.mark.parametrize("scalar", [0.3, 1.2, -0.1, 1.01171875, 1.00390625])
@pytest.mark.parametrize("case", list(CASES))
def test_bf16_scalar_is_rounded_to_nearest(case, scalar, device):
    torch.manual_seed(0)
    A = torch.randn(SHAPE, dtype=torch.bfloat16)
    B = torch.randn(SHAPE, dtype=torch.bfloat16)
    M = (torch.randn(SHAPE) > 0).to(torch.bfloat16)
    s = torch.tensor(scalar).to(torch.bfloat16).item()
    to_device = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    Z = torch.where(M != 0, A, torch.zeros_like(A))
    tensors = {"A": A, "M": M, "Z": Z, "a": to_device(A), "b": to_device(B), "m": to_device(M), "z": to_device(Z)}
    st = to_device(torch.full(SHAPE, s, dtype=torch.bfloat16))

    output_tensor, expected = CASES[case](tensors, s, st, scalar)

    assert torch.equal(ttnn.to_torch(output_tensor), expected)
