# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 softcap cases, beside the unary category suites."""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole


_GENERATED_SOFTCAP_VALUES = [1.52587890625e-05, 4.0, 5.0, 25.0, 30.0, 50.0, 65536.0, 1e30, 3e38]


_GENERATED_SOFTCAP_CLASSES = {
    "+0": 0.0,
    "-0": -0.0,
    "+inf": float("inf"),
    "-inf": -float("inf"),
    "nan": float("nan"),
    "+1": 1.0,
    "-1": -1.0,
    "-100": -100.0,
    "+200": 200.0,
}


# Input class -> the result's class at beta = 5.0: torch's, except where the stock path and torch
# differ (signed zero, subnormal inputs read as zero) and the kernel keeps the stock one.
_GENERATED_SOFTCAP_SPECIAL_RESULTS = {
    "+0": "+0",
    "+1": "finite",
    "+200": "finite",
    "+inf": "finite",
    "-0": "+0",  # torch gives -0; this kernel stores +0
    "-1": "finite",
    "-100": "finite",
    "-inf": "finite",
    "nan": "nan",  # NaN, as torch gives; the BF16 pack stores a NaN as +inf, which the device reads back
}


def _generated_softcap_run(device, values, beta):
    """ttnn.softcap of a 1-D BF16 tensor, padded with ones to whole tiles."""
    n = values.numel()
    padded = torch.cat([values, torch.ones((-n) % 1024, dtype=values.dtype)])
    tensor = ttnn.from_torch(padded.reshape(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.softcap(tensor, beta)).reshape(-1)[:n]


@pytest.mark.skipif(not is_blackhole(), reason="softcap SFPU op is implemented for Blackhole only")
@pytest.mark.parametrize("beta", _GENERATED_SOFTCAP_VALUES)
def test_softcap_sfpu_generated_bf16_sweep(device, beta):
    """Every finite BF16 encoding, in pure ULP against the exact result of the input as the SFPU reads it (a
    subnormal as 0, as stock does); the leaf has no cancellation, so float64 holds it far below 1 ULP."""
    bits = torch.arange(65536, dtype=torch.int32)
    every = (bits << 16).view(torch.float32)
    every = every[torch.isfinite(every)].to(torch.bfloat16)
    actual = _generated_softcap_run(device, every, beta).to(torch.float64)
    x = every.to(torch.float64)
    x = torch.where(x.abs() < 2.0**-126, torch.zeros_like(x), x)
    exact = beta * torch.tanh(x / beta)
    normal = (exact.abs() >= 2.0**-126) & (exact.abs() <= 3.3895313892515355e38)
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs().clamp(min=2.0**-126))) - 7)
    assert ((actual - exact).abs() / ulp)[normal].max().item() < 1.0
    # Below the smallest normal the result is 0, or the smallest normal it rounds to.
    tiny = ~normal & (exact.abs() < 2.0**-126)
    assert bool(torch.all(actual[tiny].abs() <= 2.0**-126))


@pytest.mark.skipif(not is_blackhole(), reason="softcap SFPU op is implemented for Blackhole only")
def test_softcap_sfpu_generated_specials(device):
    """NaN gives NaN (stored as +inf); the other classes as torch, or as the stock path where the two differ."""
    names = list(_GENERATED_SOFTCAP_SPECIAL_RESULTS)
    values = torch.tensor([_GENERATED_SOFTCAP_CLASSES[key] for key in names], dtype=torch.float32)
    actual = _generated_softcap_run(device, values.to(torch.bfloat16), 5.0).to(torch.float32)

    def kind(v):
        if torch.isnan(v):
            return "nan"
        if torch.isinf(v):
            return "+inf" if v > 0 else "-inf"
        if v == 0:
            return "-0" if torch.signbit(v) else "+0"
        return "finite"

    # The BF16 pack stores a NaN as +inf, and the SFPU a subnormal result as +0.
    stored = {"nan": "+inf", "subnormal": "+0"}
    got = {key: kind(actual[i]) for i, key in enumerate(names)}
    want = {key: stored.get(kind, kind) for key, kind in _GENERATED_SOFTCAP_SPECIAL_RESULTS.items()}
    assert got == want
