# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 gelu_tanh cases, beside the unary category suites."""

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.eltwise.test_gelu_fw_variants import _generated_gelu_tanh_assert


_GENERATED_GELU_TANH_CLASSES = {
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


# Input class -> the result's class: torch's, except where the stock path and torch differ (signed zero, subnormal
# inputs read as zero) and the kernel keeps the stock one.
_GENERATED_GELU_TANH_SPECIAL_RESULTS = {
    "+0": "+0",
    "+1": "finite",
    "+200": "finite",
    "+inf": "+inf",
    "-0": "+0",  # torch gives -0; this kernel stores +0
    "-1": "finite",
    "-100": "+0",  # torch gives -0; this kernel stores +0
    "-inf": "nan",  # NaN, as torch gives; the BF16 pack stores a NaN as +inf, which the device reads back
    "nan": "nan",  # NaN, as torch gives; the BF16 pack stores a NaN as +inf, which the device reads back
}


_GENERATED_GELU_TANH_TINY = torch.finfo(torch.float32).tiny  # 2^-126, the smallest normal


_GENERATED_GELU_TANH_BF16_MAX = torch.finfo(torch.bfloat16).max


# Results the kernel may store as 0: within this relative distance above the smallest normal, and at every input at or
# below the next, where the exp leaf's e^-|z| is 0.
_GENERATED_GELU_TANH_BOUNDARY_BAND = 2.0**-12


_GENERATED_GELU_TANH_LOGISTIC_FLUSH_INPUT = -10.0625


def _generated_gelu_tanh_run(device, values):
    """ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh) of a 1-D BF16 tensor x, padded with ones to whole 32x32 tiles."""
    n = values.numel()
    padded = torch.cat([values, torch.ones((-n) % (32 * 32), dtype=values.dtype)])
    tensor = ttnn.from_torch(padded.reshape(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.gelu(tensor, variant=ttnn.GeluVariant.Tanh)).reshape(-1)[:n]


@pytest.mark.skipif(
    not is_blackhole(),
    reason="gelu_tanh's generated SFPU kernel is certified on Blackhole only; Wormhole keeps the stock body",
)
def test_gelu_tanh_sfpu_generated_bf16_sweep(device):
    """Every finite BF16 encoding, scored as the certificate scores it."""
    bits = torch.arange(65536, dtype=torch.int32)
    every = (bits << 16).view(torch.float32)
    every = every[torch.isfinite(every)].to(torch.bfloat16)
    _generated_gelu_tanh_assert(every, _generated_gelu_tanh_run(device, every))


@pytest.mark.skipif(
    not is_blackhole(),
    reason="gelu_tanh's generated SFPU kernel is certified on Blackhole only; Wormhole keeps the stock body",
)
def test_gelu_tanh_sfpu_generated_bfloat8_b(device):
    """BFLOAT8_B data runs the generated kernel too (16-bit DEST holds it as BF16): the result tracks the exact one
    of the inputs as BFLOAT8_B holds them, to BFLOAT8_B's block precision."""
    generator = torch.Generator().manual_seed(0)
    values = (8 * torch.randn((1, 1, 64, 64), generator=generator)).to(torch.bfloat16)
    tensor = ttnn.from_torch(values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.to_torch(tensor).to(torch.float64).flatten()
    actual = ttnn.to_torch(ttnn.gelu(tensor, variant=ttnn.GeluVariant.Tanh)).to(torch.float64).flatten()
    exact = x / (1 + torch.exp(-(1.5957691216057308 * x + 0.07135481627260025 * x**3)))
    pcc = torch.corrcoef(torch.stack([actual, exact]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc:.6f} against the exact result"


@pytest.mark.skipif(
    not is_blackhole(),
    reason="gelu_tanh's generated SFPU kernel is certified on Blackhole only; Wormhole keeps the stock body",
)
def test_gelu_tanh_sfpu_generated_specials(device):
    """NaN gives NaN (stored as +inf); the other classes as torch, or as the stock path where the two differ."""
    names = list(_GENERATED_GELU_TANH_SPECIAL_RESULTS)
    values = torch.tensor([_GENERATED_GELU_TANH_CLASSES[key] for key in names], dtype=torch.float32)
    actual = _generated_gelu_tanh_run(device, values.to(torch.bfloat16)).to(torch.float32)

    def kind(v):
        if torch.isnan(v):
            return "nan"
        if torch.isinf(v):
            return "+inf" if v > 0 else "-inf"
        if v == 0:
            return "-0" if torch.signbit(v) else "+0"
        return "finite"

    # The BF16 pack stores a NaN as +inf.
    stored = {"nan": "+inf"}
    got = {key: kind(actual[i]) for i, key in enumerate(names)}
    want = {key: stored.get(cls, cls) for key, cls in _GENERATED_GELU_TANH_SPECIAL_RESULTS.items()}
    assert got == want
