# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Declared derivative mathematics; no fitted-output oracle.
# Unit-gradient accuracy only. Arbitrary-gradient and special parity belong
# to the existing observed-TTNN qualification flow.
import numpy as np
import pytest
import torch
import ttnn

_TARGETS = ("blackhole", "wormhole_b0")
_REFERENCE_SHA256 = "4e745f5d8d2712508ab8f6c117788448544013ffba80f164cb10a11936885ea9"


def _declared_derivative(x):
    import math

    exp = np.exp
    pi = np.pi
    sqrt = np.sqrt
    return np.broadcast_to(np.asarray(2 * exp(-(x**2)) / sqrt(pi), dtype=np.float64), x.shape)


def _bf16_round_ftz(values):
    rounded = torch.from_numpy(values).to(torch.bfloat16).to(torch.float64).numpy()
    subnormal = (np.abs(rounded) < 2.0**-126) & (rounded != 0.0)
    return np.where(subnormal, np.copysign(0.0, rounded), rounded)


def _ulp_spacing(values):
    words = (np.abs(values).astype(np.float32).view(np.uint32) >> 16).astype(np.uint32)
    upper = (np.minimum(words + 1, 0x7F80) << 16).view(np.float32)
    lower = (words << 16).view(np.float32)
    spacing = (upper - lower).astype(np.float64)
    return np.where(np.isinf(upper), np.float64(2.0**120), spacing)


def _assert_unit_gradient_math(raw, actual):
    with np.errstate(all="ignore"):
        values = (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        golden = _declared_derivative(values)
        got = (actual.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    finite = (raw & np.uint16(0x7F80)) != 0x7F80
    assert np.isfinite(golden[finite]).all()
    rounded = _bf16_round_ftz(golden[finite])
    finite_got = got[finite]
    assert np.all(np.isposinf(finite_got[np.isposinf(rounded)]))
    assert np.all(np.isneginf(finite_got[np.isneginf(rounded)]))
    numeric = np.isfinite(rounded)
    golden_ftz = np.where(rounded[numeric] == 0, 0.0, golden[finite][numeric])
    pure_ulp = np.abs(golden_ftz - finite_got[numeric]) / _ulp_spacing(rounded[numeric])
    assert np.isfinite(pure_ulp).all()
    assert not pure_ulp.size or float(pure_ulp.max()) < 1.0
    # Declared nonfinite input expectations are diagnostic, not a fourth gate.
    same_class = (
        (np.isnan(golden) & np.isnan(got))
        | (np.isposinf(golden) & np.isposinf(got))
        | (np.isneginf(golden) & np.isneginf(got))
        | (np.isfinite(golden) & np.isfinite(got))
    )
    return int(np.count_nonzero(~same_class & ~finite))


@pytest.mark.skipif(ttnn.get_arch_name() not in _TARGETS, reason="generated target unavailable")
def test_erf_bw_bf16_exhaustive(device):
    raw = np.arange(65536, dtype=np.uint16)
    host = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(256, 256)
    gradient = torch.from_numpy(np.full(raw.shape, 0x3F80, dtype=np.uint16)).view(torch.bfloat16).reshape(256, 256)
    device_input = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    device_gradient = ttnn.from_torch(gradient, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(
        ttnn.erf_bw(device_gradient, device_input, **{"memory_config": ttnn.DRAM_MEMORY_CONFIG})[0]
    ).to(torch.bfloat16)
    actual = result.contiguous().view(torch.uint16).cpu().numpy().reshape(-1)
    mismatches = _assert_unit_gradient_math(raw, actual)
    print(
        "Declared nonfinite-input expectation mismatches (diagnostic observations):",
        mismatches,
        "; observed TTNN qualification remains specials authority.",
    )
