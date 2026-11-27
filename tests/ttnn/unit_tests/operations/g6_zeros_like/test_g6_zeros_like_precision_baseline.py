# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_zeros_like on the production cases.

The op writes a constant, so the baseline is exactness, measured with the standard metrics: PCC,
max/mean abs error, ULP error and the count of elements whose bits (float32) or decoded value
(bfloat8_b) differ from +0.0. Relative RMS (RMS error / std(reference)) and the got/true ratio
are undefined against an all-zero reference; the absolute RMS error is recorded instead.

Cases are the production geometries only (BB-22); the float32 input draws are the ones a value
path through x would break (NaN payloads, inf, -0.0, subnormals).
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc, comp_ulp
from ttnn.operations.g6_zeros_like import SUPPORTED, g6_zeros_like

FP32_SHAPE = (1, 12, 128, 128)
BF8_SHAPE = (5, 1, 32, 256)
SEED = 0

_SPECIAL_BITS = [0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x00000001, 0x807FFFFF, 0x7F7FFFFF]


def _input(kind, shape):
    torch.manual_seed(SEED)
    if kind == "randn":
        return torch.randn(shape, dtype=torch.float32)
    if kind == "random_bits":
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int64).to(torch.int32).view(torch.float32)
    if kind == "special_bits":
        signed = [v - (1 << 32) if v >= (1 << 31) else v for v in _SPECIAL_BITS]
        numel = math.prod(shape)
        pattern = torch.tensor(signed, dtype=torch.int32).repeat(-(-numel // len(signed)))[:numel]
        return pattern.reshape(shape).view(torch.float32)
    raise ValueError(kind)


def _to_device(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _measure(actual, expected, dtype):
    a = actual.to(torch.float32)
    e = expected.to(torch.float32)
    _, pcc = comp_pcc(e, a)
    _, allclose_detail = comp_allclose(e, a)
    ulp_ok, ulp_detail = comp_ulp(e, a, ulp_threshold=0)
    diff = (a - e).abs()
    if dtype == ttnn.float32:
        wrong = int((actual.contiguous().view(torch.int32) != expected.contiguous().view(torch.int32)).sum().item())
    else:
        wrong = int((a != e).sum().item())
    return {
        "pcc": float(pcc),
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "rms_abs_err": float(diff.pow(2).mean().sqrt().item()),
        "ulp_exact": bool(ulp_ok),
        "ulp_detail": ulp_detail,
        "allclose_detail": allclose_detail,
        "wrong_elements": wrong,
        "numel": actual.numel(),
    }


def _report_and_gate(case, m):
    print(
        f"\nPRECISION {case}: pcc={m['pcc']:.6f} max_abs={m['max_abs_err']:.3e} mean_abs={m['mean_abs_err']:.3e} "
        f"rms_abs={m['rms_abs_err']:.3e} wrong={m['wrong_elements']}/{m['numel']} ulp_exact={m['ulp_exact']} "
        f"({m['allclose_detail']}; {m['ulp_detail']}) rel_rms=n/a ratio=n/a (all-zero reference)"
    )
    assert m["wrong_elements"] == 0, f"{case}: {m['wrong_elements']} elements are not +0.0"
    assert m["max_abs_err"] == 0.0
    assert m["ulp_exact"], m["ulp_detail"]
    assert m["pcc"] == 1.0


@pytest.mark.parametrize("input_kind", ["randn", "random_bits", "special_bits"])
@pytest.mark.parametrize("with_optional", [False, True], ids=["new_output", "into_optional"])
def test_precision_fp32(device, input_kind, with_optional):
    torch_x = _input(input_kind, FP32_SHAPE)
    x = _to_device(torch_x, device, ttnn.float32)
    if with_optional:
        o = _to_device(torch.full(FP32_SHAPE, float("nan")), device, ttnn.float32)
        out = g6_zeros_like(x, optional_tensor=o)
        assert out.buffer_address() == o.buffer_address()
    else:
        out = g6_zeros_like(x)
    m = _measure(ttnn.to_torch(out), torch.zeros_like(torch_x), ttnn.float32)
    _report_and_gate(f"float32 {FP32_SHAPE} {'into_optional' if with_optional else 'new_output'} {input_kind}", m)


def test_precision_bfloat8_b(device, expect_error):
    torch.manual_seed(SEED)
    torch_x = torch.randn(BF8_SHAPE, dtype=torch.float32).to(torch.bfloat16)
    x = _to_device(torch_x, device, ttnn.bfloat8_b)
    if ttnn.bfloat8_b not in SUPPORTED["dtype"]:
        with expect_error(NotImplementedError, ""):
            g6_zeros_like(x)
        return
    # Ones in a freed block of the output's size: a page left unwritten decodes to 1, not 0.
    ttnn.deallocate(_to_device(torch.ones(BF8_SHAPE, dtype=torch.bfloat16), device, ttnn.bfloat8_b))
    out = g6_zeros_like(x)
    m = _measure(ttnn.to_torch(out), torch.zeros_like(torch_x), ttnn.bfloat8_b)
    _report_and_gate(f"bfloat8_b {BF8_SHAPE} new_output randn", m)
