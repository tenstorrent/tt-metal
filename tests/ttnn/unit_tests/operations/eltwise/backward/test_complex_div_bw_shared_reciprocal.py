# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""div_bw on ComplexTensors builds one reciprocal of the divisor, not three.

Complex divide is multiply-by-reciprocal, and all three divisions in this op divide
by the same tensor. The reciprocal of a conjugate is the conjugate of the reciprocal,
so the conjugated division shares it too.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)
DTYPES = [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)]


def _c(re, im, dtype, torch_dtype, device):
    def t(v):
        return ttnn.from_torch(
            torch.full(SHAPE, v, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    return ttnn.complex_tensor(t(re), t(im))


@pytest.mark.parametrize("dtype, torch_dtype", DTYPES)
def test_complex_div_bw_matches_the_closed_form(device, dtype, torch_dtype):
    grad, a, b = (2.0, 1.0), (3.0, -4.0), (1.0, 2.0)
    out = ttnn.div_bw(
        _c(*grad, dtype, torch_dtype, device),
        _c(*a, dtype, torch_dtype, device),
        _c(*b, dtype, torch_dtype, device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g, ta, tb = complex(*grad), complex(*a), complex(*b)
    want = [g / tb.conjugate(), -g * (ta / tb / tb).conjugate()]
    tol = 0.05 if dtype == ttnn.bfloat16 else 1e-4
    for got, w in zip(out, want):
        r = ttnn.to_torch(got.real).float().flatten()[0].item()
        i = ttnn.to_torch(got.imag).float().flatten()[0].item()
        assert abs(r - w.real) <= tol * max(1.0, abs(w.real)), f"real {r} vs {w.real}"
        assert abs(i - w.imag) <= tol * max(1.0, abs(w.imag)), f"imag {i} vs {w.imag}"


@pytest.mark.parametrize("dtype, torch_dtype", DTYPES)
def test_complex_div_bw_still_writes_nan_when_the_divisor_is_zero(device, dtype, torch_dtype):
    out = ttnn.div_bw(
        _c(2.0, 1.0, dtype, torch_dtype, device),
        _c(3.0, -4.0, dtype, torch_dtype, device),
        _c(0.0, 0.0, dtype, torch_dtype, device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for got in out:
        for half in (got.real, got.imag):
            v = ttnn.to_torch(half).float().flatten()[0].item()
            # bfloat16 Dest cannot carry NaN and returns an infinity instead.
            assert v != v or v in (float("inf"), float("-inf")), f"expected NaN or inf, got {v}"


def test_complex_div_bw_dispatch_count(device):
    grad = _c(2.0, 1.0, ttnn.bfloat16, torch.bfloat16, device)
    a = _c(3.0, -4.0, ttnn.bfloat16, torch.bfloat16, device)
    b = _c(1.0, 2.0, ttnn.bfloat16, torch.bfloat16, device)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    try:
        ttnn.div_bw(grad, a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    finally:
        captured = ttnn.graph.end_graph_capture()
    n = sum(
        1
        for x in captured
        if x.get("node_type") == "function_start" and str(x.get("params", {}).get("name", "")).endswith("DeviceOperation")
    )
    assert n == 44, f"expected 44 device operations, got {n}"
