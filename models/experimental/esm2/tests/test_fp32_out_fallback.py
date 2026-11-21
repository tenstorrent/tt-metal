# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: MIT
"""CPU test: the fp32-output fallback only triggers on an unsupported dtype kwarg."""

from __future__ import annotations

import types

import pytest

from models.experimental.esm2.tt.esm2.ttnn_backend import TtnnEsm2


def _backend(fn_name, exc):
    calls = []

    def fn(*args, dtype=None, **kwargs):
        calls.append(dtype)
        if dtype is not None and exc is not None:
            raise exc
        return "out"

    ttnn = types.SimpleNamespace(float32="fp32", **{fn_name: fn})
    be = object.__new__(TtnnEsm2)
    be.ttnn = ttnn
    be._mm_out32_ok = None
    be._mm32_matmul_ok = None
    return be, calls


@pytest.mark.parametrize("name", ["linear", "matmul"])
def test_unsupported_dtype_kwarg_warns_and_falls_back(name):
    be, calls = _backend(name, TypeError("unexpected keyword 'dtype'"))
    call = (lambda: be._linear(1, 2, out_fp32=True)) if name == "linear" else (lambda: be._matmul(1, 2, out_fp32=True))
    with pytest.warns(RuntimeWarning):
        assert call() == "out"
    assert call() == "out"
    assert calls == ["fp32", None, None]


@pytest.mark.parametrize("name", ["linear", "matmul"])
def test_runtime_error_propagates_and_is_not_sticky(name):
    be, calls = _backend(name, RuntimeError("out of memory"))
    call = (lambda: be._linear(1, 2, out_fp32=True)) if name == "linear" else (lambda: be._matmul(1, 2, out_fp32=True))
    with pytest.raises(RuntimeError):
        call()
    flag = be._mm_out32_ok if name == "linear" else be._mm32_matmul_ok
    assert flag is None
