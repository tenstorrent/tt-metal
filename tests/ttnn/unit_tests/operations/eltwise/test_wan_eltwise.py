# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Eltwise ops at the Wan2.2 per-device activation shapes (Blackhole LoudBox 2x4: local M 8192, TP width 2560,
ffn width 6912; 2x2: local M 16384).

test_wan_eltwise checks PCC against torch. test_wan_eltwise_perf prints the device kernel duration of the op's
program from the real-time profiler (WANELT_N samples, default 10, after 2 warm-up calls)."""

import os
import statistics

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

TILE = ttnn.TILE_LAYOUT
DT = {"bf16": (torch.bfloat16, ttnn.bfloat16), "fp32": (torch.float32, ttnn.float32)}


def _tt(device, t, dtype):
    return ttnn.from_torch(t, dtype=DT[dtype][1], layout=TILE, device=device)


def _rand(shape, dtype, lo=-3.0, hi=3.0):
    return (torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo).to(DT[dtype][0])


def _gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def _unary(op, golden, lo=-3.0, hi=3.0):
    def build(device, shape, dtype):
        x = _rand(shape, dtype, lo, hi)
        tx = _tt(device, x, dtype)
        return (lambda: op(tx)), golden(x.float())

    return build


def _binary(op, golden, other_shape=None, scalar=None):
    def build(device, shape, dtype):
        a = _rand(shape, dtype)
        ta = _tt(device, a, dtype)
        if scalar is not None:
            return (lambda: op(ta, scalar)), golden(a.float(), scalar)
        b = _rand(other_shape(shape) if other_shape else shape, dtype)
        tb = _tt(device, b, dtype)
        return (lambda: op(ta, tb)), golden(a.float(), b.float())

    return build


def _row(shape):
    return (1, 1, 1, shape[-1])


def _col(shape):
    return (1, 1, shape[-2], 1)


def _addcmul(gate_shape, gate_dtype="bf16", value=1.0):
    def build(device, shape, dtype):
        a, b = _rand(shape, dtype), _rand(shape, dtype)
        c = _rand(gate_shape(shape), gate_dtype)
        ta, tb, tc = _tt(device, a, dtype), _tt(device, b, dtype), _tt(device, c, gate_dtype)
        return (lambda: ttnn.addcmul(ta, tb, tc, value=value)), a.float() + value * b.float() * c.float()

    return build


def _lerp_tensor_weight(device, shape, dtype):
    a, b = _rand(shape, dtype), _rand(shape, dtype)
    w = torch.full((1, 1, 1, 1), 5.0, dtype=torch.float32)
    ta, tb, tw = _tt(device, a, dtype), _tt(device, b, dtype), _tt(device, w, "fp32")
    return (lambda: ttnn.lerp(ta, tb, tw)), torch.lerp(a.float(), b.float(), w)


def _lerp_scalar_weight(device, shape, dtype):
    a, b = _rand(shape, dtype), _rand(shape, dtype)
    ta, tb = _tt(device, a, dtype), _tt(device, b, dtype)
    return (lambda: ttnn.lerp(ta, tb, 5.0)), torch.lerp(a.float(), b.float(), 5.0)


def _typecast(to):
    def build(device, shape, dtype):
        x = _rand(shape, dtype)
        tx = _tt(device, x, dtype)
        return (lambda: ttnn.typecast(tx, DT[to][1])), x.float()

    return build


def _cond(shape, dtype):
    return (torch.rand(shape) > 0.5).to(DT[dtype][0])


def _where(true_shape=None, false_shape=None, scalar_false=None):
    def build(device, shape, dtype):
        c = _cond(shape, dtype)
        a = _rand(true_shape(shape) if true_shape else shape, dtype)
        tc, ta = _tt(device, c, dtype), _tt(device, a, dtype)
        if scalar_false is not None:
            return (lambda: ttnn.where(tc, ta, scalar_false)), torch.where(c.bool(), a.float(), scalar_false)
        b = _rand(false_shape(shape) if false_shape else shape, dtype)
        tb = _tt(device, b, dtype)
        return (lambda: ttnn.where(tc, ta, tb)), torch.where(c.bool(), a.float(), b.float())

    return build


def _mac_tst(device, shape, dtype):
    a, c = _rand(shape, dtype), _rand(shape, dtype)
    ta, tc = _tt(device, a, dtype), _tt(device, c, dtype)
    return (lambda: ttnn.mac(ta, 0.5, tc)), a.float() * 0.5 + c.float()


def _mac_tss(device, shape, dtype):
    a = _rand(shape, dtype)
    ta = _tt(device, a, dtype)
    return (lambda: ttnn.mac(ta, 0.5, 1.25)), a.float() * 0.5 + 1.25


def _rowcol(op, golden):
    def build(device, shape, dtype):
        a, b = _rand(_col(shape), dtype), _rand(_row(shape), dtype)
        ta, tb = _tt(device, a, dtype), _tt(device, b, dtype)
        return (lambda: op(ta, tb)), golden(a.float(), b.float())

    return build


def _scalar_shape(shape):
    return (1, 1, 1, 1)


def _identity_copy(device, shape, dtype):
    x = _rand(shape, dtype)
    tx = _tt(device, x, dtype)
    return (lambda: ttnn.clone(tx)), x.float()


# name -> (builder, pcc, dtypes)
OPS = {
    "gelu": (_unary(ttnn.gelu, torch.nn.functional.gelu), 0.9995, ("bf16",)),
    "gelu_fast": (_unary(lambda x: ttnn.gelu(x, fast_and_approximate_mode=True), torch.nn.functional.gelu), 0.998, ("bf16",)),
    "gelu_tanh": (
        _unary(lambda x: ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh), _gelu_tanh),
        0.9995,
        ("bf16",),
    ),
    "silu": (_unary(ttnn.silu, torch.nn.functional.silu), 0.9995, ("bf16",)),
    "sin": (_unary(ttnn.sin, torch.sin, -3.14, 3.14), 0.999, ("bf16",)),
    "cos": (_unary(ttnn.cos, torch.cos, -3.14, 3.14), 0.999, ("bf16",)),
    "typecast_bf16_fp32": (_typecast("fp32"), 0.99999, ("bf16",)),
    "typecast_fp32_bf16": (_typecast("bf16"), 0.9999, ("fp32",)),
    "add": (_binary(ttnn.add, torch.add), 0.9999, ("bf16", "fp32")),
    "mul": (_binary(ttnn.multiply, torch.mul), 0.9999, ("bf16", "fp32")),
    "sub": (_binary(ttnn.subtract, torch.sub), 0.9999, ("bf16", "fp32")),
    "add_scalar": (_binary(ttnn.add, torch.add, scalar=1.0), 0.9999, ("bf16", "fp32")),
    "mul_scalar": (_binary(ttnn.multiply, torch.mul, scalar=0.125), 0.9999, ("bf16", "fp32")),
    "add_rowbcast": (_binary(ttnn.add, torch.add, other_shape=_row), 0.9999, ("bf16", "fp32")),
    "mul_rowbcast": (_binary(ttnn.multiply, torch.mul, other_shape=_row), 0.9999, ("bf16", "fp32")),
    "mul_colbcast": (_binary(ttnn.multiply, torch.mul, other_shape=_col), 0.9999, ("bf16", "fp32")),
    "mul_scalarbcast": (_binary(ttnn.multiply, torch.mul, other_shape=_scalar_shape), 0.9999, ("bf16", "fp32")),
    "mul_rowcol": (_rowcol(ttnn.multiply, torch.mul), 0.9999, ("bf16", "fp32")),
    "where": (_where(), 0.9999, ("bf16", "fp32")),
    "where_rowbcast": (_where(false_shape=_row), 0.9999, ("bf16", "fp32")),
    "where_tts": (_where(scalar_false=0.0), 0.9999, ("bf16", "fp32")),
    "where_tts_rowbcast": (_where(true_shape=_row, scalar_false=0.0), 0.9999, ("bf16", "fp32")),
    "mac_tst": (_mac_tst, 0.9999, ("bf16", "fp32")),
    "mac_tss": (_mac_tss, 0.9999, ("bf16", "fp32")),
    "addcmul": (_addcmul(lambda s: s), 0.9999, ("bf16",)),
    "addcmul_rowbcast": (_addcmul(_row), 0.9999, ("bf16",)),
    "addcmul_rowbcast_value": (_addcmul(_row, value=0.5), 0.9999, ("bf16",)),
    "addcmul_colbcast": (_addcmul(_col), 0.9999, ("bf16",)),
    "lerp_tensor_weight": (_lerp_tensor_weight, 0.9999, ("fp32",)),
    "lerp_scalar_weight": (_lerp_scalar_weight, 0.9999, ("bf16",)),
}

# (M, N): residual stream and ffn at LoudBox 2x4, residual at 2x2, VAE-like narrow tail, small gate-like shape
SHAPES = {
    "m8192n2560": (8192, 2560),
    "m8192n5120": (8192, 5120),
    "m8192n6912": (8192, 6912),
    "m16384n2560": (16384, 2560),
    "m8192n128": (8192, 128),
    "m32n2560": (32, 2560),
}

# Which ops run at which shapes: activations at the ffn shape, residual ops at the residual shape.
SHAPE_SETS = {
    "gelu": ["m8192n6912", "m8192n5120"],
    "gelu_fast": ["m8192n6912"],
    "gelu_tanh": ["m8192n6912", "m8192n5120"],
    "silu": ["m8192n6912", "m8192n5120", "m32n2560"],
    "sin": ["m8192n128"],
    "cos": ["m8192n128"],
    "typecast_bf16_fp32": ["m8192n2560", "m32n2560"],
    "typecast_fp32_bf16": ["m8192n2560", "m32n2560"],
    "lerp_tensor_weight": ["m8192n128", "m8192n2560"],
    "lerp_scalar_weight": ["m8192n128"],
    "mul_scalarbcast": ["m8192n2560"],
    "mul_rowcol": ["m8192n2560"],
    "where": ["m8192n2560"],
    "where_rowbcast": ["m8192n2560"],
    "where_tts": ["m8192n2560"],
    "where_tts_rowbcast": ["m8192n2560"],
    "mac_tst": ["m8192n2560"],
    "mac_tss": ["m8192n2560"],
    "addcmul_colbcast": ["m8192n2560"],
}
DEFAULT_SHAPES = ["m8192n2560", "m16384n2560", "m8192n5120"]


def _cases(perf=False):
    out = []
    for name, (_, _, dtypes) in OPS.items():
        for sh in SHAPE_SETS.get(name, DEFAULT_SHAPES):
            for dt in dtypes:
                out.append(pytest.param(name, sh, dt, id=f"{name}-{sh}-{dt}"))
    return out


@pytest.mark.parametrize("name, shape_id, dtype", _cases())
def test_wan_eltwise(device, name, shape_id, dtype):
    build, pcc, _ = OPS[name]
    m, n = SHAPES[shape_id]
    torch.manual_seed(0)
    call, golden = build(device, (1, 1, m, n), dtype)
    out = ttnn.to_torch(call())
    assert_with_pcc(golden, out.float().reshape(golden.shape) if out.shape != golden.shape else out.float(), pcc)


@pytest.mark.parametrize("name, shape_id, dtype", _cases())
def test_wan_eltwise_perf(device, name, shape_id, dtype):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active")
    build, _, _ = OPS[name]
    m, n = SHAPES[shape_id]
    n_samples = int(os.environ.get("WANELT_N", "10"))
    torch.manual_seed(0)
    call, _ = build(device, (1, 1, m, n), dtype)
    for _ in range(2):
        call()
    ttnn.synchronize_device(device)
    ns = []
    srcs = None
    for _ in range(n_samples):
        _, rec = profile_realtime_program(device, call, collect_all=True)
        rec = [r for r in rec if r["duration_ns"] > 0]
        assert rec, "no program record"
        r = max(rec, key=lambda r: r["duration_ns"])
        ns.append(r["duration_ns"])
        srcs = r["kernel_sources"]
    tag = f"{name}-{shape_id}-{dtype}"
    for i, v in enumerate(ns):
        print(f"WANELT-SAMPLE {tag} i={i} duration_ns={v:.0f}")
    print(f"WANELT-MEDIAN {tag} duration_ns={statistics.median(ns):.0f}")
    print(f"WANELT-KERNELS {tag} {[os.path.basename(s) for s in srcs]}")
