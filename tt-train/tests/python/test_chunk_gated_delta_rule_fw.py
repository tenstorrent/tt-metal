# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.ops.metal.chunk_gated_delta_rule_fw: the training forward that also returns what the backward consumes.

    q, k [B, 1, T, HK*K]      v [B, 1, T, HV*V]      g, beta [B, 1, T, HV]      initial_state [B, HV, K, V]
    o [B, 1, T, HV*V]         final_state [B, HV, K, V]
    h [B*HV, NC, K, V]        v_new [B*HV, 1, T, V]  g_cumsum [B*HV, 1, T, 1]   A [B*HV, 1, T, C]

Pinned here:
1. Every output against the float64 chunked reference, including a Qwen3.8-sized shape.
2. The head-major output is exactly the token-major output, permuted; the other outputs are unchanged.
3. Two live input sets at one shape each get their own answer from one cached program.
4. Shape-contract violations raise before dispatch.
5. A wide-V shape (K=128, V=256) fits L1 and matches the reference.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

import ttnn
import ttml

# pytest.ini uses --import-mode=importlib, which does not put the test directory on sys.path.
_TEST_DIR = str(pathlib.Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
from chunk_gated_delta_rule_reference import assert_close, chunked_forward  # noqa: E402

pytestmark = pytest.mark.requires_device

OUT_NAMES = ("o", "final_state", "h", "v_new", "g_cumsum", "A")
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# (B, T, HK, HV, K, V), chunk_size
SHAPES = [
    ((1, 32, 1, 1, 32, 32), 32),  # single tile, single head
    ((1, 128, 2, 2, 64, 64), 64),  # 2 chunks
    ((1, 128, 2, 4, 64, 64), 32),  # GVA G=2
    ((2, 64, 1, 3, 64, 64), 32),  # GVA G=3, multi-batch
    ((1, 128, 2, 2, 64, 128), 64),  # wide V: K != V
    ((1, 128, 1, 2, 128, 128), 64),  # Qwen head dims
]
SHAPE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in SHAPES]

# Qwen3.8 linear attention on a 2-way tensor-parallel chip: 16/48 key/value heads globally.
QWEN_SHAPE = ((1, 1024, 8, 24, 128, 128), 64)


@pytest.fixture(scope="module")
def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _build_inputs(shape, with_h0, seed=42):
    """Seeded inputs; q/k/v rounded to bf16 (device dtype), q/k L2-normalized along K (caller contract)."""
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    bf = lambda x: x.to(torch.bfloat16).double()
    return {
        "q": bf(l2(rn(B, T, HK, K))),
        "k": bf(l2(rn(B, T, HK, K))),
        "v": bf(rn(B, T, HV, V)),
        "g": F.logsigmoid(rn(B, T, HV)) * 0.5,
        "beta": torch.rand(B, T, HV, generator=gen, dtype=torch.float64),
        "h0": rn(B, HV, K, V) * 0.1 if with_h0 else None,
    }


def _tt(t, device, dtype):
    return ttnn.from_torch(t.to(TORCH_DTYPE[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _run_ttml(x, shape, chunk, gate_dtype, device, scale=None, output_head_major=False):
    B, T, HK, HV, K, V = shape
    flat = lambda t: t.reshape(B, 1, T, -1)
    return ttml.ops.metal.chunk_gated_delta_rule_fw(
        _tt(flat(x["q"]), device, ttnn.bfloat16),
        _tt(flat(x["k"]), device, ttnn.bfloat16),
        _tt(flat(x["v"]), device, ttnn.bfloat16),
        _tt(flat(x["g"]), device, gate_dtype),
        _tt(flat(x["beta"]), device, gate_dtype),
        K,
        initial_state=None if x["h0"] is None else _tt(x["h0"], device, ttnn.float32),
        chunk_size=chunk,
        scale=scale,
        output_head_major=output_head_major,
    )


def _to_host(outs):
    return [ttnn.to_torch(t).double() for t in outs]


def _to_head_major(outs, shape):
    """ttml outputs -> o [BH,T,V], final [B,HV,K,V], h [BH,NC,K,V], v_new [BH,T,V], g_cumsum [BH,T], A [BH,T,C]."""
    B, T, HK, HV, K, V = shape
    o, fs, h, vn, gc, a = outs
    o = o.reshape(B, T, HV, V).permute(0, 2, 1, 3).reshape(B * HV, T, V)
    return [o, fs, h, vn.reshape(B * HV, T, V), gc.reshape(B * HV, T), a.reshape(B * HV, T, -1)]


def _reference(x, shape, chunk, gate_dtype, scale=None):
    """float64 outputs in the head-major layout of `_to_head_major`."""
    B, T, HK, HV, K, V = shape
    G = HV // HK
    gate = lambda t: t.to(TORCH_DTYPE[gate_dtype]).double()
    o, fs, h, vn, gc, a = chunked_forward(
        x["q"].repeat_interleave(G, dim=2),
        x["k"].repeat_interleave(G, dim=2),
        x["v"],
        gate(x["g"]),
        gate(x["beta"]),
        initial_state=x["h0"],
        chunk_size=chunk,
        scale=scale,
    )
    tok = lambda t: t.permute(0, 2, 1, 3).reshape(B * HV, T, -1)
    return [
        tok(o),
        fs,
        h.transpose(1, 2).reshape(B * HV, -1, K, V),
        tok(vn),
        gc.transpose(1, 2).reshape(B * HV, T),
        tok(a),
    ]


def _check_vs_reference(x, shape, chunk, gate_dtype, device, with_h0):
    got = _to_head_major(_to_host(_run_ttml(x, shape, chunk, gate_dtype, device)), shape)
    exp = _reference(x, shape, chunk, gate_dtype)
    for name, g, e in zip(OUT_NAMES, got, exp):
        assert g.shape == e.shape, f"{name}: {tuple(g.shape)} != {tuple(e.shape)}"
        if name == "h" and not with_h0:
            assert torch.count_nonzero(g[:, 0]) == 0, "h[:, 0] must be exactly zero without initial_state"
        assert_close(name, g, e, pcc_min=0.999)


@pytest.mark.parametrize("gate_dtype", [ttnn.float32, ttnn.bfloat16], ids=["gate_fp32", "gate_bf16"])
@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("shape,chunk", SHAPES, ids=SHAPE_IDS)
def test_vs_reference(shape, chunk, with_h0, gate_dtype, device):
    _check_vs_reference(_build_inputs(shape, with_h0), shape, chunk, gate_dtype, device, with_h0)


def test_qwen_shape_vs_reference(device):
    shape, chunk = QWEN_SHAPE
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, ttnn.bfloat16, device, with_h0=True)


def test_wide_v_fits_l1(device):
    """K=128, V=256 at chunk 64: the prep's single-tile scratch CBs keep it inside one core's L1."""
    shape, chunk = (1, 64, 1, 1, 128, 256), 64
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, ttnn.bfloat16, device, with_h0=True)


def test_explicit_scale(device):
    shape, chunk = SHAPES[1]
    x = _build_inputs(shape, with_h0=True)
    got = _to_head_major(_to_host(_run_ttml(x, shape, chunk, ttnn.float32, device, scale=0.3)), shape)
    exp = _reference(x, shape, chunk, ttnn.float32, scale=0.3)
    assert_close("o", got[0], exp[0], pcc_min=0.999)


@pytest.mark.parametrize("shape,chunk", [SHAPES[3], SHAPES[4]], ids=[SHAPE_IDS[3], SHAPE_IDS[4]])
def test_head_major_output_is_permuted_token_major(shape, chunk, device):
    B, T, HK, HV, K, V = shape
    x = _build_inputs(shape, with_h0=True, seed=2)
    token_major = _to_host(_run_ttml(x, shape, chunk, ttnn.bfloat16, device))
    head_major = _to_host(_run_ttml(x, shape, chunk, ttnn.bfloat16, device, output_head_major=True))
    assert tuple(head_major[0].shape) == (B * HV, 1, T, V)
    tm = token_major[0].reshape(B, T, HV, V).permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    assert torch.equal(head_major[0], tm)
    for name, hm, tmj in zip(OUT_NAMES[1:], head_major[1:], token_major[1:]):
        assert torch.equal(hm, tmj), f"{name}: differs between output layouts"


def test_two_live_input_sets(device):
    shape, chunk = SHAPES[2]
    xa, xb = _build_inputs(shape, True, seed=1), _build_inputs(shape, True, seed=2)
    a1 = _to_host(_run_ttml(xa, shape, chunk, ttnn.float32, device))
    entries = device.num_program_cache_entries()
    b1 = _to_host(_run_ttml(xb, shape, chunk, ttnn.float32, device))
    a2 = _to_host(_run_ttml(xa, shape, chunk, ttnn.float32, device))
    assert device.num_program_cache_entries() == entries, "a second input set at the same shape recompiled"
    for name, ta, tb, ta2 in zip(OUT_NAMES, a1, b1, a2):
        assert torch.equal(ta, ta2), f"{name}: rerun with input set A changed"
        assert not torch.equal(ta, tb), f"{name}: input sets A and B gave the same answer"


def _zero_inputs(device, B=1, T=64, HK=1, HV=1, K=32, V=32, gate_dtype=ttnn.float32):
    z = lambda *s, dt=ttnn.bfloat16: _tt(torch.zeros(*s), device, dt)
    return [
        z(B, 1, T, HK * K),
        z(B, 1, T, HK * K),
        z(B, 1, T, HV * V),
        z(B, 1, T, HV, dt=gate_dtype),
        z(B, 1, T, HV, dt=gate_dtype),
    ]


@pytest.mark.parametrize(
    "kwargs,overrides,match",
    [
        ({"chunk_size": 48}, {}, "chunk_size"),
        ({"chunk_size": 64}, {"T": 96}, "multiple of chunk_size"),
        ({"chunk_size": 32}, {"HV": 3, "HK": 2}, "multiple of"),
        ({"chunk_size": 32}, {"HV": 33}, "HV=33"),
    ],
    ids=["chunk_48", "T_not_multiple", "HV_not_multiple_of_HK", "HV_over_32"],
)
def test_rejects_bad_shapes(kwargs, overrides, match, device, expect_error):
    dims = {"T": 64, "HK": 1, "HV": 1, "K": 32, "V": 32, **overrides}
    args = _zero_inputs(device, **dims)
    with expect_error(RuntimeError, match):
        ttml.ops.metal.chunk_gated_delta_rule_fw(*args, dims["K"], **kwargs)


def test_rejects_mixed_gate_dtypes(device, expect_error):
    q, k, v, g, _ = _zero_inputs(device)
    beta = _tt(torch.zeros(1, 1, 64, 1), device, ttnn.bfloat16)
    with expect_error(RuntimeError, "beta must be"):
        ttml.ops.metal.chunk_gated_delta_rule_fw(q, k, v, g, beta, 32)
