# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite: the forward written with existing ttnn ops,
which also returns what the backward consumes.

    q, k [B, 1, T, HK*K]      v [B, 1, T, HV*V]      g, beta [B, 1, T, HV]      initial_state [B, HV, K, V]
    o [B*HV, 1, T, V]         final_state [B, HV, K, V]
    h [B*HV, NC, K, V]        v_new [B*HV, 1, T, V]  A [B*HV, 1, T, C]

Every output is compared with the float64 reference (ttnn's recurrent golden) under per-output bounds of 2-3x the
measured error.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
import ttml
from tests.ttnn.utils_for_testing import assert_numeric_metrics

# conftest.py puts this directory on sys.path (pytest.ini uses --import-mode=importlib).
from chunk_gated_delta_rule_reference import (  # noqa: E402
    DECAY_GATE,
    QWEN_TP4_SHAPE,
    STRONG_GATE,
    WEAK_GATE,
    reference_forward,
)

pytestmark = pytest.mark.requires_device

OUT_NAMES = ("o", "final_state", "h", "v_new", "A")
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# Relative RMS (Frobenius) bound per output, 2-3x the error measured on Blackhole (o ~2.7e-3, A ~4e-4 at the Qwen3.8
# TP4 shape).
RMS_MAX = {"o": 8e-3, "final_state": 8e-3, "h": 8e-3, "v_new": 5e-3, "A": 2e-3}
PCC_MIN = 0.9999
# Last chunk's error over the first chunk's, with no decay along a 128-chunk scan (measured 1.3).
GROWTH_MAX = 2.0


def _case(
    shape,
    chunk,
    gate_dtype=ttnn.float32,
    with_h0=True,
    gate_scale=WEAK_GATE,
    scale=None,
    aligned_keys=False,
    rms_max=None,
):
    return dict(
        shape=shape,
        chunk=chunk,
        gate_dtype=gate_dtype,
        with_h0=with_h0,
        gate_scale=gate_scale,
        scale=scale,
        aligned_keys=aligned_keys,
        rms_max={**RMS_MAX, **(rms_max or {})},
    )


# (B, T, HK, HV, K, V)
CASES = [
    # The LoRA run: Qwen3.8 on one chip of 4-way tensor parallelism, bf16 gates, no initial state.
    pytest.param(_case(*QWEN_TP4_SHAPE, ttnn.bfloat16, with_h0=False), id="qwen_tp4_c64"),
    # The same shape at chunk 32 (the prep's forward-substitution inverse), fp32 gates and an initial state.
    pytest.param(_case(QWEN_TP4_SHAPE[0], 32), id="qwen_tp4_c32_h0"),
    pytest.param(_case((1, 32, 1, 1, 32, 32), 32, with_h0=False), id="one_tile"),
    # Every value head on one key head: the backward's head limit.
    pytest.param(_case((1, 64, 1, 32, 32, 32), 32, ttnn.bfloat16), id="gva32"),
    # Two batches, G = 3, and K != V with odd tile counts either way.
    pytest.param(_case((2, 128, 1, 3, 96, 160), 64), id="b2_k96_v160"),
    pytest.param(_case((1, 64, 2, 4, 160, 32), 32), id="k160_v32"),
    pytest.param(_case((1, 128, 2, 4, 64, 64), 32, gate_scale=DECAY_GATE), id="decay_gate"),
    # o measured 7.3e-3.
    pytest.param(_case((1, 128, 2, 2, 64, 64), 64, gate_scale=STRONG_GATE, rms_max={"o": 2e-2}), id="strong_gate"),
    # k_i . k_j ~ 1 for every pair, so N is dense rather than near zero (A measured 1.5e-3).
    pytest.param(_case((1, 256, 1, 2, 64, 64), 64, aligned_keys=True, rms_max={"A": 5e-3}), id="aligned_keys"),
    # o is linear in q, so a wrong or ignored scale keeps the PCC at 1; the relative RMS sees it.
    pytest.param(_case((1, 128, 2, 2, 64, 64), 64, scale=0.3), id="scale_0.3"),
]


@pytest.fixture(scope="module")
def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _build_inputs(shape, with_h0, seed=42, gate_scale=WEAK_GATE, aligned_keys=False):
    """Seeded inputs; q/k/v rounded to bf16, q/k L2-normalized along K (the caller's contract)."""
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    bf = lambda x: x.to(torch.bfloat16).double()
    q, k = l2(rn(B, T, HK, K)), l2(rn(B, T, HK, K))
    if aligned_keys:
        base = l2(rn(B, 1, HK, K))
        q, k = l2(base + 0.05 * rn(B, T, HK, K)), l2(base + 0.05 * rn(B, T, HK, K))
    return {
        "q": bf(q),
        "k": bf(k),
        "v": bf(rn(B, T, HV, V)),
        "g": F.logsigmoid(rn(B, T, HV)) * gate_scale,
        "beta": torch.rand(B, T, HV, generator=gen, dtype=torch.float64),
        "h0": rn(B, HV, K, V) * 0.1 if with_h0 else None,
    }


def _tt(t, device, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.to(TORCH_DTYPE[dtype]), dtype=dtype, layout=layout, device=device)


def _upload(x, shape, gate_dtype, device):
    B, T, HK, HV, K, V = shape
    flat = lambda t: t.reshape(B, 1, T, -1)
    return {
        "q": _tt(flat(x["q"]), device, ttnn.bfloat16),
        "k": _tt(flat(x["k"]), device, ttnn.bfloat16),
        "v": _tt(flat(x["v"]), device, ttnn.bfloat16),
        "g": _tt(flat(x["g"]), device, gate_dtype),
        "beta": _tt(flat(x["beta"]), device, gate_dtype),
        "initial_state": None if x["h0"] is None else _tt(x["h0"], device, ttnn.float32),
    }


def _dispatch(t, key_head_dim, chunk, **kw):
    return ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        key_head_dim,
        initial_state=t["initial_state"],
        chunk_size=chunk,
        **kw,
    )


def _reference(x, chunk, gate_dtype, scale=None):
    gate = lambda t: t.to(TORCH_DTYPE[gate_dtype]).double()
    return reference_forward(
        x["q"], x["k"], x["v"], gate(x["g"]), gate(x["beta"]), chunk_size=chunk, initial_state=x["h0"], scale=scale
    )


def _host(outs, shape, chunk):
    """ttml outputs as o [BH,T,V], final_state [B,HV,K,V], h [BH,NC,K,V], v_new [BH,T,V], A [BH,T,C]."""
    B, T, HK, HV, K, V = shape
    o, fs, h, vn, a = (ttnn.to_torch(t).double() for t in outs)
    return o.reshape(B * HV, T, V), fs, h, vn.reshape(B * HV, T, V), a.reshape(B * HV, T, chunk)


@pytest.mark.parametrize("case", CASES)
def test_vs_reference(case, device):
    shape, chunk = case["shape"], case["chunk"]
    B, T, HK, HV, K, V = shape
    x = _build_inputs(shape, case["with_h0"], gate_scale=case["gate_scale"], aligned_keys=case["aligned_keys"])
    outs = _dispatch(_upload(x, shape, case["gate_dtype"], device), K, chunk, scale=case["scale"])

    # The saved state is the backward's contract: these shapes, all fp32 TILE.
    expected = {
        "o": (B * HV, 1, T, V),
        "final_state": (B, HV, K, V),
        "h": (B * HV, T // chunk, K, V),
        "v_new": (B * HV, 1, T, V),
        "A": (B * HV, 1, T, chunk),
    }
    for name, tensor in zip(OUT_NAMES, outs):
        assert tuple(tensor.shape) == expected[name], f"{name}: {tuple(tensor.shape)} != {expected[name]}"
        assert tensor.dtype == ttnn.float32 and tensor.layout == ttnn.TILE_LAYOUT, name

    got = _host(outs, shape, chunk)
    if not case["with_h0"]:
        assert torch.count_nonzero(got[2][:, 0]) == 0, "h[:, 0] must be exactly zero without initial_state"
    above = torch.arange(chunk)[None, :] > (torch.arange(T) % chunk)[:, None]
    assert torch.count_nonzero(got[4][:, above]) == 0, "A has non-zeros above the diagonal"
    for name, g, e in zip(OUT_NAMES, got, _reference(x, chunk, case["gate_dtype"], case["scale"])):
        passed, message = assert_numeric_metrics(
            e,
            g,
            pcc_threshold=PCC_MIN,
            frobenius_threshold=case["rms_max"][name],
            check_allclose=False,
            assert_on_fail=False,
        )
        assert passed, f"{name}: {message}"


def test_error_does_not_grow_along_the_scan(device):
    """No decay carries every chunk's error into the next. Over 128 chunks the error of the last chunk stays within
    a small factor of the first chunk's."""
    shape, chunk = (1, 8192, 1, 2, 128, 128), 64
    B, T, HK, HV, K, V = shape
    x = _build_inputs(shape, with_h0=True, gate_scale=0.0)
    got = _host(_dispatch(_upload(x, shape, ttnn.float32, device), K, chunk), shape, chunk)
    exp = _reference(x, chunk, ttnn.float32)
    rel = lambda a, e: ((a - e).norm() / e.norm()).item()
    first, last = slice(0, chunk), slice(T - chunk, T)
    for name, idx in (("o", 0), ("v_new", 3)):
        e_first = rel(got[idx][:, first], exp[idx][:, first])
        e_last = rel(got[idx][:, last], exp[idx][:, last])
        assert e_last <= GROWTH_MAX * e_first, f"{name}: last-chunk error {e_last:.3e} vs first {e_first:.3e}"
    e_first, e_last = rel(got[2][:, 1], exp[2][:, 1]), rel(got[2][:, -1], exp[2][:, -1])
    assert e_last <= GROWTH_MAX * e_first, f"h: last-chunk error {e_last:.3e} vs first {e_first:.3e}"


def test_bf16_output_only_rounds_o(device):
    shape, chunk = (1, 128, 2, 4, 64, 64), 32
    t = _upload(_build_inputs(shape, with_h0=True, seed=11), shape, ttnn.float32, device)
    full = _dispatch(t, shape[4], chunk)
    packed = _dispatch(t, shape[4], chunk, output_dtype=ttnn.bfloat16)
    assert packed[0].dtype == ttnn.bfloat16
    o32, o16 = ttnn.to_torch(full[0]).double(), ttnn.to_torch(packed[0]).double()
    assert torch.all((o16 - o32).abs() <= o32.abs() * 2**-7), "o differs from the fp32 o by more than bf16 rounding"
    for name, a, b in zip(OUT_NAMES[1:], full[1:], packed[1:]):
        assert b.dtype == ttnn.float32, f"{name}: output_dtype changed its dtype"
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"{name}: output_dtype changed it"


def test_save_for_backward_false_skips_only_the_saved_state(device):
    shape, chunk = (1, 128, 2, 4, 64, 64), 32
    t = _upload(_build_inputs(shape, with_h0=True, seed=5), shape, ttnn.float32, device)
    saved = _dispatch(t, shape[4], chunk)
    unsaved = _dispatch(t, shape[4], chunk, save_for_backward=False)
    assert unsaved[2] is None and unsaved[3] is None and unsaved[4] is None
    for name, a, b in zip(OUT_NAMES[:2], saved[:2], unsaved[:2]):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"{name}: save_for_backward changed it"


# Every rejection starts from valid inputs (B=1, T=64, HK=1, HV=2, K=V=32, chunk 32) and changes one thing.
def _rejection_inputs(device):
    shape = (1, 64, 1, 2, 32, 32)
    x = _build_inputs(shape, with_h0=True)
    return x, _upload(x, shape, ttnn.float32, device)


def _bf16(device, shape, layout=ttnn.TILE_LAYOUT):
    return _tt(torch.zeros(shape), device, ttnn.bfloat16, layout)


def _f32(device, shape):
    return _tt(torch.zeros(shape), device, ttnn.float32)


REJECTIONS = [
    # attributes
    pytest.param(lambda t, d: {"chunk_size": 16}, "chunk_size must be 32 or 64, got 16", id="chunk16"),
    pytest.param(lambda t, d: {"chunk_size": 128}, "chunk_size must be 32 or 64, got 128", id="chunk128"),
    pytest.param(
        lambda t, d: {"key_head_dim": 48}, "key_head_dim must be a positive multiple of 32, got 48", id="key_dim48"
    ),
    pytest.param(lambda t, d: {"output_dtype": ttnn.uint32}, "output dtype must be FLOAT32 or BFLOAT16", id="out_u32"),
    # placement, dtype and layout
    pytest.param(
        lambda t, d: {"k": ttnn.from_torch(torch.zeros(1, 1, 64, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)},
        "k must be on Device",
        id="host_k",
    ),
    pytest.param(lambda t, d: {"q": _f32(d, (1, 1, 64, 32))}, "q has unsupported dtype", id="q_fp32"),
    pytest.param(
        lambda t, d: {"v": _bf16(d, (1, 1, 64, 64), ttnn.ROW_MAJOR_LAYOUT)}, "v requires TILE layout", id="v_row_major"
    ),
    pytest.param(lambda t, d: {"beta": _bf16(d, (1, 1, 64, 2))}, "beta has unsupported dtype", id="mixed_gates"),
    pytest.param(
        lambda t, d: {"initial_state": _bf16(d, (1, 2, 32, 32))}, "initial_state has unsupported dtype", id="h0_bf16"
    ),
    # shapes
    pytest.param(lambda t, d: {"q": _bf16(d, (1, 64, 32))}, "inputs must be rank 4", id="rank3_q"),
    pytest.param(
        lambda t, d: {"q": _bf16(d, (1, 1, 48, 32)), "k": _bf16(d, (1, 1, 48, 32))},
        "T=48 must be a multiple of chunk_size=32",
        id="unaligned_T",
    ),
    pytest.param(lambda t, d: {"q": _bf16(d, (1, 2, 64, 32))}, r"q must be \[B,1,T,HK\*K\]", id="q_dim1"),
    pytest.param(lambda t, d: {"k": _bf16(d, (1, 1, 64, 64))}, "k .* must match q", id="k_width"),
    pytest.param(
        lambda t, d: {"q": _bf16(d, (1, 1, 64, 96)), "k": _bf16(d, (1, 1, 64, 96))},
        "HK=3 must divide HV=2",
        id="hk3_hv2",
    ),
    pytest.param(lambda t, d: {"v": _bf16(d, (1, 1, 64, 96))}, r"v must be \[B,1,T,HV\*V\]", id="v_width"),
    pytest.param(lambda t, d: {"beta": _f32(d, (1, 1, 32, 2))}, r"g and beta must be \[B,1,T,HV\]", id="beta_T"),
    pytest.param(
        lambda t, d: {"initial_state": _f32(d, (1, 2, 32, 64))}, r"initial_state must be \[B,HV,K,V\]", id="h0_shape"
    ),
]


@pytest.mark.parametrize("change,message", REJECTIONS)
def test_rejects(change, message, device, expect_error):
    _, t = _rejection_inputs(device)
    args = dict(t, key_head_dim=32, chunk_size=32)
    args.update(change(t, device))
    with expect_error(RuntimeError, message):
        ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite(
            args["q"],
            args["k"],
            args["v"],
            args["g"],
            args["beta"],
            args["key_head_dim"],
            initial_state=args["initial_state"],
            chunk_size=args["chunk_size"],
            **({"output_dtype": args["output_dtype"]} if "output_dtype" in args else {}),
        )
