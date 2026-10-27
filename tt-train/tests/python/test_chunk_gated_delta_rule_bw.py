# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.ops.metal.chunk_gated_delta_rule_bw: the fused backward of the chunked gated delta rule.

    q, k   [B, 1, T, HK*K]      v [B, 1, T, HV*V]      g, beta [B, 1, T, HV]
    d_out  [B, 1, T, HV*V], or head-major [B*HV, 1, T, V]
    dq, dk [B, 1, T, HK*K]      dv [B, 1, T, HV*V]     dg, dbeta [B, 1, T, HV]
    dh0    [B, HV, K, V]

Pinned here:
1. Every gradient against a float64 autograd oracle of the token-by-token recurrence.
2. Two live input sets at one shape each get their own answer from one cached program.
3. Shape-contract violations raise before dispatch.
4. Fed the saved h / v_new / A of ttml.ops.metal.chunk_gated_delta_rule_fw, every gradient still
   matches the oracle, including a Qwen3.8-sized shape with head-major d_out.
5. Head-major d_out [B*HV, 1, T, V] and rank-3 [B, T, X] inputs give bit-identical gradients.
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
from chunk_gated_delta_rule_reference import assert_close, chunked_forward, recurrent_vjp  # noqa: E402

pytestmark = pytest.mark.requires_device

GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dh0")
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.99}
RMS = {ttnn.float32: 0.02, ttnn.bfloat16: 0.12}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# (B, T, HK, HV, K, V), chunk_size
SHAPES = [
    ((1, 32, 1, 1, 32, 32), 32),  # single tile, single head
    ((1, 128, 2, 2, 64, 64), 64),  # no GVA, 2 chunks
    ((1, 128, 2, 4, 64, 64), 32),  # GVA G=2
    ((2, 64, 1, 3, 64, 64), 32),  # GVA G=3, multi-batch
    ((1, 128, 2, 2, 64, 128), 64),  # wide V: K != V
]
SHAPE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in SHAPES]

# Qwen3.8 linear attention on a 2-way tensor-parallel chip: 16/48 key/value heads globally.
QWEN_SHAPE = ((1, 1024, 8, 24, 128, 128), 64)


@pytest.fixture(scope="module")
def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _build_inputs(shape, state_mode, dtype, seed=42):
    """Seeded inputs, rounded to `dtype` so the oracle sees what the device sees. q/k are L2-normalized
    along K (caller contract) and stay at the key-head count."""
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    g_scale = 0.02 if state_mode == "with_h0_and_dht" else 0.5
    x = {
        "q": l2(rn(B, T, HK, K)),
        "k": l2(rn(B, T, HK, K)),
        "v": rn(B, T, HV, V),
        "g": F.logsigmoid(rn(B, T, HV)) * g_scale,
        "beta": torch.rand(B, T, HV, generator=gen, dtype=torch.float64),
        "do": rn(B, T, HV, V),
        "h0": rn(B, HV, K, V) * 0.1 if state_mode != "do_only" else None,
        "dht": rn(B, HV, K, V) if state_mode == "with_h0_and_dht" else None,
    }
    return {n: None if t is None else t.to(TORCH_DTYPE[dtype]).double() for n, t in x.items()}


def _sum_key_heads(grads, shape):
    """dq/dk of the repeated per-value-head q/k -> per key head."""
    B, T, HK, HV, K, V = shape
    dq, dk = (d.reshape(B, T, HK, HV // HK, K).sum(3) for d in grads[:2])
    return (dq, dk, *grads[2:])


def _oracle(x, shape):
    """float64 gradients of the token-by-token recurrence; dq/dk summed over each key head's value heads."""
    G = shape[3] // shape[2]
    grads = recurrent_vjp(
        x["q"].repeat_interleave(G, dim=2),
        x["k"].repeat_interleave(G, dim=2),
        x["v"],
        x["g"],
        x["beta"],
        x["do"],
        d_final_state=x["dht"],
        initial_state=x["h0"],
    )
    return _sum_key_heads(grads, shape)


def _chunked_oracle(x, shape, chunk_size):
    """The same gradients through the chunked forward: far less autograd memory at long T."""
    G = shape[3] // shape[2]
    names = ("q", "k", "v", "g", "beta", "h0")
    leaves = {n: x[n].clone().requires_grad_(True) for n in names if x[n] is not None}
    rep = lambda t: t.repeat_interleave(G, dim=2)
    o, final_state, *_ = chunked_forward(
        rep(leaves["q"]),
        rep(leaves["k"]),
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        initial_state=leaves.get("h0"),
        chunk_size=chunk_size,
    )
    loss = (o * x["do"]).sum()
    if x["dht"] is not None:
        loss = loss + (final_state * x["dht"]).sum()
    loss.backward()
    return tuple(leaves[n].grad if n in leaves else None for n in names)


def _dev(t, device, dtype):
    if t is None:
        return None
    return ttnn.from_torch(
        t.to(TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _flat(x, shape):
    """Token-major [B, T, X] views of the inputs, heads folded into the last dim."""
    B, T, HK, HV, K, V = shape
    return {
        "q": x["q"].reshape(B, T, HK * K),
        "k": x["k"].reshape(B, T, HK * K),
        "v": x["v"].reshape(B, T, HV * V),
        "g": x["g"],
        "beta": x["beta"],
        "do": x["do"].reshape(B, T, HV * V),
    }


def _run_ttml(x, shape, chunk_size, dtype, device, *, rank=4, do_head_major=False, saved=None):
    B, T, HK, HV, K, V = shape
    f = {n: t.unsqueeze(1) if rank == 4 else t for n, t in _flat(x, shape).items()}
    if do_head_major:
        f["do"] = x["do"].permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    f = {n: _dev(t, device, dtype) for n, t in f.items()}
    return ttml.ops.metal.chunk_gated_delta_rule_bw(
        f["q"],
        f["k"],
        f["v"],
        f["g"],
        f["beta"],
        f["do"],
        K,
        initial_state=_dev(x["h0"], device, dtype),
        d_final_state=_dev(x["dht"], device, dtype),
        chunk_size=chunk_size,
        **(saved or {}),
    )


def _run_ttml_fw(x, shape, chunk_size, device):
    """The saved forward from ttml's training forward; its q/k/v are bf16, gates and h0 match x."""
    f = {n: _dev(t.unsqueeze(1), device, ttnn.bfloat16) for n, t in _flat(x, shape).items() if n != "do"}
    _, _, h, v_new, _, a = ttml.ops.metal.chunk_gated_delta_rule_fw(
        f["q"],
        f["k"],
        f["v"],
        f["g"],
        f["beta"],
        shape[4],
        initial_state=_dev(x["h0"], device, ttnn.float32),
        chunk_size=chunk_size,
    )
    return {"h": h, "v_new": v_new, "a": a}


def _out_shapes(shape):
    B, T, HK, HV, K, V = shape
    return {
        "dq": (B, 1, T, HK * K),
        "dk": (B, 1, T, HK * K),
        "dv": (B, 1, T, HV * V),
        "dg": (B, 1, T, HV),
        "dbeta": (B, 1, T, HV),
        "dh0": (B, HV, K, V),
    }


def _check_grads(got, expected, shape, dtype):
    shapes = _out_shapes(shape)
    for name, dev_t, exp in zip(GRAD_NAMES, got, expected):
        if exp is None:
            assert dev_t is None, f"{name}: expected None without initial_state"
            continue
        assert dev_t.layout == ttnn.TILE_LAYOUT and dev_t.dtype == dtype, f"{name}: {dev_t.layout} {dev_t.dtype}"
        host = ttnn.to_torch(dev_t).double()
        assert tuple(host.shape) == shapes[name], f"{name}: shape {tuple(host.shape)} != {shapes[name]}"
        assert_close(name, host, exp.reshape(host.shape), pcc_min=PCC[dtype], rms_max=RMS[dtype])


@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("state_mode", ["do_only", "with_h0", "with_h0_and_dht"])
def test_vs_oracle(shape, chunk_size, dtype, state_mode, device):
    x = _build_inputs(shape, state_mode, dtype)
    _check_grads(_run_ttml(x, shape, chunk_size, dtype, device), _oracle(x, shape), shape, dtype)


def test_two_live_input_sets(device):
    """A second call at the same shape hits the program cache and must read the new buffers."""
    shape, chunk_size, dtype = SHAPES[2][0], SHAPES[2][1], ttnn.float32
    xa = _build_inputs(shape, "with_h0", dtype, seed=1)
    xb = _build_inputs(shape, "with_h0", dtype, seed=2)
    got_a = _run_ttml(xa, shape, chunk_size, dtype, device)
    entries = device.num_program_cache_entries()
    got_b = _run_ttml(xb, shape, chunk_size, dtype, device)
    assert device.num_program_cache_entries() == entries, "a second input set at the same shape recompiled"
    _check_grads(got_a, _oracle(xa, shape), shape, dtype)
    _check_grads(got_b, _oracle(xb, shape), shape, dtype)


@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("state_mode", ["do_only", "with_h0", "with_h0_and_dht"])
def test_saved_forward_vs_oracle(shape, chunk_size, state_mode, device):
    dtype = ttnn.bfloat16
    x = _build_inputs(shape, state_mode, dtype)
    got = _run_ttml(x, shape, chunk_size, dtype, device, saved=_run_ttml_fw(x, shape, chunk_size, device))
    recomputed = _run_ttml(x, shape, chunk_size, dtype, device)
    _check_grads(got, _oracle(x, shape), shape, dtype)
    for name, dev_t, rec_t in zip(GRAD_NAMES, got, recomputed):
        if dev_t is not None:
            assert_close(
                f"{name} saved vs recomputed", ttnn.to_torch(dev_t), ttnn.to_torch(rec_t).double(), pcc_min=0.999
            )


def test_qwen_shape_saved_forward_head_major_d_out(device):
    """The training path at a Qwen3.8 per-chip shape: flat inputs, head-major d_out, saved forward."""
    shape, chunk_size = QWEN_SHAPE
    dtype = ttnn.bfloat16
    x = _build_inputs(shape, "with_h0_and_dht", dtype)
    saved = _run_ttml_fw(x, shape, chunk_size, device)
    got = _run_ttml(x, shape, chunk_size, dtype, device, do_head_major=True, saved=saved)
    _check_grads(got, _chunked_oracle(x, shape, chunk_size), shape, dtype)


@pytest.mark.parametrize(
    "shape,chunk_size", [SHAPES[2], SHAPES[3], SHAPES[4]], ids=[SHAPE_IDS[2], SHAPE_IDS[3], SHAPE_IDS[4]]
)
@pytest.mark.parametrize("saved", [False, True], ids=["recompute", "saved"])
def test_layouts_bit_identical(shape, chunk_size, saved, device):
    """Head-major d_out and rank-3 inputs read the same pages as the rank-4 token-major baseline."""
    dtype = ttnn.bfloat16
    x = _build_inputs(shape, "with_h0_and_dht", dtype, seed=3)
    fw = (lambda: _run_ttml_fw(x, shape, chunk_size, device)) if saved else (lambda: None)
    base = _run_ttml(x, shape, chunk_size, dtype, device, saved=fw())
    head_major = _run_ttml(x, shape, chunk_size, dtype, device, do_head_major=True, saved=fw())
    rank3 = _run_ttml(x, shape, chunk_size, dtype, device, rank=3, saved=fw())
    for name, tb, th, t3 in zip(GRAD_NAMES, base, head_major, rank3):
        hb, hh, h3 = (ttnn.to_torch(t) for t in (tb, th, t3))
        assert torch.equal(hb, hh), f"{name}: head-major d_out differs"
        if name != "dh0":
            assert h3.dim() == 3, f"{name}: rank-3 input gave a rank-{h3.dim()} gradient"
        assert torch.equal(hb.reshape(h3.shape), h3), f"{name}: rank-3 inputs differ"


def test_rejects_partial_saved_forward(device, expect_error):
    shape, chunk_size = SHAPES[1]
    x = _build_inputs(shape, "do_only", ttnn.bfloat16)
    saved = _run_ttml_fw(x, shape, chunk_size, device)
    del saved["a"]
    with expect_error(RuntimeError, "must be given together"):
        _run_ttml(x, shape, chunk_size, ttnn.bfloat16, device, saved=saved)


def test_rejects_bad_d_out(device, expect_error):
    z = lambda *s: _dev(torch.zeros(*s), device, ttnn.bfloat16)
    q, k, v, g, beta = z(1, 1, 64, 64), z(1, 1, 64, 64), z(1, 1, 64, 128), z(1, 1, 64, 2), z(1, 1, 64, 2)
    with expect_error(RuntimeError, "d_out must be token-major"):
        ttml.ops.metal.chunk_gated_delta_rule_bw(q, k, v, g, beta, z(1, 1, 64, 64), 64, chunk_size=32)


@pytest.mark.parametrize(
    "T,q_width,v_width,H,key_head_dim,chunk_size,match",
    [
        (96, 128, 128, 2, 64, 64, "multiple of chunk_size"),
        (128, 128, 129, 2, 64, 64, "not a multiple of HV"),  # v width / H
        (128, 96, 128, 2, 64, 64, "not a multiple of K"),  # q width / K
        (128, 192, 128, 2, 64, 64, "multiple of key heads"),  # HK=3 does not divide HV=2
        (128, 128, 96, 2, 64, 64, "multiples of 32"),  # V = 48
        (128, 96, 128, 2, 48, 64, "multiples of 32"),  # K = 48
        (128, 128, 128, 2, 64, 48, "chunk_size must be 32 or 64"),
    ],
    ids=["ragged_T", "v_width", "q_width", "gva_ratio", "V_not_tile", "K_not_tile", "bad_chunk"],
)
def test_rejects_bad_shapes(T, q_width, v_width, H, key_head_dim, chunk_size, match, device, expect_error):
    z = lambda *s: _dev(torch.zeros(*s), device, ttnn.bfloat16)
    q, k, v = z(1, 1, T, q_width), z(1, 1, T, q_width), z(1, 1, T, v_width)
    g, beta, do = z(1, 1, T, H), z(1, 1, T, H), z(1, 1, T, v_width)
    with expect_error(RuntimeError, match):
        ttml.ops.metal.chunk_gated_delta_rule_bw(q, k, v, g, beta, do, key_head_dim, chunk_size=chunk_size)
