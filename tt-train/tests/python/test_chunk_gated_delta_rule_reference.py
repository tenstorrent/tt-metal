# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of the float64 references behind the chunk_gated_delta_rule tests.

The device tests compare the kernels with `chunked_forward` (every output the backward consumes) and,
for gradients at long T, with autograd through it.  That is only meaningful if the chunked algorithm
itself equals the token-by-token definition, which is what is pinned here, without a device:

1. `chunked_forward` equals `recurrent_forward` (o, final state) and FLA's naive recurrent rule
   (o, the state entering every chunk), for several gate strengths, with and without `initial_state`.
2. Gradients through `chunked_forward` equal `recurrent_vjp`, so the chunked oracle used by the long
   backward tests is a valid gradient oracle.
3. The device tests' weak gates (scale 0.02) keep `initial_state` visible in the final state, while the
   former default (0.5) hides it completely, which is why they cannot test the state carry.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch
import torch.nn.functional as F

from chunk_gated_delta_rule_reference import chunked_forward, recurrent_forward, recurrent_vjp

# (B, T, H, K, V), chunk_size
CASES = [
    ((1, 64, 2, 32, 32), 32),
    ((2, 128, 3, 64, 32), 64),
    ((1, 96, 1, 32, 64), 32),
]
CASE_IDS = [f"B{c[0][0]}_T{c[0][1]}_H{c[0][2]}_K{c[0][3]}_V{c[0][4]}_c{c[1]}" for c in CASES]
GATE_SCALES = {"weak": 0.02, "decaying": 0.5, "strong": 4.0}

_FLA_REF = pathlib.Path(__file__).resolve().parents[3] / "models/demos/blackhole/qwen36/tests/gdn_fla_ref.py"


def _inputs(shape, gate_scale, with_h0, seed=0):
    B, T, H, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True)
    return {
        "q": l2(rn(B, T, H, K)),
        "k": l2(rn(B, T, H, K)),
        "v": rn(B, T, H, V),
        "g": F.logsigmoid(rn(B, T, H)) * gate_scale,
        "beta": torch.rand(B, T, H, generator=gen, dtype=torch.float64),
        "h0": rn(B, H, K, V) * 0.1 if with_h0 else None,
    }


def _rel(a, b):
    return ((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(1e-300)).item()


@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("gate", GATE_SCALES, ids=list(GATE_SCALES))
@pytest.mark.parametrize("shape,chunk", CASES, ids=CASE_IDS)
def test_chunked_equals_recurrent(shape, chunk, gate, with_h0):
    x = _inputs(shape, GATE_SCALES[gate], with_h0)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    o, final_state, *_ = chunked_forward(*args, initial_state=x["h0"], chunk_size=chunk)
    o_ref, final_ref = recurrent_forward(*args, initial_state=x["h0"])
    assert _rel(o, o_ref) < 1e-9, f"o: {_rel(o, o_ref):.3g}"
    assert _rel(final_state, final_ref) < 1e-9, f"final_state: {_rel(final_state, final_ref):.3g}"


def test_chunked_equals_recurrent_with_explicit_scale():
    shape, chunk = CASES[1]
    x = _inputs(shape, 0.02, True)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    o, *_ = chunked_forward(*args, initial_state=x["h0"], chunk_size=chunk, scale=0.3)
    o_ref, _ = recurrent_forward(*args, initial_state=x["h0"], scale=0.3)
    assert _rel(o, o_ref) < 1e-9
    default, *_ = chunked_forward(*args, initial_state=x["h0"], chunk_size=chunk)
    assert _rel(o, default) > 0.5, "scale had no effect"


@pytest.mark.skipif(not _FLA_REF.exists(), reason="gdn_fla_ref.py not in this checkout")
@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("gate", ["weak", "decaying"])
@pytest.mark.parametrize("shape,chunk", CASES, ids=CASE_IDS)
def test_chunked_matches_fla_naive(shape, chunk, gate, with_h0):
    """FLA's naive recurrent_gated_delta_rule is the external ground truth (float32): output and the state
    entering every chunk, which is what `h` of the forward op holds."""
    spec = importlib.util.spec_from_file_location("gdn_fla_ref", _FLA_REF)
    fla = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fla)

    B, T, H, K, V = shape
    x = _inputs(shape, GATE_SCALES[gate], with_h0)
    o, final_state, h, *_ = chunked_forward(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["h0"], chunk_size=chunk
    )
    f32 = lambda t: None if t is None else t.float()
    o_fla, states = fla.naive_recurrent_per_token_state(
        f32(x["q"]), f32(x["k"]), f32(x["v"]), f32(x["beta"]), f32(x["g"]), initial_state=f32(x["h0"])
    )
    assert _rel(o, o_fla.double()) < 1e-4, f"o: {_rel(o, o_fla.double()):.3g}"
    assert _rel(final_state, states[:, -1].double()) < 1e-4
    start = torch.zeros(B, H, K, V, dtype=torch.float64) if x["h0"] is None else x["h0"]
    assert torch.equal(h[:, 0], start)
    for i in range(1, T // chunk):
        entering = states[:, i * chunk - 1].double()
        assert _rel(h[:, i], entering) < 1e-4, f"h[:, {i}]: {_rel(h[:, i], entering):.3g}"


@pytest.mark.parametrize("shape,chunk", CASES, ids=CASE_IDS)
def test_chunked_gradients_equal_recurrent_vjp(shape, chunk):
    B, T, H, K, V = shape
    x = _inputs(shape, 0.02, True)
    gen = torch.Generator().manual_seed(5)
    d_out = torch.randn(B, T, H, V, generator=gen, dtype=torch.float64)
    d_final = torch.randn(B, H, K, V, generator=gen, dtype=torch.float64)

    names = ("q", "k", "v", "g", "beta", "h0")
    leaves = {n: x[n].clone().requires_grad_(True) for n in names}
    o, final_state, *_ = chunked_forward(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        initial_state=leaves["h0"],
        chunk_size=chunk,
    )
    ((o * d_out).sum() + (final_state * d_final).sum()).backward()

    expected = recurrent_vjp(
        x["q"], x["k"], x["v"], x["g"], x["beta"], d_out, d_final_state=d_final, initial_state=x["h0"]
    )
    for name, exp in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), expected):
        assert _rel(leaves[name[1:]].grad, exp) < 1e-8, f"{name}: {_rel(leaves[name[1:]].grad, exp):.3g}"


@pytest.mark.parametrize("shape,chunk", [CASES[0], CASES[1]], ids=[CASE_IDS[0], CASE_IDS[1]])
def test_weak_gates_keep_the_initial_state_visible(shape, chunk):
    """Inputs of the device tests (see WEAK_GATE there): `initial_state` must still change the final state
    after several chunks, otherwise a scan that drops the carry S * exp(d_last) passes."""
    B, T, H, K, V = shape
    effect = {}
    for gate in ("weak", "decaying"):
        x = _inputs(shape, GATE_SCALES[gate], True)
        args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        with_h0 = chunked_forward(*args, initial_state=x["h0"], chunk_size=chunk)[1]
        without = chunked_forward(*args, initial_state=None, chunk_size=chunk)[1]
        effect[gate] = _rel(with_h0, without)
    assert effect["weak"] > 1e-2, f"weak gates: initial_state moves the final state by only {effect['weak']:.3g}"
    assert effect["decaying"] < 1e-6, f"decaying gates unexpectedly keep the state ({effect['decaying']:.3g})"
