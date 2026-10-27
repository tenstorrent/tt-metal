# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule: the autograd op over the fused forward and backward.

    q, k [B, 1, T, HK*K]   v [B, 1, T, HV*V]   g, beta [B, 1, T, HV]   initial_state [B, HV, K, V]
    -> o [B*HV, 1, T, V], head-major

Pinned here:
1. The output and every input gradient against float64 torch autograd through the token-by-token
   recurrence.
2. The head-major output is exactly the token-major forward output, permuted.
3. Inputs that do not require grad get none, and the rest still match.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn
import ttml

# pytest.ini uses --import-mode=importlib, which does not put the test directory on sys.path.
_TEST_DIR = str(pathlib.Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
from chunk_gated_delta_rule_reference import assert_close, recurrent_forward  # noqa: E402

pytestmark = pytest.mark.requires_device

PCC = 0.99
RMS = 0.12

# (B, T, HK, HV, K, V), chunk_size
SHAPES = [
    ((1, 64, 1, 1, 32, 32), 32),
    ((1, 128, 2, 4, 64, 64), 64),  # GVA G=2
    ((2, 64, 1, 3, 64, 64), 32),  # GVA G=3, multi-batch
    ((1, 128, 2, 2, 64, 128), 64),  # K != V
    ((1, 128, 1, 3, 128, 128), 64),  # Qwen head dims, G=3
]
SHAPE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in SHAPES]
NAMES = ("q", "k", "v", "g", "beta", "h0")


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _build_inputs(shape, with_h0, seed=0):
    """Seeded bf16-representable inputs; q/k L2-normalized along K (the op's contract)."""
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True)
    bf = lambda x: x.to(torch.bfloat16).double()
    return {
        "q": bf(l2(rn(B, T, HK, K))),
        "k": bf(l2(rn(B, T, HK, K))),
        "v": bf(rn(B, T, HV, V)),
        "g": bf(F.logsigmoid(rn(B, T, HV)) * 0.5),
        "beta": bf(torch.rand(B, T, HV, generator=gen, dtype=torch.float64)),
        "h0": bf(rn(B, HV, K, V) * 0.1) if with_h0 else None,
        "w": bf(rn(B * HV, 1, T, V)),
    }


def _flat(x, shape):
    B, T, HK, HV, K, V = shape
    return {
        "q": x["q"].reshape(B, 1, T, HK * K),
        "k": x["k"].reshape(B, 1, T, HK * K),
        "v": x["v"].reshape(B, 1, T, HV * V),
        "g": x["g"].reshape(B, 1, T, HV),
        "beta": x["beta"].reshape(B, 1, T, HV),
        "h0": x["h0"],
    }


def _ttml_tensor(t, requires_grad):
    out = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(t.numpy(), dtype=np.float32), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )
    out.set_requires_grad(requires_grad)
    return out


def _run_ttml(x, shape, chunk_size, needs_grad=NAMES):
    leaves = {n: None if t is None else _ttml_tensor(t, n in needs_grad) for n, t in _flat(x, shape).items()}
    out = ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        shape[4],
        chunk_size=chunk_size,
        initial_state=leaves["h0"],
    )
    out_np = out.to_numpy()
    ttml.ops.unary.mean(ttml.ops.binary.mul(out, _ttml_tensor(x["w"], False))).backward(False)
    grads = {}
    for n, leaf in leaves.items():
        if leaf is not None and n in needs_grad:
            grads[n] = leaf.get_grad_tensor().to_numpy()
    return out_np, grads


def _oracle(x, shape):
    """float64 head-major output and the gradients of mean(o * w), in the op's input layouts."""
    B, T, HK, HV, K, V = shape
    G = HV // HK
    leaves = {n: None if t is None else t.clone().requires_grad_(True) for n, t in x.items() if n != "w"}
    o, _ = recurrent_forward(
        leaves["q"].repeat_interleave(G, dim=2),
        leaves["k"].repeat_interleave(G, dim=2),
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        initial_state=leaves["h0"],
    )
    o = o.permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    (o * x["w"]).mean().backward()
    flat = _flat({n: None if t is None else t.grad for n, t in leaves.items()}, shape)
    return o.detach(), flat


def _check(name, got, exp):
    got = torch.as_tensor(np.asarray(got), dtype=torch.float64)
    assert tuple(got.shape) == tuple(exp.shape), f"{name}: shape {tuple(got.shape)} != {tuple(exp.shape)}"
    assert_close(name, got, exp, pcc_min=PCC, rms_max=RMS)


@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
def test_vs_torch_autograd(shape, chunk_size, with_h0):
    x = _build_inputs(shape, with_h0)
    out, grads = _run_ttml(x, shape, chunk_size)
    o_ref, g_ref = _oracle(x, shape)
    _check("o", out, o_ref)
    for name in NAMES:
        if g_ref[name] is None:
            assert name not in grads
            continue
        _check(f"d{name}", grads[name], g_ref[name])


def test_partial_requires_grad():
    shape, chunk_size = SHAPES[1]
    x = _build_inputs(shape, with_h0=True, seed=1)
    needs = ("q", "v", "beta")
    _, grads = _run_ttml(x, shape, chunk_size, needs_grad=needs)
    _, g_ref = _oracle(x, shape)
    assert set(grads) == set(needs)
    for name in needs:
        _check(f"d{name}", grads[name], g_ref[name])


@pytest.mark.parametrize("shape,chunk_size", [SHAPES[2], SHAPES[3]], ids=[SHAPE_IDS[2], SHAPE_IDS[3]])
def test_head_major_output_is_permuted_token_major(shape, chunk_size):
    B, T, HK, HV, K, V = shape
    device = ttml.autograd.AutoContext.get_instance().get_device()
    x = _flat(_build_inputs(shape, with_h0=True, seed=2), shape)
    dev = lambda t, dt: ttnn.from_torch(t.to(torch.float32), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    args = [dev(x[n], ttnn.bfloat16) for n in ("q", "k", "v", "g", "beta")]
    h0 = dev(x["h0"], ttnn.float32)
    run = lambda hm: ttml.ops.metal.chunk_gated_delta_rule_fw(
        *args, K, initial_state=h0, chunk_size=chunk_size, output_head_major=hm
    )
    token_major, head_major = run(False), run(True)
    tm = ttnn.to_torch(token_major[0]).reshape(B, T, HV, V).permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    assert torch.equal(ttnn.to_torch(head_major[0]), tm)
    for i in range(1, 6):
        assert torch.equal(ttnn.to_torch(head_major[i]), ttnn.to_torch(token_major[i]))
