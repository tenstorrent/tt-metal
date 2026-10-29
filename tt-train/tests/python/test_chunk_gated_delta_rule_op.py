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
4. Weak gates (the default here) keep the inter-chunk state carry observable; one decaying-gate case
   covers the other regime.  See `WEAK_GATE`.
5. An explicit `scale` is checked against the oracle in absolute terms, not only by correlation.
6. Cache hits patch every buffer address with both input sets alive, and a changed `initial_state`
   presence or gate dtype misses the cache.

The loss is `mean(o * w)` with `w` scaled by the number of outputs, so the gradients are O(1) instead of
1e-6..1e-5 (where a vacuous comparison of near-zero tensors would pass).  Tolerances: every tensor must
reach PCC >= `PCC_MIN` and a relative RMS error <= `RMS_MAX`.  Measured on Blackhole over all cases here
the worst relative RMS is 9.4e-3 (`dg`; the output and gradients are rounded to bf16 and the gates and
state through bf16/fp32 casts) and the worst PCC is 0.99998, so the bounds sit about 2x above the
measurement.
"""

from __future__ import annotations


import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn
import ttml

# conftest.py puts this directory on sys.path (pytest.ini uses --import-mode=importlib).
from chunk_gated_delta_rule_reference import DECAY_GATE, WEAK_GATE, assert_close, recurrent_forward  # noqa: E402

pytestmark = pytest.mark.requires_device

PCC_MIN = 0.9999
RMS_MAX = 0.02

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


def _build_inputs(shape, with_h0, seed=0, gate_scale=WEAK_GATE):
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
        "g": bf(F.logsigmoid(rn(B, T, HV)) * gate_scale),
        "beta": bf(torch.rand(B, T, HV, generator=gen, dtype=torch.float64)),
        "h0": bf(rn(B, HV, K, V) * 0.1) if with_h0 else None,
        "w": bf(rn(B * HV, 1, T, V) * (B * HV * T * V)),  # mean() divides it back out of the gradient
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


def _ttml_tensor(t, requires_grad, dtype=ttnn.DataType.BFLOAT16):
    out = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(t.numpy(), dtype=np.float32), layout=ttnn.Layout.TILE, new_type=dtype
    )
    out.set_requires_grad(requires_grad)
    return out


def _leaves(x, shape, needs_grad=NAMES, gate_dtype=ttnn.DataType.BFLOAT16):
    return {
        n: (
            None
            if t is None
            else _ttml_tensor(t, n in needs_grad, gate_dtype if n in ("g", "beta") else ttnn.DataType.BFLOAT16)
        )
        for n, t in _flat(x, shape).items()
    }


def _forward_backward(leaves, x, chunk_size, key_head_dim, scale=None):
    """(o, {name: grad of a leaf that requires one}, [names of frozen leaves that got one])."""
    out = ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        key_head_dim,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=leaves["h0"],
    )
    out_np = out.to_numpy()
    ttml.ops.unary.mean(ttml.ops.binary.mul(out, _ttml_tensor(x["w"], False))).backward(False)
    grads, frozen_with_grad = {}, []
    for n, leaf in leaves.items():
        if leaf is None:
            continue
        if leaf.get_requires_grad():
            grads[n] = leaf.get_grad_tensor().to_numpy()
        elif leaf.is_grad_initialized():
            frozen_with_grad.append(n)
    return out_np, grads, frozen_with_grad


def _run_ttml(x, shape, chunk_size, needs_grad=NAMES, gate_dtype=ttnn.DataType.BFLOAT16, scale=None):
    leaves = _leaves(x, shape, needs_grad, gate_dtype)
    return _forward_backward(leaves, x, chunk_size, shape[4], scale)


def _oracle(x, shape, scale=None):
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
        scale=scale,
    )
    o = o.permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    (o * x["w"]).mean().backward()
    flat = _flat({n: None if t is None else t.grad for n, t in leaves.items()}, shape)
    return o.detach(), flat


def _check(name, got, exp):
    got = torch.as_tensor(np.asarray(got), dtype=torch.float64)
    assert tuple(got.shape) == tuple(exp.shape), f"{name}: shape {tuple(got.shape)} != {tuple(exp.shape)}"
    assert_close(name, got, exp, pcc_min=PCC_MIN, rms_max=RMS_MAX)


@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
def test_vs_torch_autograd(shape, chunk_size, with_h0):
    x = _build_inputs(shape, with_h0)
    out, grads, frozen = _run_ttml(x, shape, chunk_size)
    assert not frozen
    o_ref, g_ref = _oracle(x, shape)
    _check("o", out, o_ref)
    for name in NAMES:
        if g_ref[name] is None:
            assert name not in grads
            continue
        _check(f"d{name}", grads[name], g_ref[name])


@pytest.mark.parametrize("shape,chunk_size", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_fp32_gates(shape, chunk_size):
    """FP32 gates beside BF16 q/k/v run the forward and the backward, and dg / dbeta come back FP32."""
    B, T, HK, HV, K, V = shape
    x = _build_inputs(shape, with_h0=True)
    gen = torch.Generator().manual_seed(7)
    # Not representable in bf16, so a gate silently rounded to bf16 anywhere would be seen.
    x["g"] = (F.logsigmoid(torch.randn(B, T, HV, generator=gen, dtype=torch.float64)) * 0.5).float().double()
    x["beta"] = torch.rand(B, T, HV, generator=gen, dtype=torch.float64).float().double()
    out, grads, _ = _run_ttml(x, shape, chunk_size, gate_dtype=ttnn.DataType.FLOAT32)
    o_ref, g_ref = _oracle(x, shape)
    _check("o", out, o_ref)
    for name in NAMES:
        if g_ref[name] is None:
            continue
        _check(f"d{name}", grads[name], g_ref[name])


@pytest.mark.parametrize("shape,chunk_size", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_decaying_gate(shape, chunk_size):
    """The previous default gate (decay of e^-13 .. e^-26 per chunk): the within-chunk decay terms."""
    x = _build_inputs(shape, with_h0=True, gate_scale=DECAY_GATE)
    out, grads, _ = _run_ttml(x, shape, chunk_size)
    o_ref, g_ref = _oracle(x, shape)
    _check("o", out, o_ref)
    for name in NAMES:
        _check(f"d{name}", grads[name], g_ref[name])


@pytest.mark.parametrize("scale", [0.3, 1.0], ids=["scale_0.3", "scale_1"])
def test_explicit_scale(scale):
    """`o` and dq scale with `scale`: a wrong or ignored one keeps their PCC at 1, not their relative RMS."""
    shape, chunk_size = SHAPES[1]
    assert abs(scale - shape[4] ** -0.5) > 0.1, "scale must differ from the default K^-0.5"
    x = _build_inputs(shape, with_h0=True)
    out, grads, _ = _run_ttml(x, shape, chunk_size, scale=scale)
    o_ref, g_ref = _oracle(x, shape, scale=scale)
    _check("o", out, o_ref)
    for name in NAMES:
        _check(f"d{name}", grads[name], g_ref[name])
    default_out, default_grads, _ = _run_ttml(x, shape, chunk_size)
    for name, got, default in (("o", out, default_out), ("dq", grads["q"], default_grads["q"])):
        ratio = np.linalg.norm(np.asarray(got, dtype=np.float64)) / np.linalg.norm(
            np.asarray(default, dtype=np.float64)
        )
        assert ratio > 1.5, f"{name} barely moved with scale {scale} (ratio {ratio:.3f})"


@pytest.mark.parametrize(
    "needs",
    [("q", "v", "beta"), ("g", "h0"), ("k",), ("q", "k"), ("h0",)],
    ids=["q_v_beta", "g_h0", "k", "q_k", "h0"],
)
def test_partial_requires_grad(needs):
    """Frozen leaves get no gradient and the others still match; at G=2 the dq/dk reduction is skipped
    for a frozen q or k."""
    shape, chunk_size = SHAPES[1]
    x = _build_inputs(shape, with_h0=True, seed=1)
    _, grads, frozen_with_grad = _run_ttml(x, shape, chunk_size, needs_grad=needs)
    _, g_ref = _oracle(x, shape)
    assert set(grads) == set(needs)
    assert not frozen_with_grad, f"frozen leaves received a gradient: {frozen_with_grad}"
    for name in needs:
        _check(f"d{name}", grads[name], g_ref[name])


def test_cache_hit_with_both_input_sets_alive():
    """Forward and backward through two live input sets: the second hits every cached program and must read
    and write its own buffers.  Both sets stay alive, so a no-op address patch cannot hide behind reuse."""
    shape, chunk_size = (1, 96, 2, 4, 64, 64), 32  # a shape no other test dispatches
    device = ttml.autograd.AutoContext.get_instance().get_device()
    xa, xb = _build_inputs(shape, True, seed=1), _build_inputs(shape, True, seed=2)
    la, lb = _leaves(xa, shape), _leaves(xb, shape)
    out_a, grads_a, _ = _forward_backward(la, xa, chunk_size, shape[4])
    entries = device.num_program_cache_entries()
    assert entries > 0
    out_b, grads_b, _ = _forward_backward(lb, xb, chunk_size, shape[4])
    assert device.num_program_cache_entries() == entries, "a second input set at the same shape recompiled"
    for x, out, grads in ((xa, out_a, grads_a), (xb, out_b, grads_b)):
        o_ref, g_ref = _oracle(x, shape)
        _check("o", out, o_ref)
        for name in NAMES:
            _check(f"d{name}", grads[name], g_ref[name])
    assert not np.array_equal(out_a, out_b)


def test_attribute_changes_miss_the_cache():
    """`initial_state` presence and the gate dtype each select a different program."""
    shape, chunk_size = (1, 448, 1, 2, 64, 64), 32  # no other test (in any GDN file) dispatches this shape
    device = ttml.autograd.AutoContext.get_instance().get_device()
    keep = []

    def new_programs(with_h0, gate_dtype=ttnn.DataType.BFLOAT16):
        x = _build_inputs(shape, with_h0)
        before = device.num_program_cache_entries()
        keep.append(_run_ttml(x, shape, chunk_size, gate_dtype=gate_dtype))
        return device.num_program_cache_entries() - before

    assert new_programs(True) > 0
    assert new_programs(True) == 0, "identical call missed the cache"
    assert new_programs(False) > 0, "dropping initial_state reused the programs"
    assert new_programs(True, ttnn.DataType.FLOAT32) > 0, "an fp32 gate reused the bf16-gate programs"


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


def _forward_only(leaves, chunk_size, key_head_dim):
    return ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        key_head_dim,
        chunk_size=chunk_size,
        initial_state=leaves["h0"],
    )


@pytest.mark.parametrize("grad_mode_disabled", [False, True], ids=["no_leaf_requires_grad", "grad_mode_disabled"])
def test_forward_without_backward_matches_and_does_not_poison_the_cache(grad_mode_disabled):
    """With nothing to differentiate the forward skips the saved-state writes.  The output must equal the
    oracle, and a grad-enabled call at the same shape afterwards (a different program) must still give
    correct gradients."""
    shape, chunk_size = SHAPES[1]
    x = _build_inputs(shape, with_h0=True, seed=3)
    o_ref, g_ref = _oracle(x, shape)
    ctx = ttml.autograd.AutoContext.get_instance()
    if grad_mode_disabled:
        ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    try:
        leaves = _leaves(x, shape, needs_grad=NAMES if grad_mode_disabled else ())
        out = _forward_only(leaves, chunk_size, shape[4])
        _check("o", out.to_numpy(), o_ref)
    finally:
        ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    o, grads, _ = _run_ttml(x, shape, chunk_size)
    _check("o", o, o_ref)
    for name in NAMES:
        _check(f"d{name}", grads[name], g_ref[name])
