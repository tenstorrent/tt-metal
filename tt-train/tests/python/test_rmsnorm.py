# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RMSNorm metal ops and their autograd wrappers through the bindings, in both modes.

The plain mode normalizes the last dim of a 4D `x`, in bf16 or fp32. Gated mode multiplies by silu(gate) per V-wide
head: `x` and `dx` are head-major [B*H, 1, T, V]; `gate`, `dy`, `out` and `dgate` are head-merged [B, 1, T, H*V];
`gamma` [1, 1, 1, V] is shared by all heads. Per head h:
    inv   = rsqrt(mean_V(x_h^2) + eps)
    u     = x_h * gamma * inv
    out_h = u * silu(gate_h)
and with du = dy * silu(gate), t = sum_V(u * du) * inv / V:
    dx     = inv * (gamma * du - t * x)
    dgate  = dy * u * silu'(gate)
    dgamma = sum over rows and heads of x * inv * du
The plain mode is the same with silu(gate) = 1 and no dgate. The shape sweep is in tests/ops/rmsnorm_op_test.cpp.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttml
from bf16_ulp import assert_within_bf16_ulp
from tensor_helpers import as_stored, sigmoid, to_device, to_host, value

pytestmark = pytest.mark.requires_device

MAX_ULP = 1.0
MAX_ULP_P99 = 1.0
# rms is scaled by 1/V held in bf16, which is inexact unless V is a power of two.
MAX_ULP_P99_RMS = 1.5
# The backward takes the bf16 rms the forward returns and holds 1/rms in bf16, so 1/rms is off by up to
# 2**-7 relative before the output's own rounding. The plain RMSNorm backward measures the same.
MAX_ULP_BW = 2.5
MAX_ULP_P99_BW = 2.0
# dx = inv * (gamma * du - t * x) cancels, so elements below peak / 2**3 are bounded at that spacing.
DX_P99_FLOOR_BINADES = 3
# dgamma has only V elements, each a sum of B*T*H signed terms, so near-zero entries have unbounded
# per-element ULP; only the error at the peak is bounded.
MAX_ULP_DGAMMA = 2.0
EPS = 1e-6

# (batch, T, H, V): the Qwen3.8 TP=4 head layout. Its 384 tile-rows don't divide evenly over the cores, so both core
# groups run.
GATED_SHAPE = (1, 1024, 12, 128)
DTYPES = ["bf16", "fp32"]
AUTOGRAD_SHAPES = [(1, 64, 3, 128)]


def grad_of(tensor):
    return tensor.get_grad_tensor().to_numpy().astype(np.float64)


def merge_heads(a, heads):
    """[B*H, 1, T, V] -> [B, 1, T, H*V]."""
    rows, _, seq, dim = a.shape
    return a.reshape(rows // heads, heads, seq, dim).transpose(0, 2, 1, 3).reshape(rows // heads, 1, seq, heads * dim)


def split_heads(a, heads):
    """[B, 1, T, H*V] -> [B*H, 1, T, V]."""
    batch, _, seq, width = a.shape
    dim = width // heads
    return a.reshape(batch, seq, heads, dim).transpose(0, 2, 1, 3).reshape(batch * heads, 1, seq, dim)


def reference(x, gate, gamma, dy, eps, heads):
    """(out, rms, dx, dgate, dgamma); without a gate this is the plain RMSNorm and dgate is None."""
    gamma_v = gamma.reshape(-1)
    dim = x.shape[-1]
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    inv = 1.0 / rms
    u = x * gamma_v * inv
    if gate is None:
        du, out, dgate = dy, u, None
    else:
        z, d = split_heads(gate, heads), split_heads(dy, heads)
        s = sigmoid(z)
        du = d * z * s
        out = merge_heads(u * z * s, heads)
        dgate = merge_heads(d * u * s * (1.0 + z - z * s), heads)
    t = np.sum(u * du, axis=-1, keepdims=True) * inv / dim
    dx = inv * (gamma_v * du - t * x)
    dgamma = np.sum(x * inv * du, axis=(0, 1, 2)).reshape(1, 1, 1, dim)
    return out, rms, dx, dgate, dgamma


def inputs(batch, seq, heads, dim, seed):
    """Heads get RMS spread over 2**-2 .. 2**3, so normalizing over the wrong columns cannot pass."""
    rng = np.random.default_rng(seed)
    head_scale = np.tile(2.0 ** (np.arange(heads) % 6 - 2), batch).reshape(batch * heads, 1, 1, 1)
    merged = (batch, 1, seq, heads * dim)
    return (
        rng.normal(0.0, 1.0, (batch * heads, 1, seq, dim)) * head_scale,
        rng.uniform(-20.0, 20.0, merged),  # gate, reaching into both saturation tails of sigmoid
        rng.uniform(0.5, 1.5, (1, 1, 1, dim)),
        rng.normal(0.0, 1.0, merged),
    )


def run(x, gate, gamma, dy, dtype="bf16", compute_dgamma=True, eps=EPS):
    """Forward then backward on device -> (out, rms, dx, dgate, dgamma) on host, None where not produced."""
    x_dev, gamma_dev = value(to_device(x, dtype)), value(to_device(gamma))
    gate_dev = None if gate is None else value(to_device(gate))
    out, rms = ttml.ops.metal.rmsnorm_fw(x_dev, gamma_dev, True, eps, gate_dev)
    dx, dgamma, dgate = ttml.ops.metal.rmsnorm_bw(x_dev, gamma_dev, rms, value(to_device(dy)), compute_dgamma, gate_dev)
    host = [None if t is None else to_host(t) for t in (out, rms, dx, dgate, dgamma)]
    return tuple(host)


def assert_matches(got, x, gate, gamma, dy, heads, label, dtype="bf16", eps=EPS):
    stored_gate = None if gate is None else as_stored(gate)
    expected = reference(as_stored(x, dtype), stored_gate, as_stored(gamma), as_stored(dy), eps, heads)
    out, rms, dx, dgate, dgamma = got
    ref_out, ref_rms, ref_dx, ref_dgate, ref_dgamma = expected
    if out is not None:
        assert out.shape == ref_out.shape
        assert_within_bf16_ulp(out, ref_out, f"{label} out", MAX_ULP, MAX_ULP_P99)
    if rms is not None:
        assert_within_bf16_ulp(rms, ref_rms, f"{label} rms", MAX_ULP, MAX_ULP_P99_RMS)
    assert dx.shape == x.shape
    assert_within_bf16_ulp(dx, ref_dx, f"{label} dx", MAX_ULP_BW, MAX_ULP_P99_BW, DX_P99_FLOOR_BINADES)
    assert (dgate is None) == (gate is None)
    if dgate is not None:
        assert dgate.shape == gate.shape
        assert_within_bf16_ulp(dgate, ref_dgate, f"{label} dgate", MAX_ULP_BW, MAX_ULP_P99_BW)
    if dgamma is not None:
        assert dgamma.shape == gamma.shape
        assert_within_bf16_ulp(dgamma, ref_dgamma, f"{label} dgamma", MAX_ULP_DGAMMA)


class TestGatedMode:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matches_reference(self, dtype):
        batch, seq, heads, dim = GATED_SHAPE
        x, gate, gamma, dy = inputs(batch, seq, heads, dim, seed=100 + heads * dim)
        got = run(x, gate, gamma, dy, dtype)
        assert_matches(got, x, gate, gamma, dy, heads, f"{batch}x{seq}x{heads}x{dim} {dtype}", dtype)


class TestUngatedMode:
    """Without a gate the op is the plain RMSNorm, which also takes an fp32 input."""

    @pytest.mark.parametrize("compute_dgamma", [True, False], ids=["dgamma", "no_dgamma"])
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((2, 1, 96, 80), "bf16"),
            ((2, 1, 96, 80), "fp32"),
            # The forward keeps everything but gamma in L1 and the backward streams.
            ((1, 1, 32, 8192), "fp32"),
            # Both stream.
            ((1, 1, 32, 16384), "fp32"),
        ],
    )
    def test_matches_reference(self, shape, dtype, compute_dgamma):
        rng = np.random.default_rng(shape[-1])
        x, dy = rng.normal(0.0, 1.0, shape), rng.normal(0.0, 1.0, shape)
        gamma = rng.uniform(0.5, 1.5, (1, 1, 1, shape[-1]))
        got = run(x, None, gamma, dy, dtype, compute_dgamma)
        assert (got[4] is not None) == compute_dgamma
        assert_matches(got, x, None, gamma, dy, 1, f"ungated {shape} {dtype}", dtype)


class TestProgramCache:
    """A second launch of a shape hits the cached program, so only override_runtime_arguments hands
    the kernels the new buffer addresses. Every launch's tensors stay alive until the end so a
    later one cannot land at an earlier one's address."""

    def test_reads_the_new_buffers(self):
        alive = []
        for seed in (1, 2):
            x, gate, gamma, dy = inputs(1, 64, 3, 128, seed=seed)
            dev = [value(to_device(a)) for a in (x, gate, gamma, dy)]
            out, rms = ttml.ops.metal.rmsnorm_fw(dev[0], dev[2], True, EPS, dev[1])
            dx, dgamma, dgate = ttml.ops.metal.rmsnorm_bw(dev[0], dev[2], rms, dev[3], True, dev[1])
            alive.append((dev, out, rms, dx, dgamma, dgate))
            got = tuple(to_host(t) for t in (out, rms, dx, dgate, dgamma))
            assert_matches(got, x, gate, gamma, dy, 3, f"relaunch seed={seed}")

    def test_mode_and_dtype_are_part_of_the_program(self):
        # One input shape [2, 1, 64, 64] runs ungated, gated as 2 batches of 1 head and as 1 batch of 2 heads, with
        # either input dtype; a cache keyed on the input shape alone would reuse the first program for all of them.
        x, gate, gamma, dy = inputs(1, 64, 2, 64, seed=11)
        cases = [
            ("ungated", None, split_heads(dy, 2), 1),
            ("2x1 heads", split_heads(gate, 2), split_heads(dy, 2), 1),
            ("1x2 heads", gate, dy, 2),
        ]
        for dtype in DTYPES * 2:
            for name, case_gate, case_dy, heads in cases:
                got = run(x, case_gate, gamma, case_dy, dtype)
                assert_matches(got, x, case_gate, gamma, case_dy, heads, f"{name} {dtype}", dtype)

    def test_epsilon_and_dgamma_flag_are_part_of_the_program(self):
        x, gate, gamma, dy = inputs(1, 64, 2, 64, seed=9)
        # mean(x^2) is 6e-6 to 2.5e-5 against eps = 0.1, so dropping or misreading eps changes out by 63-126x.
        x = x * 1e-2
        for eps, compute_dgamma in ((1e-6, True), (0.1, False), (0.1, True), (1e-6, False)):
            got = run(x, gate, gamma, dy, compute_dgamma=compute_dgamma, eps=eps)
            assert (got[4] is not None) == compute_dgamma
            assert_matches(got, x, gate, gamma, dy, 2, f"eps={eps} compute_dgamma={compute_dgamma}", eps=eps)


class TestAutogradWrapper:
    """`ttml.ops.rmsnorm.gated_rmsnorm` wires the gated mode into the graph."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("batch,seq,heads,dim", AUTOGRAD_SHAPES)
    @pytest.mark.parametrize("train_gamma", [True, False], ids=["trained_gamma", "frozen_gamma"])
    def test_forward_and_backward(self, batch, seq, heads, dim, dtype, train_gamma):
        x, gate, gamma, _ = inputs(batch, seq, heads, dim, seed=400 + dim)
        x_dev = to_device(x, dtype, requires_grad=True)
        gate_dev = to_device(gate, requires_grad=True)
        gamma_dev = to_device(gamma, requires_grad=train_gamma)

        out = ttml.ops.rmsnorm.gated_rmsnorm(x_dev, gate_dev, gamma_dev, EPS)
        out_host = out.to_numpy().astype(np.float64)
        out.backward(retain_graph=False)

        assert x_dev.is_grad_initialized() and gate_dev.is_grad_initialized()
        assert gamma_dev.is_grad_initialized() == train_gamma
        got = (out_host, None, grad_of(x_dev), grad_of(gate_dev), grad_of(gamma_dev) if train_gamma else None)
        ones = np.ones(gate.shape)  # backward() seeds dL/dout with ones
        assert_matches(got, x, gate, gamma, ones, heads, f"autograd {seq}x{heads}x{dim} {dtype}", dtype)


class TestValidation:
    def test_rejects_head_width_that_is_not_tile_aligned(self, expect_error):
        x, gate, gamma = (to_device(np.zeros(s)) for s in ((2, 1, 32, 48), (1, 1, 32, 96), (1, 1, 1, 48)))
        with expect_error(RuntimeError, "V must be a multiple of 32"):
            ttml.ops.metal.rmsnorm_fw(value(x), value(gamma), True, EPS, value(gate))

    def test_rejects_gate_that_is_not_whole_heads(self, expect_error):
        x, gate, gamma = (to_device(np.zeros(s)) for s in ((2, 1, 32, 64), (1, 1, 32, 96), (1, 1, 1, 64)))
        with expect_error(RuntimeError, "whole number of heads"):
            ttml.ops.metal.rmsnorm_fw(value(x), value(gamma), True, EPS, value(gate))

    def test_rejects_gate_that_is_not_head_merged(self, expect_error):
        x, gate, gamma = (to_device(np.zeros(s)) for s in ((2, 1, 32, 64), (1, 1, 64, 128), (1, 1, 1, 64)))
        with expect_error(RuntimeError, "is not the head-merged layout"):
            ttml.ops.metal.rmsnorm_fw(value(x), value(gamma), True, EPS, value(gate))

    def test_rejects_fp32_gate(self, expect_error):
        x, gamma = to_device(np.zeros((2, 1, 32, 64))), to_device(np.ones((1, 1, 1, 64)))
        gate = to_device(np.zeros((1, 1, 32, 128)), "fp32")
        with expect_error(RuntimeError, "Gate has unsupported dtype"):
            ttml.ops.metal.rmsnorm_fw(value(x), value(gamma), True, EPS, value(gate))

    def test_rejects_upstream_grad_in_the_input_layout(self, expect_error):
        x, gate, gamma = (to_device(np.ones(s)) for s in ((2, 1, 32, 64), (1, 1, 32, 128), (1, 1, 1, 64)))
        _, rms = ttml.ops.metal.rmsnorm_fw(value(x), value(gamma), True, EPS, value(gate))
        with expect_error(RuntimeError, "dL_dout shape"):
            ttml.ops.metal.rmsnorm_bw(value(x), value(gamma), rms, value(x), True, value(gate))

    def test_rejects_gated_row_that_does_not_fit_in_l1(self, expect_error):
        # Gated mode only has the resident-row kernels. With an fp32 x, V = 5120 still fits the forward but not the
        # backward, and V = 8192 fits neither.
        def gated_inputs(dim):
            x = to_device(np.ones((1, 1, 32, dim)), "fp32")
            return x, to_device(np.ones((1, 1, 32, dim))), to_device(np.ones((1, 1, 1, dim)))

        x, gate, gamma = gated_inputs(5120)
        _, rms = ttml.ops.metal.rmsnorm_fw(value(x), value(gamma), True, EPS, value(gate))
        with expect_error(RuntimeError, "resident in L1"):
            ttml.ops.metal.rmsnorm_bw(value(x), value(gamma), rms, value(gate), True, value(gate))
        x, gate, gamma = gated_inputs(8192)
        with expect_error(RuntimeError, "resident in L1"):
            ttml.ops.metal.rmsnorm_fw(value(x), value(gamma), True, EPS, value(gate))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
