# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the grouped gated RMSNorm metal ops and their autograd wrapper.

`x`, `gate` are head-merged [B, 1, T, H*V]; `gamma` [1, 1, 1, V] is shared by all heads. Per head h:
    inv   = rsqrt(mean_V(x_h^2) + eps)
    u     = x_h * gamma * inv
    out_h = u * silu(gate_h)
and with du = dy * silu(gate), t = sum_V(u * du) * inv / V:
    dx    = inv * (gamma * du - t * x)
    dgate = dy * u * silu'(gate)
    dgamma = sum over rows and heads of x * inv * du
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from bf16_ulp import assert_within_bf16_ulp

pytestmark = pytest.mark.requires_device

# out and dgate are within 0.52 ULP over every shape below (bf16 rounding of the result), dx peaks at
# 0.50, so these leave ~2x headroom. dx's p99 reaches 2.2 on elements where gamma*du and t*x cancel.
MAX_ULP = 1.0
MAX_ULP_P99 = 1.0
MAX_ULP_P99_DX = 4.0
# dgamma has only V elements, each a sum of B*T*H signed terms, so near-zero entries have unbounded
# per-element ULP; only the error at the peak (observed <= 0.91) is bounded.
MAX_ULP_DGAMMA = 2.0
EPS = 1e-6

# (batch, T, H, V). V/32 of 1, 2, 4, 8 covers group widths of 1 to 8 tiles; (1, 256, 12, 128) and
# (1, 1024, 12, 128) are the Qwen3.8 TP=4 head layout, where cores own several items that cross from
# one head and tile-row to the next; batch 2 checks rows are counted over the batch; 131 tile-rows of
# a single head make 131 items, which no Wormhole or Blackhole grid divides evenly, so both compute
# groups run and the runtime-arg walk crosses the group boundary.
SHAPES = [
    (1, 32, 1, 32),
    (1, 64, 3, 128),
    (2, 96, 4, 64),
    (1, 64, 2, 256),
    (1, 256, 12, 128),
    (1, 1024, 12, 128),
    (1, 4192, 1, 32),
]
AUTOGRAD_SHAPES = [(1, 64, 3, 128), (2, 32, 4, 64)]


def to_device(array, requires_grad=False):
    tensor = ttml.autograd.Tensor.from_numpy(array.astype(np.float32), layout=ttnn.Layout.TILE)
    tensor.set_requires_grad(requires_grad)
    return tensor


def to_host(ttnn_tensor):
    return ttml.autograd.create_tensor(ttnn_tensor).to_numpy().astype(np.float64)


def as_stored(array):
    """``array`` as bf16 on device, so the oracle sees what the kernel received."""
    return to_host(to_device(array).get_value())


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def split_heads(a, num_heads):
    *lead, width = a.shape
    return a.reshape(*lead, num_heads, width // num_heads)


def reference_forward(x, gate, gamma, eps, num_heads):
    xh, gh = split_heads(x, num_heads), split_heads(gate, num_heads)
    inv = 1.0 / np.sqrt(np.mean(xh**2, axis=-1, keepdims=True) + eps)
    out = xh * gamma.reshape(-1) * inv * gh * sigmoid(gh)
    return out.reshape(x.shape)


def reference_backward(x, gate, gamma, dy, eps, num_heads):
    xh, gh, dyh = (split_heads(a, num_heads) for a in (x, gate, dy))
    group = xh.shape[-1]
    gamma_v = gamma.reshape(-1)
    inv = 1.0 / np.sqrt(np.mean(xh**2, axis=-1, keepdims=True) + eps)
    s = sigmoid(gh)
    silu = gh * s
    u = xh * gamma_v * inv
    du = dyh * silu
    t = np.sum(u * du, axis=-1, keepdims=True) * inv / group
    dx = inv * (gamma_v * du - t * xh)
    dgate = dyh * u * s * (1.0 + gh - silu)
    dgamma = np.sum(xh * inv * du, axis=tuple(range(xh.ndim - 1))).reshape(1, 1, 1, group)
    return dx.reshape(x.shape), dgate.reshape(x.shape), dgamma


def inputs(batch, seq, heads, group, seed):
    """Heads get RMS spread over 2**-2 .. 2**3, so normalizing over the wrong columns cannot pass."""
    rng = np.random.default_rng(seed)
    shape = (batch, 1, seq, heads * group)
    head_scale = np.repeat(2.0 ** (np.arange(heads) % 6 - 2), group)
    return (
        rng.normal(0.0, 1.0, shape) * head_scale,
        rng.uniform(-4.0, 4.0, shape),  # gate, spanning both silu saturation tails
        rng.uniform(0.5, 1.5, (1, 1, 1, group)),
        rng.normal(0.0, 1.0, shape),
    )


def run_forward(x, gate, gamma, eps=EPS):
    out = ttml.ops.metal.gated_rmsnorm_fw(
        to_device(x).get_value(), to_device(gate).get_value(), to_device(gamma).get_value(), eps
    )
    return to_host(out)


def run_backward(x, gate, gamma, dy, compute_dgamma, eps=EPS):
    dx, dgate, dgamma = ttml.ops.metal.gated_rmsnorm_bw(
        to_device(x).get_value(),
        to_device(gate).get_value(),
        to_device(gamma).get_value(),
        to_device(dy).get_value(),
        eps,
        compute_dgamma,
    )
    return to_host(dx), to_host(dgate), None if dgamma is None else to_host(dgamma)


def assert_forward(got, x, gate, gamma, heads, label, eps=EPS):
    expected = reference_forward(as_stored(x), as_stored(gate), as_stored(gamma), eps, heads)
    assert_within_bf16_ulp(got, expected, label, MAX_ULP, MAX_ULP_P99)


def assert_backward(got, x, gate, gamma, dy, heads, label, eps=EPS):
    dx, dgate, dgamma = got
    ref_dx, ref_dgate, ref_dgamma = reference_backward(
        as_stored(x), as_stored(gate), as_stored(gamma), as_stored(dy), eps, heads
    )
    assert_within_bf16_ulp(dx, ref_dx, f"{label} dx", MAX_ULP, MAX_ULP_P99_DX)
    assert_within_bf16_ulp(dgate, ref_dgate, f"{label} dgate", MAX_ULP, MAX_ULP_P99)
    if dgamma is not None:
        assert_within_bf16_ulp(dgamma, ref_dgamma, f"{label} dgamma", MAX_ULP_DGAMMA)


class TestForward:
    @pytest.mark.parametrize("batch,seq,heads,group", SHAPES)
    def test_matches_reference(self, batch, seq, heads, group):
        x, gate, gamma, _ = inputs(batch, seq, heads, group, seed=100 + heads * group)
        out = run_forward(x, gate, gamma)
        assert out.shape == x.shape
        assert_forward(out, x, gate, gamma, heads, f"fw {batch}x{seq}x{heads}x{group}")

    def test_epsilon_dominates_tiny_inputs(self):
        # mean(x^2) ~ 1e-4 against eps = 0.1: dropping or misreading eps would change out ~30x.
        x, gate, gamma, _ = inputs(1, 64, 2, 64, seed=7)
        x = x * 1e-2
        assert_forward(run_forward(x, gate, gamma, eps=0.1), x, gate, gamma, 2, "fw eps=0.1", eps=0.1)


class TestBackward:
    @pytest.mark.parametrize("batch,seq,heads,group", SHAPES)
    def test_matches_reference(self, batch, seq, heads, group):
        x, gate, gamma, dy = inputs(batch, seq, heads, group, seed=200 + heads * group)
        got = run_backward(x, gate, gamma, dy, compute_dgamma=True)
        assert got[0].shape == x.shape and got[1].shape == x.shape
        assert got[2].shape == (1, 1, 1, group)
        assert_backward(got, x, gate, gamma, dy, heads, f"bw {batch}x{seq}x{heads}x{group}")

    def test_skips_dgamma_when_not_requested(self):
        x, gate, gamma, dy = inputs(1, 64, 3, 128, seed=17)
        got = run_backward(x, gate, gamma, dy, compute_dgamma=False)
        assert got[2] is None
        assert_backward(got, x, gate, gamma, dy, 3, "bw no dgamma")

    def test_epsilon_dominates_tiny_inputs(self):
        x, gate, gamma, dy = inputs(1, 64, 2, 64, seed=8)
        x = x * 1e-2
        got = run_backward(x, gate, gamma, dy, compute_dgamma=True, eps=0.1)
        assert_backward(got, x, gate, gamma, dy, 2, "bw eps=0.1", eps=0.1)


class TestProgramCache:
    """A second launch of a shape hits the cached program, so only override_runtime_arguments hands
    the kernels the new buffer addresses. Every launch's tensors stay alive until the end so a
    later one cannot land at an earlier one's address."""

    def test_forward_reads_the_new_buffers(self):
        alive = []
        for seed in (1, 2):
            x, gate, gamma, _ = inputs(1, 64, 3, 128, seed=seed)
            dev = [to_device(a) for a in (x, gate, gamma)]
            out = ttml.ops.metal.gated_rmsnorm_fw(*(t.get_value() for t in dev), EPS)
            alive.append((dev, out))
            assert_forward(to_host(out), x, gate, gamma, 3, f"fw relaunch seed={seed}")

    def test_backward_reads_the_new_buffers(self):
        alive = []
        for seed in (1, 2):
            x, gate, gamma, dy = inputs(1, 64, 3, 128, seed=seed)
            dev = [to_device(a) for a in (x, gate, gamma, dy)]
            outs = ttml.ops.metal.gated_rmsnorm_bw(*(t.get_value() for t in dev), EPS, True)
            alive.append((dev, outs))
            got = tuple(to_host(t) for t in outs)
            assert_backward(got, x, gate, gamma, dy, 3, f"bw relaunch seed={seed}")

    def test_epsilon_is_part_of_the_program(self):
        # eps is a compile-time arg; a cache keyed without it would reuse the first eps here.
        x, gate, gamma, _ = inputs(1, 64, 2, 64, seed=9)
        x = x * 1e-2
        for eps in (1e-6, 0.1):
            assert_forward(run_forward(x, gate, gamma, eps=eps), x, gate, gamma, 2, f"fw eps={eps}", eps=eps)

    def test_dgamma_flag_is_part_of_the_program(self):
        x, gate, gamma, dy = inputs(1, 64, 3, 128, seed=10)
        for compute_dgamma in (False, True, False):
            got = run_backward(x, gate, gamma, dy, compute_dgamma=compute_dgamma)
            assert (got[2] is not None) == compute_dgamma
            assert_backward(got, x, gate, gamma, dy, 3, f"bw compute_dgamma={compute_dgamma}")


class TestAutogradWrapper:
    """`ttml.ops.gated_rmsnorm.gated_rmsnorm` wires both metal ops into the graph."""

    @pytest.mark.parametrize("batch,seq,heads,group", AUTOGRAD_SHAPES)
    def test_forward(self, batch, seq, heads, group):
        x, gate, gamma, _ = inputs(batch, seq, heads, group, seed=300 + group)
        out = ttml.ops.gated_rmsnorm.gated_rmsnorm(to_device(x), to_device(gate), to_device(gamma), EPS)
        assert_forward(out.to_numpy().astype(np.float64), x, gate, gamma, heads, f"autograd fw {seq}x{heads}x{group}")

    @pytest.mark.parametrize("batch,seq,heads,group", AUTOGRAD_SHAPES)
    @pytest.mark.parametrize("train_gamma", [True, False], ids=["trained_gamma", "frozen_gamma"])
    def test_backward(self, batch, seq, heads, group, train_gamma):
        x, gate, gamma, _ = inputs(batch, seq, heads, group, seed=400 + group)
        x_dev = to_device(x, requires_grad=True)
        gate_dev = to_device(gate, requires_grad=True)
        gamma_dev = to_device(gamma, requires_grad=train_gamma)

        ttml.ops.gated_rmsnorm.gated_rmsnorm(x_dev, gate_dev, gamma_dev, EPS).backward(retain_graph=False)

        assert x_dev.is_grad_initialized() and gate_dev.is_grad_initialized()
        assert gamma_dev.is_grad_initialized() == train_gamma
        dgamma = gamma_dev.get_grad_tensor().to_numpy().astype(np.float64) if train_gamma else None
        got = (
            x_dev.get_grad_tensor().to_numpy().astype(np.float64),
            gate_dev.get_grad_tensor().to_numpy().astype(np.float64),
            dgamma,
        )
        ones = np.ones(x.shape)  # backward() seeds dL/dout with ones
        assert_backward(got, x, gate, gamma, ones, heads, f"autograd bw {seq}x{heads}x{group}")


class TestValidation:
    def test_rejects_group_width_that_is_not_tile_aligned(self, expect_error):
        x = to_device(np.zeros((1, 1, 32, 96)))
        gamma = to_device(np.ones((1, 1, 1, 48)))
        with expect_error(RuntimeError, "multiple of 32"):
            ttml.ops.metal.gated_rmsnorm_fw(x.get_value(), x.get_value(), gamma.get_value())

    def test_rejects_width_that_is_not_whole_heads(self, expect_error):
        x = to_device(np.zeros((1, 1, 32, 96)))
        gamma = to_device(np.ones((1, 1, 1, 64)))
        with expect_error(RuntimeError, "multiple of V"):
            ttml.ops.metal.gated_rmsnorm_fw(x.get_value(), x.get_value(), gamma.get_value())

    def test_rejects_partial_tile_rows(self, expect_error):
        x = to_device(np.zeros((1, 1, 48, 64)))
        gamma = to_device(np.ones((1, 1, 1, 64)))
        with expect_error(RuntimeError, "T = 48"):
            ttml.ops.metal.gated_rmsnorm_fw(x.get_value(), x.get_value(), gamma.get_value())

    def test_rejects_gate_of_another_shape(self, expect_error):
        x = to_device(np.zeros((1, 1, 32, 128)))
        gate = to_device(np.zeros((1, 1, 64, 128)))
        gamma = to_device(np.ones((1, 1, 1, 64)))
        with expect_error(RuntimeError, "gate shape"):
            ttml.ops.metal.gated_rmsnorm_fw(x.get_value(), gate.get_value(), gamma.get_value())

    def test_rejects_upstream_grad_of_another_shape(self, expect_error):
        x = to_device(np.zeros((1, 1, 32, 128)))
        dy = to_device(np.zeros((1, 1, 32, 64)))
        gamma = to_device(np.ones((1, 1, 1, 64)))
        with expect_error(RuntimeError, "dL_dout shape"):
            ttml.ops.metal.gated_rmsnorm_bw(x.get_value(), x.get_value(), gamma.get_value(), dy.get_value())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
