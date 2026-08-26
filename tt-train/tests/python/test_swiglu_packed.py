# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the packed-SwiGLU metal ops and their autograd wrapper.

`packed` is [.., R, 2*I] = [gate | up]; the ops read the two column halves in place:
    h        = silu(gate) * up
    dL/dgate = dL/dh * up * silu'(gate),   silu'(x) = s * (1 + x * (1 - s)),  s = sigmoid(x)
    dL/dup   = dL/dh * silu(gate)
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.testing import assert_within_ulp

pytestmark = pytest.mark.requires_device

# Observed worst case is 1.24 ULP over every shape below, so this leaves ~2x headroom.
MAX_ULP = 2.5
# Per-element ULP is unbounded where the oracle vanishes: silu'(gate) -> 0 for negative gate
# drives dgate's per-element error to ~164 on a correct kernel, while its p99 stays near 6.
MAX_ULP_P99 = 2.5
MAX_ULP_P99_DGATE = 10.0

# (batch, S, I). I/32 of 1..4 covers every block size get_block_size(Wt, 4) can pick; I=512
# adds a row spanning many blocks; the last is the only case with batch > 1 and with
# total_rows (128) above the core count, which is what splits work into two compute groups.
SHAPES = [(1, 32, 32), (1, 64, 64), (1, 32, 96), (1, 128, 128), (1, 64, 512), (2, 2048, 128)]
AUTOGRAD_SHAPES = [(1, 64, 64), (2, 128, 128)]


def to_device(array):
    return ttml.autograd.Tensor.from_numpy(array.astype(np.float32), layout=ttnn.Layout.TILE)


def to_host(ttnn_tensor):
    return ttml.autograd.create_tensor(ttnn_tensor).to_numpy().astype(np.float64)


def as_stored(array):
    """``array`` as bf16 on device, so the oracle sees what the kernel received."""
    return to_host(to_device(array).get_value())


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def reference_forward(gate, up):
    return gate * sigmoid(gate) * up


def reference_backward(gate, up, dh):
    s = sigmoid(gate)
    return dh * up * (s * (1.0 + gate * (1.0 - s))), dh * gate * s


def inputs(batch, seq, inner, seed):
    rng = np.random.default_rng(seed)
    shape = (batch, 1, seq, inner)
    return (
        rng.uniform(-4.0, 4.0, shape),  # gate, spanning both silu saturation tails
        rng.uniform(-2.0, 2.0, shape),
        rng.uniform(-1.0, 1.0, shape),
    )


def pack(gate, up):
    return np.concatenate([gate, up], axis=-1)


def assert_forward(got, gate, up, label):
    assert_within_ulp(got, reference_forward(as_stored(gate), as_stored(up)), label, MAX_ULP, MAX_ULP_P99)


def assert_backward(got, gate, up, dh, inner, label):
    dgate, dup = reference_backward(as_stored(gate), as_stored(up), as_stored(dh))
    assert_within_ulp(got[..., :inner], dgate, f"{label} dgate", MAX_ULP, MAX_ULP_P99_DGATE)
    assert_within_ulp(got[..., inner:], dup, f"{label} dup", MAX_ULP, MAX_ULP_P99)


class TestForward:
    @pytest.mark.parametrize("batch,seq,inner", SHAPES)
    def test_matches_reference(self, batch, seq, inner):
        gate, up, _ = inputs(batch, seq, inner, seed=100 + inner)
        h = ttml.ops.metal.swiglu_packed_fw(to_device(pack(gate, up)).get_value())
        assert list(h.shape) == [batch, 1, seq, inner]
        assert_forward(to_host(h), gate, up, f"fw {batch}x{seq}x{inner}")

    def test_writes_into_preallocated_output(self):
        gate, up, _ = inputs(1, 64, 64, seed=3)
        preallocated = to_device(np.zeros((1, 1, 64, 64))).get_value()
        returned = ttml.ops.metal.swiglu_packed_fw(to_device(pack(gate, up)).get_value(), preallocated)
        assert_forward(to_host(returned), gate, up, "fw preallocated returned")
        assert_forward(to_host(preallocated), gate, up, "fw preallocated in place")


class TestBackward:
    @pytest.mark.parametrize("batch,seq,inner", SHAPES)
    def test_matches_reference(self, batch, seq, inner):
        gate, up, dh = inputs(batch, seq, inner, seed=200 + inner)
        dpacked = ttml.ops.metal.swiglu_packed_bw(to_device(pack(gate, up)).get_value(), to_device(dh).get_value())
        assert list(dpacked.shape) == [batch, 1, seq, 2 * inner]
        assert_backward(to_host(dpacked), gate, up, dh, inner, f"bw {batch}x{seq}x{inner}")

    def test_writes_into_preallocated_grad(self):
        gate, up, dh = inputs(1, 64, 64, seed=11)
        preallocated = to_device(np.zeros((1, 1, 64, 128))).get_value()
        returned = ttml.ops.metal.swiglu_packed_bw(
            to_device(pack(gate, up)).get_value(), to_device(dh).get_value(), preallocated
        )
        assert_backward(to_host(returned), gate, up, dh, 64, "bw preallocated returned")
        assert_backward(to_host(preallocated), gate, up, dh, 64, "bw preallocated in place")


class TestAutogradWrapper:
    """`ttml.ops.swiglu_packed.swiglu_packed` is the path LlamaMLP takes; it wires the two
    metal ops into the graph, which the direct-op tests above do not exercise."""

    @pytest.mark.parametrize("batch,seq,inner", AUTOGRAD_SHAPES)
    def test_forward(self, batch, seq, inner):
        gate, up, _ = inputs(batch, seq, inner, seed=300 + inner)
        out = ttml.ops.swiglu_packed.swiglu_packed(to_device(pack(gate, up)))
        assert_forward(out.to_numpy().astype(np.float64), gate, up, f"autograd fw {seq}x{inner}")

    @pytest.mark.parametrize("batch,seq,inner", AUTOGRAD_SHAPES)
    def test_backward(self, batch, seq, inner):
        gate, up, _ = inputs(batch, seq, inner, seed=400 + inner)
        packed = to_device(pack(gate, up))
        packed.set_requires_grad(True)

        ttml.ops.swiglu_packed.swiglu_packed(packed).backward(retain_graph=False)

        assert packed.is_grad_initialized(), "backward did not reach the packed input"
        ones = np.ones((batch, 1, seq, inner))  # backward() seeds dL/dh with ones
        assert_backward(
            packed.get_grad_tensor().to_numpy().astype(np.float64),
            gate,
            up,
            ones,
            inner,
            f"autograd bw {seq}x{inner}",
        )


class TestValidation:
    def test_rejects_half_width_that_is_not_tile_aligned(self, expect_error):
        # 96 pads to 96, whose halves are 1.5 tiles each.
        packed = to_device(np.zeros((1, 1, 32, 96)))
        with expect_error(RuntimeError, "multiple of"):
            ttml.ops.metal.swiglu_packed_fw(packed.get_value())

    def test_rejects_upstream_grad_that_is_not_half_the_packed_width(self, expect_error):
        packed = to_device(np.zeros((1, 1, 32, 128)))
        wrong = to_device(np.zeros((1, 1, 32, 32)))  # should be 64
        with expect_error(RuntimeError, "halved"):
            ttml.ops.metal.swiglu_packed_bw(packed.get_value(), wrong.get_value())

    # 100 pads to 128, so the padded halves are tile-aligned and only the logical split is wrong:
    # the kernel would read up at element 64 while the op's contract puts it at 50.
    def test_fw_rejects_a_logical_width_that_pads_into_alignment(self, expect_error):
        packed = to_device(np.zeros((1, 1, 32, 100)))
        with expect_error(RuntimeError, "logical dim"):
            ttml.ops.metal.swiglu_packed_fw(packed.get_value())

    def test_bw_rejects_a_logical_width_that_pads_into_alignment(self, expect_error):
        packed = to_device(np.zeros((1, 1, 32, 100)))
        dh = to_device(np.zeros((1, 1, 32, 50)))  # consistent with packed, both halves of a bad split
        with expect_error(RuntimeError, "logical dim"):
            ttml.ops.metal.swiglu_packed_bw(packed.get_value(), dh.get_value())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
