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

import ttml
from bf16_ulp import assert_within_bf16_ulp
from tensor_helpers import as_stored, sigmoid, to_device, to_host

pytestmark = pytest.mark.requires_device

# Observed worst case is 1.24 ULP over every shape below, so this leaves ~2x headroom.
MAX_ULP = 2.5
# Per-element ULP is unbounded where the oracle vanishes: silu'(gate) -> 0 for negative gate
# drives dgate's per-element error to ~164 on a correct kernel, while its p99 stays near 6.
MAX_ULP_P99 = 2.5
MAX_ULP_P99_DGATE = 10.0
# The transposed-IO backward's dh carries the low-rank term: the FPU's fp32 sum lands within one bf16 rounding of
# the numpy reference's, which the elementwise pass turns into up to one more ULP at the peak.
MAX_ULP_LOWRANK = 3.5

# (batch, S, I). I/32 of 1..4 covers every block size get_block_size(Wt, 4) can pick; I=512
# adds a row spanning many blocks; (2, 2048, 128) is the only case with batch > 1; 131 tile-rows
# of I=32 make 131 blocks, which neither an 8x8 nor a 13x10 grid divides evenly, so both compute
# groups run and the runtime-arg walk crosses the group boundary.
SHAPES = [(1, 32, 32), (1, 64, 64), (1, 32, 96), (1, 128, 128), (1, 64, 512), (2, 2048, 128), (1, 4192, 32)]
AUTOGRAD_SHAPES = [(1, 64, 64), (2, 128, 128)]


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
    assert_within_bf16_ulp(got, reference_forward(as_stored(gate), as_stored(up)), label, MAX_ULP, MAX_ULP_P99)


def assert_backward(got, gate, up, dh, inner, label, max_ulp=MAX_ULP):
    dgate, dup = reference_backward(as_stored(gate), as_stored(up), as_stored(dh))
    assert_within_bf16_ulp(got[..., :inner], dgate, f"{label} dgate", max_ulp, MAX_ULP_P99_DGATE)
    assert_within_bf16_ulp(got[..., inner:], dup, f"{label} dup", max_ulp, MAX_ULP_P99)


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


class TestProgramCache:
    """A second launch of a shape hits the cached program, so only override_runtime_arguments hands
    the kernels the new buffer addresses. Every launch's tensors stay alive until the end so a
    later one cannot land at an earlier one's address."""

    def test_forward_reads_the_new_buffers(self):
        alive = []
        for seed in (1, 2):
            gate, up, _ = inputs(1, 64, 64, seed=seed)
            packed = to_device(pack(gate, up))
            h = ttml.ops.metal.swiglu_packed_fw(packed.get_value())
            alive.append((packed, h))
            assert_forward(to_host(h), gate, up, f"fw relaunch seed={seed}")

    def test_backward_reads_the_new_buffers(self):
        alive = []
        for seed in (1, 2):
            gate, up, dh = inputs(1, 64, 64, seed=seed)
            packed, dh_dev = to_device(pack(gate, up)), to_device(dh)
            dpacked = ttml.ops.metal.swiglu_packed_bw(packed.get_value(), dh_dev.get_value())
            alive.append((packed, dh_dev, dpacked))
            assert_backward(to_host(dpacked), gate, up, dh, 64, f"bw relaunch seed={seed}")


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


class TestTransposedIO:
    """Iteration 39: packed^T [1, 1, 2I, T] in (the gate_up output of a transposed-orientation forward), h^T
    [1, 1, I + 64, T] out with the tail rows zeroed (the transposed wide w2 input), and the backward from the wide
    gradient dh^T [1, 1, I + 64, T] (rows [I, I + 32) the LoRA tail s_dh^T) to dpacked^T [1, 1, 2I, T] with the
    low-rank term dh^T += scale * A^T s_dh^T folded in. Checked against the token-major ops on the same data, which
    read the same bf16 tiles and run the same math (the backward's low-rank term against numpy, bf16-rounded the
    way the kernel rounds dh before the elementwise pass)."""

    # (I, T): the TP shard width at the training token count, plus small shapes where the blocks per row and the
    # cores with work vary (T = 96 gives block_size 3, I = 64 a single I-row per half).
    SHAPES_T = [(64, 32), (96, 96), (512, 1024), (8192, 1024)]
    RANK, SCALE = 16, 0.25

    @staticmethod
    def _transposed(array):
        return np.ascontiguousarray(np.swapaxes(array, -1, -2))

    @pytest.mark.parametrize("inner,seq", SHAPES_T)
    def test_forward_matches_the_token_major_op(self, inner, seq):
        gate, up, _ = inputs(1, seq, inner, seed=300 + inner)
        out_cols = inner + 64
        packed_t = to_device(self._transposed(pack(gate, up)))
        h_t = ttml.ops.metal.swiglu_packed_fw(packed_t.get_value(), None, out_cols, transposed_input=True)
        assert list(h_t.shape) == [1, 1, out_cols, seq]
        got = to_host(h_t)
        h = to_host(ttml.ops.metal.swiglu_packed_fw(to_device(pack(gate, up)).get_value()))
        assert np.array_equal(self._transposed(got[:, :, :inner, :]), h), "h^T differs from the token-major h"
        assert np.all(got[:, :, inner:, :] == 0.0), "the tail rows are not zeroed"
        assert_forward(self._transposed(got[:, :, :inner, :]), gate, up, f"fw^T {inner}x{seq}")

    @pytest.mark.parametrize("inner,seq", SHAPES_T)
    def test_backward_matches_the_token_major_op_with_the_low_rank_term(self, inner, seq):
        gate, up, dh = inputs(1, seq, inner, seed=400 + inner)
        rng = np.random.default_rng(500 + inner)
        a = rng.uniform(-0.5, 0.5, (1, 1, self.RANK, inner))
        s_dh = rng.uniform(-1.0, 1.0, (1, 1, seq, self.RANK))
        tail = np.zeros((1, 1, seq, 64))
        tail[..., : self.RANK] = s_dh
        dh_wide_t = to_device(self._transposed(np.concatenate([dh, tail], axis=-1)))  # [1, 1, I + 64, T]
        dpacked_t = ttml.ops.metal.swiglu_packed_bw(
            to_device(self._transposed(pack(gate, up))).get_value(),
            dh_wide_t.get_value(),
            lora_a=to_device(a).get_value(),
            lowrank_scale=self.SCALE,
            transposed_io=True,
        )
        assert list(dpacked_t.shape) == [1, 1, 2 * inner, seq]
        got = self._transposed(to_host(dpacked_t))
        # The kernel's dh: the bf16 tiles plus the fp32 low-rank product, rounded to bf16 once (the FPU's fp32
        # accumulation lands within one bf16 rounding of numpy's float64 sum; with lora_a = 0 the ops agree bit
        # for bit, `O/iter39_swiglu_diag.log`).
        dh_full = as_stored(as_stored(dh) + self.SCALE * as_stored(s_dh) @ as_stored(a))
        assert_backward(got, gate, up, dh_full, inner, f"bw^T {inner}x{seq}", MAX_ULP_LOWRANK)
        ref = to_host(
            ttml.ops.metal.swiglu_packed_bw(to_device(pack(gate, up)).get_value(), to_device(dh_full).get_value())
        )
        assert_within_bf16_ulp(got, ref, f"bw^T vs token-major {inner}x{seq}", MAX_ULP_LOWRANK, MAX_ULP_LOWRANK)

    def test_rejects_padding_and_mixed_modes(self, expect_error):
        packed_t = to_device(np.zeros((1, 1, 128, 32)))
        with expect_error(RuntimeError, "exclusive"):
            ttml.ops.metal.swiglu_packed_fw(
                packed_t.get_value(), None, 128, transposed_output=True, transposed_input=True
            )
        with expect_error(RuntimeError, "needs lora_a"):
            ttml.ops.metal.swiglu_packed_bw(
                packed_t.get_value(), to_device(np.zeros((1, 1, 96, 32))).get_value(), transposed_io=True
            )
        with expect_error(RuntimeError, "no padding"):
            ttml.ops.metal.swiglu_packed_fw(
                to_device(np.zeros((1, 1, 128, 20))).get_value(), None, 128, transposed_input=True
            )


class TestValidation:
    def test_rejects_half_width_that_is_not_tile_aligned(self, expect_error):
        # 96 pads to 96, whose halves are 1.5 tiles each.
        packed = to_device(np.zeros((1, 1, 32, 96)))
        with expect_error(RuntimeError, "multiple of"):
            ttml.ops.metal.swiglu_packed_fw(packed.get_value())

    def test_rejects_upstream_grad_that_is_not_half_the_packed_width(self, expect_error):
        packed = to_device(np.zeros((1, 1, 32, 128)))
        wrong = to_device(np.zeros((1, 1, 32, 32)))  # should be at least 64 (wider is the wide LoRA buffer)
        with expect_error(RuntimeError, ">= I"):
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
