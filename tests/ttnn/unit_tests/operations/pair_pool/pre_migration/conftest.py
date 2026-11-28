# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared inputs, the independent float64 reference and the checks of the pair_pool pre-migration suite.

The helpers are handed to the tests through the ``pp`` fixture (no test imports this conftest)."""

from __future__ import annotations

import types

import pytest
import torch
import ttnn

# The golden suite's tolerance (ttnn/ttnn/bringup/pair_pool_ttnn/tests/unit/test_pair_pool.py): relative error of the
# whole output against float64 <= 1e-6.
REL_TOL = 1e-6
# Per element (added here, tighter than PCC, looser than the norm): an fp32 chain of ~7 roundings (max, subtract, exp,
# add, divide, multiply, add) cannot move an element by more than ~1e-5 of the magnitude it sums.
ELEM_TOL = 1e-5


def make_input(shape, seed, spread=3.0, specials=True):
    """x [.., R, 2 hd] = [kv | gate] float32; the gate rows carry exp underflow and equal-gate pairs (as the golden)."""
    *lead, rows, width = shape
    hd = width // 2
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(*lead, rows, hd, generator=g)
    gate = torch.randn(*lead, rows, hd, generator=g) * spread
    if specials:
        gate[..., 0, : min(8, hd)] = 90.0  # the other row's weight underflows
        if hd > 8:
            gate[..., 2, 8:16] = gate[..., 3, 8:16]  # equal gates: weights 1/2
    return torch.cat([kv, gate], dim=-1).float().contiguous()


def make_selector_input(shape, seed):
    """kv in [1, 2); the gates select one row of each pair hard (|gap| 60), by a pattern that differs per column, per
    column tile and per pair, so a parity, column-tile or pair-index mix-up picks another element."""
    *lead, rows, width = shape
    hd = width // 2
    g = torch.Generator().manual_seed(seed)
    kv = 1.0 + torch.rand(*lead, rows, hd, generator=g)
    pair = torch.arange(rows // 2).view(-1, 1)
    col = torch.arange(hd).view(1, -1)
    pick_odd = ((pair + col + col // 32) % 2).bool()  # True: the pair's odd row wins
    gate = torch.zeros(*lead, rows, hd)
    gate[..., 0::2, :] = torch.where(pick_odd, -30.0, 30.0)
    gate[..., 1::2, :] = torch.where(pick_odd, 30.0, -30.0)
    return torch.cat([kv, gate], dim=-1).float().contiguous()


def split_pairs(x):
    """float64 (kv0, g0, kv1, g1), each [.., R / 2, hd]."""
    x = x.double()
    hd = x.shape[-1] // 2
    return x[..., 0::2, :hd], x[..., 0::2, hd:], x[..., 1::2, :hd], x[..., 1::2, hd:]


def reference(x):
    """float64: out[t] = kv[2t] w0 + kv[2t+1] w1, (w0, w1) = softmax(gate[2t], gate[2t+1]) per column."""
    kv0, g0, kv1, g1 = split_pairs(x)
    w = torch.softmax(torch.stack([g0, g1]), dim=0)
    return kv0 * w[0] + kv1 * w[1]


def check(got, x, label=""):
    """``got`` (torch, [.., R / 2, hd]) against the float64 reference: shape, the golden's norm tolerance, a per-element
    bound, and the golden's negative controls (swapped weights, the gate ignored) must be far off."""
    want = reference(x)
    assert list(got.shape) == list(want.shape), f"{label}: shape {list(got.shape)} != {list(want.shape)}"
    assert torch.isfinite(got).all(), f"{label}: non-finite output"
    got = got.double()
    rel = ((got - want).norm() / want.norm()).item()
    assert rel <= REL_TOL, f"{label}: rel vs float64 {rel:.3e} > {REL_TOL}"
    kv0, g0, kv1, g1 = split_pairs(x)
    w = torch.softmax(torch.stack([g0, g1]), dim=0)
    bound = ELEM_TOL * (kv0.abs() * w[0] + kv1.abs() * w[1]) + 1e-12
    err = (got - want).abs()
    bad = err > bound
    assert not bad.any(), f"{label}: {int(bad.sum())} elements off, first at {bad.nonzero()[0].tolist()}"
    swapped = kv0 * w[1] + kv1 * w[0]
    no_gate = (kv0 + kv1) / 2
    for name, other in (("swapped weights", swapped), ("gate ignored", no_gate)):
        if not torch.allclose(other, want):
            assert ((got - other).norm() / other.norm()).item() > 1e-3, f"{label}: matches the '{name}' control"


def to_dev(x, device, memory_config=None, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        x,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def out_shape(shape):
    s = list(shape)
    s[-2] //= 2
    s[-1] //= 2
    return s


def alloc_out(shape, device, memory_config=None, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    """An output buffer for input ``shape`` (the output spec by default)."""
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape(shape)), dtype, layout, device, memory_config or ttnn.DRAM_MEMORY_CONFIG
    )


def assert_output_spec(out, shape, buffer_type=ttnn.BufferType.DRAM):
    assert list(out.shape) == out_shape(shape), (list(out.shape), out_shape(shape))
    assert list(out.padded_shape) == out_shape(shape), list(out.padded_shape)
    assert out.dtype == ttnn.float32, out.dtype
    assert out.layout == ttnn.TILE_LAYOUT, out.layout
    mc = out.memory_config()
    assert mc.buffer_type == buffer_type, mc
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc


@pytest.fixture
def pp():
    return types.SimpleNamespace(
        make_input=make_input,
        make_selector_input=make_selector_input,
        reference=reference,
        check=check,
        to_dev=to_dev,
        alloc_out=alloc_out,
        out_shape=out_shape,
        assert_output_spec=assert_output_spec,
    )
