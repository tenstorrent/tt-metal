# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for gated_delta_net_backward — the immutable spec.

DO NOT MODIFY. The implementer makes this pass; it does not move to meet the
implementation.

What it pins
------------
1. The six gradients (dq, dk, dv, dg, dbeta, dh0) of the chunked gated delta
   rule, against the float64 autograd oracle in the golden suite
   (`eval/golden_tests/gated_delta_net_backward/helpers.py`). That oracle is a
   dtype-clean transcription of the in-tree torch reference and passes
   `torch.autograd.gradcheck`, so it is the definition, not an approximation.
2. `dh0 is None` — not a zero tensor — when `initial_state is None`.
3. `dq` is EXACTLY zero when `do` is zero, for every input: `q` feeds only the
   output and never the recurrent state, so the `dht` path cannot produce a
   `dq`. Gated on magnitude, because PCC against a constant-zero reference is
   undefined.
4. The `dht` boundary condition of the reverse scan. Dropping `dht` passes most
   cells (the `do` path dominates) and fails only these — they are load-bearing.
5. Ragged sequences (`T % chunk_size != 0`): exactly `T` rows out, with the
   padded positions masked out of every gradient.
6. `block_val_tiles < Vt` — the `(1,256,4,128,256)` cell is the only shape in
   this file whose L1 footprint forces the V-block loop, so it is the only one
   that exercises `num_v_blocks > 1`. Keep it.

Contract requirements honoured by every builder here
----------------------------------------------------
* `q` and `k` are L2-normalized along the last dim. Without this the UT
  transform is not contractive and the forward diverges (measured |o|max of
  2.9e18). This is a contract requirement, not a normalization convenience.
* Any case that exercises `dht` uses a weak decay (`g_scale <= 0.05`). At the
  default gate distribution `|dh0| ~= exp(sum(g)) * |dht|` lands around 1e-23,
  and the test would be asserting on numerical noise.

Tolerances are the golden suite's, keyed by dtype
(`helpers.py::TOLERANCES`): float32 -> PCC 0.999 / rel-RMS 0.02,
bfloat16 -> PCC 0.99 / rel-RMS 0.12. `dg` is the weakest gradient at every
precision — it is the only one through `exp()` of a cumulative sum.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

from eval.golden_tests.gated_delta_net_backward.helpers import (
    pytorch_gated_delta_net_backward,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout

GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dh0")

# Same bands as eval/golden_tests/.../helpers.py::TOLERANCES. Do not tighten.
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.99}
RMS = {ttnn.float32: 0.02, ttnn.bfloat16: 0.12}
ZERO_ATOL = {ttnn.float32: 1e-6, ttnn.bfloat16: 1e-2}

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# (B, T, H, K, V), chunk_size
SHAPES = [
    ((1, 32, 1, 32, 32), 32),  # single tile, single chunk, single head
    ((1, 128, 2, 64, 64), 32),  # multi-tile, 4 chunks
    ((1, 128, 2, 64, 128), 64),  # non-square heads (wide_v), chunk 64
    ((2, 64, 4, 64, 64), 32),  # multi-batch, multi-head
    ((1, 100, 2, 64, 64), 64),  # ragged tail (T % chunk_size == 36)
    ((1, 256, 4, 128, 256), 64),  # largest state; forces num_v_blocks > 1
]

SHAPE_IDS = [f"B{s[0]}_T{s[1]}_H{s[2]}_K{s[3]}_V{s[4]}_c{c}" for s, c in SHAPES]


def _build_inputs(shape, state_mode, *, g_scale, zero_do=False):
    """Seeded input set. q and k are L2-normalized along K — contract, not style."""
    torch.manual_seed(42)
    B, T, H, K, V = shape

    def l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    out = {
        "q": l2(torch.randn(B, T, H, K, dtype=torch.float64)),
        "k": l2(torch.randn(B, T, H, K, dtype=torch.float64)),
        "v": torch.randn(B, T, H, V, dtype=torch.float64),
        "g": F.logsigmoid(torch.randn(B, T, H, dtype=torch.float64)) * g_scale,
        "beta": torch.rand(B, T, H, dtype=torch.float64),
        "do": (
            torch.zeros(B, T, H, V, dtype=torch.float64) if zero_do else torch.randn(B, T, H, V, dtype=torch.float64)
        ),
        "h0": None,
        "dht": None,
    }
    if state_mode in ("with_h0", "with_h0_and_dht"):
        out["h0"] = torch.randn(B, H, K, V, dtype=torch.float64) * 0.1
    if state_mode == "with_h0_and_dht":
        out["dht"] = torch.randn(B, H, K, V, dtype=torch.float64)
    return out


def _to_device(tensor, device, dtype):
    if tensor is None:
        return None
    return ttnn.from_torch(
        tensor.to(TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _rel_rms(got, exp):
    denom = exp.double().pow(2).mean().sqrt()
    if denom == 0:
        return float(got.double().pow(2).mean().sqrt())
    return float((got.double() - exp.double()).pow(2).mean().sqrt() / denom)


def _run(shape, chunk_size, dtype, state_mode, device, *, g_scale=None, zero_do=False):
    if g_scale is None:
        g_scale = 0.02 if state_mode == "with_h0_and_dht" else 0.5

    ref_in = _build_inputs(shape, state_mode, g_scale=g_scale, zero_do=zero_do)
    expected = pytorch_gated_delta_net_backward(
        ref_in["q"],
        ref_in["k"],
        ref_in["v"],
        ref_in["g"],
        ref_in["beta"],
        ref_in["do"],
        dht=ref_in["dht"],
        initial_state=ref_in["h0"],
        chunk_size=chunk_size,
    )

    got = gated_delta_net_backward(
        _to_device(ref_in["q"], device, dtype),
        _to_device(ref_in["k"], device, dtype),
        _to_device(ref_in["v"], device, dtype),
        _to_device(ref_in["g"], device, dtype),
        _to_device(ref_in["beta"], device, dtype),
        _to_device(ref_in["do"], device, dtype),
        dht=_to_device(ref_in["dht"], device, dtype),
        initial_state=_to_device(ref_in["h0"], device, dtype),
        chunk_size=chunk_size,
    )

    assert (
        isinstance(got, (tuple, list)) and len(got) == 6
    ), f"expected a 6-tuple (dq, dk, dv, dg, dbeta, dh0), got {type(got)}"

    B, T, H, K, V = shape
    shapes = {
        "dq": (B, T, H, K),
        "dk": (B, T, H, K),
        "dv": (B, T, H, V),
        "dg": (B, T, H),
        "dbeta": (B, T, H),
        "dh0": (B, H, K, V),
    }

    for name, dev_t, exp in zip(GRAD_NAMES, got, expected):
        if exp is None:
            assert dev_t is None, f"{name}: expected None (initial_state is None), got a tensor"
            continue
        assert dev_t is not None, f"{name}: op returned None but the reference produced a gradient"

        host = ttnn.to_torch(dev_t).to(torch.float64)
        assert tuple(host.shape) == shapes[name], f"{name}: shape {tuple(host.shape)} != {shapes[name]}"

        if not exp.any():
            # An identically-zero reference gradient has no correlation
            # structure, so PCC is undefined. Gate on magnitude instead.
            max_abs = float(host.abs().max())
            assert max_abs <= ZERO_ATOL[dtype], f"{name}: reference is identically zero but |device| max is {max_abs}"
            continue

        ok, msg = check_with_pcc_without_tensor_printout(exp, host, PCC[dtype])
        assert ok, f"[{name}] {msg}"
        rms = _rel_rms(host, exp)
        assert rms <= RMS[dtype], f"[{name}] rel-RMS {rms:.4g} > {RMS[dtype]}"

    return got, expected


@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("state_mode", ["do_only", "with_h0", "with_h0_and_dht"])
def test_gated_delta_net_backward(shape, chunk_size, dtype, state_mode, device):
    _run(shape, chunk_size, dtype, state_mode, device)


@pytest.mark.parametrize("shape,chunk_size", SHAPES[:4], ids=SHAPE_IDS[:4])
def test_dh0_is_none_without_initial_state(shape, chunk_size, device):
    """`dh0` must be None, not a zero tensor, when initial_state is None.

    A zero tensor type-checks downstream and hides a missing gradient path.
    """
    got, _ = _run(shape, chunk_size, ttnn.float32, "do_only", device)
    assert got[5] is None, "dh0 must be None when initial_state is None"


@pytest.mark.parametrize(
    "shape,chunk_size",
    [((1, 128, 2, 64, 64), 64), ((1, 64, 2, 64, 128), 32)],
    ids=["T128_c64", "T64_wide_v_c32"],
)
def test_dht_only_path(shape, chunk_size, device):
    """Zero `do` + non-zero `dht`: every gradient arrives through the reverse
    scan's boundary condition.

    An implementation that drops `dht` scores 0 here instead of "slightly off".
    `g_scale` must stay small: |dh0| ~= exp(sum(g)) * |dht|, so at the default
    decay every gradient is numerical noise.

    This also pins the measured invariant that `dq` is EXACTLY zero whenever
    `do` is zero — `q` feeds only the output, never the state.
    """
    got, expected = _run(
        shape,
        chunk_size,
        ttnn.float32,
        "with_h0_and_dht",
        device,
        g_scale=0.01,
        zero_do=True,
    )
    assert not expected[0].any(), "oracle invariant broken: dq must be zero when do is zero"
    dq = ttnn.to_torch(got[0]).to(torch.float64)
    assert float(dq.abs().max()) <= ZERO_ATOL[ttnn.float32], (
        "dq must be identically zero when do is zero — state gradient is "
        f"leaking into the query path (|dq|max = {float(dq.abs().max())})"
    )
    # The dht path must actually carry signal, or this test asserts nothing.
    assert float(expected[5].abs().max()) > 1e-6, "dht path produced no dh0 signal"


@pytest.mark.parametrize(
    "shape,chunk_size",
    [((1, 33, 1, 32, 32), 32), ((1, 65, 1, 64, 64), 64), ((1, 100, 2, 64, 64), 32)],
    ids=["tail1_c32", "tail1_c64", "tail4_c32"],
)
@pytest.mark.parametrize("state_mode", ["do_only", "with_h0_and_dht"])
def test_ragged_tail(shape, chunk_size, state_mode, device):
    """`T % chunk_size != 0`: exactly T rows out, padded positions contributing
    nothing to any of the six gradients."""
    got, _ = _run(shape, chunk_size, ttnn.float32, state_mode, device)
    B, T, H, K, V = shape
    assert tuple(ttnn.to_torch(got[0]).shape) == (B, T, H, K)
    assert tuple(ttnn.to_torch(got[3]).shape) == (B, T, H)


@pytest.mark.parametrize("chunk_size", [32, 64], ids=["c32", "c64"])
def test_chunk_size_is_an_internal_tiling_choice(chunk_size, device):
    """Both chunk sizes must reach the same mathematics on the same input.

    chunk_size changes the blocking, not the function — the two results must
    agree with each other as well as with the oracle.
    """
    shape = (1, 128, 2, 64, 64)
    _run(shape, chunk_size, ttnn.float32, "with_h0", device)
