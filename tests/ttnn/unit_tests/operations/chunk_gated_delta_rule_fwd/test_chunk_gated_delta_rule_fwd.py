# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for chunk_gated_delta_rule_fwd — the immutable spec.

DO NOT MODIFY. The implementer makes this pass; it does not move to meet the
implementation.

What it pins
------------
1. All six outputs (o, final_state, h, v_new, g_cumsum, A) against the float64
   oracle `eval/golden_tests/chunk_gated_delta_rule_fwd/helpers.py::
   pytorch_chunk_gated_delta_rule_fwd`, per output, on PCC AND relative RMS.
2. Every extent case the design's work split can take (op_design.md ->
   Work Distribution -> extent-pinned tests):
     NV > 1 with Vs = 1          (1,256,4,128,256) c64   -- also the largest state
     NV > 1 with Vs > 1          (1,256,32,128,128) c64  -- H = 32, one full head tile
     NV = 1 with Vs > 1          (4,128,16,64,64) c32    -- also > 1 item per core
     > 1 scan unit per core      (4,64,32,32,32) c32     -- B*H = 128 exceeds the grid
     NS = 1 (one chunk)          (1,32,1,32,32), (1,48,3,64,64)
     NS > 1                      every multi-chunk shape
3. Ragged tails (T % chunk_size != 0, including T < chunk_size): exactly T rows
   in every token-indexed output; final_state is the state after T tokens.
4. h[:, 0] is the initial state, and EXACTLY zero when initial_state is None.
5. The saturated-gate precision path (g_scale = 8: |decay| ~ 250 inside a chunk).
6. The keyword surface: explicit scale; the validation errors.

Inputs: q and k MUST be L2-normalized along K (caller contract; without it the
UT transform is not contractive). `make_reference_inputs` does that; it draws
from torch.randn under a generator seeded with 42.

Tolerances are the golden suite's (`helpers.TOLERANCES`), keyed by dtype:
float32 -> PCC 0.999 / rel-RMS 0.02; bfloat16 -> PCC 0.99 / rel-RMS 0.12.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.chunk_gated_delta_rule_fwd import chunk_gated_delta_rule_fwd

from eval.golden_tests.chunk_gated_delta_rule_fwd.helpers import (
    TOLERANCES,
    make_reference_inputs,
    pytorch_chunk_gated_delta_rule_fwd,
    quantize,
)

SEED = 42
OUTPUT_NAMES = ("o", "final_state", "h", "v_new", "g_cumsum", "A")
DTYPES = [ttnn.float32, ttnn.bfloat16]
DTYPE_IDS = ["fp32", "bf16"]

# ((B, T, H, K, V), chunk_size)
CASES = [
    ((1, 32, 1, 32, 32), 32),  # single tile, single chunk, single head
    ((1, 128, 2, 64, 64), 32),  # multi-chunk scan
    ((2, 64, 4, 64, 64), 32),  # multi-batch, multi-head
    ((1, 128, 2, 64, 128), 64),  # wide_v, chunk 64
    ((1, 100, 2, 64, 64), 64),  # ragged tail (T % C == 36)
    ((1, 48, 3, 64, 64), 64),  # T < C: one partial chunk, odd H
    ((1, 256, 4, 128, 256), 64),  # largest state; NV > 1, Vs = 1
    ((4, 128, 16, 64, 64), 32),  # NV = 1, Vs > 1; several items per core
    ((1, 256, 32, 128, 128), 64),  # H = 32; NV > 1, Vs > 1
    ((4, 64, 32, 32, 32), 32),  # B*H = 128 > grid: several scan units per core
]


def _case_id(case):
    shape, chunk = case
    return "x".join(str(d) for d in shape) + f"_c{chunk}"


def _to_dev(t, device, dtype, memory_config=None):
    if t is None:
        return None
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _prepare(shape, state_mode, dtype, device, *, g_scale=1.0):
    torch.manual_seed(SEED)
    ref = quantize(make_reference_inputs(shape, state_mode, g_scale=g_scale, seed=SEED), dtype)
    dev = {n: _to_dev(t, device, dtype) for n, t in ref.items()}
    return ref, dev


def _run(dev, chunk_size, **kw):
    return chunk_gated_delta_rule_fwd(
        dev["q"],
        dev["k"],
        dev["v"],
        dev["g"],
        dev["beta"],
        initial_state=dev["initial_state"],
        chunk_size=chunk_size,
        **kw,
    )


def _oracle(ref, chunk_size, **kw):
    return pytorch_chunk_gated_delta_rule_fwd(
        ref["q"],
        ref["k"],
        ref["v"],
        ref["g"],
        ref["beta"],
        initial_state=ref["initial_state"],
        chunk_size=chunk_size,
        **kw,
    )


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.allclose(a, b, atol=1e-6) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rel_rms(got, exp):
    got, exp = got.flatten().double(), exp.flatten().double()
    denom = exp.std().item()
    err = (got - exp).pow(2).mean().sqrt().item()
    return err / denom if denom > 0 else err


def _expected_shapes(shape, chunk):
    B, T, H, K, V = shape
    NC = (T + chunk - 1) // chunk
    return {
        "o": [B, T, H, V],
        "final_state": [B, H, K, V],
        "h": [B, NC, H, K, V],
        "v_new": [B, T, H, V],
        "g_cumsum": [B, T, H],
        "A": [B, T, H, chunk],
    }


def _check_all(outputs, expected, shape, chunk, dtype, *, skip=()):
    assert isinstance(outputs, (tuple, list)) and len(outputs) == 6, "must return a 6-tuple"
    pcc_t, rms_t = TOLERANCES[dtype]
    shapes = _expected_shapes(shape, chunk)
    failures = []
    for idx, name in enumerate(OUTPUT_NAMES):
        got = outputs[idx]
        assert got is not None, f"{name} is None — every output is required"
        assert isinstance(got, ttnn.Tensor), f"{name} is not a ttnn.Tensor"
        assert list(got.shape) == shapes[name], f"{name}: shape {list(got.shape)} != {shapes[name]}"
        assert got.dtype == dtype, f"{name}: dtype {got.dtype} != {dtype}"
        assert got.layout == ttnn.TILE_LAYOUT, f"{name}: layout must be TILE"
        if name in skip:
            continue
        g = ttnn.to_torch(got).double()
        e = expected[idx].double()
        assert torch.isfinite(g).all(), f"{name}: non-finite values"
        pcc, rms = _pcc(g, e), _rel_rms(g, e)
        if pcc < pcc_t or rms > rms_t:
            failures.append(f"{name}: pcc={pcc:.6f} (>= {pcc_t}) rel_rms={rms:.4g} (<= {rms_t})")
    assert not failures, "; ".join(failures)


# --- the main matrix -----------------------------------------------------------


@pytest.mark.parametrize("state_mode", ["no_h0", "with_h0"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_chunk_gated_delta_rule_fwd(case, dtype, state_mode, device):
    shape, chunk = case
    ref, dev = _prepare(shape, state_mode, dtype, device)
    outputs = _run(dev, chunk)
    _check_all(outputs, _oracle(ref, chunk), shape, chunk, dtype)


# --- the meaning of h -------------------------------------------------------------


def test_no_h0_first_state_is_exactly_zero(device):
    shape, chunk = (1, 128, 2, 64, 64), 32
    _ref, dev = _prepare(shape, "no_h0", ttnn.float32, device)
    h = ttnn.to_torch(_run(dev, chunk)[2]).double()
    assert h[:, 0].abs().max().item() == 0.0, "h[:, 0] must be exactly zero when initial_state is None"


def test_h0_is_first_state(device):
    shape, chunk = (1, 100, 2, 64, 128), 64
    ref, dev = _prepare(shape, "with_h0", ttnn.float32, device)
    h = ttnn.to_torch(_run(dev, chunk)[2]).double()
    h0 = ref["initial_state"].double()
    err = (h[:, 0] - h0).abs().max().item()
    assert err <= 1e-5 * max(1.0, h0.abs().max().item()), f"h[:, 0] != initial_state (max abs err {err:.3g})"


# --- precision: saturated gate ------------------------------------------------------


@pytest.mark.parametrize("case", [((1, 128, 2, 64, 64), 32), ((1, 128, 2, 64, 64), 64)], ids=["c32", "c64"])
def test_saturated_gate(case, device):
    """g_scale = 8: |decay| reaches ~250 inside a chunk. Every L[t, s] near the
    diagonal is a difference of two large cumulative sums; the design builds it
    from g directly (D = LT @ diag(g) @ SL) so this passes at the fp32 band."""
    shape, chunk = case
    ref, dev = _prepare(shape, "with_h0", ttnn.float32, device, g_scale=8.0)
    _check_all(_run(dev, chunk), _oracle(ref, chunk), shape, chunk, ttnn.float32)


# --- keyword surface -------------------------------------------------------------------


def test_explicit_scale(device):
    shape, chunk = (1, 128, 2, 64, 64), 64
    ref, dev = _prepare(shape, "with_h0", ttnn.float32, device)
    _check_all(_run(dev, chunk, scale=0.5), _oracle(ref, chunk, scale=0.5), shape, chunk, ttnn.float32)


def test_default_compute_kernel_config_is_exported():
    from ttnn.operations.chunk_gated_delta_rule_fwd import default_compute_kernel_config

    a, b = default_compute_kernel_config(), default_compute_kernel_config()
    assert a is not b, "must be a factory (fresh descriptor per call)"
    assert a.fp32_dest_acc_en is True
    assert a.math_approx_mode is False
    assert a.math_fidelity == ttnn.MathFidelity.HiFi4


# --- validation -------------------------------------------------------------------------


def _small(device, dtype=ttnn.float32):
    _ref, dev = _prepare((1, 64, 2, 32, 32), "no_h0", dtype, device)
    return dev


def test_rejects_chunk_size_not_multiple_of_32(device, expect_error):
    dev = _small(device)
    with expect_error(ValueError, ""):
        _run(dev, 48)


def test_rejects_shape_mismatch(device, expect_error):
    dev = _small(device)
    torch.manual_seed(SEED)
    dev["v"] = _to_dev(torch.randn(1, 64, 3, 32), device, ttnn.float32)  # H mismatch
    with expect_error(ValueError, ""):
        _run(dev, 32)


def test_rejects_bad_rank(device, expect_error):
    dev = _small(device)
    torch.manual_seed(SEED)
    dev["g"] = _to_dev(torch.randn(1, 64, 2, 1), device, ttnn.float32)  # g must be rank 3
    with expect_error(ValueError, ""):
        _run(dev, 32)


def test_rejects_mixed_dtype(device, expect_error):
    from ttnn.operations._op_contract import UnsupportedAxisValue

    dev = _small(device)
    torch.manual_seed(SEED)
    dev["k"] = _to_dev(torch.nn.functional.normalize(torch.randn(1, 64, 2, 32), dim=-1), device, ttnn.bfloat16)
    with expect_error(UnsupportedAxisValue, ""):
        _run(dev, 32)
