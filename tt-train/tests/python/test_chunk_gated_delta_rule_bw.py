# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.ops.metal.chunk_gated_delta_rule_bw: the fused backward of the chunked gated delta rule.

    q, k   [B, 1, T, HK*K]      v [B, 1, T, HV*V]      g, beta [B, 1, T, HV]
    d_out  [B, 1, T, HV*V], or head-major [B*HV, 1, T, V]
    dq, dk [B, 1, T, HK*K]      dv [B, 1, T, HV*V]     dg, dbeta [B, 1, T, HV]
    dh0    [B, HV, K, V]

Pinned here:
1. Every gradient against a float64 autograd oracle of the token-by-token recurrence.
2. Two live input sets at one shape each get their own answer from one cached program.
3. Shape-contract violations raise before dispatch.
4. Fed the saved h / v_new / A of ttml.ops.metal.chunk_gated_delta_rule_fw, every gradient still
   matches the oracle, including a Qwen3.8-sized shape with head-major d_out.
5. Head-major d_out [B*HV, 1, T, V] and rank-3 [B, T, X] inputs give bit-identical gradients.
6. Several items per core at Vb > 1 (every CB keeps a uniform push size across items).
7. Strong gates (|cumsum(g)| in the hundreds within a chunk) still match the oracle.
8. FLOAT32 gates with BFLOAT16 q/k/v/d_out (the forward's dtype contract) match the oracle, with and
   without the saved forward; dg / dbeta come back in q's dtype.
9. Weak gates (the default here) keep the inter-chunk state carry observable; one decaying-gate case
   covers the other regime.  See `WEAK_GATE`.
10. An explicit `scale` is checked against the oracle in absolute terms, not only by correlation.
11. Cache hits patch every buffer address, with all input and output sets alive at once, and a changed
    `initial_state` / `d_final_state` presence, saved forward or gate dtype misses the cache.
12. `d_final_state` without `initial_state`, GVA at G = 4, 8 and 32 (HV = 32), K > V, and every V-block
    geometry (`BLOCK_SHAPES`) match the oracle; T = 4096 with no decay keeps the error of T = 256.
13. Rejections name the offending value (exact text), for shapes, dtypes, layouts, placement, T = 0,
    B = 0, states and saved tensors of the wrong dtype or shape, and a shape with no V block that fits
    L1; a tensor placed differently (L1 instead of DRAM) is rejected even with the DRAM program cached.

Tolerances: every gradient must reach PCC >= `PCC_MIN` and a relative RMS error <= `RMS_MAX`, for both
dtypes.  Measured on Blackhole over all cases here the worst relative RMS is 1.0e-2 (`dg` at strong
gates; 8.8e-3 elsewhere) and the worst PCC is 0.99998; the bf16 rounding of the output alone is
2^-8, so the bounds sit about 2x above the measurement.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn.functional as F

import ttnn
import ttml

# conftest.py puts this directory on sys.path (pytest.ini uses --import-mode=importlib).
from chunk_gated_delta_rule_reference import (  # noqa: E402
    DECAY_GATE,
    QWEN_SHAPE,
    STRONG_GATE,
    WEAK_GATE,
    assert_close,
    chunked_forward,
    recurrent_vjp,
)

pytestmark = pytest.mark.requires_device

GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dh0")
PCC_MIN = 0.9999
RMS_MAX = 0.02
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# (B, T, HK, HV, K, V), chunk_size
SHAPES = [
    ((1, 32, 1, 1, 32, 32), 32),  # single tile, single head
    ((1, 128, 2, 2, 64, 64), 64),  # no GVA, 2 chunks
    ((1, 128, 2, 4, 64, 64), 32),  # GVA G=2
    ((2, 64, 1, 3, 64, 64), 32),  # GVA G=3, multi-batch
    ((1, 128, 2, 2, 64, 128), 64),  # wide V: K != V
]
SHAPE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in SHAPES]

# Qwen3.8 linear attention on a 2-way tensor-parallel chip: 16/48 key/value heads globally.
# 512 items, so every core runs several, and K = V = 64 at chunk 64 solves to Vb = 2.
MANY_ITEMS_SHAPE = ((2, 2048, 2, 8, 64, 64), 64)

# More shapes: GVA at G = 4, 8 and 32 (HV = 8, 16, 32), and K > V.
COVERAGE_SHAPES = [
    ((1, 128, 2, 8, 64, 64), 64),  # G = 4
    ((1, 64, 2, 16, 32, 32), 32),  # G = 8
    ((1, 64, 1, 32, 32, 32), 32),  # G = 32 = HV, the largest HV
    ((1, 128, 2, 2, 128, 64), 64),  # K > V
]
COVERAGE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in COVERAGE_SHAPES]

# (K, V, chunk): one row per V-block geometry the L1 budget produces (Vt tiles of V, Vb of them per block).
# The values below were read off the factory when the rows were chosen; a budget change that moves a row to
# another Vb leaves the row valid and only makes this comment stale.
#   K=32  V=32   c32: Vt 1 Vb 1      K=32  V=128 c32: Vt 4 Vb 4 (the whole of V)
#   K=64  V=192  c64: Vt 6 Vb 3      K=96  V=96  c64: Vt 3 Vb 1, odd Kt
#   K=128 V=256  c32: Vt 8 Vb 4      K=192 V=192 c32: Vt 6 Vb 2
#   K=256 V=256  c32: Vt 8 Vb 1      K=160 V=160 c32: Vt 5 Vb 1, prime Vt
BLOCK_SHAPES = [
    (32, 32, 32),
    (32, 128, 32),
    (64, 192, 64),
    (96, 96, 64),
    (128, 256, 32),
    (192, 192, 32),
    (256, 256, 32),
    (160, 160, 32),
]
BLOCK_IDS = [f"K{k}_V{v}_c{c}" for k, v, c in BLOCK_SHAPES]


@pytest.fixture(scope="module")
def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _build_inputs(shape, state_mode, dtype, seed=42, gate_scale=WEAK_GATE):
    """Seeded inputs, rounded to `dtype` so the oracle sees what the device sees. q/k are L2-normalized
    along K (caller contract) and stay at the key-head count."""
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = {
        "q": l2(rn(B, T, HK, K)),
        "k": l2(rn(B, T, HK, K)),
        "v": rn(B, T, HV, V),
        "g": F.logsigmoid(rn(B, T, HV)) * gate_scale,
        "beta": torch.rand(B, T, HV, generator=gen, dtype=torch.float64),
        "do": rn(B, T, HV, V),
        "h0": rn(B, HV, K, V) * 0.1 if state_mode in ("with_h0", "with_h0_and_dht") else None,
        "dht": rn(B, HV, K, V) if state_mode in ("with_h0_and_dht", "dht_only") else None,
    }
    return {n: None if t is None else t.to(TORCH_DTYPE[dtype]).double() for n, t in x.items()}


def _sum_key_heads(grads, shape):
    """dq/dk of the repeated per-value-head q/k -> per key head."""
    B, T, HK, HV, K, V = shape
    dq, dk = (d.reshape(B, T, HK, HV // HK, K).sum(3) for d in grads[:2])
    return (dq, dk, *grads[2:])


def _oracle(x, shape, scale=None):
    """float64 gradients of the token-by-token recurrence; dq/dk summed over each key head's value heads."""
    G = shape[3] // shape[2]
    grads = recurrent_vjp(
        x["q"].repeat_interleave(G, dim=2),
        x["k"].repeat_interleave(G, dim=2),
        x["v"],
        x["g"],
        x["beta"],
        x["do"],
        d_final_state=x["dht"],
        initial_state=x["h0"],
        scale=scale,
    )
    return _sum_key_heads(grads, shape)


def _chunked_oracle(x, shape, chunk_size):
    """The same gradients through the chunked forward: far less autograd memory at long T."""
    G = shape[3] // shape[2]
    names = ("q", "k", "v", "g", "beta", "h0")
    leaves = {n: x[n].clone().requires_grad_(True) for n in names if x[n] is not None}
    rep = lambda t: t.repeat_interleave(G, dim=2)
    o, final_state, *_ = chunked_forward(
        rep(leaves["q"]),
        rep(leaves["k"]),
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        initial_state=leaves.get("h0"),
        chunk_size=chunk_size,
    )
    loss = (o * x["do"]).sum()
    if x["dht"] is not None:
        loss = loss + (final_state * x["dht"]).sum()
    loss.backward()
    return tuple(leaves[n].grad if n in leaves else None for n in names)


def _dev(t, device, dtype):
    if t is None:
        return None
    return ttnn.from_torch(
        t.to(TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _flat(x, shape):
    """Token-major [B, T, X] views of the inputs, heads folded into the last dim."""
    B, T, HK, HV, K, V = shape
    return {
        "q": x["q"].reshape(B, T, HK * K),
        "k": x["k"].reshape(B, T, HK * K),
        "v": x["v"].reshape(B, T, HV * V),
        "g": x["g"],
        "beta": x["beta"],
        "do": x["do"].reshape(B, T, HV * V),
    }


def _upload(x, shape, dtype, device, *, rank=4, do_head_major=False, gate_dtype=None):
    """The device tensors of `x`; callers that keep the result keep the inputs alive."""
    B, T, HK, HV, K, V = shape
    f = {n: t.unsqueeze(1) if rank == 4 else t for n, t in _flat(x, shape).items()}
    if do_head_major:
        f["do"] = x["do"].permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    f = {n: _dev(t, device, (gate_dtype or dtype) if n in ("g", "beta") else dtype) for n, t in f.items()}
    f["h0"], f["dht"] = _dev(x["h0"], device, dtype), _dev(x["dht"], device, dtype)
    return f


def _dispatch(f, shape, chunk_size, *, saved=None, scale=None):
    return ttml.ops.metal.chunk_gated_delta_rule_bw(
        f["q"],
        f["k"],
        f["v"],
        f["g"],
        f["beta"],
        f["do"],
        shape[4],
        initial_state=f["h0"],
        d_final_state=f["dht"],
        chunk_size=chunk_size,
        scale=scale,
        **(saved or {}),
    )


def _run_ttml(
    x, shape, chunk_size, dtype, device, *, rank=4, do_head_major=False, saved=None, gate_dtype=None, scale=None
):
    f = _upload(x, shape, dtype, device, rank=rank, do_head_major=do_head_major, gate_dtype=gate_dtype)
    return _dispatch(f, shape, chunk_size, saved=saved, scale=scale)


def _run_ttml_fw(x, shape, chunk_size, device, gate_dtype=ttnn.bfloat16, scale=None):
    """The saved forward from ttml's training forward; its q/k/v are bf16, h0 matches x."""
    f = {
        n: _dev(t.unsqueeze(1), device, gate_dtype if n in ("g", "beta") else ttnn.bfloat16)
        for n, t in _flat(x, shape).items()
        if n != "do"
    }
    _, _, h, v_new, _, a = ttml.ops.metal.chunk_gated_delta_rule_fw(
        f["q"],
        f["k"],
        f["v"],
        f["g"],
        f["beta"],
        shape[4],
        initial_state=_dev(x["h0"], device, ttnn.float32),
        chunk_size=chunk_size,
        scale=scale,
    )
    return {"h": h, "v_new": v_new, "a": a}


def _out_shapes(shape):
    B, T, HK, HV, K, V = shape
    return {
        "dq": (B, 1, T, HK * K),
        "dk": (B, 1, T, HK * K),
        "dv": (B, 1, T, HV * V),
        "dg": (B, 1, T, HV),
        "dbeta": (B, 1, T, HV),
        "dh0": (B, HV, K, V),
    }


def _check_grads(got, expected, shape, dtype):
    shapes = _out_shapes(shape)
    for name, dev_t, exp in zip(GRAD_NAMES, got, expected):
        if exp is None:
            assert dev_t is None, f"{name}: expected None without initial_state"
            continue
        assert dev_t.layout == ttnn.TILE_LAYOUT and dev_t.dtype == dtype, f"{name}: {dev_t.layout} {dev_t.dtype}"
        host = ttnn.to_torch(dev_t).double()
        assert tuple(host.shape) == shapes[name], f"{name}: shape {tuple(host.shape)} != {shapes[name]}"
        assert_close(name, host, exp.reshape(host.shape), pcc_min=PCC_MIN, rms_max=RMS_MAX)


@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("state_mode", ["do_only", "with_h0", "with_h0_and_dht", "dht_only"])
def test_vs_oracle(shape, chunk_size, dtype, state_mode, device):
    x = _build_inputs(shape, state_mode, dtype)
    _check_grads(_run_ttml(x, shape, chunk_size, dtype, device), _oracle(x, shape), shape, dtype)


def _addresses(tensors):
    return {n: t.buffer_address() for n, t in tensors.items() if t is not None}


def test_cache_hit_patches_buffer_addresses(device):
    """Both input sets and both gradient sets stay alive, so the second dispatch cannot land on the first's
    addresses by reuse: a no-op `override_runtime_arguments` would leave it reading and writing set A."""
    shape, chunk_size, dtype = (1, 96, 2, 4, 64, 64), 32, ttnn.float32  # a shape no other test dispatches
    xa = _build_inputs(shape, "with_h0_and_dht", dtype, seed=1)
    xb = _build_inputs(shape, "with_h0_and_dht", dtype, seed=2)
    ta, tb = _upload(xa, shape, dtype, device), _upload(xb, shape, dtype, device)
    ia, ib = _addresses(ta), _addresses(tb)
    assert all(ia[n] != ib[n] for n in ia), "input sets share a buffer address"

    got_a = _dispatch(ta, shape, chunk_size)
    entries = device.num_program_cache_entries()
    assert entries > 0
    got_b = _dispatch(tb, shape, chunk_size)
    assert device.num_program_cache_entries() == entries, "a second input set at the same shape recompiled"
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(got_a, got_b)), "gradients share a buffer"

    _check_grads(got_a, _oracle(xa, shape), shape, dtype)
    _check_grads(got_b, _oracle(xb, shape), shape, dtype)
    for name, ha, hb in zip(GRAD_NAMES, got_a, got_b):
        assert not torch.equal(ttnn.to_torch(ha), ttnn.to_torch(hb)), f"{name}: sets A and B gave the same answer"
    again = _dispatch(ta, shape, chunk_size)
    assert device.num_program_cache_entries() == entries
    for name, first, second in zip(GRAD_NAMES, got_a, again):
        assert torch.equal(ttnn.to_torch(first), ttnn.to_torch(second)), f"{name}: rerun with input set A changed"


def test_attribute_changes_miss_the_cache(device):
    """`initial_state` / `d_final_state` presence, the saved forward, the gate dtype and the chunk size each
    select a different program."""
    # No other test (in any GDN file) dispatches this shape.
    shape, chunk_size, dtype = (1, 384, 1, 2, 64, 64), 32, ttnn.bfloat16
    keep = []

    def run(state_mode, *, saved=False, gate_dtype=None, chunk=chunk_size):
        x = _build_inputs(shape, state_mode, dtype)
        sv = _run_ttml_fw(x, shape, chunk, device) if saved else None
        before = device.num_program_cache_entries()
        t = _upload(x, shape, dtype, device, gate_dtype=gate_dtype)
        keep.append((t, sv, _dispatch(t, shape, chunk, saved=sv)))
        return device.num_program_cache_entries() - before

    assert run("with_h0") > 0
    assert run("with_h0") == 0, "identical call missed the cache"
    assert run("do_only") > 0, "dropping initial_state reused the program"
    assert run("with_h0_and_dht") > 0, "adding d_final_state reused the program"
    assert run("with_h0", gate_dtype=ttnn.float32) > 0, "an fp32 gate reused the bf16-gate program"
    assert run("with_h0", chunk=64) > 0, "a different chunk size reused the program"
    # The forward's own programs are built before the count starts: only the backward's may be new.
    assert run("with_h0", saved=True) > 0, "the saved forward reused the recompute program"


@pytest.mark.parametrize("shape,chunk_size", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("state_mode", ["do_only", "with_h0", "with_h0_and_dht", "dht_only"])
def test_saved_forward_vs_oracle(shape, chunk_size, state_mode, device):
    dtype = ttnn.bfloat16
    x = _build_inputs(shape, state_mode, dtype)
    got = _run_ttml(x, shape, chunk_size, dtype, device, saved=_run_ttml_fw(x, shape, chunk_size, device))
    recomputed = _run_ttml(x, shape, chunk_size, dtype, device)
    _check_grads(got, _oracle(x, shape), shape, dtype)
    for name, dev_t, rec_t in zip(GRAD_NAMES, got, recomputed):
        if dev_t is not None:
            assert_close(
                f"{name} saved vs recomputed",
                ttnn.to_torch(dev_t),
                ttnn.to_torch(rec_t).double(),
                pcc_min=PCC_MIN,
                rms_max=RMS_MAX,
            )


@pytest.mark.parametrize("shape,chunk_size", COVERAGE_SHAPES, ids=COVERAGE_IDS)
@pytest.mark.parametrize("saved", [False, True], ids=["fp32_recompute", "bf16_saved"])
def test_coverage_shapes_vs_oracle(shape, chunk_size, saved, device):
    dtype = ttnn.bfloat16 if saved else ttnn.float32
    x = _build_inputs(shape, "with_h0_and_dht", dtype)
    sv = _run_ttml_fw(x, shape, chunk_size, device) if saved else None
    _check_grads(_run_ttml(x, shape, chunk_size, dtype, device, saved=sv), _oracle(x, shape), shape, dtype)


@pytest.mark.parametrize("saved", [False, True], ids=["fp32_recompute", "bf16_saved"])
@pytest.mark.parametrize("K,V,chunk_size", BLOCK_SHAPES, ids=BLOCK_IDS)
def test_block_shapes_vs_oracle(K, V, chunk_size, saved, device):
    shape = (1, 2 * chunk_size, 1, 2, K, V)
    dtype = ttnn.bfloat16 if saved else ttnn.float32
    x = _build_inputs(shape, "with_h0_and_dht", dtype)
    sv = _run_ttml_fw(x, shape, chunk_size, device) if saved else None
    _check_grads(_run_ttml(x, shape, chunk_size, dtype, device, saved=sv), _oracle(x, shape), shape, dtype)


def test_long_sequence_error_does_not_grow(device):
    """No decay is the worst case for error carried along the recurrence (TF32 operands, fp32 state): at
    T = 4096 (64 chunks) the gradients keep the error of T = 256 (measured flat from T = 256 to 4096:
    0.3% .. 0.8% relative RMS)."""
    shape, chunk_size, dtype = (1, 4096, 1, 2, 64, 64), 64, ttnn.float32
    x = _build_inputs(shape, "with_h0_and_dht", dtype, gate_scale=0.0)
    _check_grads(_run_ttml(x, shape, chunk_size, dtype, device), _chunked_oracle(x, shape, chunk_size), shape, dtype)


def test_qwen_shape_saved_forward_head_major_d_out(device):
    """The training path at a Qwen3.8 per-chip shape: flat inputs, head-major d_out, saved forward."""
    shape, chunk_size = QWEN_SHAPE
    dtype = ttnn.bfloat16
    x = _build_inputs(shape, "with_h0_and_dht", dtype)
    saved = _run_ttml_fw(x, shape, chunk_size, device)
    got = _run_ttml(x, shape, chunk_size, dtype, device, do_head_major=True, saved=saved)
    _check_grads(got, _chunked_oracle(x, shape, chunk_size), shape, dtype)


@pytest.mark.parametrize("saved", [False, True], ids=["recompute", "saved"])
def test_fp32_gates_with_bf16_inputs(saved, device):
    """The forward takes BF16 or FP32 gates beside BF16 q/k/v; the backward must take the same."""
    shape, chunk_size = SHAPES[2]
    dtype = ttnn.bfloat16
    x = _build_inputs(shape, "with_h0", dtype)
    x32 = _build_inputs(shape, "with_h0", ttnn.float32)
    x["g"], x["beta"] = x32["g"], x32["beta"]  # not representable in bf16
    sv = _run_ttml_fw(x, shape, chunk_size, device, ttnn.float32) if saved else None
    got = _run_ttml(x, shape, chunk_size, dtype, device, saved=sv, gate_dtype=ttnn.float32)
    _check_grads(got, _oracle(x, shape), shape, dtype)


@pytest.mark.parametrize(
    "dtype,saved",
    [(ttnn.bfloat16, False), (ttnn.bfloat16, True), (ttnn.float32, False)],
    ids=["bf16_recompute", "bf16_saved", "fp32_recompute"],
)
def test_many_items_per_core(dtype, saved, device):
    """Items > cores at Vb > 1.  The fp32 row has the tight tolerance that a slightly corrupted dg fails."""
    shape, chunk_size = MANY_ITEMS_SHAPE
    x = _build_inputs(shape, "with_h0", dtype)
    sv = _run_ttml_fw(x, shape, chunk_size, device) if saved else None
    got = _run_ttml(x, shape, chunk_size, dtype, device, saved=sv)
    _check_grads(got, _chunked_oracle(x, shape, chunk_size), shape, dtype)


@pytest.mark.parametrize("shape,chunk_size", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_decaying_gate_vs_oracle(shape, chunk_size, device):
    """The previous default gate (decay of e^-13 .. e^-26 per chunk): the within-chunk decay terms."""
    dtype = ttnn.float32
    x = _build_inputs(shape, "with_h0_and_dht", dtype, gate_scale=DECAY_GATE)
    _check_grads(_run_ttml(x, shape, chunk_size, dtype, device), _oracle(x, shape), shape, dtype)


@pytest.mark.parametrize("shape,chunk_size", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_strong_gate_vs_oracle(shape, chunk_size, device):
    """Within-chunk decay spans of ~200: a decay difference formed as cumsum_i - cumsum_j cancels at the
    ~tf32 width of the FPU source registers, and exp() magnifies what survives."""
    dtype = ttnn.float32
    x = _build_inputs(shape, "with_h0", dtype, gate_scale=STRONG_GATE)
    _check_grads(_run_ttml(x, shape, chunk_size, dtype, device), _oracle(x, shape), shape, dtype)


@pytest.mark.parametrize("scale", [0.3, 1.0], ids=["scale_0.3", "scale_1"])
@pytest.mark.parametrize("saved", [False, True], ids=["recompute", "saved"])
def test_explicit_scale(scale, saved, device):
    """dq, and every gradient through o, scale with `scale`: a wrong or ignored one keeps the PCC of dq at 1."""
    shape, chunk_size = SHAPES[1]
    dtype = ttnn.bfloat16
    assert abs(scale - shape[4] ** -0.5) > 0.1, "scale must differ from the default K^-0.5"
    x = _build_inputs(shape, "with_h0_and_dht", dtype)
    sv = _run_ttml_fw(x, shape, chunk_size, device, scale=scale) if saved else None
    got = _run_ttml(x, shape, chunk_size, dtype, device, saved=sv, scale=scale)
    _check_grads(got, _oracle(x, shape, scale=scale), shape, dtype)
    default = _run_ttml(x, shape, chunk_size, dtype, device, saved=sv)
    ratio = ttnn.to_torch(got[0]).double().norm() / ttnn.to_torch(default[0]).double().norm()
    assert ratio > 1.5, f"dq barely moved with scale {scale} (ratio {ratio:.3f})"


@pytest.mark.parametrize(
    "shape,chunk_size", [SHAPES[2], SHAPES[3], SHAPES[4]], ids=[SHAPE_IDS[2], SHAPE_IDS[3], SHAPE_IDS[4]]
)
@pytest.mark.parametrize("saved", [False, True], ids=["recompute", "saved"])
def test_layouts_bit_identical(shape, chunk_size, saved, device):
    """Head-major d_out and rank-3 inputs read the same pages as the rank-4 token-major baseline."""
    dtype = ttnn.bfloat16
    x = _build_inputs(shape, "with_h0_and_dht", dtype, seed=3)
    fw = (lambda: _run_ttml_fw(x, shape, chunk_size, device)) if saved else (lambda: None)
    base = _run_ttml(x, shape, chunk_size, dtype, device, saved=fw())
    head_major = _run_ttml(x, shape, chunk_size, dtype, device, do_head_major=True, saved=fw())
    rank3 = _run_ttml(x, shape, chunk_size, dtype, device, rank=3, saved=fw())
    for name, tb, th, t3 in zip(GRAD_NAMES, base, head_major, rank3):
        hb, hh, h3 = (ttnn.to_torch(t) for t in (tb, th, t3))
        assert torch.equal(hb, hh), f"{name}: head-major d_out differs"
        if name != "dh0":
            assert h3.dim() == 3, f"{name}: rank-3 input gave a rank-{h3.dim()} gradient"
        assert torch.equal(hb.reshape(h3.shape), h3), f"{name}: rank-3 inputs differ"


def test_rejects_partial_saved_forward(device, expect_error):
    shape, chunk_size = SHAPES[1]
    x = _build_inputs(shape, "do_only", ttnn.bfloat16)
    saved = _run_ttml_fw(x, shape, chunk_size, device)
    del saved["a"]
    with expect_error(RuntimeError, "h, v_new and a are the saved forward and must be given together"):
        _run_ttml(x, shape, chunk_size, ttnn.bfloat16, device, saved=saved)


def test_rejects_bad_d_out(device, expect_error):
    z = lambda *s: _dev(torch.zeros(*s), device, ttnn.bfloat16)
    q, k, v, g, beta = z(1, 1, 64, 64), z(1, 1, 64, 64), z(1, 1, 64, 128), z(1, 1, 64, 2), z(1, 1, 64, 2)
    with expect_error(RuntimeError, "d_out must be token-major"):
        ttml.ops.metal.chunk_gated_delta_rule_bw(q, k, v, g, beta, z(1, 1, 64, 64), 64, chunk_size=32)


@pytest.mark.parametrize(
    "T,q_width,v_width,H,key_head_dim,chunk_size,match",
    [
        (96, 128, 128, 2, 64, 64, "T=96 must be a multiple of chunk_size=64"),
        (128, 128, 129, 2, 64, 64, "v width 129 is not a multiple of HV=2"),
        (128, 96, 128, 2, 64, 64, "q width 96 is not a multiple of K=64"),
        (128, 192, 128, 2, 64, 64, "value heads HV=2 must be a multiple of key heads HK=3"),
        (128, 128, 96, 2, 64, 64, "K=64 and V=48 must be multiples of 32"),
        (128, 96, 128, 2, 48, 64, "K=48 and V=64 must be multiples of 32"),
        (128, 128, 128, 2, 64, 48, "chunk_size must be 32 or 64, got 48"),
        (0, 128, 128, 2, 64, 64, "B=1 must be at least 1 and T=0 at least chunk_size=64"),
        (128, 128, 128, 33, 64, 64, re.escape("HV=33 must be in [1, 32]")),
    ],
    ids=["ragged_T", "v_width", "q_width", "gva_ratio", "V_not_tile", "K_not_tile", "bad_chunk", "T_0", "HV_over_32"],
)
def test_rejects_bad_shapes(T, q_width, v_width, H, key_head_dim, chunk_size, match, device, expect_error):
    z = lambda *s: _dev(torch.zeros(*s), device, ttnn.bfloat16)
    if H == 33:  # q/v widths of one 33-head group
        q_width, v_width = 64, 33 * 32
    q, k, v = z(1, 1, T, q_width), z(1, 1, T, q_width), z(1, 1, T, v_width)
    g, beta, do = z(1, 1, T, H), z(1, 1, T, H), z(1, 1, T, v_width)
    with expect_error(RuntimeError, match):
        ttml.ops.metal.chunk_gated_delta_rule_bw(q, k, v, g, beta, do, key_head_dim, chunk_size=chunk_size)


def test_rejects_zero_batch(device, expect_error):
    z = lambda *s: _dev(torch.zeros(*s), device, ttnn.bfloat16)
    args = [z(0, 1, 64, 32), z(0, 1, 64, 32), z(0, 1, 64, 32), z(0, 1, 64, 1), z(0, 1, 64, 1), z(0, 1, 64, 32)]
    with expect_error(RuntimeError, "B=0 must be at least 1 and T=64 at least chunk_size=32"):
        ttml.ops.metal.chunk_gated_delta_rule_bw(*args, 32, chunk_size=32)


@pytest.mark.parametrize("key_head_dim", [256, 512], ids=["K256", "K512"])
def test_rejects_shapes_with_no_fitting_v_block(key_head_dim, device, expect_error):
    """K = V = 256 at chunk 64 is accepted by the forward but leaves the backward no V block that fits L1."""
    K = key_head_dim
    z = lambda *s: _dev(torch.zeros(*s), device, ttnn.bfloat16)
    args = [z(1, 1, 64, K), z(1, 1, 64, K), z(1, 1, 64, K), z(1, 1, 64, 1), z(1, 1, 64, 1), z(1, 1, 64, K)]
    with expect_error(RuntimeError, "no V block fits L1"):
        ttml.ops.metal.chunk_gated_delta_rule_bw(*args, K, chunk_size=64)


def _bad_input(name, device):
    """Valid zero bf16 inputs (B=1, T=64, K=V=32, chunk 32) with one tensor replaced; (args, kwargs)."""
    z = lambda *s, dt=ttnn.bfloat16, **kw: ttnn.from_torch(
        torch.zeros(*s),
        dtype=dt,
        layout=kw.get("layout", ttnn.TILE_LAYOUT),
        device=device,
        memory_config=kw.get("mem", ttnn.DRAM_MEMORY_CONFIG),
    )
    q, k, v, g, beta, do = (
        z(1, 1, 64, 32),
        z(1, 1, 64, 32),
        z(1, 1, 64, 32),
        z(1, 1, 64, 1),
        z(1, 1, 64, 1),
        z(1, 1, 64, 32),
    )
    kwargs = {"chunk_size": 32}
    f32 = ttnn.float32
    if name == "k_dtype":
        k = z(1, 1, 64, 32, dt=f32)
    elif name == "v_row_major":
        v = z(1, 1, 64, 32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif name == "k_in_l1":
        k = z(1, 1, 64, 32, mem=ttnn.L1_MEMORY_CONFIG)
    elif name == "beta_dtype":
        beta = z(1, 1, 64, 1, dt=f32)
    elif name == "h0_dtype":
        kwargs["initial_state"] = z(1, 1, 32, 32, dt=f32)
    elif name == "h0_shape":
        kwargs["initial_state"] = z(1, 1, 64, 32)
    elif name == "dht_shape":
        kwargs["d_final_state"] = z(1, 2, 32, 32)
    elif name == "h_shape":
        kwargs.update(h=z(1, 1, 32, 32, dt=f32), v_new=z(1, 1, 64, 32, dt=f32), a=z(1, 1, 64, 32, dt=f32))
    elif name == "h_dtype":
        kwargs.update(h=z(1, 2, 32, 32), v_new=z(1, 1, 64, 32, dt=f32), a=z(1, 1, 64, 32, dt=f32))
    return [q, k, v, g, beta, do], kwargs


@pytest.mark.parametrize(
    "name,match",
    [
        ("k_dtype", "k has unsupported dtype FLOAT32"),
        ("v_row_major", "v requires TILE layout. Got: ROW_MAJOR"),
        ("k_in_l1", "k must be in DRAM. Got: L1"),
        ("beta_dtype", "beta has unsupported dtype FLOAT32"),
        ("h0_dtype", "initial_state has unsupported dtype FLOAT32"),
        ("h0_shape", re.escape("initial_state must be Shape([1, 1, 32, 32]), got Shape([1, 1, 64, 32])")),
        ("dht_shape", re.escape("d_final_state must be Shape([1, 1, 32, 32]), got Shape([1, 2, 32, 32])")),
        ("h_shape", re.escape("h must be Shape([1, 2, 32, 32]), got Shape([1, 1, 32, 32])")),
        ("h_dtype", "h has unsupported dtype BFLOAT16"),
    ],
)
def test_rejects_bad_inputs(name, match, device, expect_error):
    args, kwargs = _bad_input(name, device)
    with expect_error(RuntimeError, match):
        ttml.ops.metal.chunk_gated_delta_rule_bw(*args, 32, **kwargs)


def test_placement_is_rejected_after_the_program_is_cached(device, expect_error):
    """With the DRAM program cached, k in L1 must still fail: validation runs on cache hits too, and the
    program key carries the placement, so the DRAM program is neither reused nor trusted for it."""
    args, kwargs = _bad_input("none", device)
    ttml.ops.metal.chunk_gated_delta_rule_bw(*args, 32, **kwargs)  # fills the cache with the DRAM variant
    args[1] = ttnn.to_memory_config(args[1], ttnn.L1_MEMORY_CONFIG)
    with expect_error(RuntimeError, "k must be in DRAM. Got: L1"):
        ttml.ops.metal.chunk_gated_delta_rule_bw(*args, 32, **kwargs)
