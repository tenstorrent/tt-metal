# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.ops.metal.chunk_gated_delta_rule_fw: the training forward that also returns what the backward consumes.

    q, k [B, 1, T, HK*K]      v [B, 1, T, HV*V]      g, beta [B, 1, T, HV]      initial_state [B, HV, K, V]
    o [B, 1, T, HV*V]         final_state [B, HV, K, V]
    h [B*HV, NC, K, V]        v_new [B*HV, 1, T, V]  g_cumsum [B*HV, 1, T, 1]   A [B*HV, 1, T, C]

Pinned here:
1. Every output against the float64 chunked reference, including a Qwen3.8-sized shape.
2. The head-major output is exactly the token-major output, permuted; the other outputs are unchanged.
3. Two live input sets at one shape each get their own answer from one cached program.
4. Shape-contract violations raise before dispatch.
5. A wide-V shape (K=128, V=256) fits L1 and matches the reference.
6. Strong gates (|cumsum(g)| in the hundreds within a chunk) still match the reference.
7. More value heads than worker cores (a core runs several scans back to back), including an odd
   number of key tiles (K = V = 96), match the reference.
8. Weak gates (the default here) keep the inter-chunk state carry observable; one decaying-gate case
   covers the other regime.  See `WEAK_GATE`.
9. An explicit `scale` is checked in absolute terms (relative RMS), not only by correlation.
10. Cache hits patch every buffer address, with all input and output sets alive at once, and a changed
    `initial_state` presence or gate dtype misses the cache.
11. Shapes outside the earlier ones: GVA at HV = 8 / 16 / 32, K > V, and every V-block geometry the
    backward can pick (`BLOCK_SHAPES`); T = 4096 with no decay keeps the same error as T = 256.
12. Rejections name the offending value (exact text), for shapes, dtypes, layouts, placement, T = 0,
    B = 0, an `initial_state` of the wrong dtype or shape, and a shape that overflows L1.
13. A tensor placed differently (L1 instead of DRAM) is rejected even with the DRAM program cached.

Tolerances: every output must reach PCC >= `PCC_MIN` and a relative RMS error <= `RMS_MAX`.  Measured
on Blackhole over all cases here the worst relative RMS is 4.1e-3 (`o`, 2^-8 is the bf16 rounding of
the output; the host `q * scale` rounds q to bf16 once more and is not mirrored in the float64
reference) and the worst PCC is 0.999997, so the bounds sit about 2.4x above the measurement.
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
)

pytestmark = pytest.mark.requires_device

PCC_MIN = 0.9999
RMS_MAX = 0.01

OUT_NAMES = ("o", "final_state", "h", "v_new", "g_cumsum", "A")
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# (B, T, HK, HV, K, V), chunk_size
SHAPES = [
    ((1, 32, 1, 1, 32, 32), 32),  # single tile, single head
    ((1, 128, 2, 2, 64, 64), 64),  # 2 chunks
    ((1, 128, 2, 4, 64, 64), 32),  # GVA G=2
    ((2, 64, 1, 3, 64, 64), 32),  # GVA G=3, multi-batch
    ((1, 128, 2, 2, 64, 128), 64),  # wide V: K != V
    ((1, 128, 1, 2, 128, 128), 64),  # Qwen head dims
]
SHAPE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in SHAPES]

# Qwen3.8 linear attention on a 2-way tensor-parallel chip: 16/48 key/value heads globally.

# More shapes: GVA at G = 4, 8 and 32 (HV = 8, 16, 32), and K > V.
COVERAGE_SHAPES = [
    ((1, 128, 2, 8, 64, 64), 64),  # G = 4
    ((1, 64, 2, 16, 32, 32), 32),  # G = 8
    ((1, 64, 1, 32, 32, 32), 32),  # G = 32 = HV, the largest HV
    ((1, 128, 2, 2, 128, 64), 64),  # K > V
]
COVERAGE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in COVERAGE_SHAPES]

# (K, V, chunk) -> (Vt, Vb), the key-tile count Kt = K/32, the backward's V block: one geometry per row, so a
# change to the L1 budget that moves a row to another Vb shows up as a comment to update, not as a gap.
#   K=32  V=32   c32: Vt 1 Vb 1      K=32  V=128 c32: Vt 4 Vb 4 (whole V)
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


def _build_inputs(shape, with_h0, seed=42, gate_scale=WEAK_GATE):
    """Seeded inputs; q/k/v rounded to bf16 (device dtype), q/k L2-normalized along K (caller contract)."""
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    bf = lambda x: x.to(torch.bfloat16).double()
    return {
        "q": bf(l2(rn(B, T, HK, K))),
        "k": bf(l2(rn(B, T, HK, K))),
        "v": bf(rn(B, T, HV, V)),
        "g": F.logsigmoid(rn(B, T, HV)) * gate_scale,
        "beta": torch.rand(B, T, HV, generator=gen, dtype=torch.float64),
        "h0": rn(B, HV, K, V) * 0.1 if with_h0 else None,
    }


def _tt(t, device, dtype):
    return ttnn.from_torch(t.to(TORCH_DTYPE[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _upload(x, shape, gate_dtype, device):
    """The device tensors of `x`; callers that keep the result keep the inputs alive."""
    B, T, HK, HV, K, V = shape
    flat = lambda t: t.reshape(B, 1, T, -1)
    return {
        "q": _tt(flat(x["q"]), device, ttnn.bfloat16),
        "k": _tt(flat(x["k"]), device, ttnn.bfloat16),
        "v": _tt(flat(x["v"]), device, ttnn.bfloat16),
        "g": _tt(flat(x["g"]), device, gate_dtype),
        "beta": _tt(flat(x["beta"]), device, gate_dtype),
        "h0": None if x["h0"] is None else _tt(x["h0"], device, ttnn.float32),
    }


def _dispatch(t, shape, chunk, scale=None, output_head_major=False, save_for_backward=True):
    return ttml.ops.metal.chunk_gated_delta_rule_fw(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        shape[4],
        initial_state=t["h0"],
        chunk_size=chunk,
        scale=scale,
        output_head_major=output_head_major,
        save_for_backward=save_for_backward,
    )


def _run_ttml(x, shape, chunk, gate_dtype, device, scale=None, output_head_major=False):
    return _dispatch(_upload(x, shape, gate_dtype, device), shape, chunk, scale, output_head_major)


def _to_host(outs):
    return [ttnn.to_torch(t).double() for t in outs]


def _to_head_major(outs, shape):
    """ttml outputs -> o [BH,T,V], final [B,HV,K,V], h [BH,NC,K,V], v_new [BH,T,V], g_cumsum [BH,T], A [BH,T,C]."""
    B, T, HK, HV, K, V = shape
    o, fs, h, vn, gc, a = outs
    o = o.reshape(B, T, HV, V).permute(0, 2, 1, 3).reshape(B * HV, T, V)
    return [o, fs, h, vn.reshape(B * HV, T, V), gc.reshape(B * HV, T), a.reshape(B * HV, T, -1)]


def _reference(x, shape, chunk, gate_dtype, scale=None):
    """float64 outputs in the head-major layout of `_to_head_major`."""
    B, T, HK, HV, K, V = shape
    G = HV // HK
    gate = lambda t: t.to(TORCH_DTYPE[gate_dtype]).double()
    o, fs, h, vn, gc, a = chunked_forward(
        x["q"].repeat_interleave(G, dim=2),
        x["k"].repeat_interleave(G, dim=2),
        x["v"],
        gate(x["g"]),
        gate(x["beta"]),
        initial_state=x["h0"],
        chunk_size=chunk,
        scale=scale,
    )
    tok = lambda t: t.permute(0, 2, 1, 3).reshape(B * HV, T, -1)
    return [
        tok(o),
        fs,
        h.transpose(1, 2).reshape(B * HV, -1, K, V),
        tok(vn),
        gc.transpose(1, 2).reshape(B * HV, T),
        tok(a),
    ]


def _check_vs_reference(x, shape, chunk, gate_dtype, device, with_h0, scale=None):
    got = _to_head_major(_to_host(_run_ttml(x, shape, chunk, gate_dtype, device, scale=scale)), shape)
    _compare(got, _reference(x, shape, chunk, gate_dtype, scale=scale), chunk, with_h0)


def _compare(got, exp, chunk, with_h0):
    for name, g, e in zip(OUT_NAMES, got, exp):
        assert g.shape == e.shape, f"{name}: {tuple(g.shape)} != {tuple(e.shape)}"
        if name == "h" and not with_h0:
            assert torch.count_nonzero(g[:, 0]) == 0, "h[:, 0] must be exactly zero without initial_state"
        if name == "A":
            # The backward consumes A as stored: row r of a chunk has nothing right of the diagonal.
            T = g.shape[1]
            above = torch.arange(chunk)[None, :] > (torch.arange(T) % chunk)[:, None]
            assert torch.count_nonzero(g[:, above]) == 0, "A has non-zeros above the diagonal"
        assert_close(name, g, e, pcc_min=PCC_MIN, rms_max=RMS_MAX)


@pytest.mark.parametrize("gate_dtype", [ttnn.float32, ttnn.bfloat16], ids=["gate_fp32", "gate_bf16"])
@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("shape,chunk", SHAPES, ids=SHAPE_IDS)
def test_vs_reference(shape, chunk, with_h0, gate_dtype, device):
    _check_vs_reference(_build_inputs(shape, with_h0), shape, chunk, gate_dtype, device, with_h0)


@pytest.mark.parametrize("shape,chunk", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_decaying_gate_vs_reference(shape, chunk, device):
    """The previous default gate (decay of e^-13 .. e^-26 per chunk): the within-chunk decay terms."""
    x = _build_inputs(shape, with_h0=True, gate_scale=DECAY_GATE)
    _check_vs_reference(x, shape, chunk, ttnn.float32, device, with_h0=True)


@pytest.mark.parametrize("shape,chunk", COVERAGE_SHAPES, ids=COVERAGE_IDS)
@pytest.mark.parametrize("gate_dtype", [ttnn.float32, ttnn.bfloat16], ids=["gate_fp32", "gate_bf16"])
def test_coverage_shapes_vs_reference(shape, chunk, gate_dtype, device):
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, gate_dtype, device, with_h0=True)


@pytest.mark.parametrize("K,V,chunk", BLOCK_SHAPES, ids=BLOCK_IDS)
def test_block_shapes_vs_reference(K, V, chunk, device):
    shape = (1, 2 * chunk, 1, 2, K, V)
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, ttnn.float32, device, with_h0=True)


def test_long_sequence_error_does_not_grow(device):
    """No decay is the worst case for error carried along the scan (TF32 operands, fp32 state): at T = 4096
    (64 chunks) the error matches the one at T = 256 (measured flat from T = 256 to 8192: 2.6e-3 .. 3.4e-3
    for the state)."""
    shape, chunk = (1, 4096, 1, 2, 128, 128), 64
    x = _build_inputs(shape, with_h0=True, gate_scale=0.0)
    _compare(
        _to_head_major(_to_host(_run_ttml(x, shape, chunk, ttnn.float32, device)), shape),
        _reference(x, shape, chunk, ttnn.float32),
        chunk,
        with_h0=True,
    )


def test_qwen_shape_vs_reference(device):
    shape, chunk = QWEN_SHAPE
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, ttnn.bfloat16, device, with_h0=True)


def test_wide_v_fits_l1(device):
    """K=128, V=256 at chunk 64: the prep's single-tile scratch CBs keep it inside one core's L1."""
    shape, chunk = (1, 64, 1, 1, 128, 256), 64
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, ttnn.bfloat16, device, with_h0=True)


@pytest.mark.parametrize("shape,chunk", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_strong_gate_vs_reference(shape, chunk, device):
    """Within-chunk decay spans of ~200: a decay difference formed as cumsum_i - cumsum_j cancels at the
    ~tf32 width of the FPU source registers, so L_mask, T_inv and the state decay would drift."""
    x = _build_inputs(shape, with_h0=True, gate_scale=STRONG_GATE)
    _check_vs_reference(x, shape, chunk, ttnn.float32, device, with_h0=True)


@pytest.mark.parametrize("K,V", [(32, 32), (96, 96)], ids=["K32_V32", "K96_V96"])
def test_more_heads_than_cores(K, V, device):
    """B*HV one past the worker-core count: the scan units no longer fit one per core.  K = V = 96 is
    three key tiles, a push size that does not divide the scratch CB's old, larger capacity."""
    grid = device.compute_with_storage_grid_size()
    HV = 32
    B = (grid.x * grid.y) // HV + 1
    shape = (B, 128, 2, HV, K, V)
    assert B * HV > grid.x * grid.y
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, 64, ttnn.float32, device, with_h0=True)


@pytest.mark.parametrize("scale", [0.3, 1.0], ids=["scale_0.3", "scale_1"])
def test_explicit_scale(scale, device):
    """`o` is linear in q, so a wrong or ignored scale keeps PCC at 1; only the absolute error sees it."""
    shape, chunk = SHAPES[1]
    assert abs(scale - shape[4] ** -0.5) > 0.1, "scale must differ from the default K^-0.5"
    x = _build_inputs(shape, with_h0=True)
    default = _to_head_major(_to_host(_run_ttml(x, shape, chunk, ttnn.float32, device)), shape)
    got = _to_head_major(_to_host(_run_ttml(x, shape, chunk, ttnn.float32, device, scale=scale)), shape)
    exp = _reference(x, shape, chunk, ttnn.float32, scale=scale)
    _compare(got, exp, chunk, with_h0=True)
    # Only `o` carries q; every other output is independent of the scale.
    ratio = got[0].norm() / default[0].norm()
    assert abs(ratio - scale / shape[4] ** -0.5) < 0.05 * ratio, f"o scaled by {ratio:.4f}"
    for name, g, d in zip(OUT_NAMES[1:], got[1:], default[1:]):
        assert torch.equal(g, d), f"{name} depends on scale"


@pytest.mark.parametrize("shape,chunk", [SHAPES[3], SHAPES[4]], ids=[SHAPE_IDS[3], SHAPE_IDS[4]])
def test_head_major_output_is_permuted_token_major(shape, chunk, device):
    B, T, HK, HV, K, V = shape
    x = _build_inputs(shape, with_h0=True, seed=2)
    token_major = _to_host(_run_ttml(x, shape, chunk, ttnn.bfloat16, device))
    head_major = _to_host(_run_ttml(x, shape, chunk, ttnn.bfloat16, device, output_head_major=True))
    assert tuple(head_major[0].shape) == (B * HV, 1, T, V)
    tm = token_major[0].reshape(B, T, HV, V).permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    assert torch.equal(head_major[0], tm)
    for name, hm, tmj in zip(OUT_NAMES[1:], head_major[1:], token_major[1:]):
        assert torch.equal(hm, tmj), f"{name}: differs between output layouts"


def _addresses(tensors):
    return {n: t.buffer_address() for n, t in tensors.items() if t is not None}


def test_cache_hit_patches_buffer_addresses(device):
    """Both input sets and both output sets stay alive, so the second dispatch cannot land on the first's
    addresses by reuse: a no-op `override_runtime_arguments` would leave it reading and writing set A."""
    shape, chunk = (1, 96, 2, 4, 64, 64), 32  # a shape no other test dispatches
    xa, xb = _build_inputs(shape, True, seed=1), _build_inputs(shape, True, seed=2)
    ta, tb = _upload(xa, shape, ttnn.float32, device), _upload(xb, shape, ttnn.float32, device)
    ia, ib = _addresses(ta), _addresses(tb)
    assert all(ia[n] != ib[n] for n in ia), "input sets share a buffer address"

    out_a = _dispatch(ta, shape, chunk)
    entries = device.num_program_cache_entries()
    assert entries > 0
    out_b = _dispatch(tb, shape, chunk)
    assert device.num_program_cache_entries() == entries, "a second input set at the same shape recompiled"
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(out_a, out_b)), "outputs share a buffer"

    host_a, host_b = _to_host(out_a), _to_host(out_b)
    _compare(_to_head_major(host_a, shape), _reference(xa, shape, chunk, ttnn.float32), chunk, True)
    _compare(_to_head_major(host_b, shape), _reference(xb, shape, chunk, ttnn.float32), chunk, True)
    for name, ha, hb in zip(OUT_NAMES, host_a, host_b):
        assert not torch.equal(ha, hb), f"{name}: input sets A and B gave the same answer"
    again = _to_host(_dispatch(ta, shape, chunk))
    assert device.num_program_cache_entries() == entries
    for name, h1, h2 in zip(OUT_NAMES, host_a, again):
        assert torch.equal(h1, h2), f"{name}: rerun with input set A changed"


def test_attribute_changes_miss_the_cache(device):
    """A different `initial_state` presence, gate dtype or chunk size is a different program."""
    shape, chunk = (1, 320, 1, 2, 64, 64), 32  # no other test (in any GDN file) dispatches this shape
    keep = []

    def run(with_h0, gate_dtype, chunk_size=chunk):
        t = _upload(_build_inputs(shape, with_h0), shape, gate_dtype, device)
        keep.append((t, _dispatch(t, shape, chunk_size)))
        return device.num_program_cache_entries()

    base = run(True, ttnn.float32)
    assert run(True, ttnn.float32) == base, "identical call missed the cache"
    with_no_h0 = run(False, ttnn.float32)
    assert with_no_h0 > base, "dropping initial_state reused the program"
    with_bf16_gates = run(True, ttnn.bfloat16)
    assert with_bf16_gates > with_no_h0, "a bf16 gate reused the fp32-gate program"
    assert run(True, ttnn.float32, chunk_size=64) > with_bf16_gates, "a different chunk size reused the program"


def _zero_inputs(device, B=1, T=64, HK=1, HV=1, K=32, V=32, gate_dtype=ttnn.float32):
    z = lambda *s, dt=ttnn.bfloat16: _tt(torch.zeros(*s), device, dt)
    return [
        z(B, 1, T, HK * K),
        z(B, 1, T, HK * K),
        z(B, 1, T, HV * V),
        z(B, 1, T, HV, dt=gate_dtype),
        z(B, 1, T, HV, dt=gate_dtype),
    ]


@pytest.mark.parametrize(
    "kwargs,overrides,match",
    [
        ({"chunk_size": 48}, {}, "chunk_size must be 32 or 64, got 48"),
        ({"chunk_size": 64}, {"T": 96}, "T=96 must be a multiple of chunk_size=64"),
        ({"chunk_size": 32}, {"HV": 3, "HK": 2}, "value heads HV=3 must be a multiple of key heads HK=2"),
        ({"chunk_size": 32}, {"HV": 33}, re.escape("HV=33 must be in [1, 32]")),
        ({"chunk_size": 32}, {"T": 0}, "B=1 must be at least 1 and T=0 at least chunk_size=32"),
        ({"chunk_size": 32}, {"B": 0}, "B=0 must be at least 1 and T=64 at least chunk_size=32"),
        ({"chunk_size": 32}, {"K": 48, "V": 48}, "K=48 and V=48 must be multiples of 32"),
        ({"chunk_size": 64}, {"K": 512, "V": 512}, "the prep at chunk_size=64 K=512 V=512 needs 4190208 B of L1"),
    ],
    ids=[
        "chunk_48",
        "T_not_multiple",
        "HV_not_multiple_of_HK",
        "HV_over_32",
        "T_0",
        "B_0",
        "K_V_not_tile",
        "L1_overflow",
    ],
)
def test_rejects_bad_shapes(kwargs, overrides, match, device, expect_error):
    dims = {"T": 64, "HK": 1, "HV": 1, "K": 32, "V": 32, **overrides}
    args = _zero_inputs(device, **dims)
    with expect_error(RuntimeError, match):
        ttml.ops.metal.chunk_gated_delta_rule_fw(*args, dims["K"], **kwargs)


def _bad_input(name, device):
    """Valid zero inputs (B=1, T=64, K=V=32) with one tensor replaced; returns (args, kwargs)."""
    q, k, v, g, beta = _zero_inputs(device)
    kwargs = {"chunk_size": 32}
    z = lambda *s, dt=ttnn.bfloat16, **kw: ttnn.from_torch(
        torch.zeros(*s),
        dtype=dt,
        layout=kw.get("layout", ttnn.TILE_LAYOUT),
        device=device,
        memory_config=kw.get("mem", ttnn.DRAM_MEMORY_CONFIG),
    )
    if name == "q_fp32":
        q = z(1, 1, 64, 32, dt=ttnn.float32)
    elif name == "q_row_major":
        q = z(1, 1, 64, 32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif name == "k_in_l1":
        k = z(1, 1, 64, 32, mem=ttnn.L1_MEMORY_CONFIG)
    elif name == "k_shape":
        k = z(1, 1, 64, 64)
    elif name == "gate_T":
        g, beta = z(1, 1, 32, 1, dt=ttnn.float32), z(1, 1, 32, 1, dt=ttnn.float32)
    elif name == "beta_dtype":
        beta = z(1, 1, 64, 1, dt=ttnn.bfloat16)
    elif name == "h0_bf16":
        kwargs["initial_state"] = z(1, 1, 32, 32)
    elif name == "h0_shape":
        kwargs["initial_state"] = z(1, 1, 64, 32, dt=ttnn.float32)
    return [q, k, v, g, beta], kwargs


@pytest.mark.parametrize(
    "name,match",
    [
        ("q_fp32", "q has unsupported dtype FLOAT32"),
        ("q_row_major", "q requires TILE layout. Got: ROW_MAJOR"),
        ("k_in_l1", "k must be in DRAM. Got: L1"),
        ("k_shape", "shapes disagree"),
        ("gate_T", "shapes disagree"),
        ("beta_dtype", "beta has unsupported dtype BFLOAT16"),
        ("h0_bf16", "initial_state has unsupported dtype BFLOAT16"),
        ("h0_shape", re.escape("initial_state must be Shape([1, 1, 32, 32]), got Shape([1, 1, 64, 32])")),
    ],
)
def test_rejects_bad_inputs(name, match, device, expect_error):
    args, kwargs = _bad_input(name, device)
    with expect_error(RuntimeError, match):
        ttml.ops.metal.chunk_gated_delta_rule_fw(*args, 32, **kwargs)


def test_placement_is_rejected_after_the_program_is_cached(device, expect_error):
    """With the DRAM program cached, k in L1 must still fail: validation runs on cache hits too, and the
    program key carries the placement, so the DRAM program is neither reused nor trusted for it."""
    args, kwargs = _bad_input("none", device)
    ttml.ops.metal.chunk_gated_delta_rule_fw(*args, 32, **kwargs)  # fills the cache with the DRAM variant
    args[1] = ttnn.to_memory_config(args[1], ttnn.L1_MEMORY_CONFIG)
    with expect_error(RuntimeError, "k must be in DRAM. Got: L1"):
        ttml.ops.metal.chunk_gated_delta_rule_fw(*args, 32, **kwargs)


@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize(
    "shape,chunk",
    [SHAPES[1], ((1, 96, 2, 6, 64, 128), 32), ((1, 128, 1, 3, 128, 128), 64)],
    ids=["B1_T128_HK2_HV4_K64_V64_c64", "B1_T96_HK2_HV6_K64_V128_c32", "B1_T128_HK1_HV3_K128_V128_c64"],
)
def test_save_for_backward_false_skips_only_the_saved_state(shape, chunk, with_h0, device):
    """o and final_state must be bit-identical with and without the saved-state writes; h, v_new, g_cumsum and
    A come back None.  Both orders of the two modes run on one cache, so a program built for one mode cannot
    be served to the other."""
    x = _build_inputs(shape, with_h0, seed=7)
    t = _upload(x, shape, ttnn.float32, device)
    lean = _dispatch(t, shape, chunk, save_for_backward=False)
    assert all(r is None for r in lean[2:]), "skipped outputs were returned"
    full = _dispatch(t, shape, chunk)
    assert all(r is not None for r in full)
    lean_again = _dispatch(t, shape, chunk, save_for_backward=False)
    for name, a, b, c in zip(OUT_NAMES[:2], lean[:2], full[:2], lean_again[:2]):
        a, b, c = ttnn.to_torch(a), ttnn.to_torch(b), ttnn.to_torch(c)
        assert torch.equal(a, b), f"{name}: save_for_backward=False changed the result"
        assert torch.equal(a, c), f"{name}: rerun after the saved-state program changed the result"
    got = _to_head_major(_to_host(full), shape)
    _compare(got, _reference(x, shape, chunk, ttnn.float32), chunk, with_h0)


def test_save_for_backward_is_part_of_the_program_hash(device):
    shape, chunk = (1, 160, 1, 2, 64, 64), 32  # a shape no other test dispatches
    t = _upload(_build_inputs(shape, True), shape, ttnn.float32, device)
    keep = [_dispatch(t, shape, chunk)]
    entries = device.num_program_cache_entries()
    keep.append(_dispatch(t, shape, chunk, save_for_backward=False))
    assert device.num_program_cache_entries() > entries, "toggling save_for_backward reused the program"
    entries = device.num_program_cache_entries()
    keep.append(_dispatch(t, shape, chunk, save_for_backward=False))
    assert device.num_program_cache_entries() == entries, "an identical lean call missed the cache"
