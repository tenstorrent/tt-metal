# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite: the forward written with existing ttnn ops,
which also returns what the backward consumes.

    q, k [B, 1, T, HK*K]      v [B, 1, T, HV*V]      g, beta [B, 1, T, HV]      initial_state [B, HV, K, V]
    o [B, 1, T, HV*V]         final_state [B, HV, K, V]
    h [B*HV, NC, K, V]        v_new [B*HV, 1, T, V]  A [B*HV, 1, T, C]

Pinned here:
1. Every output against the float64 chunked reference, with and without `initial_state`, bf16 and fp32 gates, GVA,
   K != V, odd key-tile counts, decaying and strong gates, and a Qwen3.8-sized shape.
2. The per-chip shape of Qwen3.8 on 4-way tensor parallelism (12 value heads, 4 key heads, K = V = 128) at seq 8192,
   where the scan runs 128 chunks: the error does not grow along the scan.
3. Strongly correlated keys, where (I + N)^-1 has large entries: the unit-lower solve must be exact substitution (a
   Neumann product overflows there).
4. The saved state is what the backward reads: `h[:, 0]` is the initial state (exactly zero without one), `A` is the
   inverse with nothing above the diagonal, and `save_for_backward=False` changes nothing else.
5. The head-major output is exactly the token-major output, permuted; `output_dtype=bfloat16` only rounds `o`.
6. An explicit `scale` is checked in absolute terms (relative RMS), not only by correlation.
7. Rejections name the offending value.

Tolerances: PCC >= `PCC_MIN` and a relative RMS error <= `RMS_MAX` for every output.
"""

from __future__ import annotations

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


@pytest.fixture
def expect_error():
    """tt-train python tests do not load the metal root conftest, so provide the same helper here."""

    def check(error, message):
        return pytest.raises(error, match=message)  # allow-pytest.raises: wrapped by the local expect_error fixture

    return check


PCC_MIN = 0.9999
RMS_MAX = 0.01

OUT_NAMES = ("o", "final_state", "h", "v_new", "A")
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

# More shapes: GVA at G = 4, 8 and 32, and K > V.
COVERAGE_SHAPES = [
    ((1, 128, 2, 8, 64, 64), 64),  # G = 4
    ((1, 64, 2, 16, 32, 32), 32),  # G = 8
    ((1, 64, 1, 32, 32, 32), 32),  # G = 32 = HV
    ((1, 128, 2, 2, 128, 64), 64),  # K > V
]
COVERAGE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in COVERAGE_SHAPES]

# (K, V, chunk): odd and prime tile counts, K > V and wide V.
DIM_SHAPES = [(32, 32, 32), (32, 128, 32), (64, 192, 64), (96, 96, 64), (128, 256, 32), (160, 160, 32)]
DIM_IDS = [f"K{k}_V{v}_c{c}" for k, v, c in DIM_SHAPES]

# Qwen3.8 per chip on 4-way tensor parallelism: 4 key heads, 12 value heads, K = V = 128, seq 8192.
TP4_SHAPE = (1, 8192, 4, 12, 128, 128)


@pytest.fixture(scope="module")
def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _build_inputs(shape, with_h0, seed=42, gate_scale=WEAK_GATE, aligned_keys=False):
    """Seeded inputs; q/k/v rounded to bf16 (device dtype), q/k L2-normalized along K (caller contract).

    aligned_keys: every key is a small perturbation of one shared direction, so k_i . k_j ~ 1 and (I + N)^-1 has
    entries far above 1.
    """
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    bf = lambda x: x.to(torch.bfloat16).double()
    q, k = l2(rn(B, T, HK, K)), l2(rn(B, T, HK, K))
    if aligned_keys:
        base = l2(rn(B, 1, HK, K))
        q, k = l2(base + 0.05 * rn(B, T, HK, K)), l2(base + 0.05 * rn(B, T, HK, K))
    return {
        "q": bf(q),
        "k": bf(k),
        "v": bf(rn(B, T, HV, V)),
        "g": F.logsigmoid(rn(B, T, HV)) * gate_scale,
        "beta": torch.rand(B, T, HV, generator=gen, dtype=torch.float64),
        "h0": rn(B, HV, K, V) * 0.1 if with_h0 else None,
    }


def _tt(t, device, dtype):
    return ttnn.from_torch(t.to(TORCH_DTYPE[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _upload(x, shape, gate_dtype, device):
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


def _dispatch(t, shape, chunk, scale=None, output_head_major=False, save_for_backward=True, output_dtype=ttnn.float32):
    return ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite(
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
        output_dtype=output_dtype,
        save_for_backward=save_for_backward,
    )


def _run(x, shape, chunk, gate_dtype, device, scale=None, output_head_major=False):
    return _dispatch(_upload(x, shape, gate_dtype, device), shape, chunk, scale, output_head_major)


def _to_host(outs):
    return [ttnn.to_torch(t).double() for t in outs]


def _to_head_major(outs, shape):
    """ttml outputs -> o [BH,T,V], final [B,HV,K,V], h [BH,NC,K,V], v_new [BH,T,V], A [BH,T,C]."""
    B, T, HK, HV, K, V = shape
    o, fs, h, vn, a = outs
    o = o.reshape(B, T, HV, V).permute(0, 2, 1, 3).reshape(B * HV, T, V)
    return [o, fs, h, vn.reshape(B * HV, T, V), a.reshape(B * HV, T, -1)]


def _reference(x, shape, chunk, gate_dtype, scale=None):
    """float64 outputs in the head-major layout of `_to_head_major`."""
    B, T, HK, HV, K, V = shape
    G = HV // HK
    gate = lambda t: t.to(TORCH_DTYPE[gate_dtype]).double()
    o, fs, h, vn, _, a = chunked_forward(
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
    return [tok(o), fs, h.transpose(1, 2).reshape(B * HV, -1, K, V), tok(vn), tok(a)]


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


def _check_vs_reference(x, shape, chunk, gate_dtype, device, with_h0, scale=None):
    got = _to_head_major(_to_host(_run(x, shape, chunk, gate_dtype, device, scale=scale)), shape)
    _compare(got, _reference(x, shape, chunk, gate_dtype, scale=scale), chunk, with_h0)


@pytest.mark.parametrize("gate_dtype", [ttnn.float32, ttnn.bfloat16], ids=["gate_fp32", "gate_bf16"])
@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("shape,chunk", SHAPES, ids=SHAPE_IDS)
def test_vs_reference(shape, chunk, with_h0, gate_dtype, device):
    _check_vs_reference(_build_inputs(shape, with_h0), shape, chunk, gate_dtype, device, with_h0)


@pytest.mark.parametrize("shape,chunk", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_decaying_gate_vs_reference(shape, chunk, device):
    """The within-chunk decay terms (decay of e^-13 .. e^-26 per chunk)."""
    x = _build_inputs(shape, with_h0=True, gate_scale=DECAY_GATE)
    _check_vs_reference(x, shape, chunk, ttnn.float32, device, with_h0=True)


@pytest.mark.parametrize("shape,chunk", [SHAPES[1], SHAPES[2]], ids=[SHAPE_IDS[1], SHAPE_IDS[2]])
def test_strong_gate_vs_reference(shape, chunk, device):
    """Within-chunk decay spans of ~200: the masked exponent keeps the upper triangle at exactly 0."""
    x = _build_inputs(shape, with_h0=True, gate_scale=STRONG_GATE)
    _check_vs_reference(x, shape, chunk, ttnn.float32, device, with_h0=True)


@pytest.mark.parametrize("shape,chunk", COVERAGE_SHAPES, ids=COVERAGE_IDS)
@pytest.mark.parametrize("gate_dtype", [ttnn.float32, ttnn.bfloat16], ids=["gate_fp32", "gate_bf16"])
def test_coverage_shapes_vs_reference(shape, chunk, gate_dtype, device):
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, gate_dtype, device, with_h0=True)


@pytest.mark.parametrize("K,V,chunk", DIM_SHAPES, ids=DIM_IDS)
def test_head_dims_vs_reference(K, V, chunk, device):
    shape = (1, 2 * chunk, 1, 2, K, V)
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, ttnn.float32, device, with_h0=True)


@pytest.mark.parametrize("chunk", [32, 64], ids=["c32", "c64"])
def test_aligned_keys_vs_reference(chunk, device):
    """k_i . k_j ~ 1: T_inv = (I + N)^-1 has entries far above 1.  Exact substitution handles it; the Neumann product
    (I - N)(I + N^2)(I + N^4)... overflows to inf/nan here."""
    shape = (1, 4 * chunk, 1, 2, 64, 64)
    x = _build_inputs(shape, with_h0=True, aligned_keys=True)
    _check_vs_reference(x, shape, chunk, ttnn.float32, device, with_h0=True)


def test_long_sequence_error_does_not_grow(device):
    """No decay is the worst case for error carried along the scan: at T = 4096 (64 chunks) the error matches the
    short-sequence one."""
    shape, chunk = (1, 4096, 1, 2, 128, 128), 64
    x = _build_inputs(shape, with_h0=True, gate_scale=0.0)
    _check_vs_reference(x, shape, chunk, ttnn.float32, device, with_h0=True)


def test_qwen_shape_vs_reference(device):
    shape, chunk = QWEN_SHAPE
    _check_vs_reference(_build_inputs(shape, with_h0=True), shape, chunk, ttnn.bfloat16, device, with_h0=True)


@pytest.mark.parametrize("chunk", [32, 64], ids=["c32", "c64"])
def test_qwen_tp4_per_chip_shape_vs_reference(chunk, device):
    """12 value heads, 4 key heads, K = V = 128 at seq 8192: 256 (c32) / 128 (c64) scan steps."""
    x = _build_inputs(TP4_SHAPE, with_h0=True)
    _check_vs_reference(x, TP4_SHAPE, chunk, ttnn.float32, device, with_h0=True)


@pytest.mark.parametrize("scale", [0.3, 1.0], ids=["scale_0.3", "scale_1"])
def test_explicit_scale(scale, device):
    """`o` is linear in q, so a wrong or ignored scale keeps PCC at 1; only the absolute error sees it."""
    shape, chunk = SHAPES[1]
    assert abs(scale - shape[4] ** -0.5) > 0.1, "scale must differ from the default K^-0.5"
    x = _build_inputs(shape, with_h0=True)
    default = _to_head_major(_to_host(_run(x, shape, chunk, ttnn.float32, device)), shape)
    got = _to_head_major(_to_host(_run(x, shape, chunk, ttnn.float32, device, scale=scale)), shape)
    _compare(got, _reference(x, shape, chunk, ttnn.float32, scale=scale), chunk, with_h0=True)
    ratio = got[0].norm() / default[0].norm()
    assert abs(ratio - scale / shape[4] ** -0.5) < 0.05 * ratio, f"o scaled by {ratio:.4f}"
    for name, g, d in zip(OUT_NAMES[1:], got[1:], default[1:]):
        assert torch.equal(g, d), f"{name} depends on scale"


@pytest.mark.parametrize("shape,chunk", [SHAPES[3], SHAPES[4]], ids=[SHAPE_IDS[3], SHAPE_IDS[4]])
def test_head_major_output_is_permuted_token_major(shape, chunk, device):
    B, T, HK, HV, K, V = shape
    x = _build_inputs(shape, with_h0=True, seed=2)
    t = _upload(x, shape, ttnn.bfloat16, device)
    token_major = _to_host(_dispatch(t, shape, chunk))
    head_major = _to_host(_dispatch(t, shape, chunk, output_head_major=True))
    assert tuple(head_major[0].shape) == (B * HV, 1, T, V)
    tm = token_major[0].reshape(B, T, HV, V).permute(0, 2, 1, 3).reshape(B * HV, 1, T, V)
    assert torch.equal(head_major[0], tm)
    for name, hm, tmj in zip(OUT_NAMES[1:], head_major[1:], token_major[1:]):
        assert torch.equal(hm, tmj), f"{name}: differs between output layouts"


@pytest.mark.parametrize("head_major", [False, True], ids=["token_major", "head_major"])
def test_bf16_output_only_rounds_o(head_major, device):
    shape, chunk = SHAPES[2]
    x = _build_inputs(shape, with_h0=True, seed=11)
    t = _upload(x, shape, ttnn.float32, device)
    full = _dispatch(t, shape, chunk, output_head_major=head_major)
    packed = _dispatch(t, shape, chunk, output_head_major=head_major, output_dtype=ttnn.bfloat16)
    assert full[0].dtype == ttnn.float32 and packed[0].dtype == ttnn.bfloat16
    assert all(p.dtype == ttnn.float32 for p in packed[1:]), "only o follows output_dtype"
    o32, o16 = ttnn.to_torch(full[0]).double(), ttnn.to_torch(packed[0]).double()
    assert torch.all((o16 - o32).abs() <= o32.abs() * 2**-7), "o differs from the fp32 o by more than bf16 rounding"
    for name, a, b in zip(OUT_NAMES[1:], full[1:], packed[1:]):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"{name}: output_dtype changed it"


def test_save_for_backward_false_skips_only_the_saved_state(device):
    shape, chunk = SHAPES[2]
    t = _upload(_build_inputs(shape, with_h0=True, seed=5), shape, ttnn.float32, device)
    saved = _dispatch(t, shape, chunk)
    unsaved = _dispatch(t, shape, chunk, save_for_backward=False)
    assert unsaved[2] is None and unsaved[3] is None and unsaved[4] is None
    for name, a, b in zip(OUT_NAMES[:2], saved[:2], unsaved[:2]):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"{name}: save_for_backward changed it"


def test_saved_state_has_the_layout_the_backward_reads(device):
    """h [BH,NC,K,V], v_new [BH,1,T,V], A [BH,1,T,C], all fp32 TILE: the contract of chunk_gated_delta_rule_bw."""
    shape, chunk = SHAPES[2]
    B, T, HK, HV, K, V = shape
    t = _upload(_build_inputs(shape, with_h0=True), shape, ttnn.float32, device)
    o, fs, h, vn, a = _dispatch(t, shape, chunk)
    for name, tensor, expected in (
        ("h", h, (B * HV, T // chunk, K, V)),
        ("v_new", vn, (B * HV, 1, T, V)),
        ("A", a, (B * HV, 1, T, chunk)),
        ("final_state", fs, (B, HV, K, V)),
    ):
        assert tuple(tensor.shape) == expected, f"{name}: {tuple(tensor.shape)} != {expected}"
        assert tensor.dtype == ttnn.float32 and tensor.layout == ttnn.TILE_LAYOUT, name


def _bad_inputs(device, **over):
    shape = (1, 64, 1, 1, 32, 32)
    t = _upload(_build_inputs(shape, with_h0=False), shape, ttnn.float32, device)
    args = dict(t, key_head_dim=32, chunk_size=32)
    args.update(over)
    return args


def _call(args):
    return ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite(
        args["q"],
        args["k"],
        args["v"],
        args["g"],
        args["beta"],
        args["key_head_dim"],
        initial_state=args["h0"],
        chunk_size=args["chunk_size"],
    )


@pytest.mark.parametrize(
    "over,message",
    [
        ({"chunk_size": 16}, "chunk_size must be 32 or 64, got 16"),
        ({"chunk_size": 128}, "chunk_size must be 32 or 64, got 128"),
        ({"key_head_dim": 48}, "key_head_dim must be a positive multiple of 32, got 48"),
    ],
    ids=["chunk16", "chunk128", "key_head_dim48"],
)
def test_rejects_bad_attributes(over, message, device, expect_error):
    with expect_error(RuntimeError, message):
        _call(_bad_inputs(device, **over))


def test_rejects_unaligned_sequence(device, expect_error):
    shape = (1, 48, 1, 1, 32, 32)
    t = _upload(_build_inputs(shape, with_h0=False), shape, ttnn.float32, device)
    with expect_error(RuntimeError, "T=48 must be a multiple of chunk_size=32"):
        _call(dict(t, key_head_dim=32, chunk_size=32))


def test_rejects_non_bf16_q(device, expect_error):
    shape = (1, 64, 1, 1, 32, 32)
    x = _build_inputs(shape, with_h0=False)
    t = _upload(x, shape, ttnn.float32, device)
    t["q"] = _tt(x["q"].reshape(1, 1, 64, -1), device, ttnn.float32)
    with expect_error(RuntimeError, "q has unsupported dtype"):
        _call(dict(t, key_head_dim=32, chunk_size=32))


def test_rejects_mixed_gate_dtypes(device, expect_error):
    shape = (1, 64, 1, 1, 32, 32)
    x = _build_inputs(shape, with_h0=False)
    t = _upload(x, shape, ttnn.float32, device)
    t["beta"] = _tt(x["beta"].reshape(1, 1, 64, -1), device, ttnn.bfloat16)
    with expect_error(RuntimeError, "beta has unsupported dtype"):
        _call(dict(t, key_head_dim=32, chunk_size=32))


def test_rejects_bad_initial_state(device, expect_error):
    shape = (1, 64, 1, 1, 32, 32)
    x = _build_inputs(shape, with_h0=True)
    t = _upload(x, shape, ttnn.float32, device)
    t["h0"] = _tt(x["h0"], device, ttnn.bfloat16)
    with expect_error(RuntimeError, "initial_state has unsupported dtype"):
        _call(dict(t, key_head_dim=32, chunk_size=32))


def test_rejects_hv_not_multiple_of_hk(device, expect_error):
    shape = (1, 64, 2, 3, 32, 32)  # 2 key heads, 3 value heads
    t = _upload(_build_inputs(shape, with_h0=False), shape, ttnn.float32, device)
    with expect_error(RuntimeError, "HK=2 must divide HV=3"):
        _call(dict(t, key_head_dim=32, chunk_size=32))


def test_rejects_host_tensor(device, expect_error):
    shape = (1, 64, 1, 1, 32, 32)
    x = _build_inputs(shape, with_h0=False)
    t = _upload(x, shape, ttnn.float32, device)
    t["k"] = ttnn.from_torch(x["k"].reshape(1, 1, 64, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "k must be on Device"):
        _call(dict(t, key_head_dim=32, chunk_size=32))


def test_rejects_row_major_input(device, expect_error):
    shape = (1, 64, 1, 1, 32, 32)
    x = _build_inputs(shape, with_h0=False)
    t = _upload(x, shape, ttnn.float32, device)
    t["v"] = ttnn.from_torch(
        x["v"].reshape(1, 1, 64, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error(RuntimeError, "v requires TILE layout"):
        _call(dict(t, key_head_dim=32, chunk_size=32))
