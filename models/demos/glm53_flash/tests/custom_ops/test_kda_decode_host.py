# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of ``custom_ops/kda_decode`` and the ``tt/kda.py`` knobs (no device; _agents/kda_roofline):

* the kernel ABI of both variants: every ``get_compile_time_arg_val`` / ``get_arg_val`` index of the default kernels
  (``*_kda_decode.cpp``, today's layouts: 11 compute args, reader accessors from 11, 15 reader runtime args) and of the
  knob variant (``*_kda_decode_v2.cpp``: arg 11 ``GATES_ACT``, accessors from 12, 20 runtime args) against the vectors
  ``KdaDecode.compile_time_args`` / ``runtime_args_for_core`` pack for that variant, and the CB / semaphore indices of all
  five sources against ``KdaDecode.CB`` / ``SEM_*``;
* the placements (column runs = today's, 2-column rectangles) on the grids this op can meet: the kernels' item decode
  ``(head, tile row, part)`` vs the core lists, the multicast geometry the reader derives (``n_a + n_b = parts - 1``,
  part-0 top-left, rectangle B one column), the fall-back to the column layout, the L1 budget;
* the ``gates="act"`` math: the pre-arranged g2 weights + bias reproduce ``a (f + dt)`` exactly in fp32 (the bf16
  rounding of the scaled columns bounded by half a bf16 ulp), and the golden in ``gates_activated`` mode is
  bit-identical to the ``pre`` golden on host-activated gates (the same fp32 ops in the same order);
* the knob plumbing: ``kda_decode_options`` / ``kda_mm_override`` / ``mcast1d_config`` / ``decode_linear_for``;
* the gated variant (``KdaDecode(gated=True)``, ``*_kda_decode_gated.cpp`` + the v2 compute; the speculative commit of
  ``tt/spec_decode.py``): its kernel ABI (compile-time scalars as v2, one more accessor block, the gate address at
  reader 20 / writer 7) against the vectors the builders pack, the vectors as extensions of the v2 / v1 ones, the CB
  plan's scratch pages, the kernels' scratch constants, the gated golden, the call-site checks.

    pytest --noconftest models/demos/glm53_flash/tests/custom_ops/test_kda_decode_host.py -q
"""

from __future__ import annotations

import contextlib
import re

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.custom_ops.kda_decode.op import (
    GATES_MODES,
    KERNEL_DIR,
    ONES_PAGE,
    PLACEMENTS,
    KdaDecode,
    f32_bits,
    prearrange_gate_weights,
)
from models.demos.glm53_flash.tt.kda import (
    KDA_DECODE_FIDELITIES,
    BiasedDecodeLinear,
    decode_linear_for,
    kda_decode_options,
    kda_mm_override,
    mcast1d_config,
)

K = V = 128
GRIDS = [(12, 10), (13, 10), (14, 10), (8, 8), (10, 10), (11, 6)]
ACC_WORDS = [2, 2048]  # one interleaved DRAM accessor: (config = IsDram, aligned page size)


class Grid:
    def __init__(self, x, y):
        self.x, self.y = x, y


@contextlib.contextmanager
def expect_error(error, message=""):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` (the root conftest opens the UMD cluster at collection)."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


def _src(name: str) -> str:
    return (KERNEL_DIR / name).read_text()


def _ct_indices(src: str) -> dict[str, int]:
    pat = re.compile(r"constexpr\s+(?:uint32_t|bool)\s+(\w+)\s*=\s*get_compile_time_arg_val\((\d+)\)")
    return {m.group(1): int(m.group(2)) for m in pat.finditer(src)}


def _rt_indices(src: str) -> dict[str, int]:
    pat = re.compile(r"const\s+uint32_t\s+(\w+)\s*=\s*get_arg_val<uint32_t>\((\d+)\)")
    return {m.group(1): int(m.group(2)) for m in pat.finditer(src)}


def _accessor_start(src: str) -> int:
    m = re.search(r"TensorAccessorArgs<(\d+)>\(\)", src)
    assert m, "no TensorAccessorArgs<N>() in the kernel"
    return int(m.group(1))


def _cb_consts(src: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"constexpr uint32_t cb_(\w+) = (\d+)", src)}


def _noc(core):
    return 100 + int(core.x), 200 + int(core.y)


def _op(gates="pre", placement="col", H=16):
    return KdaDecode(num_heads=H, gates=gates, placement=placement)


VARIANT_OPS = {"v1": dict(gates="pre", placement="col"), "v2": dict(gates="act", placement="rect")}


def test_variant_selection():
    assert _op().variant == "v1" and _op(gates="act").variant == "v2" and _op(placement="rect").variant == "v2"
    assert KdaDecode(num_heads=16, math_fidelity=ttnn.MathFidelity.HiFi2).variant == "v1"  # fidelity: no kernel change
    assert _op().kernel_sources() == ("reader_kda_decode.cpp", "writer_kda_decode.cpp", "compute_kda_decode.cpp")
    assert _op(gates="act").kernel_sources() == (
        "reader_kda_decode_v2.cpp",
        "writer_kda_decode.cpp",
        "compute_kda_decode_v2.cpp",
    )
    for name in ("reader_kda_decode.cpp", "compute_kda_decode.cpp"):  # the default sources know nothing of the variant
        assert "GATES_ACT" not in _src(name) and "rb_x0" not in _src(name), name
    assert (KERNEL_DIR / "writer_kda_decode.cpp").exists()


# ---------------------------------------------------------------------------------------------- kernel ABI
@pytest.mark.parametrize("variant", ["v1", "v2"])
def test_reader_abi(variant):
    op = _op(**VARIANT_OPS[variant])
    src = _src(op.kernel_sources()[0])
    ct = _ct_indices(src)
    expect = dict(Kt=0, Vt=1, Bt=2, B=3, H=4, Wq_t=5, Wg_t=6, HK_t=7, HV_t=8, parts=9, ONES_PAGE=10)
    if variant == "v2":
        expect["GATES_ACT"] = 11
    assert ct == expect, (variant, ct)
    assert _accessor_start(src) == len(expect) == KdaDecode.READER_CT_SCALARS[variant]
    reader, _, _ = op.compile_time_args(32, 6, ACC_WORDS * 8, ACC_WORDS * 2)
    head = [4, 4, 1, 32, 16, 192, 64 + 64 + 1, 64, 64, 6, ONES_PAGE]
    if variant == "v2":
        head.append(1)
        assert _op(gates="pre", placement="rect").compile_time_args(32, 6, ACC_WORDS * 8, ACC_WORDS * 2)[0][11] == 0
    assert reader == head + ACC_WORDS * 8
    rt = _rt_indices(src)
    names = {k[: -len("_addr")] if k.endswith("_addr") else k: v for k, v in rt.items()}
    n = KdaDecode.READER_RT_LEN[variant]
    assert names == {k: v for k, v in KdaDecode.READER_RT.items() if v < n}, (variant, names)
    assert max(rt.values()) + 1 == n


def test_writer_abi():
    src = _src("writer_kda_decode.cpp")
    ct = _ct_indices(src)
    expect = dict(Kt=0, Vt=1, Bt=2, B=3, H=4, parts=5, HV_t=6)
    assert {k: ct[k] for k in expect} == expect
    assert _accessor_start(src) == len(expect)
    _, writer, _ = _op().compile_time_args(40, 3, ACC_WORDS * 8, ACC_WORDS * 2)
    assert writer == [4, 4, 2, 40, 16, 3, 64] + ACC_WORDS * 2
    rt = _rt_indices(src)
    names = {("out" if k == "o_addr" else k.replace("_addr", "")): v for k, v in rt.items()}
    assert names == KdaDecode.WRITER_RT, names


@pytest.mark.parametrize("variant", ["v1", "v2"])
def test_compute_abi(variant):
    op = KdaDecode(num_heads=16, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5, **VARIANT_OPS[variant])
    src = _src(op.kernel_sources()[2])
    ct = _ct_indices(src)
    expect = dict(
        Kt=0, Vt=1, Bt=2, B=3, parts=4, lb_bits=5, l2eps_bits=6, normeps_bits=7, qscale_bits=8, invv_bits=9, IT_H=10,
    )  # fmt: skip
    if variant == "v2":
        expect["GATES_ACT"] = 11
    assert ct == expect, (variant, ct)
    _, _, comp = op.compile_time_args(4, 5, [], [])
    want = [4, 4, 1, 4, 5, f32_bits(-5.0), f32_bits(1e-6), f32_bits(1e-5), f32_bits(K**-0.5), f32_bits(1 / V), 2]
    if variant == "v2":
        want.append(1)
    assert comp == want and len(comp) == KdaDecode.COMPUTE_CT_LEN[variant]
    assert op.compile_time_args(32, 5, [], [])[2][10] == 8  # 16 live rows per face pair at B = 32 (full tiles)
    assert op.compile_time_args(10, 5, [], [])[2][10] == 5
    assert _rt_indices(src) == KdaDecode.COMPUTE_RT


def test_cb_and_semaphore_indices_match_kernels():
    for name in ("compute_kda_decode.cpp", "compute_kda_decode_v2.cpp"):
        comp = _cb_consts(_src(name))
        assert comp == KdaDecode.CB, (name, comp, KdaDecode.CB)
    for name in ("reader_kda_decode.cpp", "reader_kda_decode_v2.cpp", "writer_kda_decode.cpp"):
        for cb, idx in _cb_consts(_src(name)).items():
            assert KdaDecode.CB[cb] == idx, (name, cb, idx)
        src = _src(name)
        for sem, idx in re.findall(r"constexpr uint32_t (SEM_\w+) = (\d+)", src):
            assert getattr(KdaDecode, sem) == int(idx), (name, sem)
    plan = _op().cb_plan(5)
    assert [n for n, _, _ in plan] == list(KdaDecode.CB), "CB plan order = CB index order"
    assert len({KdaDecode.CB[n] for n, _, _ in plan}) == 32


def test_runtime_arg_vectors_match_layout():
    op = _op(placement="rect")
    parts, core_of, rect_h = op.placement(Grid(12, 10), 16, "rect")
    addrs = dict(qkv=1, hist=2, taps=3, gates=4, cc=5, onw=6, sel=7, st=8, out=9)
    r, w, c = op.runtime_args_for_core(7, 1, len(core_of), addrs, _noc, core_of, parts, rect_h)
    assert len(r) == len(KdaDecode.READER_RT) and len(w) == len(KdaDecode.WRITER_RT) and len(c) == 3
    rr = KdaDecode.READER_RT
    assert [r[rr[k]] for k in ("qkv", "hist", "taps", "gates", "cc", "onw", "sel", "st")] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert (w[KdaDecode.WRITER_RT["out"]], w[KdaDecode.WRITER_RT["st"]]) == (9, 8)
    assert (r[rr["p0_x"]], r[rr["p0_y"]]) == (w[KdaDecode.WRITER_RT["p0_x"]], w[KdaDecode.WRITER_RT["p0_y"]])
    assert r[:3] == w[:3] == c == [7, 1, len(core_of)]
    # the default variant packs today's 15-entry reader vector (no second column), bit for bit the v2 prefix
    op1 = _op()
    parts1, core1, h1 = op1.placement(Grid(12, 10), 16, "col")
    r1, w1, c1 = op1.runtime_args_for_core(7, 1, len(core1), addrs, _noc, core1, parts1, h1)
    assert len(r1) == 15 and h1 == 0
    r1_v2 = _op(gates="act").runtime_args_for_core(7, 1, len(core1), addrs, _noc, core1, parts1, 0)[0]
    assert r1_v2[:15] == r1 and r1_v2[15:] == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------------------------- placement
@pytest.mark.parametrize("gx, gy", GRIDS)
@pytest.mark.parametrize("slots", [16, 32, 64, 128])
@pytest.mark.parametrize("mode", PLACEMENTS)
def test_placement_geometry(gx, gy, slots, mode):
    op = _op(placement=mode)
    Bt = 1
    parts, core_of, rect_h = op.placement(Grid(gx, gy), slots, mode)
    assert parts >= 1 and len(core_of) <= gx * gy
    assert len({(c.x, c.y) for c in core_of}) == len(core_of), "a core holds one item"
    assert all(0 <= c.x < gx and 0 <= c.y < gy for c in core_of)
    if parts == 1:
        assert rect_h == 0 and len(core_of) == min(slots, gx * gy)
        return
    assert len(core_of) == slots * parts
    for c in range(len(core_of)):
        h, tb, part = op.item_of(c, parts, Bt)  # the kernels' decode of work item c (one item per core)
        s = h * Bt + tb
        assert c == s * parts + part
        p0, pe, rb0, rb1 = op.slot_geometry(core_of, parts, rect_h, s)
        core = core_of[c]
        if rect_h == 0:
            assert core.x == p0.x and core.y == p0.y + part and pe.y == p0.y + parts - 1 and rb0 is None
        else:
            assert parts == 2 * rect_h
            col, row = (0, part) if part < rect_h else (1, part - rect_h)
            assert (core.x, core.y) == (p0.x + col, p0.y + row)
            assert (pe.x, pe.y) == (p0.x, p0.y + rect_h - 1)
            assert (rb0.x, rb0.y, rb1.x, rb1.y) == (p0.x + 1, p0.y, p0.x + 1, p0.y + rect_h - 1)
        # the reader's multicast arithmetic: parts below the sender in its column + the second column = parts - 1
        n_b = rect_h
        n_a = parts - 1 - n_b
        assert n_a == pe.y - p0.y and n_a + n_b == parts - 1 and n_a >= 0
    assert op.l1_bytes(parts) <= 1_200_000, "per-core CBs within the 1.5 MB L1 minus headroom"


def test_rect_placement_wins_only_with_more_parts():
    op = _op(placement="rect")
    parts, cores, h = op.placement(Grid(12, 10), 16, "rect")
    assert (parts, len(cores), h) == (6, 96, 3)
    pc, cc, hc = op.placement(Grid(12, 10), 16, "col")
    assert (pc, len(cc), hc) == (5, 80, 0)
    # two tile rows (B = 40 at 16 heads): the column layout gives 3 parts, the rectangle layout only 2: column kept
    for g in (Grid(12, 10), Grid(8, 8)):
        for slots in (32, 64):
            assert op.placement(g, slots, "rect")[0] == op.placement(g, slots, "col")[0]
            assert op.placement(g, slots, "rect")[2] == 0
    assert KdaDecode.placement_rect(Grid(1, 10), 16) is None
    assert KdaDecode.placement_rect(Grid(12, 10), 16) == (6, KdaDecode.placement_rect(Grid(12, 10), 16)[1], 3)


def test_rect_runtime_args_two_multicast_columns():
    op = _op(placement="rect")
    parts, core_of, rect_h = op.placement(Grid(12, 10), 16, "rect")
    addrs = {k: 0 for k in ("qkv", "hist", "taps", "gates", "cc", "onw", "sel", "st", "out")}
    rr = KdaDecode.READER_RT
    for s in range(16):
        p0, pe, rb0, rb1 = op.slot_geometry(core_of, parts, rect_h, s)
        vecs = [
            op.runtime_args_for_core(s * parts + p, 1, 96, addrs, _noc, core_of, parts, rect_h)[0] for p in range(parts)
        ]
        assert all(
            v[rr["p0_x"] :] == vecs[0][rr["p0_x"] :] for v in vecs
        ), "every part of a slot sees the same geometry"
        v = vecs[0]
        assert (v[rr["p0_x"]], v[rr["p0_y"]]) == _noc(p0) and (v[rr["pe_x"]], v[rr["pe_y"]]) == _noc(pe)
        assert (v[rr["rb_x0"]], v[rr["rb_y0"]]) == _noc(rb0) and (v[rr["rb_x1"]], v[rr["rb_y1"]]) == _noc(rb1)
        assert v[rr["n_b"]] == rect_h == 3 and v[rr["rb_x0"]] == v[rr["rb_x1"]], "rectangle B is one NoC column"
    # column layout under the v2 kernels (gates="act" alone): no rectangle B, 5 zeros appended to today's values
    opc = _op(gates="act")
    parts, core_of, rect_h = opc.placement(Grid(12, 10), 16, "col")
    v = opc.runtime_args_for_core(3, 1, 80, addrs, _noc, core_of, parts, rect_h)[0]
    assert v[rr["rb_x0"] :] == [0, 0, 0, 0, 0] and v[rr["pe_y"]] - v[rr["p0_y"]] == parts - 1
    assert _op().runtime_args_for_core(3, 1, 80, addrs, _noc, core_of, parts, rect_h)[0] == v[:15]


def test_single_part_round_robin_item_counts():
    op = _op(H=64)
    parts, core_of, rect_h = op.placement(Grid(12, 10), 64 * 4, "col")  # 64 heads x 4 tile rows (B = 128)
    assert parts == 1 and len(core_of) == 120
    NI, NC = 256, 120
    counts = [(NI - c + NC - 1) // NC for c in range(NC)]
    assert sum(counts) == NI and max(counts) - min(counts) <= 1


# ---------------------------------------------------------------------------------------------- gate math
def _inputs(B, H, seed):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    W = 3 * H * K
    Hb = -(-H // 32) * 32
    qkv = r(1, 1, B, W).to(torch.bfloat16).float()
    gates = torch.zeros(1, 1, B, H * K + H * V + Hb)
    gates[..., : H * K] = r(B, H * K) * 2.0
    gates[..., H * K : H * K + H * V] = r(B, H * V) * 2.0
    gates[..., H * K + H * V : H * K + H * V + H] = r(B, H) * 2.0
    conv = (r(B, 3, W) * 0.8).to(torch.bfloat16).float()
    S = 0.05 * r(B, H, K, V)
    taps = (r(4, W) * 0.5).to(torch.bfloat16).float()
    dt_bias = r(H * K) * 0.5
    a_exp = torch.rand(H, generator=gen).repeat_interleave(K) * 4.0 + 0.5
    onw = torch.rand(V, generator=gen) * 0.5 + 0.75
    return dict(qkv=qkv, gates=gates, conv=conv, S=S, taps=taps, dt_bias=dt_bias, a_exp=a_exp, onw=onw)


@pytest.mark.parametrize("B, H", [(4, 16), (32, 16), (1, 64)])
def test_golden_activated_gates_equivalent(B, H):
    x = _inputs(B, H, seed=B + H)
    common = dict(num_heads=H, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5)
    o, S, conv = KdaDecode.golden(
        x["qkv"][0, 0], x["gates"][0, 0], x["conv"], x["S"], x["taps"], x["dt_bias"], x["a_exp"], x["onw"], **common
    )
    g_act = KdaDecode.activate_gates(x["gates"][0, 0], x["dt_bias"], x["a_exp"], H * K)
    o2, S2, conv2 = KdaDecode.golden(
        x["qkv"][0, 0], g_act, x["conv"], x["S"], x["taps"], None, None, x["onw"], gates_activated=True, **common
    )
    # the decay path (the only math the program itself changes) is bit-identical: the same fp32 ops in the same order
    f = x["gates"][0, 0][:, : H * K]
    assert torch.equal(
        torch.exp(-5.0 * torch.sigmoid(x["a_exp"] * (f + x["dt_bias"]))), torch.exp(-5.0 * g_act[:, : H * K])
    )
    # the output gate / beta sigmoids are the same function on differently strided views (torch picks a different
    # vectorised kernel per layout: 1-ulp flips on a few elements), so the end-to-end comparison is at fp32 ulps
    assert torch.equal(conv, conv2)
    torch.testing.assert_close(o2, o, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(S2, S, rtol=1e-6, atol=1e-7)
    assert (o2 - o).abs().max() <= 4 * torch.finfo(torch.float32).eps * o.abs().max()


def test_prearranged_gate_weights_fp32_exact():
    gen = torch.Generator().manual_seed(3)
    H, d, Hb = 16, 288, 32
    hk, N = H * K, H * K + H * V + Hb
    w = (torch.randn(d, N, generator=gen) * 0.05).to(torch.bfloat16).float()  # the g2 block is bf16-valued
    a_exp = (torch.rand(H, generator=gen) * 4 + 0.5).repeat_interleave(K)
    dt = torch.randn(hk, generator=gen) * 0.5
    w_act, bias = prearrange_gate_weights(w, a_exp, dt, hk)
    assert w_act.dtype == torch.float32 and bias.dtype == torch.float32 and bias.shape == (N,)
    assert torch.equal(w_act[:, hk:], w[:, hk:]) and torch.equal(bias[hk:], torch.zeros(N - hk))
    assert torch.equal(w_act[:, :hk], w[:, :hk] * a_exp[None, :]) and torch.equal(bias[:hk], a_exp * dt)
    g1 = (torch.randn(32, d, generator=gen) * 0.3).to(torch.bfloat16).double()
    direct = a_exp.double() * (g1 @ w[:, :hk].double() + dt.double())
    folded = g1 @ w_act[:, :hk].double() + bias[:hk].double()
    torch.testing.assert_close(folded, direct, rtol=1e-6, atol=1e-6)  # fp32 products of a bf16 weight and fp32 a
    # the bf16 rounding of the scaled columns (what the matmul reads): half a bf16 ulp relative, 2^-8
    rel = (w_act[:, :hk].to(torch.bfloat16).float() - w_act[:, :hk]).abs() / w_act[:, :hk].abs().clamp_min(1e-30)
    assert rel.max() <= 2.0**-8 + 1e-7
    with expect_error(AssertionError):
        prearrange_gate_weights(w, a_exp[:10], dt, hk)


def test_prearranged_weights_per_chip_slices_match_layer_construction():
    """What ``TtKDA`` builds per TP column: chip c scales its f columns with the channels of heads c*H..(c+1)*H."""
    gen = torch.Generator().manual_seed(5)
    tp, H, d = 4, 4, 16
    H_total = tp * H
    hk, Hb = H * K, 32
    fb_t = torch.randn(d, H_total * K, generator=gen)
    a_flat = (torch.rand(H_total, generator=gen) + 0.5).repeat_interleave(K)
    dt_flat = torch.randn(H_total * K, generator=gen)
    for c in range(tp):
        w_g2 = torch.zeros(2 * d + Hb, hk + H * V + Hb)
        w_g2[:d, :hk] = fb_t[:, c * hk : (c + 1) * hk]
        sl = slice(c * hk, (c + 1) * hk)
        w_act, bias = prearrange_gate_weights(w_g2, a_flat[sl], dt_flat[sl], hk)
        want = fb_t[:, sl] * a_flat[sl][None, :]
        assert torch.equal(w_act[:d, :hk], want) and torch.equal(bias[:hk], a_flat[sl] * dt_flat[sl])
        assert torch.equal(w_act[d:, :], w_g2[d:, :])


# ---------------------------------------------------------------------------------------------- knobs
def test_decode_options_defaults_bundle_and_overrides():
    assert kda_decode_options({}) == dict(gates="pre", placement="col", fidelity="hifi4")
    assert kda_decode_options({"GLM53_KDA_DECODE_V2": "1"}) == dict(gates="act", placement="rect", fidelity="hifi4")
    assert kda_decode_options({"GLM53_KDA_DECODE_V2": "0"}) == kda_decode_options({})
    assert kda_decode_options({"GLM53_KDA_DECODE_V2": "1", "GLM53_KDA_DECODE_GATES": "pre"})["gates"] == "pre"
    assert kda_decode_options({"GLM53_KDA_DECODE_PLACEMENT": "rect"}) == dict(
        gates="pre", placement="rect", fidelity="hifi4"
    )
    assert kda_decode_options({"GLM53_KDA_DECODE_FIDELITY": "HiFi2"})["fidelity"] == "hifi2"
    for bad in (
        {"GLM53_KDA_DECODE_GATES": "x"},
        {"GLM53_KDA_DECODE_PLACEMENT": "row"},
        {"GLM53_KDA_DECODE_FIDELITY": "lofi"},
    ):
        with expect_error(ValueError):
            kda_decode_options(bad)
    assert set(KDA_DECODE_FIDELITIES) == {"hifi4", "hifi2"} and tuple(GATES_MODES) == ("pre", "act")
    with expect_error(AssertionError):
        KdaDecode(num_heads=16, gates="bogus")
    with expect_error(AssertionError):
        KdaDecode(num_heads=16, placement="row")
    assert KdaDecode(num_heads=16, gates="act").gates_activated and not KdaDecode(num_heads=16).gates_activated


def test_mm_override_and_config():
    assert kda_mm_override("QKV", {}) is None
    assert kda_mm_override("O", {"GLM53_KDA_MM_O": "12,6,2,8"}) == (12, 6, 2, 8)
    for bad in ("12,6,2", "0,6,2,8", "a,b,c,d"):
        with expect_error(ValueError):
            kda_mm_override("G1", {"GLM53_KDA_MM_G1": bad})
    with expect_error(AssertionError):
        kda_mm_override("ROUTER", {})
    cfg = mcast1d_config(12, 8, 2, 4)
    assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (12, 8)
    assert (cfg.per_core_N, cfg.in0_block_w, cfg.out_subblock_w, cfg.per_core_M, cfg.mcast_in0) == (2, 4, 2, 1, True)
    assert mcast1d_config(9, 1, 1, 16).out_subblock_w == 1 and mcast1d_config(12, 2, 6, 8).out_subblock_w == 2
    act = mcast1d_config(12, 8, 2, 9, activation=ttnn.UnaryOpType.SIGMOID)
    assert act.fused_activation is not None and act.fused_activation.op_type == ttnn.UnaryOpType.SIGMOID


class _FakeDevice:
    def compute_with_storage_grid_size(self):
        return Grid(12, 10)


class _FakeTensor:
    def __init__(self, *shape):
        self.shape = list(shape)


def test_decode_linear_for_applies_override_and_bias():
    dev, w, b = _FakeDevice(), _FakeTensor(288, 4128), _FakeTensor(1, 1, 1, 4128)
    plain = decode_linear_for(dev, w, "G2", env={})
    assert not isinstance(plain, BiasedDecodeLinear) and plain.config().fused_activation is None
    assert plain.config().compute_with_storage_grid_size.x * plain.config().compute_with_storage_grid_size.y >= 65
    biased = decode_linear_for(dev, w, "G2", activation=ttnn.UnaryOpType.SIGMOID, bias=b, env={})
    assert isinstance(biased, BiasedDecodeLinear) and biased.bias is b and biased._act_name == "sigmoid"
    assert biased.config().fused_activation.op_type == ttnn.UnaryOpType.SIGMOID
    over = decode_linear_for(
        dev, w, "G2", activation=ttnn.UnaryOpType.SIGMOID, bias=b, env={"GLM53_KDA_MM_G2": "12,6,2,9"}
    )
    c = over.config()
    assert (c.compute_with_storage_grid_size.x, c.compute_with_storage_grid_size.y, c.per_core_N, c.in0_block_w) == (
        12,
        6,
        2,
        9,
    )
    assert c.fused_activation.op_type == ttnn.UnaryOpType.SIGMOID
    qkv = decode_linear_for(dev, _FakeTensor(4096, 6144), "QKV", env={"GLM53_KDA_MM_QKV": "12,8,2,4"}).config()
    assert (qkv.per_core_N, qkv.in0_block_w) == (2, 4)


# ---------------------------------------------------------------------------------------------- gated variant
GATED_ACC = ACC_WORDS * 8 + [2, 128]  # the reader's 8 blocks + the gate page accessor (RM int32, 128 B pages)
GATED_WACC = ACC_WORDS * 2 + [2, 128]


def test_gated_variant_selection_and_twin():
    g = KdaDecode(num_heads=16, gated=True)
    assert g.variant == "gated" and g.gated
    assert g.kernel_sources() == (
        "reader_kda_decode_gated.cpp",
        "writer_kda_decode_gated.cpp",
        "compute_kda_decode_v2.cpp",
    )
    for base in (
        _op(),
        _op(gates="act", placement="rect"),
        KdaDecode(num_heads=16, math_fidelity=ttnn.MathFidelity.HiFi2),
    ):
        tw = base.gated_twin()
        assert tw.gated and tw.variant == "gated" and not base.gated
        assert (tw.H, tw.K, tw.lb, tw.l2_eps, tw.norm_eps, tw.fidelity, tw.gates, tw.placement_mode) == (
            base.H, base.K, base.lb, base.l2_eps, base.norm_eps, base.fidelity, base.gates, base.placement_mode,
        )  # fmt: skip
        assert tw.gates_activated == base.gates_activated
    # the default and v2 sources know nothing of the gate
    for name in (
        "reader_kda_decode.cpp",
        "writer_kda_decode.cpp",
        "compute_kda_decode.cpp",
        "reader_kda_decode_v2.cpp",
    ):
        s = _src(name)
        assert "gate_a" not in s and "GATE_BYTES" not in s and "gate_addr" not in s, name
    # the gated variant reuses the v2 compute kernel unchanged: no write-gate ABI in it (its "gates" are the KDA gates)
    s = _src("compute_kda_decode_v2.cpp")
    assert "gate_a" not in s and "GATE_BYTES" not in s and "gate_addr" not in s and "GATED" not in s


def test_gated_reader_abi():
    op = KdaDecode(num_heads=16, gated=True)
    src = _src(op.kernel_sources()[0])
    ct = _ct_indices(src)
    expect = dict(Kt=0, Vt=1, Bt=2, B=3, H=4, Wq_t=5, Wg_t=6, HK_t=7, HV_t=8, parts=9, ONES_PAGE=10, GATES_ACT=11)
    assert ct == expect, ct
    assert _accessor_start(src) == 12 == KdaDecode.READER_CT_SCALARS["gated"]
    # nine accessor blocks: the eight of v2 + the gate, each chained with next_compile_time_args_offset
    assert (
        src.count("TensorAccessorArgs<") == 9
        and "gate_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>" in src
    )
    reader, _, comp = op.compile_time_args(32, 6, GATED_ACC, GATED_WACC)
    v2 = _op(gates="pre", placement="rect").compile_time_args(32, 6, ACC_WORDS * 8, ACC_WORDS * 2)
    assert reader[:12] == v2[0][:12] and reader == v2[0] + [2, 128], "the v2 vector + the gate block"
    assert comp == v2[2] and len(comp) == KdaDecode.COMPUTE_CT_LEN["gated"] == 12
    assert KdaDecode(num_heads=16, gates="act", gated=True).compile_time_args(4, 5, GATED_ACC, GATED_WACC)[0][11] == 1
    rt = _rt_indices(src)
    names = {k[: -len("_addr")] if k.endswith("_addr") else k: v for k, v in rt.items()}
    assert names == KdaDecode.READER_RT_GATED and max(rt.values()) + 1 == KdaDecode.READER_RT_LEN["gated"] == 21
    assert (
        KdaDecode.READER_RT_GATED["gate"] == 20
        and {k: v for k, v in KdaDecode.READER_RT_GATED.items() if v < 20} == KdaDecode.READER_RT
    )


def test_gated_writer_abi():
    op = KdaDecode(num_heads=16, gated=True)
    src = _src(op.kernel_sources()[1])
    ct = _ct_indices(src)
    expect = dict(Kt=0, Vt=1, Bt=2, B=3, H=4, parts=5, HV_t=6)
    assert {k: ct[k] for k in expect} == expect and _accessor_start(src) == 7
    assert (
        src.count("TensorAccessorArgs<") == 3
        and "gate_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>" in src
    )
    _, writer, _ = op.compile_time_args(40, 3, GATED_ACC, GATED_WACC)
    assert writer == [4, 4, 2, 40, 16, 3, 64] + ACC_WORDS * 2 + [2, 128]
    rt = _rt_indices(src)
    names = {("out" if k == "o_addr" else k.replace("_addr", "")): v for k, v in rt.items()}
    assert names == KdaDecode.WRITER_RT_GATED and max(rt.values()) + 1 == KdaDecode.WRITER_RT_LEN["gated"] == 8
    assert (
        KdaDecode.WRITER_RT_GATED["gate"] == 7
        and {k: v for k, v in KdaDecode.WRITER_RT_GATED.items() if v < 7} == KdaDecode.WRITER_RT
    )
    for name in ("reader_kda_decode_gated.cpp", "writer_kda_decode_gated.cpp"):
        s = _src(name)
        for cb, idx in _cb_consts(s).items():
            assert KdaDecode.CB[cb] == idx, (name, cb, idx)
        assert (
            "noc_async_full_barrier();" in s.rstrip().splitlines()[-2]
        ), f"{name}: every DM kernel ends with the full barrier"


def test_gated_runtime_vectors_extend_v2_and_v1():
    addrs = dict(qkv=1, hist=2, taps=3, gates=4, cc=5, onw=6, sel=7, st=8, out=9, gate=77)
    for base_opts in (dict(), dict(gates="act", placement="rect")):
        base = _op(**base_opts)
        gated = base.gated_twin()
        mode = base.placement_mode
        parts, core_of, rect_h = base.placement(Grid(12, 10), 16, mode)
        for c in (0, 7, len(core_of) - 1):
            r, w, k = gated.runtime_args_for_core(c, 1, len(core_of), addrs, _noc, core_of, parts, rect_h)
            assert len(r) == 21 and len(w) == 8 and len(k) == 3
            assert r[20] == 77 and w[7] == 77
            rb, wb, kb = base.runtime_args_for_core(c, 1, len(core_of), addrs, _noc, core_of, parts, rect_h)
            assert w[:7] == wb and k == kb
            if base.variant == "v2":
                assert r[:20] == rb
            else:  # the v1 base: today's 15 entries, then the (zero) second column of the v2 layout
                assert r[:15] == rb and r[15:20] == [0, 0, 0, 0, 0]
    # a gated launch without the gate address cannot build its vectors
    raised = False
    try:
        _op().gated_twin().runtime_args_for_core(
            0, 1, 80, {k: v for k, v in addrs.items() if k != "gate"}, _noc, core_of, parts, rect_h
        )
    except KeyError:
        raised = True
    assert raised


def test_gated_cb_plan_scratch_pages_and_kernel_constants():
    base, gated = _op(), _op().gated_twin()
    pb, pg = base.cb_plan(5), gated.cb_plan(5)
    assert [n for n, _, _ in pg] == [n for n, _, _ in pb] == list(KdaDecode.CB)
    for (n, dt, pages), (n2, dt2, pages2) in zip(pb, pg):
        assert n == n2 and dt == dt2
        if n == "onw":
            assert pages2 == pages + KdaDecode.GATED_SCRATCH_PAGES and pages == 4
        else:
            assert pages2 == pages, n
    assert KdaDecode.GATED_SCRATCH_PAGES == 19  # 1 gate page + 9 Kt bf16 tiles (72 KB) as 18 fp32 pages
    assert gated.l1_bytes(5) - base.l1_bytes(5) == 19 * 4096 and gated.l1_bytes(6) <= 1_200_000
    r, w = _src("reader_kda_decode_gated.cpp"), _src("writer_kda_decode_gated.cpp")
    for s in (r, w):
        assert "constexpr uint32_t GATE_BYTES = 32 * 4;" in s
    assert "GATE_WRITER_OFFSET = 2048" in w and 2048 >= 128 and 2048 + 128 <= 4096, "the writer's copy inside page 0"
    assert "scratch = CircularBuffer(cb_onw).get_write_ptr() + Vt * t32" in r
    assert "CircularBuffer(cb_onw).get_write_ptr() + Vt * t32 + GATE_WRITER_OFFSET" in w
    assert "merge_l1 = scratch + t32" in r, "the merged history tiles start at the second scratch page"
    # the merged tiles: 3 branches x 3 destination taps x Kt tiles of t16 bytes fit the 18 fp32 pages after the gate page
    assert 3 * 3 * 4 * 2048 == 18 * 4096
    # the row copy: 2 x 8 words (two 16-value runs of a bf16 tile row, one per face of the row's face pair)
    assert "FACE_BYTES = 16 * 16 * 2" in r and "d[FACE_BYTES / 4 + i] = s[FACE_BYTES / 4 + i]" in r
    assert "noc_async_write_barrier();  // the copies landed before the row patches below" in r


def test_gated_golden_selects_per_user():
    B, H = 4, 16
    x = _inputs(B, H, seed=21)
    common = dict(num_heads=H, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5)
    args = (x["qkv"][0, 0], x["gates"][0, 0], x["conv"], x["S"], x["taps"], x["dt_bias"], x["a_exp"], x["onw"])
    o, S, conv = KdaDecode.golden(*args, **common)
    gate = torch.tensor([0, 1, 1, 0])
    og, Sg, convg = KdaDecode.golden_gated(*args, gate, **common)
    assert torch.equal(og, o)
    for u in range(B):
        if int(gate[u]):
            assert torch.equal(Sg[u], S[u]) and torch.equal(convg[u], conv[u])
        else:
            assert torch.equal(Sg[u], x["S"][u]) and torch.equal(convg[u], x["conv"][u])
    raised = False
    try:
        KdaDecode.golden_gated(*args, torch.ones(B + 1), **common)
    except AssertionError:
        raised = True
    assert raised


def test_gated_call_requires_gate_tensor():
    fake = _FakeTensor(4, 16, 128, 128)
    g = _op().gated_twin()
    raised = False
    try:
        g(fake, fake, fake, fake, fake, fake, fake, fake, fake)  # no gate: refused before any device access
    except AssertionError as e:
        raised = "write gate" in str(e)
    assert raised
    raised = False
    try:
        g(fake, fake, fake, fake, fake, fake, fake, fake, fake, gate=_FakeTensor(1, 16))  # wrong page width
    except AssertionError:
        raised = True
    assert raised
    raised = False
    try:
        _op()(fake, fake, fake, fake, fake, fake, fake, fake, fake, gate=_FakeTensor(1, 32))  # a gate needs gated=True
    except AssertionError as e:
        raised = "gated=True" in str(e)
    assert raised
