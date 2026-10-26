# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the fused T = K + 1 ``kda_decode`` variant (``KdaDecode(fused_steps=T)``, the three
``*_kda_decode_fused.cpp`` sources; SP2, ``_agents/spec_sp2/STATUS.md``), no device:

* the kernel ABI: every compile-time / runtime index of the fused reader / writer / compute against the vectors
  ``compile_time_args`` / ``runtime_args_for_core`` pack (``USERS, STEPS, DRY`` at 12..14 / 7..9, the accessor blocks
  from 15 / 10, the step page address at reader 20 / writer 7), the CB / semaphore indices against ``CB_FUSED``, the
  scratch-region constants, the full barrier at the end of both DM kernels, the untouched default / v2 / gated sources;
* the CB plan and the per-core L1 budget at parts = 5 and 6, B = 4 and 8, K = 1 .. 3;
* ``golden_fused``: dry mode returns the input state with the K + 1 sequential launches' outputs, commit mode ends every
  user at the plain state after its n_u launches (bits), untouched users keep their bytes, unapplied rows are zero;
* a torch model of the reader's tap-block assembly and history write-back (the rules the kernel implements) against
  the sequential golden's conv inputs / history.

    pytest --noconftest models/demos/glm53_flash/tests/custom_ops/test_kda_decode_fused_host.py -q
"""

from __future__ import annotations

import re

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.custom_ops.kda_decode.op import (  # noqa: E402
    GATE_WORDS,
    KERNEL_DIR,
    ONES_PAGE,
    KdaDecode,
    f32_bits,
)
from models.demos.glm53_flash.tests.custom_ops.test_kda_decode_host import (  # noqa: E402
    ACC_WORDS,
    Grid,
    _accessor_start,
    _cb_consts,
    _ct_indices,
    _inputs,
    _noc,
    _op,
    _rt_indices,
    _src,
)

K = V = 128
COMMON = dict(num_heads=16, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5)
FUSED_ACC = ACC_WORDS * 8 + [2, 128]  # the reader's 8 blocks + the step page accessor (RM int32, 128 B pages)
FUSED_WACC = ACC_WORDS * 2 + [2, 128]
SOURCES = ("reader_kda_decode_fused.cpp", "writer_kda_decode_fused.cpp", "compute_kda_decode_fused.cpp")


def _fused(T=4, dry=False, **opts):
    return KdaDecode(num_heads=16, fused_steps=T, fused_dry=dry, **opts)


# ---------------------------------------------------------------------------------------------- variant
def test_fused_variant_selection_and_twin():
    f = _fused(4, True)
    assert f.variant == "fused" and f.fused and f.fused_steps == 4 and f.fused_dry and not f.gated
    assert f.kernel_sources() == SOURCES
    assert _op().variant == "v1" and not _op().fused and _op().fused_steps == 0
    for base in (
        _op(),
        _op(gates="act", placement="rect"),
        KdaDecode(num_heads=16, math_fidelity=ttnn.MathFidelity.HiFi2),
    ):
        for dry in (True, False):
            tw = base.fused_twin(3, dry)
            assert tw.fused and tw.variant == "fused" and tw.fused_steps == 3 and tw.fused_dry == dry
            assert (tw.H, tw.K, tw.lb, tw.l2_eps, tw.norm_eps, tw.fidelity, tw.gates, tw.placement_mode) == (
                base.H, base.K, base.lb, base.l2_eps, base.norm_eps, base.fidelity, base.gates, base.placement_mode,
            )  # fmt: skip
            assert tw.gates_activated == base.gates_activated and not tw.gated
    raised = False
    try:
        KdaDecode(num_heads=16, gated=True, fused_steps=2)
    except AssertionError as e:
        raised = "exclusive" in str(e)
    assert raised
    raised = False
    try:
        KdaDecode(num_heads=16, fused_steps=33)
    except AssertionError:
        raised = True
    assert raised
    # the default, v2 and gated sources know nothing of the fused variant
    for name in (
        "reader_kda_decode.cpp",
        "writer_kda_decode.cpp",
        "compute_kda_decode.cpp",
        "reader_kda_decode_v2.cpp",
        "compute_kda_decode_v2.cpp",
        "reader_kda_decode_gated.cpp",
        "writer_kda_decode_gated.cpp",
    ):
        s = _src(name)
        assert "STEPS" not in s and "steps_a" not in s and "cb_loop" not in s and "fused_steps" not in s, name


# ---------------------------------------------------------------------------------------------- kernel ABI
def test_fused_reader_abi():
    op = _fused(4, False)
    src = _src(op.kernel_sources()[0])
    ct = _ct_indices(src)
    expect = dict(
        Kt=0, Vt=1, Bt=2, B=3, H=4, Wq_t=5, Wg_t=6, HK_t=7, HV_t=8, parts=9, ONES_PAGE=10, GATES_ACT=11, USERS=12,
        STEPS=13, DRY=14,
    )  # fmt: skip
    assert ct == expect, ct
    assert _accessor_start(src) == 15 == KdaDecode.READER_CT_SCALARS["fused"]
    assert (
        src.count("TensorAccessorArgs<") == 9
        and "steps_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>" in src
    )
    reader, writer, comp = op.compile_time_args(16, 5, FUSED_ACC, FUSED_WACC)
    v2 = _op(gates="pre", placement="rect").compile_time_args(16, 5, ACC_WORDS * 8, ACC_WORDS * 2)
    assert reader[:12] == v2[0][:12] and reader == v2[0][:12] + [4, 4, 0] + FUSED_ACC, "v2 scalars + USERS/STEPS/DRY"
    assert _fused(4, True).compile_time_args(16, 5, FUSED_ACC, FUSED_WACC)[0][12:15] == [4, 4, 1]
    assert _fused(2, False).compile_time_args(16, 5, FUSED_ACC, FUSED_WACC)[0][12:15] == [8, 2, 0]
    assert _fused(4, False, gates="act").compile_time_args(16, 5, FUSED_ACC, FUSED_WACC)[0][11] == 1
    rt = _rt_indices(src)
    names = {k[: -len("_addr")] if k.endswith("_addr") else k: v for k, v in rt.items()}
    assert names == KdaDecode.READER_RT_FUSED and max(rt.values()) + 1 == KdaDecode.READER_RT_LEN["fused"] == 21
    assert (
        KdaDecode.READER_RT_FUSED["steps"] == 20
        and {k: v for k, v in KdaDecode.READER_RT_FUSED.items() if v < 20} == KdaDecode.READER_RT
    )
    # the rows rule: B = USERS x STEPS <= 32
    for bad in (15, 33, 36):
        raised = False
        try:
            op.compile_time_args(bad, 5, FUSED_ACC, FUSED_WACC)
        except AssertionError:
            raised = True
        assert raised, bad


def test_fused_writer_abi():
    op = _fused(4, False)
    src = _src(op.kernel_sources()[1])
    ct = _ct_indices(src)
    expect = dict(Kt=0, Vt=1, Bt=2, B=3, H=4, parts=5, HV_t=6, USERS=7, STEPS=8, DRY=9)
    assert ct == expect and _accessor_start(src) == 10 == KdaDecode.WRITER_CT_SCALARS["fused"]
    assert (
        src.count("TensorAccessorArgs<") == 3
        and "steps_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>" in src
    )
    _, writer, _ = op.compile_time_args(32, 5, FUSED_ACC, FUSED_WACC)
    assert writer == [4, 4, 1, 32, 16, 5, 64, 8, 4, 0] + ACC_WORDS * 2 + [2, 128]
    assert _fused(4, True).compile_time_args(32, 5, FUSED_ACC, FUSED_WACC)[1][7:10] == [8, 4, 1]
    rt = _rt_indices(src)
    names = {("out" if k == "o_addr" else k.replace("_addr", "")): v for k, v in rt.items()}
    assert names == KdaDecode.WRITER_RT_FUSED and max(rt.values()) + 1 == KdaDecode.WRITER_RT_LEN["fused"] == 8
    assert KdaDecode.WRITER_CT_SCALARS == {"v1": 7, "v2": 7, "gated": 7, "fused": 10}


def test_fused_compute_abi():
    op = KdaDecode(fused_steps=4, **COMMON)
    src = _src(op.kernel_sources()[2])
    ct = _ct_indices(src)
    expect = dict(
        Kt=0, Vt=1, Bt=2, B=3, parts=4, lb_bits=5, l2eps_bits=6, normeps_bits=7, qscale_bits=8, invv_bits=9, IT_H=10,
        GATES_ACT=11, USERS=12, STEPS=13, DRY=14,
    )  # fmt: skip
    assert ct == expect, ct
    _, _, comp = op.compile_time_args(16, 5, [], [])
    want = [
        4,
        4,
        1,
        16,
        5,
        f32_bits(-5.0),
        f32_bits(1e-6),
        f32_bits(1e-5),
        f32_bits(K**-0.5),
        f32_bits(1 / V),
        8,
        0,
        4,
        4,
        0,
    ]
    assert comp == want and len(comp) == KdaDecode.COMPUTE_CT_LEN["fused"] == 15
    assert op.compile_time_args(32, 5, [], [])[2][10:15] == [8, 0, 8, 4, 0]  # R = 32: full-tile SFPU
    assert op.compile_time_args(8, 5, [], [])[2][10:15] == [4, 0, 2, 4, 0]  # R = 8: 4 iterations per face
    assert KdaDecode(fused_steps=2, fused_dry=True, gates="act", **COMMON).compile_time_args(8, 5, [], [])[2][11:15] == [
        1, 4, 2, 1,
    ]  # fmt: skip
    assert _rt_indices(src) == KdaDecode.COMPUTE_RT
    # the step page reaches the three compute threads through the unpacker's mailbox broadcast
    assert "read_tile_value(cb_selT, 0, uu)" in src and "static_assert(Bt == 1 && USERS * STEPS == B" in src


def test_fused_cb_semaphores_scratch_and_barriers():
    comp = _cb_consts(_src("compute_kda_decode_fused.cpp"))
    assert comp == KdaDecode.CB_FUSED, (comp, KdaDecode.CB_FUSED)
    assert KdaDecode.CB_FUSED["loop"] == 28 == KdaDecode.CB["oq"] and "oq" not in KdaDecode.CB_FUSED
    assert {k: v for k, v in KdaDecode.CB_FUSED.items() if k != "loop"} == {
        k: v for k, v in KdaDecode.CB.items() if k != "oq"
    }
    for name in SOURCES[:2]:
        s = _src(name)
        for cb, idx in _cb_consts(s).items():
            assert KdaDecode.CB_FUSED[cb] == idx, (name, cb, idx)
        for sem, idx in re.findall(r"constexpr uint32_t (SEM_\w+) = (\d+)", s):
            assert getattr(KdaDecode, sem) == int(idx), (name, sem)
        assert (
            "noc_async_full_barrier();" in s.rstrip().splitlines()[-2]
        ), f"{name}: the DM kernel ends with the barrier"
        assert "constexpr uint32_t STEP_BYTES = 32 * 4;" in s
    r, w = _src(SOURCES[0]), _src(SOURCES[1])
    assert "scratch = CircularBuffer(cb_onw).get_write_ptr() + Vt * t32" in r
    assert "orig_l1 = scratch + t32" in r, "the history tiles / merged tiles start at the second scratch page"
    assert (
        "STEP_WRITER_OFFSET = 2048" in w
        and "CircularBuffer(cb_onw).get_write_ptr() + Vt * t32 + STEP_WRITER_OFFSET" in w
    )
    assert GATE_WORDS * 4 == 128 <= 2048 and 2048 + 128 <= 4096, "both page copies inside the first scratch page"
    # the 9 Kt bf16 history / merged tiles fit the 18 fp32 pages after the page
    assert 3 * 3 * 4 * 2048 == (KdaDecode.FUSED_SCRATCH_PAGES - 1) * 4096 and KdaDecode.FUSED_SCRATCH_PAGES == 19
    # the row-run copy helper: contiguous face runs, two NoC writes per piece; the per-row word copy
    assert "copy_rows(" in r and "piece * 32u" in r and "FACE_BYTES = 16 * 16 * 2" in r
    assert "d[FACE_BYTES / 4 + i] = s[FACE_BYTES / 4 + i]" in r
    assert "noc_async_write_barrier();  // the tap blocks are complete before the push / multicast" in r
    assert "noc_async_write_barrier();  // the copies landed before the row patches below" in r
    # the compute's loop state: two kv slots at index 28, the oq matmul rides cb_vread
    c = _src(SOURCES[2])
    assert "mm_row<Kt, Vt>(cb_qd, 0, s_cb, cb_vread)" in c and "cb_loop = 28" in c and "cb_oq =" not in c


def test_fused_runtime_vectors_extend_v2_and_v1():
    addrs = dict(qkv=1, hist=2, taps=3, gates=4, cc=5, onw=6, sel=7, st=8, out=9, steps=77)
    for base_opts in (dict(), dict(gates="act", placement="rect")):
        base = _op(**base_opts)
        fused = base.fused_twin(4, False)
        parts, core_of, rect_h = base.placement(Grid(12, 10), 16, base.placement_mode)
        for c in (0, 7, len(core_of) - 1):
            r, w, k = fused.runtime_args_for_core(c, 1, len(core_of), addrs, _noc, core_of, parts, rect_h)
            assert len(r) == 21 and len(w) == 8 and len(k) == 3
            assert r[20] == 77 and w[7] == 77
            rb, wb, kb = base.runtime_args_for_core(c, 1, len(core_of), addrs, _noc, core_of, parts, rect_h)
            assert w[:7] == wb and k == kb
            if base.variant == "v2":
                assert r[:20] == rb
            else:
                assert r[:15] == rb and r[15:20] == [0, 0, 0, 0, 0]
            g = base.gated_twin().runtime_args_for_core(
                c, 1, len(core_of), dict(addrs, gate=77), _noc, core_of, parts, rect_h
            )
            assert g[0] == r and g[1] == w, "the fused vectors have the gated layout (the page address last)"
    raised = False
    try:
        _op().fused_twin(4, False).runtime_args_for_core(
            0, 1, 80, {k: v for k, v in addrs.items() if k != "steps"}, _noc, core_of, parts, rect_h
        )
    except KeyError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------------------------- CB plan / L1
def test_fused_cb_plan_and_l1_budget():
    base = _op()
    pb = base.cb_plan(5)
    for parts in (5, 6):
        for T in (2, 3, 4):
            f = _fused(T, False)
            pf = f.cb_plan(parts)
            assert [n for n, _, _ in pf] == list(KdaDecode.CB_FUSED) and len(
                {KdaDecode.CB_FUSED[n] for n, _, _ in pf}
            ) == 32
            for (n, dt, pages), (n2, dt2, pages2) in zip(base.cb_plan(parts), pf):
                assert dt == dt2
                if n == "onw":
                    assert n2 == n and pages2 == pages + KdaDecode.FUSED_SCRATCH_PAGES
                elif n == "selT":
                    assert n2 == n and pages2 == KdaDecode.FUSED_SELT_PAGES == 3 and pages == 2
                elif n == "oq":
                    assert n2 == "loop" and pages2 == 2 * 16 and pages == 4
                else:
                    assert n2 == n and pages2 == pages, n
    rows = []
    for parts in (5, 6):
        for B in (4, 8):
            for Kd in (1, 2, 3):
                T = Kd + 1
                R = B * T
                f = _fused(T, False)
                f.compile_time_args(R, parts, FUSED_ACC, FUSED_WACC)  # the rows rule holds at these points
                l1 = f.l1_bytes(parts)
                rows.append((parts, B, Kd, R, l1))
                assert l1 <= 1_200_000, (parts, B, Kd, l1)
    base5, gated5 = base.l1_bytes(5), base.gated_twin().l1_bytes(5)
    assert _fused(4).l1_bytes(5) - base5 == 19 * 4096 + 2048 + (2 * 16 - 4) * 4096  # scratch + selT page + loop CB
    logger.info(
        "fused kda_decode L1 per core: "
        + "; ".join(f"parts {p} B {b} K {k} R {r}: {l1 / 1024:.0f} KB" for p, b, k, r, l1 in rows)
    )
    logger.info(f"v1 {base5 / 1024:.0f} KB, gated {gated5 / 1024:.0f} KB at parts 5")
    assert _fused(4).l1_bytes(6) <= 1_200_000


# ---------------------------------------------------------------------------------------------- call site
class _FakeTensor:
    def __init__(self, *shape):
        self.shape = list(shape)


def test_fused_call_requires_step_page_and_rows():
    st = _FakeTensor(4, 16, 128, 128)
    qkv16 = _FakeTensor(1, 1, 16, 3 * 16 * 128)
    f = _op().fused_twin(4, False)
    raised = False
    try:
        f(qkv16, st, st, st, st, st, st, st, st)  # no step page: refused before any device access
    except AssertionError as e:
        raised = "step page" in str(e)
    assert raised
    raised = False
    try:
        f(_FakeTensor(1, 1, 12, 3 * 16 * 128), st, st, st, st, st, st, st, st, steps=_FakeTensor(1, 32))  # R != B T
    except AssertionError:
        raised = True
    assert raised
    raised = False
    try:
        f(qkv16, st, st, st, st, st, st, st, st, steps=_FakeTensor(1, 16))  # wrong page width
    except AssertionError:
        raised = True
    assert raised
    raised = False
    try:
        f(qkv16, st, st, st, st, st, st, st, st, steps=_FakeTensor(1, 32), gate=_FakeTensor(1, 32))
    except AssertionError as e:
        raised = "not a write gate" in str(e)
    assert raised
    raised = False
    try:
        _op()(qkv16, st, st, st, st, st, st, st, st, steps=_FakeTensor(1, 32))  # a step page needs fused_steps
    except AssertionError as e:
        raised = "fused_steps" in str(e)
    assert raised


# ---------------------------------------------------------------------------------------------- golden
def _slabs(B, H, T, seed):
    gen = torch.Generator().manual_seed(seed)
    W = 3 * H * K
    Hb = -(-H // 32) * 32
    qkv = torch.randn(T * B, W, generator=gen).to(torch.bfloat16).float()
    gates = torch.zeros(T * B, H * K + H * V + Hb)
    gates[:, : H * K] = torch.randn(T * B, H * K, generator=gen) * 2.0
    gates[:, H * K : H * K + H * V] = torch.randn(T * B, H * V, generator=gen) * 2.0
    gates[:, H * K + H * V : H * K + H * V + H] = torch.randn(T * B, H, generator=gen) * 2.0
    return qkv, gates


def _sequential(x, qkv, gates, B, T, H):
    """The plain golden on the per-candidate slabs, snapshots (o_j, S, conv) after every launch."""
    S, conv = x["S"], x["conv"]
    out = []
    for j in range(T):
        o, S, conv = KdaDecode.golden(
            qkv[j * B : (j + 1) * B], gates[j * B : (j + 1) * B], conv, S, x["taps"], x["dt_bias"], x["a_exp"], x["onw"],
            num_heads=H, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5,
        )  # fmt: skip
        out.append((o, S, conv))
    return out


@pytest.mark.parametrize("B, T", [(4, 4), (8, 2), (1, 3)])
def test_golden_fused_dry_and_commit(B, T):
    H = 16
    x = _inputs(B, H, seed=B * 7 + T)
    qkv, gates = _slabs(B, H, T, seed=B + T)
    seq = _sequential(x, qkv, gates, B, T, H)
    args = (qkv, gates, x["conv"], x["S"], x["taps"], x["dt_bias"], x["a_exp"], x["onw"])
    kw = dict(num_heads=H, gate_lower_bound=-5.0, l2_eps=1e-6, norm_eps=1e-5)
    # dry: the input state back, every row's output = the sequential launches' (candidate-major)
    o, S, conv = KdaDecode.golden_fused(*args, steps=None, dry=True, steps_per_user=T, **kw)
    assert torch.equal(S, x["S"]) and torch.equal(conv, x["conv"])
    assert tuple(o.shape) == (B * T, H * V)
    for j in range(T):
        assert torch.equal(o[j * B : (j + 1) * B], seq[j][0]), f"dry output of candidate {j}"
    # commit: n_u steps per user, the plain state after n_u launches bit for bit; n_u = 0 untouched
    n = torch.tensor([(u * 5 + 3) % (T + 1) for u in range(B)])
    o, S, conv = KdaDecode.golden_fused(*args, steps=n, dry=False, steps_per_user=T, **kw)
    for u in range(B):
        nu = int(n[u])
        if nu == 0:
            assert torch.equal(S[u], x["S"][u]) and torch.equal(conv[u], x["conv"][u]), f"user {u} untouched"
        else:
            assert torch.equal(S[u], seq[nu - 1][1][u]) and torch.equal(conv[u], seq[nu - 1][2][u]), f"user {u}"
        for j in range(T):
            row = o[j * B + u]
            if j < nu:
                assert torch.equal(row, seq[j][0][u]), f"user {u} step {j}: output row"
            else:
                assert torch.equal(row, torch.zeros_like(row)), f"user {u} step {j}: unapplied row is zero"
    # all T = the plain launches' end state; a page given in dry mode is ignored
    o_all, S_all, conv_all = KdaDecode.golden_fused(*args, steps=torch.full((B,), T), dry=False, steps_per_user=T, **kw)
    assert torch.equal(S_all, seq[-1][1]) and torch.equal(conv_all, seq[-1][2])
    o_d, S_d, _ = KdaDecode.golden_fused(*args, steps=n, dry=True, steps_per_user=T, **kw)
    assert torch.equal(S_d, x["S"]) and torch.equal(o_d, o_all)
    raised = False
    try:
        KdaDecode.golden_fused(*args, steps=torch.full((B,), T + 1), dry=False, steps_per_user=T, **kw)
    except AssertionError:
        raised = True
    assert raised


# ------------------------------------------------------------------------------- the reader's assembly rules
def tap_blocks(hist: torch.Tensor, raw: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """The reader's three conv tap blocks for R = B T candidate-major rows: ``[3, R, W]``, row ``(j, u)`` of tap ``t``
    = ``hist[j + t, u]`` for ``j + t < 3`` else ``raw[(j + t - 3) B + u]`` (``hist [3, B, W]`` tap-major, oldest first).
    """
    R = B * T
    out = torch.zeros(3, R, raw.shape[-1], dtype=raw.dtype)
    for t in range(3):
        for j in range(T):
            for u in range(B):
                q = j + t
                out[t, j * B + u] = hist[q, u] if q < 3 else raw[(q - 3) * B + u]
    return out


def new_history(hist: torch.Tensor, raw: torch.Tensor, n: torch.Tensor, B: int) -> torch.Tensor:
    """The commit's history write-back: row ``u`` of tap ``i`` = ``seq[i + n_u]`` with ``seq = [hist0, hist1, hist2,
    raw_0, raw_1, ...]`` of user ``u`` (``n_u = 0``: its own old rows)."""
    out = hist.clone()
    for u in range(B):
        for i in range(3):
            q = i + int(n[u])
            out[i, u] = hist[q, u] if q < 3 else raw[(q - 3) * B + u]
    return out


def copy_row_pieces(dst_r0: int, src_r0: int, n: int) -> list[tuple[int, int, int]]:
    """The kernel's ``copy_rows`` split: pieces ``(dst_r0, src_r0, rows)`` that never cross a 16-row tile half."""
    out = []
    while n > 0:
        piece = min(n, 16 - dst_r0 % 16, 16 - src_r0 % 16)
        out.append((dst_r0, src_r0, piece))
        dst_r0, src_r0, n = dst_r0 + piece, src_r0 + piece, n - piece
    return out


@pytest.mark.parametrize("B, T", [(4, 4), (8, 4), (2, 3), (1, 4), (16, 2), (3, 5)])
def test_tap_block_assembly_matches_sequential_conv_inputs(B, T):
    """The conv of the fused launch on the assembled tap blocks sees, for row (j, u), exactly the four rows the
    sequential launch j sees for user u (its history after launches < j, then its raw row); the write-back rule
    equals the sequential history after n_u launches."""
    H = 2
    W = 3 * H * K
    gen = torch.Generator().manual_seed(B * 10 + T)
    hist = torch.randn(3, B, W, generator=gen)
    raw = torch.randn(B * T, W, generator=gen)
    blocks = tap_blocks(hist, raw, B, T)
    # the sequential launches: launch j's conv rows = [hist (shifted j times) | raw_j]
    h = hist.clone()
    for j in range(T):
        for u in range(B):
            rows = torch.stack([blocks[0, j * B + u], blocks[1, j * B + u], blocks[2, j * B + u], raw[j * B + u]])
            want = torch.stack([h[0, u], h[1, u], h[2, u], raw[j * B + u]])
            assert torch.equal(rows, want), (j, u)
        h = torch.cat([h[1:], raw[j * B : (j + 1) * B][None]])  # the plain shift
        # the write-back after n_u = j + 1 steps for every user equals this history
        got = new_history(hist, raw, torch.full((B,), j + 1), B)
        assert torch.equal(got, h)
    assert torch.equal(new_history(hist, raw, torch.zeros(B, dtype=torch.long), B), hist)
    n = torch.tensor([(u * 3 + 1) % (T + 1) for u in range(B)])
    got = new_history(hist, raw, n, B)
    for u in range(B):
        if int(n[u]) == 0:
            assert torch.equal(got[:, u], hist[:, u])
    # the row-run pieces of the assembly cover the runs and never cross a half
    R = B * T
    for t in range(3):
        s = 3 - t
        runs = []
        if R > s * B:
            runs.append((s * B, 0, R - s * B))
        for jj in range(min(s, T)):
            runs.append((jj * B, 0, B))
        if R < 32:
            runs.append((R, R, 32 - R))
        covered = []
        for d0, s0, n_ in runs:
            for d, sr, p in copy_row_pieces(d0, s0, n_):
                assert d // 16 == (d + p - 1) // 16 and sr // 16 == (sr + p - 1) // 16 and p >= 1
                covered += list(range(d, d + p))
        assert sorted(covered) == list(range(32)), (t, sorted(covered))
