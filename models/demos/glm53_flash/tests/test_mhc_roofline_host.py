# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the mHC roofline work (_agents/mhc_roofline/STATUS.md; no device):

* the work-split planners (``delegate_units``, ``plan_positions``, ``MhcPreBlock.plan_work_extras``) on the 13x10 /
  12x10 / 8x8 / 10x10 / 11x10 grids: coverage (every output tile / stream tile exactly once), the critical path, the
  recipients (never a column leader or the coordinator in the pre-block), the prefill shapes left whole;
* the runtime-arg tails of the extras split against the kernel sources (``EXTRA_BASE`` indices, the define names), the
  CB plan with extras, the default program unchanged (no defines, the old arg vectors) when the knobs are off;
* the fused Sinkhorn pass sequence (``MHC_SINKHORN_FUSED``) as a torch model, pass for pass, against the LLK sequence
  (``parametrize_torch``): the same math, so bit-identical in float64, ``2 * iters + 3`` passes;
* A9: the ``fast_reduce_nc`` rounding model (every fp32 partial through the 19-bit SrcA before the fp32 accumulation)
  on the K-split projection: the relative error on the mixes and its effect on pre / post / comb;
* D13: the column-sharded residual math (per-chip partial projections + exact 4-way sum, local h + gather, the
  post-block on a column chunk) against the replicated math;
* the knob plumbing: module defaults, ``TtMHCFast`` refusing the unimplemented layout, the custom-op options.

    pytest --noconftest -p no:cacheprovider models/demos/glm53_flash/tests/test_mhc_roofline_host.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import KERNEL_DIR as POST_KERNEL_DIR  # noqa: E402
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import (
    POST_EXTRA_COST,
    POST_MAX_EXTRA,
    POST_UNIT_COST,
    MhcPostMix,
    delegate_units,
    extra_args,
    plan_positions,
    split_work,
)
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import BF16_TILE, FP32_TILE
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import KERNEL_DIR as PRE_KERNEL_DIR  # noqa: E402
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import MhcPreBlock
from models.demos.glm53_flash.tt import mhc as mhc_mod  # noqa: E402
from models.demos.glm53_flash.tt.mhc import (  # noqa: E402
    EXP_CAP,
    MHCKernelConfig,
    TtMHCFast,
    W,
    build_consts,
    const_tile_specs,
    parametrize_torch,
)

N, C = 4, 4096
CT = C // 32
GRIDS = [(12, 10), (13, 10), (8, 8), (10, 10), (11, 10), (14, 10)]
ITERS, EPS = 20, 1e-6


class _Grid:
    def __init__(self, x, y):
        self.x, self.y = x, y


# ------------------------------------------------------------------------------------------------- planners
def test_delegate_units_basics():
    # 128 units on 120 cores (cost 6 / 3): 8 donors, 32 extras, critical path 12 -> 9
    counts = [2] * 8 + [1] * 112
    crit, new_counts, extras = delegate_units(counts, 4, 6, 3, 4, [True] * 120)
    assert crit == 9 and new_counts == [1] * 120
    assert sum(len(e) for e in extras) == 32 and max(len(e) for e in extras) == 1
    assert all(len(extras[k]) == 0 for k in range(8))  # donors receive nothing
    assert all(len(extras[k]) == 1 for k in range(88, 120))  # prefer_last: the last cores first
    for k, ex in enumerate(extras):
        for owner, part in ex:
            assert owner < 8 and 0 <= part < 4
    # every (donor, part) exactly once
    seen = sorted((o, p) for ex in extras for o, p in ex)
    assert seen == [(o, p) for o in range(8) for p in range(4)]
    # all donors (8x8: every core two units) -> nothing to delegate, the plan is the input
    crit, nc, ex = delegate_units([2] * 64, 4, 6, 3, 4, [True] * 64)
    assert crit == 12 and nc == [2] * 64 and not any(ex)
    # max_extra 0 -> identity; empty -> identity
    assert delegate_units(counts, 4, 6, 3, 0, [True] * 120)[0] == 12
    assert delegate_units([], 4, 6, 3, 4, [])[0] == 0
    # a delegation that cannot lower the critical path is not made (every recipient would reach the old max)
    counts = [2] * 28 + [1] * 72  # 10x10, 128 positions: 112 extras over 72 cores -> >= 2 each on 40 cores = 12
    crit, nc, ex = delegate_units(counts, 4, 6, 3, 4, [True] * 100)
    assert crit == 12 and nc == counts and not any(ex)
    # eligibility honoured: 32 eligible cores at one extra each -> 9; 20 eligible cores would need two extras each
    # (= a full unit's cost): no gain, the plan stays whole
    elig = [k >= 88 for k in range(120)]
    crit, nc, ex = delegate_units([2] * 8 + [1] * 112, 4, 6, 3, 1, elig)
    assert crit == 9 and all(not ex[k] for k in range(88)) and sum(len(e) for e in ex) == 32
    crit, nc, ex = delegate_units([2] * 8 + [1] * 112, 4, 6, 3, 2, [k >= 100 for k in range(120)])
    assert crit == 12 and not any(ex)
    # prefer_last=False takes the first eligible cores
    _, _, ex = delegate_units([2] * 8 + [1] * 112, 4, 6, 3, 1, [True] * 120, prefer_last=False)
    assert all(len(ex[k]) == 1 for k in range(8, 40)) and all(len(ex[k]) == 0 for k in range(40, 120))


@pytest.mark.parametrize("gx,gy", GRIDS)
@pytest.mark.parametrize("total", [128, 1024, 4])
def test_plan_positions_covers_every_output_tile(gx, gy, total):
    grid = _Grid(gx, gy)
    plan, crit = plan_positions(total, grid, N, "extras")
    plan0, crit0 = plan_positions(total, grid, N, "tile")
    assert crit <= crit0
    produced = []
    for (core, start, count, extras), (core0, start0, count0, ex0) in zip(plan, plan0):
        assert core == core0 and start == start0 and not ex0 and count in (count0, count0 - 1)
        for p in range(start, start + count):
            produced += [(p, j) for j in range(N)]
        produced += [(p, j) for p, j in extras]
        assert len(extras) <= POST_MAX_EXTRA
        # an extra is the LAST position of its donor (the donor's count dropped by one)
        for p, j in extras:
            owner = [k for k, (c, s, cnt, _) in enumerate(plan0) if s <= p < s + cnt][0]
            assert plan[owner][2] == plan0[owner][2] - 1 and p == plan0[owner][1] + plan0[owner][2] - 1
    assert sorted(produced) == [(p, j) for p in range(total) for j in range(N)]
    if (gx, gy) == (12, 10) and total == 128:
        assert crit0 == 2 * POST_UNIT_COST and crit == POST_UNIT_COST + POST_EXTRA_COST
        assert sum(len(e) for *_, e in plan) == 32
    if total == 1024 or (gx, gy) in ((13, 10), (14, 10), (8, 8)):
        assert crit == crit0 and not any(e for *_, e in plan)  # prefill shapes / no-gain grids stay whole


def test_plan_positions_modes():
    grid = _Grid(12, 10)
    plan, _ = plan_positions(128, grid, N, "extras", allow_extras=False)  # collapse / h-mode: never split
    assert not any(e for *_, e in plan)
    plan, _ = plan_positions(128, grid, N, "tile")
    assert [(c.x, c.y, s, n) for c, s, n, _ in plan] == [(c.x, c.y, s, n) for c, s, n in split_work(128, grid)]
    rejected = False
    try:
        plan_positions(128, grid, N, "half")
    except AssertionError:
        rejected = True
    assert rejected
    assert extra_args([(5, 1), (7, 3)], 4) == [2, 5, 1, 7, 3, 0, 0, 0, 0]
    assert extra_args([], 4) == [0] + [0, 0] * 4


@pytest.mark.parametrize("gx,gy", GRIDS)
def test_pre_block_plan_work_extras(gx, gy):
    ncores = gx * gy
    crit, plan = MhcPreBlock.plan_work_extras(ncores, CT, MhcPreBlock.MAX_COLS, gy, MhcPreBlock.MAX_EXTRA, N)
    base = MhcPreBlock.plan_work(ncores, CT, MhcPreBlock.MAX_COLS)
    assert len(plan) == ncores
    projected = []
    for k, ((count, c0, count_proj, extras), (count0, c00)) in enumerate(zip(plan, base)):
        assert (count, c0) == (count0, c00)  # the h / norm phases: today's column plan
        assert count_proj in (count, count - 1) and (count_proj == count or count == 2)
        assert len(extras) <= MhcPreBlock.MAX_EXTRA
        for col in range(count_proj):
            projected += [(i, c0 + col) for i in range(N)]
        projected += list(extras)
        if extras:
            assert count == 1 and k % gy != 0 and k != 0, (k, extras)  # single-column, not a leader / coordinator
        if count_proj < count:  # a donor delegated exactly its second column
            assert count_proj == 1 and count == 2
    assert sorted(projected) == [(i, c) for i in range(N) for c in range(CT)]  # every stream tile exactly once
    two = sum(1 for c, _ in base if c == 2)
    if two and crit < two and crit:
        pass
    if (gx, gy) == (12, 10):
        assert crit == MhcPreBlock.COL_COST + MhcPreBlock.EXTRA_COST and sum(len(e) for *_, e in plan) == 32
        assert sum(1 for c, _, cp, _ in plan if cp < c) == 8
    if (gx, gy) in ((13, 10), (14, 10), (8, 8)):
        assert not any(e for *_, e in plan) and all(cp == c for c, _, cp, _ in plan)


# ------------------------------------------------------------------------------------------------- ABI / program
def _src(d: Path, name: str) -> str:
    return (d / name).read_text()


def test_post_mix_extras_abi_and_defines():
    """The extras tail sits right after the base runtime args of every kernel; the define name is the op's."""
    reader = _src(POST_KERNEL_DIR, "reader_mhc_post_mix_v2.cpp")
    writer = _src(POST_KERNEL_DIR, "writer_mhc_post_mix_v2.cpp")
    compute = _src(POST_KERNEL_DIR, "compute_mhc_post_mix_v2.cpp")
    assert re.search(r"constexpr uint32_t EXTRA_BASE = 6;", reader)  # [x, y, post, comb, start, count] + tail
    assert re.search(r"constexpr uint32_t EXTRA_BASE = 3;", writer)  # [out, start, count] + tail
    assert re.search(r"constexpr uint32_t EXTRA_BASE = 2;", compute)  # [start, count] + tail
    for src in (reader, writer, compute):
        assert "#ifndef MHC_MAX_EXTRA" in src and "#define MHC_MAX_EXTRA 0" in src
        assert re.search(r"get_arg_val<uint32_t>\(EXTRA_BASE\)", src)  # n_extra
        assert re.search(r"EXTRA_BASE \+ 1 \+ 2 \* e", src)  # position of extra e
    assert re.search(r"EXTRA_BASE \+ 2 \+ 2 \* e", writer) and re.search(r"EXTRA_BASE \+ 2 \+ 2 \* e", compute)
    assert "static_assert(MAX_EXTRA == 0 || (!COLLAPSE && !HMODE)" in compute
    # the extras come after the positions on every kernel (CB order)
    for src in (reader, writer, compute):
        assert src.index("for (uint32_t p = start_pos") < src.index("if constexpr (MAX_EXTRA > 0)")
    # every data-movement kernel ends with the full barrier
    for src in (reader, writer):
        assert src.rstrip().endswith("noc_async_full_barrier();\n}".rstrip())
    op_src = (POST_KERNEL_DIR.parent / "op.py").read_text()
    assert '("MHC_MAX_EXTRA", str(POST_MAX_EXTRA))' in op_src
    assert MhcPostMix(N, version=2, split="extras").split == "extras" and MhcPostMix(N).split == "tile"
    rejected = False
    try:
        MhcPostMix(N, version=2, split="half")
    except AssertionError:
        rejected = True
    assert rejected


def test_pre_block_extras_abi_and_defines():
    reader = _src(PRE_KERNEL_DIR, "reader_mhc_pre_block.cpp")
    compute = _src(PRE_KERNEL_DIR, "compute_mhc_pre_block.cpp")
    writer = _src(PRE_KERNEL_DIR, "writer_mhc_pre_block.cpp")
    assert re.search(r"constexpr uint32_t EXTRA_BASE = 10;", reader)  # the 10 base reader args
    assert re.search(r"count_proj = get_arg_val<uint32_t>\(3\);", compute)
    assert re.search(r"n_extra = get_arg_val<uint32_t>\(4\);", compute)
    for src in (reader, compute):
        assert "#ifndef MHC_MAX_EXTRA" in src and "#define MHC_MAX_EXTRA 0" in src
    assert "#ifndef MHC_SINKHORN_FUSED" in compute and "MHC_MAX_EXTRA" not in writer
    assert "noc_async_full_barrier();" in reader and "noc_async_full_barrier();" in writer
    # the runtime-arg tails
    noc_x, noc_y = [100 + x for x in range(12)], [200 + y for y in range(10)]
    addrs = dict(x_addr=1, fn_addr=2, consts_addr=3, w_addr=4, mask_addr=5, post_addr=6, comb_addr=7, hn_addr=8)
    r0, w0, c0 = MhcPreBlock.runtime_args_for_core(xi=3, yi=2, count=1, c0=7, noc_x=noc_x, noc_y=noc_y, **addrs)
    r, w, c = MhcPreBlock.runtime_args_for_core(
        xi=3, yi=2, count=1, c0=7, noc_x=noc_x, noc_y=noc_y, count_proj=1, extras=((2, 127),), max_extra=2, **addrs
    )
    assert len(r0) == 10 and len(c0) == 3 and w == w0  # the writer never changes
    assert r[:10] == r0 and r[10:] == [1, 1, 2, 127, 0, 0]
    assert c[:3] == c0 and c[3:] == [1, 1]
    r2, _, c2 = MhcPreBlock.runtime_args_for_core(
        xi=0, yi=0, count=2, c0=0, noc_x=noc_x, noc_y=noc_y, count_proj=1, extras=(), max_extra=2, **addrs
    )
    assert r2[10:] == [1, 0, 0, 0, 0, 0] and c2[3:] == [1, 0]  # a donor: projects one of its two columns
    # the CB plan with extras: X / FN / PARTS grow by the extras, the rest is unchanged
    plain = MhcPreBlock.cb_plan(N, 12, 10, 1)
    ext = MhcPreBlock.cb_plan(N, 12, 10, 1, max_extra=2)
    assert ext["X"][0] == plain["X"][0] + 2 and ext["FN"][0] == plain["FN"][0] + 2
    assert ext["PARTS"][0] == plain["PARTS"][0] + 1
    assert {k: v for k, v in ext.items() if k not in ("X", "FN", "PARTS")} == {
        k: v for k, v in plain.items() if k not in ("X", "FN", "PARTS")
    }
    total = sum(t * (FP32_TILE if d == ttnn.float32 else BF16_TILE) for t, d in ext.values())
    assert total <= 512 * 1024, total
    op_src = (PRE_KERNEL_DIR.parent / "op.py").read_text()
    assert '("MHC_SINKHORN_FUSED", "1")' in op_src and '("MHC_MAX_EXTRA", str(self.max_extra))' in op_src
    assert MhcPreBlock.SINKHORN_MODES == ("llk", "fused") and MhcPreBlock.SPLIT_MODES == ("cols", "extras")


def test_fused_sinkhorn_kernel_structure():
    """The fused pass sequence in the kernel source: the comb head, the row softmax (+eps out), the first column
    normalisation, iters-1 (row, column) pairs with the last one landing in CB_SCOMB, then the pre and post heads."""
    src = _src(PRE_KERNEL_DIR, "compute_mhc_pre_block.cpp")
    body = src[src.index("FORCE_INLINE void sinkhorn_fused()") :]
    body = body[: body.index("\n}\n") + 3]
    calls = re.findall(r"(coef_fused|normalize_fused)\(([^;]*)\);", body)
    assert [c[0] for c in calls] == [
        "coef_fused",
        "normalize_fused",
        "normalize_fused",
        "normalize_fused",
        "normalize_fused",
        "coef_fused",
        "coef_fused",
    ]
    assert "SEL_COMB, BASE_COMB, COEF_EXP, CB_MB" in calls[0][1]
    assert "CB_MB, RB, false, CB_MA, true" in calls[1][1]  # row softmax, no eps in the denominator, + eps out
    assert "CB_MA, CBC, true, ITERS == 1 ? CB_SCOMB : CB_MB, false" in calls[2][1]
    assert "CB_MB, RB, true, CB_MA, false" in calls[3][1]
    assert "CB_MA, CBC, true, (i + 1 == ITERS) ? CB_SCOMB : CB_MB, false" in calls[4][1]
    assert (
        "SEL_PRE, BASE_PRE, COEF_PRE, CB_SPRE" in calls[5][1]
        and "SEL_POST, BASE_POST, COEF_POST, CB_SPOST" in calls[6][1]
    )
    assert "for (uint32_t i = 1; i < ITERS; ++i)" in body
    # one dest pass per normalisation: matmul, [+eps], recip, m copied in, SFPU multiply, [+eps], one pack
    nf = src[src.index("FORCE_INLINE void normalize_fused") :]
    nf = nf[: nf.index("\n}\n")]
    order = [
        m.group(0)
        for m in re.finditer(
            r"matmul_tiles|add_unary_tile\(0|recip_tile\(0\)|copy_tile\(m_in, 0, 1\)|mul_binary_tile\(1, 0, 1\)|add_unary_tile\(1|pack_one\(1, m_out\)",
            nf,
        )
    ]
    assert order == [
        "matmul_tiles",
        "add_unary_tile(0",
        "recip_tile(0)",
        "copy_tile(m_in, 0, 1)",
        "mul_binary_tile(1, 0, 1)",
        "add_unary_tile(1",
        "pack_one(1, m_out)",
    ]
    assert nf.count("tile_regs_acquire()") == 1 and nf.count("tile_regs_release()") == 1
    # the dispatch and the untouched default
    assert "if constexpr (SINKHORN_FUSED) {\n        sinkhorn_fused();\n    } else {\n        sinkhorn();" in src


# ------------------------------------------------------------------------------------------------- fused Sinkhorn math
def sinkhorn_fused_passes(mixes: torch.Tensor, consts: torch.Tensor, iters: int, eps: float):
    """The fused kernel's passes (``sinkhorn_fused``) as torch ops over the whole token tile, one function call per
    dest pass: returns (pre, post, comb) tiles [T, 32] and the pass count."""
    k = {name: t.reshape(shape[-2], shape[-1]).to(mixes.dtype) for name, (t, shape) in const_tile_specs(consts).items()}
    passes = 0

    def coef(sel, bias, head):
        nonlocal passes
        passes += 1
        z = mixes @ sel + bias
        if head == "exp":
            return torch.exp(torch.clamp(z, max=EXP_CAP))
        s = torch.sigmoid(z)
        return s + eps if head == "pre" else s * 2.0

    def normalize(m, kt, use_eps, eps_out):
        nonlocal passes
        passes += 1
        s = m @ kt
        r = torch.reciprocal(s + eps) if use_eps else torch.reciprocal(s)
        out = m * r
        return out + eps if eps_out else out

    m = coef(k["sel_comb"], k["base_comb"], "exp")
    m = normalize(m, k["rb"], False, True)
    m = normalize(m, k["cb"], True, False)
    for _ in range(iters - 1):
        m = normalize(m, k["rb"], True, False)
        m = normalize(m, k["cb"], True, False)
    comb = m
    pre = coef(k["sel_pre"], k["base_pre"], "pre")
    post = coef(k["sel_post"], k["base_post"], "post")
    return pre, post, comb, passes


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("scale_val", [0.01, 1.0])
def test_fused_sinkhorn_equals_llk_sequence(dtype, scale_val):
    g = torch.Generator().manual_seed(1)
    mix_hc = (2 + N) * N
    mixes = torch.zeros(32, W)
    mixes[:, :mix_hc] = torch.randn(32, mix_hc, generator=g) * 3.0
    cfg = MHCKernelConfig(dim=64, n=N, sinkhorn_iters=ITERS, eps=EPS, norm_eps=1e-5)
    consts = build_consts(cfg, torch.full((3,), scale_val), torch.randn(mix_hc, generator=g)).to(dtype)
    pre_l, post_l, comb_l, n_llk = parametrize_torch(mixes.to(dtype), consts, ITERS, EPS)
    pre_f, post_f, comb_f, n_fused = sinkhorn_fused_passes(mixes.to(dtype), consts, ITERS, EPS)
    assert n_fused == 2 * ITERS + 3 and n_llk == 6 * (ITERS - 1) + 17
    # the same operations in the same order: bit-identical (the device paths differ only by where the operands are
    # rounded: 19-bit FPU operands in the LLK multiply, fp32 SFPU operands in the fused one)
    assert torch.equal(pre_l, pre_f) and torch.equal(post_l, post_f) and torch.equal(comb_l, comb_f)
    rows = comb_f[:, : N * N].reshape(32, N, N)
    # the last pass is a column normalisation: the column sums are 1 (up to eps); the row sums converge with iters
    assert torch.allclose(rows.sum(-2), torch.ones(32, N, dtype=dtype), atol=1e-4)
    assert torch.allclose(rows.sum(-1), torch.ones(32, N, dtype=dtype), atol=5e-2)


# ------------------------------------------------------------------------------------------------- A9 rounding
def tf32_round(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> the 19-bit SrcA format (1 sign, 8 exponent, 10 mantissa bits; round to nearest even), as fp32."""
    bits = t.float().contiguous().view(torch.int32)
    lsb = (bits >> 13) & 1
    rounded = (bits + 0x0FFF + lsb) & ~0x1FFF
    return rounded.view(torch.float32)


def test_tf32_round_model():
    x = torch.tensor(
        [1.0, 1.0 + 2**-10, 1.0 + 2**-11, 1.0 + 2**-11 + 2**-20, 1.0 + 3 * 2**-11], dtype=torch.float32
    )
    y = tf32_round(x)
    assert y.tolist() == [1.0, 1.0 + 2**-10, 1.0, 1.0 + 2**-10, 1.0 + 2**-9]  # ties to even, above the tie up
    r = torch.randn(10000)
    assert ((tf32_round(r) - r).abs() / r.abs()).max() <= 2**-11 + 1e-9


def test_a9_ksum_rounding_analysis():
    """``fast_reduce_nc`` over the 32 K-split partials: each fp32 partial rounds to 19 bits on unpack
    (``UnpackToSrc`` with ``add_tiles`` against a zero tile, fp32 dest accumulation), the sum itself is fp32. On the
    projection partials (random signs, comparable magnitude) the relative error of the sum is ~2^-11 (the ledger's
    "4e-4"); on the all-positive sum of squares it is bounded by 2^-11 and the rsqrt by 2^-12."""
    g = torch.Generator().manual_seed(3)
    T, KS = 32, 32
    x = torch.randn(T, N * C, generator=g) * 0.5
    mix_hc = (2 + N) * N
    fn = (torch.randn(mix_hc, N * C, generator=g) / (N * C) ** 0.5).to(torch.bfloat16).float()
    fn_t = torch.zeros(N * C, W)
    fn_t[:, :mix_hc] = fn.t() * (N * C) ** 0.5
    xk = x.reshape(T, KS, N * C // KS)
    fk = fn_t.reshape(KS, N * C // KS, W)
    part = torch.einsum("tkd,kdw->ktw", xk.double(), fk.double()).float()  # the 32 partial tiles (fp32 as the device)
    ssp = (xk.double() ** 2).sum(-1).t().float()  # [KS, T] partial sums of squares
    exact_mix = part.double().sum(0)
    exact_ss = ssp.double().sum(0)
    fast_mix = tf32_round(part).double().sum(0)
    fast_ss = tf32_round(ssp).double().sum(0)
    rel_mix = ((fast_mix - exact_mix)[:, :mix_hc].norm() / exact_mix[:, :mix_hc].norm()).item()
    rel_ss = ((fast_ss - exact_ss).abs() / exact_ss).max().item()
    assert 2**-14 < rel_mix < 2**-9, rel_mix  # ~2^-11 (4.9e-4): the measured 4e-4 of the ledger
    assert rel_ss <= 2**-11 + 1e-12, rel_ss
    # through the parametrization (scale 1.0: the worst case for the comb logits)
    norm_eps, eps = 1e-5, 1e-6
    rs_e = torch.rsqrt(exact_ss / (N * C) + norm_eps)
    rs_f = torch.rsqrt(fast_ss / (N * C) + norm_eps)
    mixes_e = exact_mix / (N * C) ** 0.5 * rs_e[:, None]
    mixes_f = fast_mix / (N * C) ** 0.5 * rs_f[:, None]
    cfg = MHCKernelConfig(dim=C, n=N, sinkhorn_iters=ITERS, eps=eps, norm_eps=norm_eps)
    base = torch.randn(mix_hc, generator=g)
    worst = {}
    for scale_val in (0.01, 1.0):
        consts = build_consts(cfg, torch.full((3,), scale_val), base).double()
        pre_e, post_e, comb_e, _ = parametrize_torch(mixes_e, consts, ITERS, eps)
        pre_f, post_f, comb_f, _ = parametrize_torch(mixes_f, consts, ITERS, eps)
        worst[scale_val] = (
            (pre_f - pre_e)[:, :N].abs().max().item(),
            (post_f - post_e)[:, :N].abs().max().item(),
            (comb_f - comb_e)[:, : N * N].abs().max().item(),
        )
    # bounded by the mixes' ~5e-4 relative error times the scale: at scale 1.0 the coefficients move by < 2e-3 absolute
    # (values in [0, 2]), at scale 0.01 by < 2e-5: far below one bf16 ulp of the sublayer input, within the chain-PCC
    # band the 45-layer bfp4 model shows run to run -- the logits gate is the arbiter, not this bound
    for scale_val, (dpre, dpost, dcomb) in worst.items():
        bound = 2e-3 if scale_val == 1.0 else 2e-5
        assert max(dpre, dpost, dcomb) < bound, (scale_val, dpre, dpost, dcomb)


# ------------------------------------------------------------------------------------------------- D13 math
def test_d13_colsharded_math_matches_replicated():
    """Column chip q holds columns [q*C/4, (q+1)*C/4) of every stream (packed-local [T, 4*C/4]): its projection partial
    + sum of squares over those columns, summed exactly over the 4 chips (one [T,32] tile each), equals the replicated
    projection; h = sum_i pre_i x_i is column-local (the gather of the 4 chunks IS h); the post-block on a chunk IS the
    chunk of the replicated post-block output (row-local elementwise math). float64 to isolate the formulation."""
    g = torch.Generator().manual_seed(7)
    T, cols = 32, 4
    Cq = C // cols
    x = torch.randn(T, N, C, generator=g).double()
    mix_hc = (2 + N) * N
    fn = torch.randn(mix_hc, N * C, generator=g).double()
    fn_s = fn.reshape(mix_hc, N, C)
    # replicated: the whole projection
    full_mix = x.reshape(T, N * C) @ fn.t()
    full_ss = (x**2).sum((1, 2))
    # column-sharded: per-chip partials over its chunk of every stream, exact 4-way sum in chip order
    parts, sss, h_chunks = [], [], []
    for q in range(cols):
        xq = x[:, :, q * Cq : (q + 1) * Cq]  # [T, N, Cq] = packed-local [T, N*Cq]
        fq = fn_s[:, :, q * Cq : (q + 1) * Cq]
        parts.append(xq.reshape(T, N * Cq) @ fq.reshape(mix_hc, N * Cq).t())
        sss.append((xq**2).sum((1, 2)))
    mix = parts[0] + parts[1] + parts[2] + parts[3]
    ss = sss[0] + sss[1] + sss[2] + sss[3]
    assert torch.allclose(mix, full_mix, rtol=1e-12, atol=1e-9) and torch.allclose(ss, full_ss, rtol=1e-12)
    # the mixes every chip computes from the same gathered partials are identical by construction (same order)
    pre = torch.sigmoid(mix[:, :N]) + 1e-6
    h_full = (pre[:, :, None] * x).sum(1)
    for q in range(cols):
        xq = x[:, :, q * Cq : (q + 1) * Cq]
        h_chunks.append((pre[:, :, None] * xq).sum(1))
    assert torch.equal(torch.cat(h_chunks, dim=-1), h_full)  # the all-gather of local h IS h (bit-identical)
    # the post-block on a chunk
    post = 2 * torch.sigmoid(mix[:, N : 2 * N])
    comb = torch.softmax(mix[:, 2 * N :].reshape(T, N, N), -1)
    y = torch.randn(T, C, generator=g).double()

    def post_block(xs, ys):  # the kernel's fixed per-row order: sum_i comb[i,j] x_i, then + post_j y
        out = torch.zeros(T, N, xs.shape[-1], dtype=torch.float64)
        for j in range(N):
            acc = comb[:, 0, j, None] * xs[:, 0]
            for i in range(1, N):
                acc = acc + comb[:, i, j, None] * xs[:, i]
            out[:, j] = acc + post[:, j, None] * ys
        return out

    out_full = post_block(x, y)
    for q in range(cols):
        xq = x[:, :, q * Cq : (q + 1) * Cq]
        yq = y[:, q * Cq : (q + 1) * Cq]
        assert torch.equal(post_block(xq, yq), out_full[:, :, q * Cq : (q + 1) * Cq])  # row-local: bit-identical
    # the elementwise work per chip is 1/4 of the replicated one; the Sinkhorn (one tile per chip) is not
    assert sum(p.numel() for p in parts) == cols * T * mix_hc  # 4 tiny partial tiles cross the columns


# ------------------------------------------------------------------------------------------------- knobs
def test_knob_defaults_are_todays_path():
    # the defaults since 3722af90ca0: the fused pre-Sinkhorn and the "extras" splits (+14 % decode, gates green)
    assert mhc_mod.MHC_KSUM == "exact" and mhc_mod.MHC_PRE_SINKHORN == "fused"
    assert (
        mhc_mod.MHC_PRE_SPLIT == "extras" and mhc_mod.MHC_POST_SPLIT == "extras" and mhc_mod.MHC_LAYOUT == "replicated"
    )
    assert mhc_mod.MHC_NORM == "sharded" and mhc_mod.MHC_PREFILL_NORM == "single"
    src = Path(mhc_mod.__file__).read_text()
    for knob in (
        "GLM53_MHC_KSUM",
        "GLM53_MHC_PRE_SINKHORN",
        "GLM53_MHC_PRE_SPLIT",
        "GLM53_MHC_POST_SPLIT",
        "GLM53_MHC_LAYOUT",
        "GLM53_MHC_PREFILL_NORM",
    ):
        assert f'os.environ.get("{knob}"' in src, knob


def test_colsharded_layout_is_refused_at_construction():
    """D13 is a design: the constructor refuses the layout before any device upload (the first statements)."""
    cfg = MHCKernelConfig(dim=C, n=N, sinkhorn_iters=ITERS, eps=EPS, norm_eps=1e-5)
    rejected = False
    try:
        TtMHCFast(None, cfg, torch.zeros(24, N * C), torch.zeros(24), torch.zeros(3), layout="colsharded")
    except NotImplementedError as e:
        rejected = "D13" in str(e)
    assert rejected


def test_reduce_batches_knob(monkeypatch):
    """``_reduce_batches``: exact -> ttnn.sum(dim=1); fast (A9) or MHC_DECODE_EXACT=0 -> fast_reduce_nc."""
    calls = []
    monkeypatch.setattr(mhc_mod.ttnn, "sum", lambda t, **k: calls.append(("sum", k.get("dim"))) or "sum")
    monkeypatch.setattr(
        mhc_mod.ttnn.experimental,
        "fast_reduce_nc",
        lambda t, **k: calls.append(("fast", tuple(k.get("dims")))) or "fast",
    )
    hc = object.__new__(TtMHCFast)
    hc.ckc = None
    monkeypatch.setattr(mhc_mod, "MHC_DECODE_EXACT", True)
    hc.ksum_mode = "exact"
    assert hc._reduce_batches("t") == "sum" and calls[-1] == ("sum", 1)
    hc.ksum_mode = "fast"
    assert hc._reduce_batches("t") == "fast" and calls[-1] == ("fast", (1,))
    monkeypatch.setattr(mhc_mod, "MHC_DECODE_EXACT", False)
    hc.ksum_mode = "exact"
    assert hc._reduce_batches("t") == "fast"
