# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MKX_XROWS_COAL (GLM53_MKX_XROWS_COAL=1, opt-in): the routed experts' activation mover (kernels/mkx/xa.cpp) reads the
T decode rows of x as ONE span per face instead of one 32-B read per row per face. Host only (no ttnn, no device):

* the byte model of both read schedules on bf16 tile pages (mkx_common.hpp's tile_elem face layout): for every
  T = 1..16 the coalesced reads leave CB_X byte-identical to the per-row reads (zero-filled tiles, garbage in the
  source's padding rows and faces 2 / 3), and CB_X untilizes to x's T rows over zeros -> every downstream byte (gate_up,
  SwiGLU, the m gather, the down, the all-reduce) is the knob-off one: bitwise, no reduction order changes;
* the issue counts (2 T KT_GU -> 2 KT_GU, KT_GU at T = 16) and the B = 4 / B = 16 probe fit they remove;
* the kernel source: the #else branch is the 44adcb9ba40 per-row loop verbatim, the coalesced branch's offsets / sizes;
* the host knob (mk_experts.xcoal, exercised from its source without importing ttnn): off by default, only with
  MKX_XROWS, added to xa's defines alone (xw / xc keep their define lists, hence their JIT hashes).

    pytest --noconftest models/demos/glm53_flash/tests/test_mkx_xrows_coal_host.py -q
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest
import torch

TT = Path(__file__).resolve().parents[3] / "autoports" / "zai_org_glm_5_3_flash" / "tt"
XA = TT / "kernels" / "mkx" / "xa.cpp"
COMMON = TT / "kernels" / "mkx" / "mkx_common.hpp"
MKX = TT / "mk_experts.py"
KT_GU = 128  # hidden 4096 / 32: x's K tiles (xa's KT_GU)
PAGE = 2048  # bf16 tile page
FACE = 512  # bf16 face (16 x 16 x 2 B)


def tile_elem(row: int, col: int) -> int:
    """mkx_common.hpp tile_elem: element (row, col) of a 32x32 tile in face layout, in elements"""
    return (((row >> 4) << 1) + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15)


def tilize(x: torch.Tensor) -> torch.Tensor:
    """[32, 32 KT] bf16 -> [KT, 2048] uint8 tile pages (face layout)"""
    kt = x.shape[1] // 32
    idx = torch.tensor([tile_elem(r, c) for r in range(32) for c in range(32)])
    pages = torch.empty(kt, 1024, dtype=torch.bfloat16)
    for k in range(kt):
        pages[k, idx] = x[:, 32 * k : 32 * (k + 1)].reshape(-1)
    return pages.view(torch.uint8).reshape(kt, PAGE)


def untilize(pages: torch.Tensor) -> torch.Tensor:
    kt = pages.shape[0]
    idx = torch.tensor([tile_elem(r, c) for r in range(32) for c in range(32)])
    el = pages.reshape(kt, PAGE).contiguous().view(torch.bfloat16).reshape(kt, 1024)
    return torch.cat([el[k, idx].reshape(32, 32) for k in range(kt)], dim=1)


def per_row_reads(t: int):
    """the knob-off schedule (xa.cpp #else): per tile, per row r < T, face 0 row r and face 1 row r -> (src, dst, bytes)"""
    return [(off, off, 32) for r in range(t) for off in (r * 32, FACE + r * 32)]


def coalesced_reads(t: int):
    """MKX_XROWS_COAL: one span per face; T = 16 = the tile's first 1 KB (faces 0 + 1)"""
    if t == 16:
        return [(0, 0, 2 * t * 32)]
    return [(0, 0, t * 32), (FACE, FACE, t * 32)]


def run_schedule(src: torch.Tensor, reads, t: int) -> torch.Tensor:
    """xa's CB_X after the zero fill (xrows_prologue / the standalone fill) and the schedule's NoC reads"""
    dst = torch.zeros_like(src)
    for k in range(src.shape[0]):
        for s, d, n in reads:
            assert s % 16 == 0 and d % 16 == 0 and n % 16 == 0, "NoC L1 reads: 16-B aligned"
            dst[k, d : d + n] = src[k, s : s + n]
    return dst


@pytest.mark.parametrize("t", list(range(1, 17)))
def test_coalesced_cb_x_bitwise(t):
    g = torch.Generator().manual_seed(1000 + t)
    x = torch.randn(32, 32 * KT_GU, generator=g).to(torch.bfloat16)
    # the source tile rows >= T are not x's: stale / garbage bytes (NaN / Inf patterns included); they must not land
    x[t:] = (
        torch.randint(0, 1 << 16, (32 - t, 32 * KT_GU), generator=g, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    src = tilize(x)
    a = run_schedule(src, per_row_reads(t), t)
    b = run_schedule(src, coalesced_reads(t), t)
    assert torch.equal(a, b), f"T={t}: CB_X bytes differ"
    want = torch.zeros(32, 32 * KT_GU, dtype=torch.bfloat16)
    want[:t] = x[:t]
    got = untilize(b)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), f"T={t}: CB_X is not x's rows over zeros"


@pytest.mark.parametrize("t", list(range(1, 17)))
def test_span_union_is_the_row_set(t):
    """the coalesced spans cover exactly the per-row byte set: rows < T of faces 0 / 1 by tile_elem, nothing else"""
    rows = {2 * tile_elem(r, c) + b for r in range(t) for c in range(32) for b in (0, 1)}
    old = {s + i for s, _, n in per_row_reads(t) for i in range(n)}
    new = {s + i for s, _, n in coalesced_reads(t) for i in range(n)}
    assert old == new == rows


def test_issue_counts_and_probe_fit():
    """reads per reader: 2 T KT_GU (knob off) vs 2 KT_GU (KT_GU at T = 16). The probe pair val_pk84_b4 / b16 (chip 0,
    layer 4, reader0_a xa marks 1 -> 2: 22.7 / 79.1 us; the same at layers 5 / 6 and on reader 2) fits a + b x reads
    with b = 18.4 ns per 32-B read (~25 cycles at 1.35 GHz: issue-bound) and a = 3.9 us (y2 seed, list, xinv)"""
    old = {t: KT_GU * len(per_row_reads(t)) for t in (1, 4, 16)}
    new = {t: KT_GU * len(coalesced_reads(t)) for t in (1, 4, 16)}
    assert old == {1: 256, 4: 1024, 16: 4096} and new == {1: 256, 4: 256, 16: 128}
    b = (79.1 - 22.7) / (old[16] - old[4])
    a = 22.7 - b * old[4]
    assert 0.0175 < b < 0.0195 and 3.0 < a < 4.5, (a, b)
    # knob-off issue time removed per MoE layer (the bytes themselves: 128 KB per reader at T = 16, ~3 us at 32 B/clk)
    assert round(b * (old[16] - new[16])) == 73 and round(b * (old[4] - new[4])) == 14


def _branches(src: str):
    m = re.search(r"#if defined\(MKX_XROWS_COAL\)\n(.*?)#else\n(.*?)#endif\n", src, re.S)
    assert m, "xa.cpp: the MKX_XROWS_COAL #if / #else / #endif block"
    return m.group(1), m.group(2)


OLD_LOOP = """        for (uint32_t k = 0; k < KT_GU; ++k) {
            const uint32_t dst = xl1 + k * xpage;
            for (uint32_t r = 0; r < MKX_XROWS; ++r) {
                noc_async_read(x.get_noc_addr(k, r * 32), dst + r * 32, 32);
                noc_async_read(x.get_noc_addr(k, 512 + r * 32), dst + 512 + r * 32, 32);
            }
        }
"""


def test_kernel_source():
    src = XA.read_text()
    coal, off = _branches(src)
    assert off == OLD_LOOP, "the knob-off branch must stay the 44adcb9ba40 per-row loop verbatim"
    assert src.count("#if defined(MKX_XROWS_COAL)") == 1
    # the block sits inside MKX_XROWS, between the zero-fill barrier and the push's barrier
    i = src.index("#if defined(MKX_XROWS_COAL)")
    assert src.index("#if defined(MKX_XROWS)\n    // decode rows only") < i < src.index("#elif defined(MKX_XPROG)")
    assert "the zeros land before the row reads" in src[i - 200 : i]
    assert src[src.index("#endif\n", i) + 7 :].lstrip().startswith("noc_async_read_barrier();")
    assert "constexpr uint32_t XROW_BYTES = MKX_XROWS * 32;" in coal
    reads = re.findall(r"noc_async_read\(x\.get_noc_addr\(k, (\d+)\), (dst(?: \+ \d+)?), ([^)]+)\);", coal)
    assert reads == [("0", "dst", "2 * XROW_BYTES"), ("0", "dst", "XROW_BYTES"), ("512", "dst + 512", "XROW_BYTES")]
    assert "if constexpr (MKX_XROWS == 16)" in coal
    # the face layout the byte model uses is the kernels' one
    assert "return (((row >> 4) << 1) + (col >> 4)) * 256 + (row & 15u) * 16 + (col & 15u);" in COMMON.read_text()


def _xcoal():
    tree = ast.parse(MKX.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "xcoal")
    ns = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MKX), "exec"), ns)  # noqa: S102 (our own source)
    return ns["xcoal"]


def test_host_knob(monkeypatch):
    xcoal = _xcoal()
    monkeypatch.delenv("GLM53_MKX_XROWS_COAL", raising=False)
    assert not xcoal([("MKX_XROWS", "16")]), "default off"
    monkeypatch.setenv("GLM53_MKX_XROWS_COAL", "0")
    assert not xcoal([("MKX_XROWS", "16")])
    monkeypatch.setenv("GLM53_MKX_XROWS_COAL", "1")
    assert xcoal([("MKX_XROWS", "16")]) and xcoal([("MKX_XROWS", "1")])
    assert not xcoal([]) and not xcoal([("MKX_XPROG", "1")]), "no MKX_XROWS (T > 16 / GLM53_MKX_XROWS=0): no define"
    src = MKX.read_text()
    split = src.index("wdefs, adefs = list(defs), list(defs)")
    guard = src.index("if xcoal(xprog):")
    add = src.index('adefs.append(("MKX_XROWS_COAL", "1"))')
    assert split < guard < add < src.index("wtail, atail = [], []"), "xa's defines alone, after the w / a split"
    assert src.count('MKX_XROWS_COAL", "1"') == 1 and "cdefs, a_ct_y2, y2_rt = list(defs)" in src
    # no revision bump: a knob-off build keeps MKX_REV (every expert kernel's define list unchanged)
    assert re.search(r"^MKX_REV = 24 ", src, re.M)
