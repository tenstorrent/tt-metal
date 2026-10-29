# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The dispatch ops' pass budget (the 2026-10-27 (2,4) block hang: an over-budget routing deadlocked
``ExpertDownDispatch`` because its writer clamped its own loop while the compute consumed the unclamped metadata):

* ``max_passes`` is a bound for a top-k routing and not for rows with more non-zeros (33 rows x 72 non-zeros, NaN rows
  through the selector matmul), the host mirror ``clamp_passes_host`` truncates exactly like the kernels, and every
  kernel-side derivation (the readers' expert set, the writers' pass / table loops, the metadata page, the LPT
  assignment) agrees on the clamped enumeration and stays within ``P``;
* the C++ ``clamp_passes`` of ``kernels/dispatch_common.hpp`` built with the host g++ (when present) matches the mirror;
* the kernel sources: every count-deriving kernel calls ``clamp_passes`` right after ``pass_bases`` (outside the
  REGIONS_SRC blocks, so both variants and the ``_units`` twins are covered), the down writer's partial clamp is gone
  and its budget is pinned to the formula, the gate|up writers set the table flag, the exit barriers stay;
* the pad-row mask (``GLM53_ROUTER_PAD_MASK``) and the eager guard (``GLM53_DISPATCH_GUARD``) helpers of ``tt/moe.py``:
  a NaN pad row gives zero counts with the mask, rows before ``seq_len`` are untouched bit for bit, the guard's row
  finder names the offending rows; the plumbing of the mask through the decoder / MTP / traced prefill.

``import ttnn`` only (no device); ``pytest --noconftest``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream.op import (  # noqa: E402
    KERNEL_DIR,
    MAX_PICKS,
    TABLE_FLAG_WORD,
    TABLE_WIDTH,
    TILE,
    clamp_passes_host,
    dispatch_contributions,
    dispatch_passes,
    expert_dispatch_golden,
    max_passes,
    table_clamped,
)
from models.demos.glm53_flash.tt import moe as M  # noqa: E402

TT = Path(M.__file__).parent
T, E_LOCAL, E_TOTAL = 128, 72, 288
P = max_passes(T, E_LOCAL)  # 104 on the (2,4) block / 8x4 column slices
COUNT_KERNELS = (  # the kernels that derive the pass enumeration from the counts themselves
    "reader_gate_up_dispatch",
    "writer_gate_up_dispatch",
    "reader_gate_up_dispatch_units",
    "writer_gate_up_dispatch_units",
    "reader_down_dispatch",
    "writer_down_dispatch",
)


@contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


def _src(name: str) -> str:
    return (KERNEL_DIR / f"{name}.cpp").read_text()


def _inside_regions_block(src: str, needle: str) -> bool:
    """Whether the line holding ``needle`` sits inside an ``#ifdef REGIONS_SRC`` ... ``#else`` / ``#endif`` arm (a
    clamp there would cover one variant only)."""
    depth = []  # True while inside the ifdef arm of a REGIONS_SRC block, False in its else arm
    for line in src.split("\n"):
        t = line.strip()
        if t.startswith("#ifdef REGIONS_SRC"):
            depth.append(True)
        elif t.startswith("#ifdef") or t.startswith("#ifndef") or t.startswith("#if "):
            depth.append(None)  # another conditional: neutral
        elif t.startswith("#else") and depth:
            depth[-1] = False if depth[-1] is True else depth[-1]
        elif t.startswith("#endif") and depth:
            depth.pop()
        elif needle in line:
            return any(d is not None for d in depth)
    raise AssertionError(f"{needle!r} not found")


def _topk_routing(T: int, E: int, picks: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.zeros(T, E)
    for t in range(T):
        ids = torch.randperm(E, generator=g)[:picks]
        r[t, ids] = torch.rand(picks, generator=g) + 0.2
    return r.bfloat16()


def _over_budget_routing(kind: str, seed: int = 7) -> torch.Tensor:
    """[T, E_LOCAL] bf16 as the hang saw it: a top-8 routing selected to one column (~2 picks per row land there) plus
    33 rows that are non-zero at every expert (``dense``: what a NaN row becomes after the selector matmul; ``nan``:
    the NaN rows themselves, ``NaN != 0`` on device)."""
    r = _topk_routing(T, E_LOCAL, 2, seed).float()
    bad = range(T - 33, T)
    r[list(bad), :] = 0.5 if kind == "dense" else float("nan")
    return r.bfloat16()


def _counts(routing: torch.Tensor) -> list[int]:
    return (routing != 0).sum(0).tolist()  # routing_counts: NaN != 0 counts


def _assign_group(passes: list[int], groups: int, group: int) -> list[int]:
    """Host model of dispatch_common.hpp ``assign_group``: LPT over the active experts (passes descending, id
    ascending) to the least loaded group (lowest id on ties); this group's experts in assignment order."""
    order = sorted((e for e, p in enumerate(passes) if p), key=lambda e: (-passes[e], e))
    load = [0] * groups
    mine = []
    for e in order:
        g = min(range(groups), key=lambda j: (load[j], j))
        load[g] += passes[e]
        if g == group:
            mine.append(e)
    return mine


# --------------------------------------------------------------------------------------------- the budget
def test_budget_bounds_topk_routings():
    """A routing with <= MAX_PICKS non-zeros per row never exceeds ``max_passes`` (random draws and the adversarial
    71 singletons + one fat expert), so the clamp never fires on a well-formed router output."""
    for seed in range(8):
        r = _topk_routing(T, E_LOCAL, MAX_PICKS, seed)
        _, _, total, clamped = clamp_passes_host(_counts(r), P)
        assert total <= P and not clamped, (seed, total)
    counts = [1] * (E_LOCAL - 1) + [MAX_PICKS * T - (E_LOCAL - 1)]  # 1024 slots: the worst split
    _, _, total, clamped = clamp_passes_host(counts, P)
    assert total == E_LOCAL - 1 + -(-counts[-1] // TILE) <= P and not clamped


@pytest.mark.parametrize("kind", ["dense", "nan"])
def test_clamp_mirror_and_kernel_derivations_agree(kind):
    """33 rows x 72 non-zeros on top of a top-k routing: the enumeration exceeds P = 104 (the hang's arithmetic), the
    mirror clamps it to exactly P, and every kernel-side derivation of the clamped arrays agrees and stays in bounds:
    the down reader's expert set (n != 0), the down writer's metadata page (passes of the active experts) and
    n_pass, the gate|up writers' table rows / act pages (base + p < P), the gate|up readers' LPT assignment."""
    r = _over_budget_routing(kind)
    counts = _counts(r)
    assert sum(counts) > MAX_PICKS * T  # the top-k contract is broken
    raw_total = sum(-(-c // TILE) for c in counts)
    assert raw_total > P, raw_total
    passes, base, total, clamped = clamp_passes_host(counts, P)
    assert clamped and total == P
    n_after = [min(c, p * TILE) if p * TILE < c else c for c, p in zip(counts, passes)]  # the kernels' n[e] after
    # the writer-side meta page and n_pass vs the reader-side expert set
    active_writer = [e for e in range(E_LOCAL) if passes[e]]
    active_reader = [e for e in range(E_LOCAL) if n_after[e]]
    assert active_writer == active_reader
    assert sum(passes[e] for e in active_writer) == total == P
    assert len(active_writer) + 1 <= 512 // 4  # the metadata page holds [n_active, passes...]
    # the gate|up writers' table rows / act pages: pass k = base[e] + p, all < P, a permutation of range(P)
    ks = [base[e] + p for e in range(E_LOCAL) for p in range(passes[e])]
    assert sorted(ks) == list(range(P))
    table = dispatch_passes(r, p_max=P)
    assert len(table) == P and [e for e, _ in table] == [e for e in range(E_LOCAL) for _ in range(passes[e])]
    for (e, rows), k in zip(table, ks):
        assert len(rows) <= TILE and all(rows[i] < rows[i + 1] for i in range(len(rows) - 1)), (e, k)
    # the truncated expert keeps its first passes (full rows), later experts none
    first_cut = next(e for e in range(E_LOCAL) if passes[e] * TILE < counts[e])
    assert all(passes[e] == 0 for e in range(first_cut + 1, E_LOCAL))
    assert all(len(rows) == TILE for e, rows in table if e == first_cut)
    # the gate|up readers' / writers' LPT assignment over the clamped passes covers the active set once
    groups = 12
    covered = [e for g in range(groups) for e in _assign_group(passes, groups, g)]
    assert sorted(covered) == active_writer
    # the unclamped enumeration (today's kernels) overflowed the resident h ring (P * cols pages) and the act pages
    assert raw_total * 2 > P * 2 and max(base[e] + passes[e] for e in range(E_LOCAL)) == P


def test_truncated_golden_matches_enumeration():
    """``expert_dispatch_golden(..., p_max)`` sums exactly the (expert, row) contributions the kernels gather
    (``dispatch_contributions``): the rows an expert lost through the budget do not reach its output, and a token
    keeps its first MAX_PICKS contributions in expert order (the down writer's gather slots, ``if (j < J)``): the
    33 dense rows of the hang's routing keep experts 0..7 of the 52 surviving ones. The chip-0 run of 2026-10-27
    returned pcc 0.57 on those rows against a golden without the slot cap and 0.99998 on the others."""
    r = _over_budget_routing("dense")
    K, I, H = 64, 32, 64
    x = torch.randn(T, K).bfloat16()
    w13 = torch.randn(E_LOCAL, K, 2 * I) * 0.1
    w2 = torch.randn(E_LOCAL, I, H) * 0.1
    act_pass, y, passes = expert_dispatch_golden(x, r, w13, w2, p_max=P)
    assert act_pass.shape[0] == len(passes) == P
    _, kept = dispatch_contributions(r, P)
    surviving = sorted({e for e, _ in passes})
    assert surviving == list(range(52))  # experts 0..51 keep their 2 passes, 52..71 are dropped
    for t in range(T - 33, T):  # the dense rows: 52 contributions on the table, 8 gathered (experts 0..7)
        assert kept[t].sum() == MAX_PICKS and kept[t, :MAX_PICKS].all()
    for t in range(T - 33):  # the top-2 rows: their experts below 52 only
        nz = torch.nonzero(r[t] != 0).flatten().tolist()
        assert kept[t].nonzero().flatten().tolist() == [e for e in nz if e < 52]
    r_kept = torch.where(kept, r, torch.zeros((), dtype=r.dtype))
    _, y_kept, _ = expert_dispatch_golden(x, r_kept, w13, w2)
    assert torch.equal(y, y_kept)
    assert not torch.equal(y, expert_dispatch_golden(x, r, w13, w2)[1])  # the budget changed the result
    # act_pass is not slot-capped: every enumerated pass carries all of its rows (the gate|up op's output)
    assert all(bool((act_pass[k, : len(rows)].abs().sum(-1) > 0).all()) for k, (_, rows) in enumerate(passes))


def test_gather_slot_cap_within_the_budget():
    """A routing within the pass budget but with more than MAX_PICKS contributions per token (12 experts on every
    row): no clamp, yet the down op gathers only the first 8 (expert ascending) -- the mirror keeps exactly those;
    a well-formed top-k routing is untouched (its mask is its non-zero pattern)."""
    r = torch.zeros(64, E_LOCAL)
    picks = [3, 5, 8, 13, 21, 34, 40, 41, 42, 55, 60, 71]
    r[:, picks] = 0.25
    r = r.bfloat16()
    _, _, total, clamped = clamp_passes_host(_counts(r), max_passes(64, E_LOCAL))
    assert total == 2 * len(picks) and not clamped
    passes, kept = dispatch_contributions(r)
    assert len(passes) == 2 * len(picks)
    assert (
        kept.sum(-1).tolist() == [MAX_PICKS] * 64
        and kept[:, picks[:MAX_PICKS]].all()
        and not kept[:, picks[MAX_PICKS:]].any()
    )
    ok = _topk_routing(T, E_LOCAL, MAX_PICKS, 5)
    _, kept_ok = dispatch_contributions(ok)
    assert torch.equal(kept_ok, ok != 0)
    assert (
        "if (j < J) {" in _src("writer_down_dispatch")
        and "self.J = MAX_PICKS" in (KERNEL_DIR.parent / "op.py").read_text()
    )


# ------------------------------------------------------------------------------------------- the C++ clamp
HARNESS = r"""
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#define tt_l1_ptr
#include "dispatch_common.hpp"
int main(int argc, char** argv) {
    constexpr uint32_t E = HE;
    const uint32_t p_max = static_cast<uint32_t>(atoi(argv[1]));
    uint16_t n[E], passes[E], base[E];
    for (uint32_t e = 0; e < E; ++e) {
        unsigned v = 0;
        if (scanf("%u", &v) != 1) return 2;
        n[e] = static_cast<uint16_t>(v);
    }
    pass_bases<E>(n, passes, base);
    bool clamped = false;
    const uint32_t total = clamp_passes<E>(n, passes, base, p_max, clamped);
    printf("%u %u\n", total, clamped ? 1u : 0u);
    for (uint32_t e = 0; e < E; ++e) printf("%u %u %u\n", passes[e], base[e], n[e]);
    return 0;
}
"""


def _build_clamp_harness(tmp: Path, E: int) -> Path | None:
    gxx = shutil.which("g++") or shutil.which("c++")
    if gxx is None:
        return None
    src = tmp / "clamp_harness.cpp"
    src.write_text(HARNESS)
    exe = tmp / f"clamp_{E}"
    subprocess.run(
        [gxx, "-std=c++17", "-O1", f"-I{KERNEL_DIR}", f"-DHE={E}", str(src), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    return exe


def test_cpp_clamp_matches_mirror(tmp_path):
    """``kernels/dispatch_common.hpp`` ``pass_bases`` + ``clamp_passes`` compiled natively against the host mirror:
    random counts, the hang's counts, a budget crossed mid-expert and a zero budget."""
    exe = _build_clamp_harness(tmp_path, E_LOCAL)
    if exe is None:
        pytest.skip("no host C++ compiler")
    g = torch.Generator().manual_seed(1)
    cases = [_counts(_over_budget_routing("dense")), _counts(_topk_routing(T, E_LOCAL, MAX_PICKS, 3))]
    cases += [torch.randint(0, 70, (E_LOCAL,), generator=g).tolist() for _ in range(6)]
    cases += [[33] * E_LOCAL, [0] * E_LOCAL, [T] * E_LOCAL]
    for counts in cases:
        for p_max in (P, 7, 0, 10_000):
            out = subprocess.run(
                [str(exe), str(p_max)], input=" ".join(map(str, counts)), capture_output=True, check=True, text=True
            ).stdout.split("\n")
            total, clamped = (int(v) for v in out[0].split())
            rows = [tuple(int(v) for v in line.split()) for line in out[1 : 1 + E_LOCAL]]
            passes, base, total_py, clamped_py = clamp_passes_host(counts, p_max)
            assert (total, bool(clamped)) == (total_py, clamped_py), (counts, p_max)
            assert [r[0] for r in rows] == passes and [r[1] for r in rows] == base, (counts, p_max)
            for e, (p, _, n) in enumerate(rows):  # n[e] cut to the kept rows, 0 when the expert lost every pass
                assert n == (min(counts[e], p * TILE) if p * TILE < counts[e] else counts[e]), (e, counts, p_max)
                assert (n != 0) == (p != 0)


# -------------------------------------------------------------------------------------- kernel sources
def test_kernel_sources_apply_one_budget():
    """Every count-deriving kernel clamps right after ``pass_bases`` (once, outside any REGIONS_SRC block: the
    regions-sourced twins share the line), the down writer pins its compile-time budget to the formula and no longer
    clamps ``n_pass`` alone, the gate|up writers flag the clamp in the table, the computes take the clamped counts,
    and the header's constants are op.py's."""
    hdr = (KERNEL_DIR / "dispatch_common.hpp").read_text()
    assert f"constexpr uint32_t DISPATCH_MAX_PICKS = {MAX_PICKS};" in hdr
    assert f"constexpr uint32_t TABLE_FLAG_WORD = {TABLE_FLAG_WORD};" in hdr and TABLE_FLAG_WORD < TABLE_WIDTH
    assert "return DISPATCH_MAX_PICKS * T / 32 + E;" in hdr  # == op.max_passes
    assert re.search(r"inline uint32_t clamp_passes\(uint16_t\* n, uint16_t\* passes, uint16_t\* base", hdr)
    for name in COUNT_KERNELS:
        src = _src(name)
        assert src.count("clamp_passes<E>(") == 1, name
        assert src.index("pass_bases<E>(n_tok, passes, base);") < src.index("clamp_passes<E>("), name
        assert not _inside_regions_block(src, "clamp_passes<E>("), name  # both variants share the line
        body = src[src.index("void kernel_main()") :]
        assert body.rstrip().endswith(
            "noc_async_full_barrier();  // exit contract: every NOC read / write / atomic acknowledged before the next kernel\n}".rstrip()
        ), name
    for name in (
        "reader_gate_up_dispatch",
        "writer_gate_up_dispatch",
        "reader_gate_up_dispatch_units",
        "writer_gate_up_dispatch_units",
        "reader_down_dispatch",
    ):
        assert "clamp_passes<E>(n_tok, passes, base, dispatch_max_passes(T, E), clamped);" in _src(name), name
    dw = _src("writer_down_dispatch")
    assert "static_assert(P_MAX==dispatch_max_passes(T,E)," in "".join(dw.split())  # clang-format wraps it
    assert "const uint32_t n_pass = clamp_passes<E>(n_tok, passes, base, P_MAX, clamped);" in dw
    assert "n_pass > P_MAX" not in dw and "meta[1 + n_active++] = passes[e];" in dw
    for name in ("writer_gate_up_dispatch", "writer_gate_up_dispatch_units"):
        assert "tab[TABLE_FLAG_WORD] = clamped ? 1 : 0;" in _src(name), name
    assert "clamped to the\n    // pass budget" in _src("compute_down_dispatch") or "clamp_passes" in _src(
        "compute_down_dispatch"
    )
    # the ops' P is the kernels' budget
    import models.demos.glm53_flash.custom_ops.expert_stream.op as O

    for t in (64, 128, 256):
        for e in (9, 36, 72):
            assert max_passes(t, e) == MAX_PICKS * t // 32 + e
    assert "self.P = max_passes(T, E)" in Path(O.__file__).read_text()


# ---------------------------------------------------------------------------------- the pad-row mask
def test_pad_mask_knobs(monkeypatch):
    # the pad mask is on by default; the eager dispatch guard (a per-layer host readback: 128-token eager TTFT 1548 ->
    # 879 ms without it) is off by default and opted in with GLM53_DISPATCH_GUARD=1
    for env, fn, default in (
        (M.ROUTER_PAD_MASK_ENV, M.pad_mask_enabled, True),
        (M.DISPATCH_GUARD_ENV, M.dispatch_guard_enabled, False),
    ):
        assert fn({}) is default and fn({env: "1"}) is True and fn({env: "0"}) is False
        monkeypatch.setenv(env, "0" if default else "1")
        assert fn() is (not default)
        monkeypatch.delenv(env)
        assert fn() is default


def test_pad_row_mask_host_shape_and_values():
    m = M.pad_row_mask_host(128, 48)
    assert tuple(m.shape) == (1, 1, 128, TILE) and m.dtype == torch.bfloat16
    assert torch.equal(m[0, 0, :, 0].float(), (torch.arange(128) < 48).float())
    assert m[0, 0, :, 1:].abs().sum() == 0
    assert M.pad_row_mask_host(64, 64)[0, 0, :, 0].sum() == 64 and M.pad_row_mask_host(64, 0).sum() == 0
    with expect_error(ValueError, "outside"):
        M.pad_row_mask_host(64, 65)
    # the expansion the device does: mask[:, 0:1] x ones[0:1, :] (one product per element, exact in bf16)
    ones = M.pad_mask_ones_host(E_TOTAL)
    assert tuple(ones.shape) == (1, 1, TILE, E_TOTAL) and ones[0, 0, 0].sum() == E_TOTAL and ones[0, 0, 1:].sum() == 0
    full = m[0, 0].float() @ ones[0, 0].float()
    assert torch.equal(full, (torch.arange(128) < 48).float()[:, None].expand(128, E_TOTAL))
    # the row-slab split a 2-row mesh does (rows=2): row r's slab, 48 valid rows all on row 0 at T = 128
    assert m[0, 0, :64, 0].sum() == 48 and m[0, 0, 64:, 0].sum() == 0


def test_nan_pad_row_gives_zero_counts_with_the_mask():
    """The hang's mechanism on the host: a NaN pad row of the [T, 288] dense matrix becomes 72 non-zeros of the
    [T, 72] selected matrix (NaN * 0 = NaN, NaN != 0) and the counts cross the budget; with the mask the pad rows
    are zero, the counts are the top-k's and rows before seq_len are untouched bit for bit."""
    seq_len = 98
    dense = _topk_routing(T, E_TOTAL, MAX_PICKS, 11).float()
    dense[seq_len:, :] = 0
    dense[seq_len : seq_len + 30, 5] = float("nan")  # 30 unspecified rows holding a NaN somewhere
    dense = dense.bfloat16()
    sel = torch.zeros(E_TOTAL, E_LOCAL)
    sel[torch.arange(E_LOCAL), torch.arange(E_LOCAL)] = 1.0  # column 0's experts
    selected = (dense.float() @ sel).bfloat16()
    assert torch.isnan(selected[seq_len : seq_len + 30]).all()
    counts = _counts(selected)
    assert sum(counts) > MAX_PICKS * T and clamp_passes_host(counts, P)[3]
    bad = M.routing_rows_over_budget(selected)
    assert [t for t, _, _ in bad] == list(range(seq_len, seq_len + 30))
    assert all(nz == E_LOCAL and not finite for _, nz, finite in bad)
    masked = M.mask_dense_host(dense, M.pad_row_mask_host(T, seq_len))
    assert masked[seq_len:].abs().sum() == 0 and not torch.isnan(masked).any()
    assert torch.equal(masked[:seq_len].view(torch.int16), dense[:seq_len].view(torch.int16))  # bit-identical
    counts_masked = _counts((masked.float() @ sel).bfloat16())
    assert sum(counts_masked) <= MAX_PICKS * seq_len and not clamp_passes_host(counts_masked, P)[3]
    assert M.routing_rows_over_budget((masked.float() @ sel).bfloat16()) == []
    assert M.routing_rows_over_budget(_over_budget_routing("dense")) == [(t, E_LOCAL, True) for t in range(T - 33, T)]


def test_table_clamped_reader():
    tab = torch.zeros(1, 1, P, TABLE_WIDTH, dtype=torch.int32)
    assert not table_clamped(tab)
    tab[0, 0, 0, TABLE_FLAG_WORD] = 1
    assert table_clamped(tab)
    assert not table_clamped(torch.ones(P, TABLE_WIDTH) * 2)  # only 1 flags


def test_mask_and_guard_plumbing():
    """Source pins: the mask reaches the MoE from the decoder's eager chunks (rows past seq_len), the traced prefill
    (a per-length persistent input refreshed per prompt, behind the knob at capture) and the MTP prefill; the guard
    runs in ``_block_dispatch`` only outside a capture; the ones constant is allocated at construction (before any
    capture)."""
    moe = (TT / "moe.py").read_text()
    assert "def _mask_pad_rows(self, dense: ttnn.Tensor, row_mask: ttnn.Tensor)" in moe
    assert (
        "out = ttnn.where(full, dense, 0.0)" in moe and moe.count("dense = self._mask_pad_rows(dense, row_mask)") == 2
    )
    assert "if dispatch_guard_enabled() and not _capture_active(self.device):" in moe
    assert 'fn = getattr(ttnn, "is_trace_capture_active", None)' in moe
    squeezed = "".join(moe.split())  # black wraps the long call
    assert (
        'self.pad_ones=self.mesh.const(("pad_mask_ones",E),lambda:self.mesh.upload(pad_mask_ones_host(E),ttnn.bfloat16))'
        in squeezed
    )
    dec = (TT / "decoder_layer.py").read_text()
    assert "elif mask is None and not wave_mode and seq_len < T and pad_mask_enabled():" in dec
    assert "mask, own = self.mesh.upload(pad_row_mask_host(T, seq_len), ttnn.bfloat16, rows=2), True" in dec
    assert "ffn = lambda t, m=mask: self.mlp(t, prefill=True, row_mask=m)" in dec
    assert 'self.mlp.experts.call_site = f"layer {layer_idx} MoE"' in dec
    mtp = (TT / "mtp.py").read_text()
    assert "self._residual_block(x, attention, prefill=True, row_mask=mask)" in mtp
    assert "self.mlp(h, prefill=True, row_mask=row_mask) if prefill else self.mlp(h)" in mtp
    fd = (TT / "functional_decoder.py").read_text()
    assert "row_mask: ttnn.Tensor | None = None" in fd and fd.count("row_mask=trace.row_mask,") == 2
    tp = (TT / "trace_prefill.py").read_text()
    assert "row_mask=_upload(mesh, pad_row_mask_host(T, T), ttnn.bfloat16, ttnn.TILE_LAYOUT, rows_dim=2)," in tp
    assert "_refresh(mesh, pad_row_mask_host(self.T, S), self.row_mask, rows_dim=2)" in tp
    mm = (TT / "mesh_model.py").read_text()
    assert "row_mask=inp.row_mask if pad_mask_enabled() else None," in mm
