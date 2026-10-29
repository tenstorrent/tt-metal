# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of decode roofline action E1 (2026-10-26, _agents/expert_e1/STATUS.md): no device is opened.

* The block kernels' ABI (compile-time arg indices, accessor offsets, the ROUTING_FOLD / FUSED runtime tails) against
  ``ExpertGateUpStream.descriptors`` / ``ExpertDownStream.descriptors`` / ``ExpertStreamFused.forward``; the default
  programs carry no defines and the same vectors as before (the literal orders are pinned).
* The folded selector: ``routing_scan.hpp`` compiled with the host g++ (``scan_active`` / ``scan_active_fold`` /
  ``scan_active_words`` + ``fold_elem``) on random routing tiles vs the torch selector-matmul path -- active sets and
  the diagonal weights bit-identical for every column base; the fp32 0/1 matmul is exact (the fold's premise).
* The fused op's arrival-counter protocol as a host model (random per-core progress interleavings: the down writer
  never loads an expert before all of its bank tiles landed; the 4-bit slots never alias or overflow; <= 16 L1
  semaphores for every feasible group count); the fused program reuses the two compute kernels unchanged.
* The planners on the 13x10 and 8x8 grids (``fused_core_plan``, ``plan_for_32row``), the knob plumbing
  (``e1_stream_knobs``, ``experts_32row_mode``, ``expert_layout_from_env``, ``_fold_serves``, ``_stream_x_into_l1``),
  the per-phase launch model (``launch_phase_table``: the 28-active point reproduces the measured 351 us).
* The moe.py contract: the hybrid glue knobs, ``_stream_serves`` / ``prefill_experts_for`` / ``stream_experts_default``
  rules and the coordinator's block-source release are intact.

    pytest --noconftest models/demos/glm53_flash/tests/test_expert_e1_host.py -q
"""

from __future__ import annotations

import inspect
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream import op as O  # noqa: E402
from models.demos.glm53_flash.custom_ops.expert_stream import plan as P  # noqa: E402
from models.demos.glm53_flash.custom_ops.expert_stream.op import KERNEL_DIR  # noqa: E402
from models.demos.glm53_flash.tt import moe as M  # noqa: E402

E_TOTAL, E_LOCAL, H, I_LOC, K_PICKS = 288, 72, 4096, 256, 8
GRIDS = [(13, 10), (8, 8)]
BLOCK_KERNELS = ("reader_gate_up", "writer_gate_up", "reader_down", "writer_down")


# ---------------------------------------------------------------------------------------------- helpers
def _src(name: str) -> str:
    return (KERNEL_DIR / f"{name}.cpp").read_text()


def _ct_indices(src: str) -> dict[str, str]:
    pat = re.compile(r"constexpr\s+(?:uint32_t|bool|auto)\s+(\w+)\s*=\s*get_compile_time_arg_val\(([^)]*)\)")
    return {m.group(1): m.group(2).strip() for m in pat.finditer(src)}


def _sq(s: str) -> str:
    """Whitespace-insensitive source text (black wraps the long literals the ABI pins)."""
    return re.sub(r",\]", "]", re.sub(r"\s+", "", s))  # black's trailing comma before a wrapped `]` too


def _accessor_offset(src: str) -> int:
    m = re.search(r"TensorAccessorArgs<(\d+)>\(\)", src)
    assert m, "no TensorAccessorArgs<N> in the kernel"
    return int(m.group(1))


def tile_elem(row: int, col: int) -> int:
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15)


def tilize_u16(mat: torch.Tensor) -> torch.Tensor:
    """[32, 32 * n] bf16 -> [n, 1024] uint16 bit patterns in the four-face tile layout the kernels scan."""
    assert mat.shape[0] == 32, tuple(mat.shape)
    if mat.shape[1] % 32:  # a selected [32, 72] matrix: its 3 tiles, zero-padded (today's tile tensor)
        mat = torch.cat([mat, torch.zeros(32, 32 - mat.shape[1] % 32, dtype=mat.dtype)], dim=1)
    bits = mat.contiguous().view(torch.int16)
    n = mat.shape[1] // 32
    out = torch.zeros(n, 1024, dtype=torch.int16)
    for i in range(n):
        for t in range(32):
            for c in range(32):
                out[i, tile_elem(t, c)] = bits[t, 32 * i + c]
    return out


def random_routing(T: int, picks: int, seed: int, E: int = E_TOTAL) -> torch.Tensor:
    """[32, E] bf16: rows < T route to `picks` random experts with random positive weights, the pad rows zero."""
    g = torch.Generator().manual_seed(seed)
    r = torch.zeros(32, E)
    for t in range(T):
        ids = torch.randperm(E, generator=g)[:picks]
        r[t, ids] = torch.rand(picks, generator=g) + 0.2
    return r.bfloat16()


def selector_columns(routing: torch.Tensor, col_base: int, E_local: int) -> torch.Tensor:
    """Today's path: the [32, E] bf16 routing x the identity block [E, E_local] in fp32 (HiFi4 keeps the bf16
    weights intact; one product per output element) -> bf16 [32, E_local]."""
    sel = torch.zeros(routing.shape[1], E_local)
    for e in range(E_local):
        sel[col_base + e, e] = 1.0
    return (routing.float() @ sel).bfloat16()


HARNESS = r"""
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define tt_l1_ptr
#include "routing_scan.hpp"
int main(int argc, char** argv) {
    const char* mode = argv[1];
    const uint32_t col_base = static_cast<uint32_t>(atoi(argv[2]));
    constexpr uint32_t HE_ = HE, HM_ = HM, HR_ = HR;  // the kernels' E / M / R_TILES
    std::vector<uint16_t> buf(HR_ * 1024, 0xFFFF);  // unread tiles hold garbage on purpose
    const size_t n_in = fread(buf.data(), 2, buf.size(), stdin);
    (void)n_in;
    uint16_t active[HE_];
    const uint32_t col_off = fold_col_off(col_base);
    const uint32_t n_tiles = fold_tiles<HE_>(col_base);
    uint32_t cnt;
    if (strcmp(mode, "classic") == 0) {
        cnt = scan_active<HE_, HM_>(buf.data(), active);
    } else if (strcmp(mode, "fold") == 0) {
        cnt = scan_active_fold<HE_, HM_>(buf.data(), active, col_off);
    } else {
        cnt = scan_active_words<HE_, HM_, HR_>(buf.data(), active, col_off, n_tiles);
    }
    printf("%u\n", cnt);
    for (uint32_t i = 0; i < cnt; ++i) {
        printf("%u", static_cast<unsigned>(active[i]));
        for (uint32_t t = 0; t < HM_; ++t) {
            const uint32_t off = strcmp(mode, "classic") == 0 ? 0 : col_off;
            printf(" %u", static_cast<unsigned>(buf[fold_elem(t, active[i], off)]));
        }
        printf("\n");
    }
    return 0;
}
"""


def _build_harness(tmp: Path, E: int, M: int, r_tiles: int) -> Path | None:
    gxx = shutil.which("g++") or shutil.which("c++")
    if gxx is None:
        return None
    src = tmp / "scan_harness.cpp"
    src.write_text(HARNESS)
    exe = tmp / f"scan_{E}_{M}_{r_tiles}"
    cmd = [
        gxx,
        "-std=c++17",
        "-O1",
        f"-I{KERNEL_DIR}",
        f"-DHE={E}",
        f"-DHM={M}",
        f"-DHR={r_tiles}",
        str(src),
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return exe


def _run_scan(exe: Path, mode: str, col_base: int, tiles: torch.Tensor) -> tuple[list[int], dict[int, list[int]]]:
    out = subprocess.run(
        [str(exe), mode, str(col_base)], input=tiles.contiguous().numpy().tobytes(), capture_output=True, check=True
    ).stdout.decode()
    lines = out.strip().split("\n")
    n = int(lines[0])
    active, diag = [], {}
    for line in lines[1 : 1 + n]:
        vals = [int(v) for v in line.split()]
        active.append(vals[0])
        diag[vals[0]] = vals[1:]
    return active, diag


# ---------------------------------------------------------------------------------------------- kernel ABI
def test_block_kernel_abi_and_default_vectors():
    """The four block kernels' compile-time positions against the lists ``descriptors()`` packs (pinned literally,
    the pre-E1 order), the ROUTING_FOLD base accessor after today's blocks, the runtime tails, the exit barriers."""
    rg, wg, rd, wd = (_src(n) for n in BLOCK_KERNELS)
    assert [
        _ct_indices(rg)[n]
        for n in ("cb_w", "cb_r", "kt", "E", "M", "groups", "expert_stride", "tile_bytes", "page_bytes", "ppb")
    ] == [str(i) for i in range(10)]
    assert _accessor_offset(rg) == 10
    assert [
        _ct_indices(wg)[n] for n in ("cb_x", "cb_r", "cb_d", "cb_out", "kt", "E", "M", "groups", "i_tiles", "ppb")
    ] == [str(i) for i in range(10)]
    assert _accessor_offset(wg) == 10
    assert [
        _ct_indices(rd)[n]
        for n in ("cb_w", "cb_r", "seg_tiles", "E", "M", "cols", "bank_expert_stride", "tile_bytes", "page_bytes")
    ] == [str(i) for i in range(9)]
    assert _accessor_offset(rd) == 9
    assert [_ct_indices(wd)[n] for n in ("cb_r", "cb_a", "cb_out", "seg_tiles", "E", "M", "cols")] == [
        str(i) for i in range(7)
    ]
    assert _accessor_offset(wd) == 7
    # the fold's base page: the last accessor block, read behind the define; its runtime arg right after today's
    for src, last, arg in ((rg, "routing_args", 5), (wg, "act_args", 5), (rd, "routing_args", 5), (wd, "out_args", 4)):
        assert f"constexpr auto base_args = TensorAccessorArgs<{last}.next_compile_time_args_offset()>();" in src
        assert f"const uint32_t base_addr = get_arg_val<uint32_t>({arg});" in src
        assert src.count("#ifdef ROUTING_FOLD") >= 2 and "#if defined(SCAN_WORDS)" in src
    # the fused tails: writer_gate_up [n_down, sem_base, sems_per_group, xy...] from ARG_NEXT (5, 6 with the fold)
    assert "constexpr uint32_t ARG_NEXT = 6;" in wg and "constexpr uint32_t ARG_NEXT = 5;" in wg
    assert "const uint32_t n_down = get_arg_val<uint32_t>(ARG_NEXT);" in wg
    assert "get_arg_val<uint32_t>(ARG_NEXT + 1)" in wg and "get_arg_val<uint32_t>(ARG_NEXT + 2)" in wg
    assert "constexpr uint32_t ARG_XY = ARG_NEXT + 3;" in wg and "get_arg_val<uint32_t>(ARG_XY + d)" in wg
    # writer_down [gate_up_groups, sem_base, sems_per_group, incs_per_expert] from ARG_NEXT (4, 5 with the fold)
    assert "constexpr uint32_t ARG_NEXT = 5;" in wd and "constexpr uint32_t ARG_NEXT = 4;" in wd
    for k in range(4):
        assert f"get_arg_val<uint32_t>(ARG_NEXT{'' if k == 0 else f' + {k}'})" in wd
    # op.py packs exactly those tails
    src = _sq(inspect.getsource(O.ExpertStreamFused.forward))
    assert _sq("fused_rt=[len(self.down_xy), self.SEM_BASE, self.sems_per_group] + self.down_xy,") in src
    assert _sq("fused_rt=[self.groups, self.SEM_BASE, self.sems_per_group, self.banks],") in src
    # the defines the op emits are the ones the kernels gate on
    assert O._e1_defines(False, False) == []
    assert O._e1_defines(True, True, "FUSED_SIGNAL") == [
        ("ROUTING_FOLD", "1"),
        ("SCAN_WORDS", "1"),
        ("FUSED_SIGNAL", "1"),
    ]
    for name, define in (("writer_gate_up", "FUSED_SIGNAL"), ("writer_down", "FUSED_WAIT")):
        assert f"#ifdef {define}" in _src(name)
    # exit contract on every data-movement kernel (writer_down: both exit paths)
    for name in BLOCK_KERNELS:
        assert "noc_async_full_barrier();" in _src(name), name
    assert wd.count("noc_async_full_barrier();") == 2
    # the default descriptors: today's literal orders and no defines (whitespace-insensitive: black wraps them)
    desc = _sq(inspect.getsource(O.ExpertGateUpStream.descriptors))
    # (2026-10-27: ``w_addr`` is the stacked tensor's address, 0 under ``regions_src`` where the per-expert addresses
    # ride as common runtime args; the stacked program's words are unchanged)
    assert _sq("w_addr = 0 if src is not None else w13.buffer_address()") in desc
    assert _sq("reader_rt[core.x][core.y] = [w_addr, routing.buffer_address(), bank, vc, r] + base_rt") in desc
    assert _sq("[x.buffer_address(), routing.buffer_address(), act.buffer_address(), bank, r] + base_rt + tail") in desc
    assert (
        _sq(
            "reader_ct = [cb_w, cb_r0, self.kt, self.E, TILE, self.groups, self.expert_stride, self.w_tile, self.page_bytes, self.ppb,]"
        )
        in desc
    )
    assert (
        _sq("writer_ct = [cb_x, cb_r1, cb_d, cb_out, self.kt, self.E, TILE, self.groups, self.i_tiles, self.ppb]")
        in desc
    )
    assert _sq("compute_ct = [cb_x, cb_w, cb_d, cb_h, cb_out, self.kt, self.ppb]") in desc
    desc = _sq(inspect.getsource(O.ExpertDownStream.descriptors))
    assert _sq("w_addr = 0 if src is not None else w2.buffer_address()") in desc
    assert _sq("[w_addr, routing.buffer_address(), bank, vc, r * self.cols] + base_rt") in desc
    assert (
        _sq(
            "[routing.buffer_address(), act.buffer_address(), out.buffer_address(), bank * self.C + r * self.cols,] + base_rt + tail"
        )
        in desc
    )
    assert (
        _sq(
            "reader_ct = [cb_w, cb_r0, self.seg_tiles, self.E, TILE, self.cols, self.bank_expert_stride, BFP8_TILE_BYTES, self.page_bytes,]"
        )
        in desc
    )
    assert _sq("writer_ct = [cb_r1, cb_a, cb_out, self.seg_tiles, self.E, TILE, self.cols]") in desc
    assert _sq("compute_ct = [cb_a, cb_w, cb_out, self.seg_tiles, self.cols]") in desc
    kern = _sq(inspect.getsource(O._BankStreamOp._kernels))
    assert _sq('extra = {"defines": [tuple(d) for d in defines]} if defines else {}') in kern
    # the compute kernels are untouched by E1 (the fused program runs them as they are)
    for name in ("compute_gate_up", "compute_down"):
        s = _src(name)
        assert "ROUTING_FOLD" not in s and "SCAN_WORDS" not in s and "FUSED" not in s


def test_fold_window_and_base_page():
    """The host mirror of the kernels' window arithmetic and the per-column base page."""
    for c in range(4):
        tb, off, n = O.fold_window(c * E_LOCAL, E_LOCAL)
        assert tb == (c * E_LOCAL) // 32 and off == (c * E_LOCAL) % 32 and n == 3
        assert tb + n <= E_TOTAL // 32  # never past the routing tensor's 9 tiles
    assert O.fold_window(31, 72) == (0, 31, 4) and O.fold_r_tiles(72) == 4  # the worst case fits R_TILES
    assert O.fold_r_tiles(288) == 10 and O.fold_window(0, 288) == (0, 0, 9)
    base = O.fold_base_host(4, E_LOCAL)
    assert tuple(base.shape) == (1, 4, 1, 16) and base[0, :, 0, 0].tolist() == [0, 72, 144, 216]
    assert int(base.abs().sum()) == 72 + 144 + 216  # only element 0 of each page is set


# ---------------------------------------------------------------------------------------------- folded selector goldens
def test_selector_matmul_is_exact_column_read():
    """The fold's premise: the fp32 product of a bf16 routing matrix with a 0/1 identity block (one product per
    output element) rounds back to the same bf16 bits as reading the columns."""
    for seed, T in ((0, 4), (1, 32), (2, 1)):
        r = random_routing(T, K_PICKS, seed)
        for c in range(4):
            sel = selector_columns(r, c * E_LOCAL, E_LOCAL)
            direct = r[:, c * E_LOCAL : (c + 1) * E_LOCAL]
            assert torch.equal(sel.view(torch.int16), direct.view(torch.int16)), (seed, c)


@pytest.mark.parametrize("T", [4, 32])
def test_scan_goldens_host_cpp(T, tmp_path):
    """routing_scan.hpp on the host: scan_active on today's selected window == scan_active_fold / scan_active_words
    on the full-matrix window at every column base; the active sets and D_e diagonals equal the torch selector
    path bit for bit; the words scan equals the element scan with col_off = 0 (the default-program case)."""
    exe = _build_harness(tmp_path, E_LOCAL, 32, O.fold_r_tiles(E_LOCAL))
    if exe is None:
        pytest.skip("no host C++ compiler")
    for seed in range(3):
        r = random_routing(T, K_PICKS, 100 * T + seed)
        full = tilize_u16(r)  # [9, 1024]
        for c in range(4):
            col_base = c * E_LOCAL
            sel = selector_columns(r, col_base, E_LOCAL)  # today's [32, 72]
            ref_active = [e for e in range(E_LOCAL) if int(sel[:, e].float().abs().sum() > 0)]
            ref_diag = {e: sel[:, e].contiguous().view(torch.int16).tolist() for e in ref_active}
            ref_diag = {e: [v & 0xFFFF for v in vals] for e, vals in ref_diag.items()}
            # today's program: the selected matrix's 3 tiles, col_off 0
            classic_active, classic_diag = _run_scan(exe, "classic", 0, tilize_u16(sel))
            assert classic_active == ref_active and classic_diag == ref_diag, (T, seed, c, "classic")
            # the fold: the window of the FULL matrix
            tb, off, n = O.fold_window(col_base, E_LOCAL)
            window = full[tb : tb + n]
            for mode in ("fold", "words"):
                act, diag = _run_scan(exe, mode, col_base, window)
                assert act == ref_active, (T, seed, c, mode)
                assert diag == ref_diag, (T, seed, c, mode)
        # the default program with SCAN_WORDS: col_off 0 on the selected window == the element scan
        sel = selector_columns(r, 0, E_LOCAL)
        a0, d0 = _run_scan(exe, "classic", 0, tilize_u16(sel))
        a1, d1 = _run_scan(exe, "words", 0, tilize_u16(sel))
        assert (a0, d0) == (a1, d1)


def test_scan_goldens_single_chip_and_wide(tmp_path):
    """E = 288 on one chip (base 0, 9 of 10 window tiles read) and E = 24 (I = 2048 single-chip test shape)."""
    exe = _build_harness(tmp_path, E_TOTAL, 32, O.fold_r_tiles(E_TOTAL))
    if exe is None:
        pytest.skip("no host C++ compiler")
    r = random_routing(32, K_PICKS, 7)
    ref = [e for e in range(E_TOTAL) if int(r[:, e].float().abs().sum() > 0)]
    tiles = tilize_u16(r)
    assert _run_scan(exe, "classic", 0, tiles)[0] == ref
    assert _run_scan(exe, "words", 0, tiles)[0] == ref
    assert _run_scan(exe, "fold", 0, tiles)[0] == ref
    exe = _build_harness(tmp_path, 24, 32, O.fold_r_tiles(24))
    r = random_routing(32, 2, 9, E=96)
    for base in (0, 24, 48, 72):
        tb, off, n = O.fold_window(base, 24)
        ref = [e for e in range(24) if int(r[:, base + e].float().abs().sum() > 0)]
        assert _run_scan(exe, "words", base, tilize_u16(r)[tb : tb + n])[0] == ref, base
        assert _run_scan(exe, "fold", base, tilize_u16(r)[tb : tb + n])[0] == ref, base


# ---------------------------------------------------------------------------------------------- fused protocol model
def _fused_sim(G: int, banks: int, n_active: int, E_local: int, seed: int) -> None:
    """Random interleaving of per-core progress: the down writer's wait (rank i -> group i % G, slot i // G) passes
    only once all `banks` cores of that group finished that expert; slots never alias."""
    spg, n_sems = P.fused_semaphores(G, E_local)
    assert n_sems <= P.NUM_L1_SEMAPHORES
    rng = torch.Generator().manual_seed(seed)
    # per core (g, b): the list of expert ranks it handles, in order; progress pointer
    mine = {(g, b): list(range(g, n_active, G)) for g in range(G) for b in range(banks)}
    done = {c: 0 for c in mine}
    sems = [0] * n_sems
    completed = set()  # (g, k) with all banks done
    landed = {}  # (g, k) -> banks that incremented
    pending = [c for c in mine if mine[c]]
    rank = 0  # the down writer's next expert
    steps = 0
    while rank < n_active:
        # the down writer polls its slot
        g, k = rank % G, rank // G
        sem = sems[g * spg + k // 7]
        field = (sem >> (4 * (k % 7))) & 0xF
        if field >= banks:
            assert (g, k) in completed, "wait passed before every bank's tile landed"
            assert len(landed[(g, k)]) == banks
            rank += 1
            continue
        # some gate|up core finishes its next expert
        assert pending, "deadlock: nothing in flight but the wait is unsatisfied"
        idx = int(torch.randint(len(pending), (1,), generator=rng))
        core = pending[idx]
        kk = done[core]
        r = mine[core][kk]
        assert r % G == core[0] and r // G == kk  # the kernels' rank <-> (group, slot) map
        sid = core[0] * spg + kk // 7
        shift = 4 * (kk % 7)
        before = (sems[sid] >> shift) & 0xF
        sems[sid] += 1 << shift
        assert ((sems[sid] >> shift) & 0xF) == before + 1 and before + 1 <= banks  # no carry into the neighbour
        assert sid < n_sems and shift + 4 <= 28  # bit 31 never touched
        landed.setdefault((core[0], kk), set()).add(core[1])
        if len(landed[(core[0], kk)]) == banks:
            completed.add((core[0], kk))
        done[core] += 1
        if done[core] == len(mine[core]):
            pending.remove(core)
        steps += 1
    # at the end every slot was polled to full count: the writer's end reset leaves zeros consistent with that
    for (g, k), who in landed.items():
        assert len(who) == banks


@pytest.mark.parametrize("G, banks", [(12, 8), (6, 8), (15, 8), (16, 8), (8, 8), (4, 8), (12, 4)])
def test_fused_arrival_protocol_model(G, banks):
    for n_active in (0, 1, 7, 28, 43, 72):
        for seed in range(3):
            _fused_sim(G, banks, n_active, E_LOCAL, seed)


def test_fused_semaphore_budget():
    """Every feasible group count stays within the 16 L1 semaphores; 9 / 10 groups at E = 72 do not (3 slots of 7
    overflow the second semaphore); the planner never picks them."""
    assert P.fused_group_choices(E_LOCAL) == [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
    assert P.fused_semaphores(12, E_LOCAL) == (1, 12) and P.fused_semaphores(6, E_LOCAL) == (2, 12)
    try:
        P.fused_semaphores(9, E_LOCAL)
        raise AssertionError("9 groups must be refused")
    except ValueError as err:
        assert "semaphores" in str(err)
    for E_local in (9, 24, 36, 72):
        for g in P.fused_group_choices(E_local):
            spg, n = P.fused_semaphores(g, E_local)
            assert n <= 16 and spg * 7 * g >= E_local


def test_fused_program_structure():
    """The fused op = the two descriptor sets on disjoint cores + the arrival semaphores on the down cores; the
    compute kernels and the per-expert math are the two-op form's (same sources, same CT args per core set)."""
    src = _sq(inspect.getsource(O.ExpertStreamFused))
    assert (
        _sq("exclude=self.down.core_set") in src
        and _sq("assert not (self.gate_up.core_set & self.down.core_set)") in src
    )
    assert (
        _sq("cb=(0, 1, 2, 3, 4, 5, 16)") in src and _sq("cb=(8, 9, 10, 11, 17)") in src
    )  # distinct CB ids per core set
    assert _sq("ttnn.SemaphoreDescriptor(self.SEM_BASE + i, ttnn.CoreType.WORKER, self.down.cores, 0)") in src
    assert _sq("weights = [w13, w2]") in src and _sq("ttnn.generic_op(weights + [x, routing, act, out]") in src
    k = inspect.getsource(O._BankStreamOp._kernels)
    assert 'KERNEL_DIR / f"compute_{name}.cpp"' in k  # the same compute sources for both forms
    # the down cores are placed first (nearest their banks), the gate|up cores around them
    init = inspect.getsource(O.ExpertStreamFused.__init__)
    assert init.index("self.down = ExpertDownStream(") < init.index("self.gate_up = ExpertGateUpStream(")
    # ExpertStream.forward routes to the fused program and threads the base page
    fwd = inspect.getsource(O.ExpertStream.forward)
    assert "return self.fused_op.forward(x, routing, w13, w2, act, out, base=base)" in fwd
    assert (
        "self.gate_up(x, routing, w13, act, base=base)" in fwd and "self.down(act, routing, w2, out, base=base)" in fwd
    )


# ---------------------------------------------------------------------------------------------- planners on both grids
@pytest.mark.parametrize("grid", GRIDS)
def test_fused_core_plan_grids(grid):
    plan = P.fused_core_plan(grid, E_LOCAL, I_LOC, H)
    assert plan["down_readers"] == 2 and plan["down_cores"] == 16
    assert plan["groups"] == (12 if grid == (13, 10) else 6)
    assert plan["gate_up_cores"] + plan["down_cores"] <= grid[0] * grid[1]
    assert plan["semaphores"] <= 16 and plan["groups"] in P.fused_group_choices(E_LOCAL)
    # the (2,4) block width (I = 1024: 4 readers) and the single-chip width (I = 2048: 8 readers)
    p1024 = P.fused_core_plan(grid, E_LOCAL, 1024, H)
    assert p1024["down_readers"] == 4 and p1024["gate_up_cores"] + p1024["down_cores"] <= grid[0] * grid[1]
    if grid == (13, 10):
        p2048 = P.fused_core_plan(grid, 24, 2048, H)
        assert p2048["down_readers"] == 8 and p2048["groups"] == 8
    else:
        try:
            P.fused_core_plan(grid, 24, 2048, H)
            raise AssertionError("64 down cores leave no room on 8x8")
        except AssertionError as err:
            assert "fits" in str(err) or "room" in str(err)
    # explicit groups are validated against the room and the semaphore budget
    try:
        P.fused_core_plan(grid, E_LOCAL, I_LOC, H, groups=9)
        raise AssertionError("9 groups must be refused")
    except AssertionError as err:
        assert "semaphores" in str(err) or "fit" in str(err)


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("units", [False, True])
@pytest.mark.parametrize("roofline", [False, True])
def test_plan_for_32row_grids(grid, units, roofline):
    for I_loc in (I_LOC, 1024):
        plan = P.plan_for_32row(grid, E_LOCAL, I_loc, H, T=256, units=units, roofline=roofline)
        kw = plan.op_kwargs()
        ppb = (I_loc // 32) // 8
        assert kw["down_readers"] == 8 and 1 <= kw["groups"] <= 32 and kw["prefetch_pairs"] >= 1
        assert ppb % kw["unit_pairs"] == 0
        assert kw["units"] == (units or (roofline and kw["units"]))
        if not roofline:
            assert kw["groups"] == (12 if grid == (13, 10) else 7) and kw["gather_rotate"] is False
            assert "measured" in plan.notes[0]
        # the gate|up L1 of the planned program fits (prefetch chosen by max_prefetch_pairs for the non-EP op)
        assert (
            P.gate_up_l1_bytes(128, P.BFP8_TILE_BYTES, kw["prefetch_pairs"], 256, E_LOCAL, False)
            <= P.L1_BYTES - P.L1_RESERVED_BYTES
        )


# ---------------------------------------------------------------------------------------------- knob plumbing
def test_e1_knobs_and_layout_rule():
    assert P.e1_stream_knobs({}) == dict(routing_fold=False, scan_words=False, fused=False)
    assert P.e1_stream_knobs(
        {"GLM53_STREAM_FOLD": "1", "GLM53_STREAM_SCAN": "words", "GLM53_STREAM_FUSED": "1"}
    ) == dict(routing_fold=True, scan_words=True, fused=True)
    try:
        P.e1_stream_knobs({"GLM53_STREAM_SCAN": "fast"})
        raise AssertionError("an unknown scan mode must be refused")
    except ValueError:
        pass
    assert P.experts_32row_mode({}) == "dense"
    try:
        P.experts_32row_mode({"GLM53_EXPERTS_32ROW": "stream"})
        raise AssertionError("an unknown 32-row mode must be refused")
    except ValueError:
        pass
    # the layout rule: GLM53_EXPERTS_32ROW over GLM53_DENSE_EXPERTS / GLM53_STREAM_EXPERTS
    assert M.expert_layout_from_env({}) == (False, True, "dense")
    assert M.expert_layout_from_env({"GLM53_DENSE_EXPERTS": "1"}) == (True, False, "dense")
    assert M.expert_layout_from_env({"GLM53_STREAM_EXPERTS": "0"}) == (False, False, "dense")
    assert M.expert_layout_from_env({"GLM53_DENSE_EXPERTS": "1", "GLM53_EXPERTS_32ROW": "dispatch"}) == (
        False,
        True,
        "dispatch",
    )
    assert M.expert_layout_from_env({"GLM53_STREAM_EXPERTS": "0", "GLM53_EXPERTS_32ROW": "units"}) == (
        False,
        True,
        "units",
    )
    # the older rules are untouched
    assert M.stream_experts_default(False, {}) and not M.stream_experts_default(True, {})
    assert M.stream_experts_default(True, {"GLM53_STREAM_EXPERTS": "1"})
    assert M.prefill_experts_for(True, True, {"GLM53_PREFILL_EXPERTS": "regions"}) == "blocks"
    assert M.prefill_experts_for(False, True, {"GLM53_PREFILL_EXPERTS": "regions"}) == "regions"
    # the decode glue defaults since 2026-10-29 (H1 / H2: the compact gather + the all-reduce combine); the padded
    # gather + reduce-scatter stays the explicit opt-out
    assert M.decode_glue_modes(8, 4, has_idx=False, in_prefill=False, env={}) == (True, "allreduce")
    assert M.decode_glue_modes(
        8, 4, has_idx=False, in_prefill=False, env={"GLM53_HYBRID_GATHER": "padded", "GLM53_HYBRID_COMBINE": "rs"}
    ) == (False, "rs")
    # TtMoE reads the composed rule in both places
    assert "expert_layout_from_env()" in inspect.getsource(M.TtMoE.__init__)
    assert "expert_layout_from_env()" in inspect.getsource(M.TtMoE.expert_cache_complete)


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _T:
    def __init__(self, dtype, last):
        self.dtype, self.shape = dtype, (1, 1, 32, last)


def test_fold_serves_and_x_into_l1():
    stub = _Stub(
        stream_fold=True,
        stream_base=object(),
        E=E_TOTAL,
        stream=True,
        stream_x_l1=True,
        w13=object(),
        stream_dispatch=True,
    )
    stub._stream_serves = lambda T: M.TtRoutedExperts._stream_serves(stub, T)
    fs = M.TtRoutedExperts._fold_serves
    assert fs(stub, _T(ttnn.bfloat16, E_TOTAL))
    assert not fs(stub, _T(ttnn.float32, E_TOTAL))  # the composite router's fp32 matrix keeps the selector matmul
    assert not fs(stub, _T(ttnn.bfloat16, E_LOCAL))  # an already-selected matrix
    stub.stream_fold = False
    assert not fs(stub, _T(ttnn.bfloat16, E_TOTAL))
    stub.stream_fold = True
    xl1 = M.TtRoutedExperts._stream_x_into_l1
    assert xl1(stub, 32) and xl1(stub, 8) and not xl1(stub, 64)  # block shapes only
    stub.stream_x_l1 = False
    assert not xl1(stub, 32)
    # _stream_serves is the hybrid-glue rule, untouched
    stub.stream_x_l1 = True
    assert (
        stub._stream_serves(32)
        and stub._stream_serves(256)
        and not stub._stream_serves(288)
        and not stub._stream_serves(40)
    )
    src = inspect.getsource(M.TtRoutedExperts._block_stream)
    assert (
        "base = self.stream_base" in src
        and "op, w13, w2 = self._stream_source()" in src
        and "op.forward(xb, dense_w, w13, w2, act, out, base=base)" in src
    )
    # the stacked op and tensors unless GLM53_REGIONS_DECODE=stream (2026-10-27)
    assert "return self.stream_op, self.w13, self.w2" in inspect.getsource(M.TtRoutedExperts._stream_source)
    src = inspect.getsource(M.TtRoutedExperts.forward)
    assert "x = self._gather_rows_compact(x, l1=self._stream_x_into_l1(self.mesh.rows * T))" in src
    src = inspect.getsource(M.TtRoutedExperts._gather_rows_compact)
    assert "memory_config=ttnn.L1_MEMORY_CONFIG" in src
    src = inspect.getsource(M.TtRoutedExperts._dispatch_op)
    assert "plan_for_32row(" in src and 'if mode == "units":' in src


def test_moe_contract_intact():
    """The coordinator's block-source release and the hybrid glue knobs survive the E1 edits."""
    src = inspect.getsource(M.TtRoutedExperts.__init__)
    assert "self._block_source = None" in src and src.index("_build_dispatch_path(fabric=") < src.index(
        "self._block_source = None"
    )
    assert "self.stream_base = (" in src and "fold_base_host(ep, E_loc)" in src
    src = inspect.getsource(M.TtRoutedExperts._build_dispatch_path)
    assert "if self.stream and self._block_source is None:" in src and "raise RuntimeError(" in src
    assert M.HYBRID_GATHER_MODES == ("padded", "compact") and M.HYBRID_COMBINE_MODES == (
        "rs",
        "allreduce",
        "allreduce_ccl",
    )
    assert M.STREAM_DISPATCH_MAX_T == 256


# ---------------------------------------------------------------------------------------------- the phase model
def test_launch_phase_table_anchors():
    t = P.launch_phase_table(32, 28)
    assert abs(t["total_us"] - 351.0) < 1.0, t["total_us"]  # the measured 341 traced + the two host ops
    assert t["floor_us"] < t["total_us"] and t["bytes"] == 28 * sum(P.expert_bytes(H, I_LOC, H))
    ladder = P.e1_expected_us(28)
    assert ladder["today"] > ladder["fold"] > ladder["fold+words"] > ladder["fold+words+fused"]
    assert ladder["fold+words+fused"] > t["floor_us"]
    for T, active in ((4, 8), (32, 28), (32, 43)):
        tab = P.launch_phase_table(T, active)
        phases = [r["phase"] for r in tab["rows"]]
        assert "gate|up weight stream" in phases and "down weight stream" in phases
        assert all(r["modelled_us"] >= 0 for r in tab["rows"])
    folded = P.launch_phase_table(32, 28, routing_fold=True)
    assert sum(r["modelled_us"] for r in folded["rows"] if "(host op)" in r["phase"]) == 0.0
    fused = P.launch_phase_table(32, 28, routing_fold=True, scan_words=True, fused=True)
    assert fused["total_us"] < 0.75 * t["total_us"]


def test_scan_model_direction():
    """The word scan model is below the element scan at the decode activities (the lever of the prelude); at full
    activity (every expert active, the element scan's early exits) the two tie (the model says so too)."""
    for n in (0, 8, 28, 43):
        assert P.scan_us(E_LOCAL, 32, n, 3, True) < P.scan_us(E_LOCAL, 32, n, 3, False), n
    assert P.scan_us(E_LOCAL, 32, 72, 3, True) < 1.5 * P.scan_us(E_LOCAL, 32, 72, 3, False)
