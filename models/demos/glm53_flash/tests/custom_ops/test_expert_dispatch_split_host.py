# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The dispatch down op's expert-parallel split (decode roofline lever E3, ``ExpertDownDispatch(down_split=
"experts")``; ``custom_ops/expert_stream/README.md`` "Expert-parallel down split"), host side:

* the geometry / L1 plan (``down_split_layout``): the 8x4 point (I = 256: 8-column units, 2 per expert, 2 owned
  columns per core, one producer per owner per pass) and the wider slices, the budget reductions, the refusals;
* the unit LPT (``down_split_units``, the mirror of ``kernels/dispatch_common.hpp`` ``for_each_unit``) against the
  C++ template built with the host g++ (when present): every unit on exactly one core, the reader / writer / compute
  of a core agree on the unit order, the ``cols`` form's per-expert LPT is the ``upe = 1`` case;
* the slot-index table (``down_slot_table``) against a mirror of the ``cols`` writer's per-row-tile cursor scan
  (the slot every (pass, row) contribution lands in, the J cap, the used slot count per row tile): the two writers
  fill the same slots with the same rows, which is why the fp32 sums and the outputs are bit-identical;
* the owner / column map of the scatter: every (owner, column) of a bank receives every pass exactly once and the
  arrival count the owner waits for is what the producers signal;
* the knob (``plan.down_split_mode``) and the cost model (``plan.down_split_expected_us``) at the README's points;
* an end-to-end emulator of the ``experts`` data flow (unit assignment -> per-(unit, pass) h rows -> scatter
  destinations (owner core, slot, tile row, owned column) -> per-row-tile slot sums) against an emulation of the
  ``cols`` writer / compute on the same random activations and weights: ``torch.equal`` (the bookkeeping moves the
  same numbers to the same places; the sums run in the same slot order);
* the kernel sources: the three twins exist, replay ``for_each_unit`` with the same template arguments, keep the
  exit barrier, run the unit's columns through DST ``dst_batch`` (= 4, half of the 8-fp32-tile Blackhole DEST in
  half-sync mode: the 2026-11-03 exactness failure) at a time, and the ``cols`` kernels are untouched by the split.

``import ttnn`` only (no device); run with ``pytest --noconftest``:

    python3 -m pytest models/demos/glm53_flash/tests/custom_ops/test_expert_dispatch_split_host.py --noconftest -q
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream import plan as P  # noqa: E402
from models.demos.glm53_flash.custom_ops.expert_stream.op import (  # noqa: E402
    DOWN_L1_BUDGET,
    DST_FP32_HALF_TILES,
    KERNEL_DIR,
    MAX_PICKS,
    TILE,
    clamp_passes_host,
    dispatch_contributions,
    dispatch_passes,
    down_slot_table,
    down_split_layout,
    down_split_units,
    down_unit_cols,
    max_passes,
)

K, H = 4096, 4096
E_LOCAL = 72


def _topk_routing(T: int, E: int, picks: int, seed: int, popular: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pref = 1.0 / (torch.arange(E, dtype=torch.float) + 1.0) ** 1.3
    pref = pref[torch.randperm(E, generator=g)]
    r = torch.zeros(T, E)
    for t in range(T):
        ids = torch.multinomial(pref, picks, replacement=False, generator=g)
        r[t, ids] = torch.rand(picks, generator=g) + 0.2
    for e in range(popular):
        r[:, e] = torch.rand(T, generator=g) + 0.2
    return r.bfloat16()


def _passes_of(routing: torch.Tensor, p_max: int) -> list[int]:
    counts = (routing != 0).sum(0).tolist()
    return clamp_passes_host(counts, p_max)[0]


# ------------------------------------------------------------------------------------------ geometry / L1
def test_layout_8x4_point():
    """I = 256, H = 4096, 8 banks x 8 cores: 16 columns per bank, 8-column units (70 KB), 2 per expert, 2 owned
    columns per core, one producer per owner per pass; ~650 KB of L1 at T = 256 with the full slot counts."""
    lay = down_split_layout(E_LOCAL, 256, H, 256)
    assert (lay["C"], lay["cols_own"], lay["unit_cols"], lay["upe"], lay["incs_per_pass"]) == (16, 2, 8, 2, 1)
    assert lay["dst_batch"] == DST_FP32_HALF_TILES == 4  # two DST batches per pass for the 8-column unit
    assert lay["unit_tiles"] == 64 and lay["unit_bytes"] == 64 * 1088
    assert lay["P"] == max_passes(256, E_LOCAL) == 136 and lay["RT"] == 8 and lay["J"] == MAX_PICKS
    assert (lay["a_slots"], lay["w_slots"]) == (6, 3)
    cb = lay["cb_bytes"]
    assert cb["slots"] == 8 * MAX_PICKS * 2 * 2048 == 262144
    assert cb["h_staging"] == 2 * 8 * 2048 and cb["weights"] == 3 * 64 * 1088
    assert cb["meta"] >= 4 * (1 + E_LOCAL * 2) and cb["slot_index"] == (136 + 1) * 64
    assert lay["l1_bytes"] == sum(cb.values()) <= DOWN_L1_BUDGET
    assert 600_000 < lay["l1_bytes"] < 700_000, lay["l1_bytes"]


@pytest.mark.parametrize("I, unit_cols, upe", [(256, 8, 2), (1024, 4, 4), (2048, 2, 8)])
def test_layout_widths(I, unit_cols, upe):
    """The unit width follows the block op's rule (<= 8 DST tiles, <= 128 tiles per unit block): 8 / 4 / 2 columns
    at I = 256 / 1024 / 2048; the wider slices give the activation / weight slots away to the budget but fit."""
    seg = I // TILE
    assert down_unit_cols(16, seg) == unit_cols
    lay = down_split_layout(E_LOCAL, I, H, 256)
    assert (lay["unit_cols"], lay["upe"]) == (unit_cols, upe)
    assert lay["unit_cols"] * seg <= 128
    assert lay["dst_batch"] == min(unit_cols, DST_FP32_HALF_TILES) and unit_cols % lay["dst_batch"] == 0
    assert lay["l1_bytes"] <= DOWN_L1_BUDGET
    if I == 2048:
        assert lay["a_slots"] < 6  # 64-tile passes: the activation slots gave way to the budget
    else:
        assert (lay["a_slots"], lay["w_slots"]) == (6, 3)  # I = 1024 fits as is (1.14 MB)


def test_layout_owner_wider_than_unit():
    """One core per bank (16 owned columns) is refused by the DST rule; 2 cores per bank own 8 columns = one unit
    each (the resident slots of 8 columns x 8 row tiles = 1 MB do not fit at T = 256: refused; T = 64 fits); 4
    cores own 4 columns: unit 8 nests 2 owners -> still one producer per owner per pass; the nesting assertion
    covers the general case."""
    with pytest.raises(ValueError, match="exceed the 8 fp32 DST tiles"):  # allow-pytest.raises: host contract
        down_split_layout(E_LOCAL, 256, H, 256, readers=1)
    # 2 cores per bank own 8 columns: phase 2 would need 8 fp32 DST tiles per acquire (half of the DEST holds 4)
    with pytest.raises(ValueError, match="fp32 DST tiles of a half"):  # allow-pytest.raises: host contract
        down_split_layout(E_LOCAL, 256, H, 64, readers=2)
    lay4 = down_split_layout(E_LOCAL, 256, H, 256, readers=4)
    assert (lay4["cols_own"], lay4["unit_cols"], lay4["incs_per_pass"]) == (4, 8, 1)
    # a slice whose unit is narrower than the owned width (I = 2048 at 4 cores per bank): several producers per pass
    lay = down_split_layout(9, 2048, H, 64, readers=4)
    assert (lay["cols_own"], lay["unit_cols"], lay["incs_per_pass"]) == (4, 2, 2)


def test_layout_refusals():
    with pytest.raises(ValueError, match="do not split"):  # allow-pytest.raises: host contract
        down_split_layout(E_LOCAL, 256, H, 256, readers=3)


# ------------------------------------------------------------------------------------------ the unit LPT
def _all_units(passes: list[int], groups: int, upe: int) -> dict[tuple[int, int], int]:
    owner = {}
    for g in range(groups):
        for u in down_split_units(passes, groups, g, upe):
            assert u not in owner, u
            owner[u] = g
    return owner


@pytest.mark.parametrize("seed", range(4))
def test_unit_lpt_partition_and_balance(seed):
    """Every (active expert, q) unit lands on exactly one of the bank's 8 cores; the loads (passes) are balanced to
    within the largest expert's pass count; the units of one core come in the global LPT order."""
    T = 256
    r = _topk_routing(T, E_LOCAL, 2, seed, popular=seed % 2)
    passes = _passes_of(r, max_passes(T, E_LOCAL))
    owner = _all_units(passes, 8, 2)
    active = [e for e, p in enumerate(passes) if p]
    assert set(owner) == {(e, q) for e in active for q in range(2)}
    load = [0] * 8
    for (e, _), g in owner.items():
        load[g] += passes[e]
    assert max(load) - min(load) <= max(passes), (load, max(passes))
    order = sorted(owner, key=lambda u: (-passes[u[0]], u[0], u[1]))
    for g in range(8):
        mine = down_split_units(passes, 8, g, 2)
        assert mine == [u for u in order if owner[u] == g]


def test_unit_lpt_upe1_is_the_expert_lpt():
    """``upe = 1`` is ``assign_group`` (the gate|up dispatch's whole-expert LPT, the pass-budget test's mirror)."""
    from models.demos.glm53_flash.tests.test_dispatch_pass_budget_host import _assign_group

    r = _topk_routing(128, E_LOCAL, 3, 11, popular=1)
    passes = _passes_of(r, max_passes(128, E_LOCAL))
    for g in range(12):
        assert [e for e, _ in down_split_units(passes, 12, g, 1)] == _assign_group(passes, 12, g)


HARNESS = r"""
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#define tt_l1_ptr
#include "dispatch_common.hpp"
int main(int argc, char** argv) {
    constexpr uint32_t E = HE;
    const uint32_t groups = static_cast<uint32_t>(atoi(argv[1]));
    const uint32_t group = static_cast<uint32_t>(atoi(argv[2]));
    uint16_t passes[E];
    for (uint32_t e = 0; e < E; ++e) {
        unsigned v = 0;
        if (scanf("%u", &v) != 1) return 2;
        passes[e] = static_cast<uint16_t>(v);
    }
    for_each_unit<E, HUPE, 32>(passes, groups, group, [&](uint32_t e, uint32_t q) { printf("%u %u\n", e, q); });
    return 0;
}
"""


def _build_unit_harness(tmp: Path, E: int, upe: int) -> Path | None:
    gxx = shutil.which("g++") or shutil.which("c++")
    if gxx is None:
        return None
    src = tmp / "unit_harness.cpp"
    src.write_text(HARNESS)
    exe = tmp / f"units_{E}_{upe}"
    subprocess.run(
        [gxx, "-std=c++17", "-O1", f"-I{KERNEL_DIR}", f"-DHE={E}", f"-DHUPE={upe}", str(src), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    return exe


@pytest.mark.parametrize("upe", [2, 4])
def test_cpp_for_each_unit_matches_mirror(tmp_path, upe):
    """``kernels/dispatch_common.hpp`` ``for_each_unit`` (the kernels' replay) compiled natively against
    ``down_split_units`` on random and adversarial pass counts, every core of a bank."""
    exe = _build_unit_harness(tmp_path, E_LOCAL, upe)
    if exe is None:
        pytest.skip("no host C++ compiler")
    g = torch.Generator().manual_seed(3)
    cases = [_passes_of(_topk_routing(256, E_LOCAL, 2, s, popular=s % 3), 136) for s in range(4)]
    cases += [torch.randint(0, 5, (E_LOCAL,), generator=g).tolist() for _ in range(3)]
    cases += [[0] * E_LOCAL, [1] * E_LOCAL, [0] * 71 + [40], [3] + [0] * 71]
    for passes in cases:
        for group in range(8):
            out = subprocess.run(
                [str(exe), "8", str(group)],
                input=" ".join(map(str, passes)),
                capture_output=True,
                check=True,
                text=True,
            ).stdout.split("\n")
            got = [tuple(int(v) for v in line.split()) for line in out if line.strip()]
            assert got == down_split_units(passes, 8, group, upe), (passes, group)


# ------------------------------------------------------------------------------------------ the slot table
def _cols_writer_slots(passes: list[tuple[int, list[int]]], T: int, J: int) -> tuple[dict, list[int]]:
    """Mirror of writer_down_dispatch.cpp phase 2: per output row tile the passes are scanned in order with a
    per-pass cursor; the row's slot is the token's running count; rows past J are dropped; j_max per tile."""
    slots, j_max = {}, []
    cursor = [0] * len(passes)
    for r in range(T // TILE):
        cnt = [0] * TILE
        jm = 0
        for k, (_, rows) in enumerate(passes):
            i = cursor[k]
            while i < len(rows) and rows[i] < (r + 1) * TILE:
                tl = rows[i] - r * TILE
                j = cnt[tl]
                cnt[tl] += 1
                if j < J:
                    jm = max(jm, j + 1)
                    slots[(k, i)] = (r, j, tl)
                i += 1
            cursor[k] = i
        j_max.append(jm)
    return slots, j_max


@pytest.mark.parametrize("T, picks, popular", [(64, 2, 0), (256, 2, 0), (256, 2, 2), (256, 8, 3)])
def test_slot_table_matches_cols_writer(T, picks, popular):
    """The experts writer's slot of every (pass, row) (``down_slot_table``: earlier passes holding the token) is the
    one the cols writer's cursor scan assigns, the same rows are dropped past J, and the used slot counts per row
    tile agree -- so both forms sum the same slots in the same order."""
    r = _topk_routing(T, E_LOCAL, picks, seed=T + picks, popular=popular)
    p_max = max_passes(T, E_LOCAL)
    passes = dispatch_passes(r, p_max=p_max)
    slots, j_max = down_slot_table(passes, T)
    ref, ref_jmax = _cols_writer_slots(passes, T, MAX_PICKS)
    assert j_max == ref_jmax
    got = {}
    for k, (_, rows) in enumerate(passes):
        for i, t in enumerate(rows):
            j = slots[k][i]
            if j < MAX_PICKS:
                got[(k, i)] = (t // TILE, j, t % TILE)
    assert got == ref
    # the kept contributions are the golden's (dispatch_contributions): the same drop rule
    _, kept = dispatch_contributions(r, p_max)
    kept_pairs = {(passes[k][0], passes[k][1][i]) for k, i in got}
    assert kept_pairs == {(int(e), int(t)) for t, e in torch.nonzero(kept).tolist()}
    if popular >= 3 and picks == 8:  # some token exceeds J slots: the cap is exercised
        assert any(j >= MAX_PICKS for ks in slots for j in ks)


def test_slot_table_over_budget_routing():
    """An over-budget routing (33 rows x 72 non-zeros at T = 128, I = 1024): the enumeration clamps to P and the
    slot table follows the clamped passes (the same rows the cols writer keeps)."""
    T = 128
    r = _topk_routing(T, E_LOCAL, 2, 5).float()
    r[T - 33 :, :] = 0.5
    r = r.bfloat16()
    p_max = max_passes(T, E_LOCAL)
    passes = dispatch_passes(r, p_max=p_max)
    assert len(passes) == p_max
    slots, j_max = down_slot_table(passes, T)
    ref, ref_jmax = _cols_writer_slots(passes, T, MAX_PICKS)
    assert j_max == ref_jmax and max(j_max) == MAX_PICKS
    assert sum(1 for ks in slots for j in ks if j < MAX_PICKS) == len(ref)


# ------------------------------------------------------------------------------------------ owner / column map
@pytest.mark.parametrize("readers, I", [(8, 256), (4, 256), (4, 2048), (8, 1024)])
def test_scatter_owner_map_and_arrival_count(readers, I):
    """For every unit (e, q) and its columns, the owner r = cc / cols_own and the owned column cc % cols_own cover
    every (owner, column) of the bank exactly once per pass, and the producers' per-owner pass counts (the writer's
    ``incr[o]``: the units whose columns touch owner o) sum to what the owner waits for (``n_pass x incs_per_pass``)."""
    E = E_LOCAL if I < 2048 else 9
    T = 256 if I < 2048 else 64
    lay = down_split_layout(E, I, H, T, readers=readers)
    C, cols_own, unit_cols, upe = lay["C"], lay["cols_own"], lay["unit_cols"], lay["upe"]
    r = _topk_routing(T, E, 2, seed=readers + I)
    passes = _passes_of(r, max_passes(T, E))
    n_pass = sum(passes)
    covered = {}
    incr = [[0] * readers for _ in range(readers)]  # [producer][owner]
    for g in range(readers):
        for e, q in down_split_units(passes, readers, g, upe):
            o_lo, o_hi = (q * unit_cols) // cols_own, ((q + 1) * unit_cols - 1) // cols_own
            for o in range(o_lo, o_hi + 1):
                incr[g][o] += passes[e]
            for c in range(unit_cols):
                cc = q * unit_cols + c
                key = (e, cc // cols_own, cc % cols_own)
                assert key not in covered, key
                covered[key] = g
    active = [e for e, p in enumerate(passes) if p]
    assert set(covered) == {(e, o, c) for e in active for o in range(readers) for c in range(cols_own)}
    assert len(covered) == len(active) * C
    for o in range(readers):
        assert sum(incr[g][o] for g in range(readers)) == n_pass * lay["incs_per_pass"]


# ------------------------------------------------------------------------------------------ knob + cost model
def test_down_split_knob():
    assert (
        P.down_split_mode({}) == "experts"
    )  # the default since 2026-11-03 (e2e 98.85 -> 87.18 ms at 32 users/row, exact)
    assert P.down_split_mode({P.DOWN_SPLIT_ENV: "experts"}) == "experts"
    with pytest.raises(ValueError, match="GLM53_STREAM_DOWN_SPLIT"):  # allow-pytest.raises: host contract
        P.down_split_mode({P.DOWN_SPLIT_ENV: "rows"})
    assert (
        P.DOWN_SPLIT_ENV == "GLM53_STREAM_DOWN_SPLIT" and os.environ.get(P.DOWN_SPLIT_ENV, "cols") in P.DOWN_SPLIT_MODES
    )


def test_cost_model_points():
    """The README's two points: ~37 active / 55 passes (the 8x4 profile's gate|up implies it) and 58 / 72 (the unit
    test routing): W2 bytes once, the stream at the block op's rate dominates the per-core compute, prelude + tail."""
    a = P.down_split_expected_us(37, 55, 256, H)
    b = P.down_split_expected_us(58, 72, 256, H)
    assert a["w2_bytes"] == 37 * 1024 * 1088 and b["w2_bytes"] == 58 * 1024 * 1088
    assert (a["unit_cols"], a["upe"]) == (8, 2)
    assert 120 < a["stream_us"] < 130 and 190 < b["stream_us"] < 200
    assert a["compute_us"] < a["stream_us"] and b["compute_us"] < b["stream_us"]
    assert 150 < a["total_us"] < 160 and 220 < b["total_us"] < 230, (a["total_us"], b["total_us"])
    assert a["act_bytes"] == 55 * 2 * 8 * 2048 * 8  # 1.8 MB per bank x 8 banks = 14 MB (56 MB in the cols form)
    assert a["bytes_per_core"] == a["w2_bytes"] / 64


# ------------------------------------------------------------------------------------------ the emulator
def _col_product(act_k: torch.Tensor, w2_e: torch.Tensor, col: int) -> torch.Tensor:
    """One output tile column of one pass: the same call (operands, shapes) in both emulations, so the bits agree."""
    return (act_k @ w2_e[:, col * TILE : (col + 1) * TILE]).bfloat16().float()


def _emulate_cols(act_pass, w2, passes, T, readers=8, banks=8, J=MAX_PICKS):
    """writer_down_dispatch.cpp + compute_down_dispatch.cpp: core (b, r) owns columns [b C + r cols_own, ..) of
    every row; h_k per pass for them; the cursor scan places rows into slots; identity sums in slot order."""
    H = w2.shape[-1]
    C = H // TILE // banks
    cols_own = C // readers
    RT = T // TILE
    y = torch.zeros(T, H)
    ref_slots, j_max = _cols_writer_slots(passes, T, J)
    for b in range(banks):
        for r in range(readers):
            col0 = b * C + r * cols_own
            slot = torch.zeros(RT, J, TILE, cols_own * TILE)
            for k, (e, rows) in enumerate(passes):
                h = torch.cat([_col_product(act_pass[k], w2[e], col0 + c) for c in range(cols_own)], dim=1)
                for i in range(len(rows)):
                    if (k, i) in ref_slots:
                        rt, j, tl = ref_slots[(k, i)]
                        slot[rt, j, tl] = h[i]
            for rt in range(RT):
                acc = torch.zeros(TILE, cols_own * TILE)
                for j in range(j_max[rt]):
                    acc = acc + slot[rt, j]
                y[rt * TILE : (rt + 1) * TILE, col0 * TILE : (col0 + cols_own) * TILE] = acc
    return y


def _emulate_experts(act_pass, w2, passes, T, readers=8, banks=8, J=MAX_PICKS):
    """The experts split: per bank the (expert, q) units LPT-assigned over the cores (down_split_units); the
    producing core computes h for the unit's columns and scatters row i of pass k into owner cc // cols_own's slot
    (t // 32, j = down_slot_table, t % 32, owned column cc % cols_own); every owner sums its slots in order."""
    H = w2.shape[-1]
    lay = down_split_layout(len(w2), act_pass.shape[-1], H, T, readers=readers, banks=banks)
    C, cols_own, unit_cols, upe = lay["C"], lay["cols_own"], lay["unit_cols"], lay["upe"]
    RT = T // TILE
    per_expert = [0] * len(w2)
    base = {}
    for k, (e, _) in enumerate(passes):
        base.setdefault(e, k)
        per_expert[e] += 1
    jtab, j_max = down_slot_table(passes, T, J)
    y = torch.zeros(T, H)
    for b in range(banks):
        slots = [torch.zeros(RT, J, TILE, cols_own * TILE) for _ in range(readers)]
        arrivals = [0] * readers
        for g in range(readers):
            for e, q in down_split_units(per_expert, readers, g, upe):
                for p in range(per_expert[e]):
                    k = base[e] + p
                    rows = passes[k][1]
                    h = torch.cat(
                        [_col_product(act_pass[k], w2[e], b * C + q * unit_cols + c) for c in range(unit_cols)], dim=1
                    )
                    for i, t in enumerate(rows):
                        j = jtab[k][i]
                        if j >= J:
                            continue
                        for c in range(unit_cols):
                            cc = q * unit_cols + c
                            slots[cc // cols_own][
                                t // TILE, j, t % TILE, (cc % cols_own) * TILE : (cc % cols_own + 1) * TILE
                            ] = h[i, c * TILE : (c + 1) * TILE]
                    for o in range((q * unit_cols) // cols_own, ((q + 1) * unit_cols - 1) // cols_own + 1):
                        arrivals[o] += 1
        for r in range(readers):
            assert arrivals[r] == len(passes) * lay["incs_per_pass"], (b, r, arrivals[r])  # the owner's wait count
            col0 = b * C + r * cols_own
            for rt in range(RT):
                acc = torch.zeros(TILE, cols_own * TILE)
                for j in range(j_max[rt]):
                    acc = acc + slots[r][rt, j]
                y[rt * TILE : (rt + 1) * TILE, col0 * TILE : (col0 + cols_own) * TILE] = acc
    return y


@pytest.mark.parametrize("T, picks, popular, I", [(128, 2, 1, 256), (128, 8, 2, 256), (64, 2, 1, 1024)])
def test_emulated_data_flow_matches_cols_form(T, picks, popular, I):
    """The experts split's bookkeeping end to end on random data: the same output bits as the cols form's flow
    (the same h rows reach the same (owner, slot, row, column) and the slot sums run in the same order), and both
    agree with the torch golden to the PCC bar."""
    from models.demos.glm53_flash.tests.utils import pcc

    E = 16
    g = torch.Generator().manual_seed(T + picks + I)
    r = _topk_routing(T, E, picks, seed=T + picks, popular=popular)
    passes = dispatch_passes(r, p_max=max_passes(T, E))
    w2 = (torch.randn(E, I, H, generator=g) * 0.02).bfloat16().float()
    act_pass = torch.zeros(len(passes), TILE, I)
    for k, (_, rows) in enumerate(passes):
        act_pass[k, : len(rows)] = (torch.randn(len(rows), I, generator=g) * 0.5).bfloat16().float()
    y_cols = _emulate_cols(act_pass, w2, passes, T)
    y_exp = _emulate_experts(act_pass, w2, passes, T)
    assert torch.equal(y_cols, y_exp), float((y_cols - y_exp).abs().max())
    # the golden: sum over the kept contributions (the J cap) of act @ W2
    _, kept = dispatch_contributions(r, max_passes(T, E))
    y_ref = torch.zeros(T, H)
    for k, (e, rows) in enumerate(passes):
        for i, t in enumerate(rows):
            if kept[t, e]:
                y_ref[t] += act_pass[k, i] @ w2[e]
    assert pcc(y_ref, y_cols) > 0.9999


# ------------------------------------------------------------------------------------------ kernel sources
def _src(name: str) -> str:
    return (KERNEL_DIR / f"{name}.cpp").read_text()


def test_kernel_sources_experts_split():
    """The three twins replay the same unit LPT (``for_each_unit<E, UPE, 32>`` with the pass counts, readers,
    reader in the reader and the writer; the compute takes the writer's metadata page), keep the exit contract, the
    writer's scatter addresses the owners' slots and signals after a write barrier, and the ``cols`` kernels carry
    no trace of the split (the default program is untouched)."""
    rd, wr, cp = (
        _src("reader_down_dispatch_experts"),
        _src("writer_down_dispatch_experts"),
        _src("compute_down_dispatch_experts"),
    )
    for name, src in (("reader", rd), ("writer", wr)):
        assert "for_each_unit<E, UPE, 32>(passes, readers, reader, [&](uint32_t e, uint32_t q) {" in src, name
        assert src.rstrip().endswith(
            "// exit contract: every NOC read / write / atomic acknowledged before the next kernel\n}"
        ), name
    assert "meta.read_tile_value(0, 1 + u)" in cp and "cb_wait_front(cb_w, unit_tiles);" in cp
    assert "matmul_tiles(cb_a, cb_w, k, c * seg_tiles + k, d);" in cp  # the cols compute's product order
    assert "matmul_tiles(cb_a, cb_w, k, k * unit_cols + c, d);" in cp  # the regions-sourced order
    # the DST half rule: dst_batch tiles per acquire (fill, the batch's columns, commit, pack, release), never
    # unit_cols fp32 tiles at once (the 2026-11-03 failure), and phase 2 within the half too
    assert "for (uint32_t c0 = 0; c0 < unit_cols; c0 += dst_batch) {" in cp
    assert "cb_reserve_back(cb_hs, dst_batch);" in cp and "cb_push_back(cb_hs, dst_batch);" in cp
    assert "static_assert(dst_batch >= 1 && dst_batch <= 4" in cp and "static_assert(cols_own <= 4" in cp
    assert "fill_tile(c, 0.0f)" not in cp.split("---- phase 2")[0]
    assert "my_x[" not in wr and "const uint64_t self_base = core_base[reader];" in wr
    assert "noc_async_write_barrier();\n    for (uint32_t o = 0; o < readers; ++o) {\n        if (incr[o] != 0) {" in wr
    assert (
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr), n_pass * incs_per_pass);" in wr
    )
    assert (
        "cb_pages_reservable_at_back(cb_a, seg_tiles)" in wr and "cb_pages_available_at_front(cb_hs, unit_cols)" in wr
    )
    assert "noc_async_writes_flushed();" in wr  # the staging tiles are read out before the packer reuses them
    assert re.search(r"noc_async_write\(src, dst, 32\);\s*noc_async_write\(src \+ 512, dst \+ 512, 32\);", wr)
    for name in ("reader_down_dispatch", "writer_down_dispatch", "compute_down_dispatch"):
        src = _src(name)
        assert "EXPERT" not in src and "for_each_unit" not in src and "cb_hs" not in src, name
