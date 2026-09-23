# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 / hop_aware_coread: which sticks each RISC-V co-reads, by DRAM-bank geometry.

HAC_CASES="4,5,3,8,w256"   LOOSE_CASES indices, or w<W>[_l1][_f32][_h<H>][_rs|_ol1] = [1,1,H(2048),W] interleaved -> DRAM (_rs: HEIGHT_SHARDED L1 out, _ol1: L1 out)
HAC_VARIANTS="head,pos,bal,inv"
  head    the op as is (kernels/, CO_READ_SEGMENT_BYTES gate as is)
  off     the op's kernels with co-read disabled (CO_READ_SHARE = 0)
  pos     kernels_mask, gate widened (DRAM unbounded), POSITIONAL mask == HEAD's split
  geo     kernels_mask, widened, BRISC reads exactly the sticks whose bank is NoC1-response-shorter
  geo_t<T>c<C>  geo with a margin (NoC1 path > T hops shorter) and a balance cap (|BRISC - 16| <= C sticks)
  geoinv  kernels_mask, widened, BRISC reads exactly the sticks whose bank is NoC0-response-shorter
  tb<K>   makespan split (hop-weighted per-RISC-V time model; K = hop weight x4; see step_mask)
  tl<K>   issue-imbalance vs total-link-load split (see step_mask) -- the unified candidate
  tp<K>_<P> tl<K> with a P-hop penalty per NoC1 read
  bk      whole banks per NoC, balanced (prefix of NoC1-favoured banks closest to co_read sticks)
  bkgeo   whole banks per NoC, only NoC1-favoured banks (bank-granular geo, capped at the balance point)
  bal     kernels_mask, widened, BRISC reads the co_read (16) sticks MOST NoC1-favoured (balanced)
  inv     kernels_mask, widened, BRISC reads the co_read (16) sticks LEAST NoC1-favoured (control)
  pos128  kernels_mask, gate as is, positional (the mask loop's own cost vs head)
  grad    THE GRADUATION CANDIDATE (graduate/: host + kernels exactly as the patches would land)
  N<mode> the same (balanced) split through kernels_list16 (CT count, list unpacked in registers)
  W<mode> the same split through kernels_listw (THE CANDIDATE: runtime count, packed words)
  S<mode> kernels_listsel: W's lists for 3 candidate physical rows, the kernel picks by NOC_NODE_ID
          (host needs no probe: geometry from the logical grid + WH's fixed column map)
  L<mode> the same split through kernels_list (explicit per-RISC-V stick list RT args, no mask loop)
Zones: TT_METAL_KERNEL_PERF_ZONES=1.
Each (case, variant) runs the op once (golden contract check on: bit-exact for bf16 -> bf16).
Opt-in: TILIZE_PERF_EXPERIMENTS=1.  Pair ns with perf_experiments/p2_breakdown/label_ns.py.
"""
import json
import os
import sys
from pathlib import Path

import pytest

import ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from eval.feature_matrix import cartesian
from eval.golden_tests.tilize import helpers
from eval.golden_tests.tilize.feature_spec import LOOSE_CASES, TARGET, _il, _sh, _DRAM, _L1, _G64, _ROW, _HEIGHT
from ttnn.operations.tilize import INPUT_TAGGERS  # type: ignore

pytestmark = pytest.mark.skipif(os.environ.get("TILIZE_PERF_EXPERIMENTS") != "1", reason="opt-in perf experiment")

EXP = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_coread"
CASES = os.environ.get("HAC_CASES", "4,5,3,8").split(",")
VARIANTS = os.environ.get("HAC_VARIANTS", "head,pos,bal,inv").split(",")
MASK_VARIANTS_FIXED = {"pos", "geoinv", "bal", "inv", "pos128"}


def _is_mask(v):
    v = v[1:] if v[0] in "LNWS" else v
    return v in MASK_VARIANTS_FIXED or v.startswith(("geo", "tb", "bk", "tl", "tp"))


# ---- geometry (WH B0 n150; measured by geom_probe/dump_geometry.py on this box) ----
_GEOM = json.load(open(EXP / "geom_probe/geometry.json"))
_BANKS = [tuple(b) for b in _GEOM["banks"]["noc0"]]  # physical NoC0 (x, y) of DRAM bank b
_GX, _GY = 10, 12  # WH NoC grid (torus)


# HAC_GEOM=approx: derive physical coords WITHOUT the probe, from the logical grid alone, assuming
# WH's fixed column map and NO harvested row above the grid (physical rows 1..5, 7..11 in order).
# On this box rows 5 and 11 are not in the grid, so logical rows 4..7 are off by 1..2 physical rows.
_XS = [1, 2, 3, 4, 6, 7, 8, 9]
_YS_NOMINAL = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
_APPROX = os.environ.get("HAC_GEOM", "exact") == "approx"


_FORCE_Y = {}  # kernels_listsel: (core.x, core.y) -> the candidate physical row being evaluated


def _phys(core):
    if (core.x, core.y) in _FORCE_Y:
        return _XS[core.x], _FORCE_Y[(core.x, core.y)]
    if _APPROX:
        return _XS[core.x], _YS_NOMINAL[core.y]
    return tuple(_GEOM["cores"][f"{core.x},{core.y}"]["phys0"])


def resp_hops(core, bank):
    """Hops of the READ RESPONSE (data) path bank -> core on NoC0 (east, south) and NoC1 (west, north).
    The request goes the other way round the same ring, so request + response is a full loop on
    either NoC (22 hops when x and y both differ): only the data path's length differs."""
    cx, cy = _phys(core)
    dx, dy = _BANKS[bank]
    return (cx - dx) % _GX + (cy - dy) % _GY, (dx - cx) % _GX + (dy - cy) % _GY


def step_mask(mode, core, row_start, rotation, co_read, tile_h=32, dram=True):
    """Bit s set = sequence step s (stick (s + rotation) mod tile_h of tile-row row_start) is BRISC's."""
    steps = list(range(tile_h))
    if mode in ("pos", "pos128") or not dram:
        chosen = steps[tile_h - co_read :]
    else:
        score = {}
        for s in steps:
            j = (s + rotation) & (tile_h - 1)
            h0, h1 = resp_hops(core, (row_start * tile_h + j) % len(_BANKS))
            score[s] = h0 - h1  # > 0: NoC1 (BRISC) has the shorter data path
        if mode.startswith("geo") and mode != "geoinv":
            # geo[_t<T>][c<C>]: BRISC takes the sticks whose NoC1 data path is > T hops shorter
            # (T = 0 by default), then the split is pulled to within C sticks of co_read (C = inf)
            # by moving the lowest-margin sticks across.
            spec = mode[3:].lstrip("_")
            t = int(spec[1:].split("c")[0]) if spec.startswith("t") else 0
            cap = int(spec.split("c")[1]) if "c" in spec else None
            chosen = [s for s in steps if score[s] > t]
            if cap is not None:
                order = sorted(steps, key=lambda s: (-score[s], s))
                n = min(max(len(chosen), co_read - cap), co_read + cap)
                chosen = order[:n]
        elif mode == "geoinv":
            chosen = [s for s in steps if score[s] < 0]
        elif mode.startswith("tb"):
            # tb<K>: makespan split. BRISC takes the k most NoC1-favoured sticks, k chosen to minimize
            # max(T_ncrisc, T_brisc), T_r = n_r * ISSUE + w * (sum of r's data-path hops), with the
            # per-hop weight w = K/4 * flits(segment) * active_cores / 64 (K = 4 -> w = 1 cycle per
            # 32-B flit-hop at full load). K -> 0 is `bal`, K -> inf is `geo`.
            w = int(mode[2:]) / 4 * ctx_seg["flits"] * ctx_seg["cores"] / 64
            order = sorted(steps, key=lambda s: (-score[s], s))
            h = {}
            for s in steps:
                h[s] = resp_hops(core, (row_start * tile_h + ((s + rotation) & (tile_h - 1))) % len(_BANKS))
            best = None
            for k in range(tile_h + 1):
                b, n = order[:k], order[k:]
                t1 = k * ISSUE + w * sum(h[s][1] for s in b)
                t0 = (tile_h - k) * ISSUE + w * sum(h[s][0] for s in n)
                cost = (max(t0, t1), abs(k - co_read))
                if best is None or cost < best[0]:
                    best = (cost, k)
            chosen = order[: best[1]]
        elif mode.startswith("tp"):
            # tp<K>_<P>: tl<K> plus a fixed NoC1 penalty of P hops per BRISC read (measured: a stick
            # with EQUAL data-path hops is cheaper on NoC0 -- case 8 tl4 vs geo, +13 %). Ties go to NoC0.
            K, P = (int(x) for x in mode[2:].split("_"))
            w = K / 4 * ctx_seg["flits"] * ctx_seg["cores"] / 64
            order = sorted(steps, key=lambda s: (-score[s], s))
            h = {
                s: resp_hops(core, (row_start * tile_h + ((s + rotation) & (tile_h - 1))) % len(_BANKS)) for s in steps
            }
            best = None
            for k in range(tile_h + 1):
                total = sum(h[s][1] + P for s in order[:k]) + sum(h[s][0] for s in order[k:])
                cost = (max(k, tile_h - k) * ISSUE + w * total, abs(k - co_read))
                if best is None or cost < best[0]:
                    best = (cost, k)
            chosen = order[: best[1]]
        elif mode.startswith("tl"):
            # tl<K>: issue imbalance vs total link load. BRISC takes the k most NoC1-favoured sticks,
            # k minimizing max(n_ncrisc, n_brisc) * ISSUE + w * (total data-path hops of the tile-row)
            # -- the issue chain is per RISC-V, the link load is shared by both -- with the per-hop
            # weight w = K/4 * flits(segment) * active_cores / 64. K -> 0: bal; K -> inf: geo.
            w = int(mode[2:]) / 4 * ctx_seg["flits"] * ctx_seg["cores"] / 64
            order = sorted(steps, key=lambda s: (-score[s], s))
            h = {
                s: resp_hops(core, (row_start * tile_h + ((s + rotation) & (tile_h - 1))) % len(_BANKS)) for s in steps
            }
            best = None
            for k in range(tile_h + 1):
                total = sum(h[s][1] for s in order[:k]) + sum(h[s][0] for s in order[k:])
                cost = (max(k, tile_h - k) * ISSUE + w * total, abs(k - co_read))
                if best is None or cost < best[0]:
                    best = (cost, k)
            chosen = order[: best[1]]
        elif mode.startswith("bk"):
            # bk: whole banks per NoC. Banks sorted by NoC1 preference; BRISC takes the prefix of
            # banks whose stick count is closest to co_read (ties: the smaller prefix). No bank is
            # read by both RISC-Vs of a core. bkgeo: stop the prefix at the first NoC0-favoured bank.
            bank_of = {s: (row_start * tile_h + ((s + rotation) & (tile_h - 1))) % len(_BANKS) for s in steps}
            banks = sorted(set(bank_of.values()), key=lambda b: (-next(score[s] for s in steps if bank_of[s] == b), b))
            best, acc, chosen_banks = (abs(0 - co_read), 0), 0, []
            prefix = []
            for b in banks:
                if mode == "bkgeo" and next(score[s] for s in steps if bank_of[s] == b) <= 0:
                    break
                prefix.append(b)
                acc += sum(1 for s in steps if bank_of[s] == b)
                if abs(acc - co_read) < best[0]:
                    best, chosen_banks = (abs(acc - co_read), acc), list(prefix)
            chosen = [s for s in steps if bank_of[s] in chosen_banks]
        elif mode == "bal":
            chosen = sorted(steps, key=lambda s: (-score[s], s))[:co_read]
        elif mode == "inv":
            chosen = sorted(steps, key=lambda s: (score[s], s))[:co_read]
        else:
            raise ValueError(mode)
    m = 0
    for s in chosen:
        m |= 1 << s
    return m


STATS = {}
ISSUE = 45  # cycles per stick read issued (NCRISC reader_issue zone: ~708 cycles / 16 reads)
ctx_seg = {"flits": 1, "cores": 64}


def _install(monkeypatch, mode, rewrite=True, listed=False, selected=False):
    tilize_mod = sys.modules["ttnn.operations.tilize.tilize"]
    orig_cpd = tilize_mod.create_program_descriptor
    ctx = {}

    def cpd(input_tensor, output_tensor, **kw):
        ctx["dram"] = input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        ctx["elem_bytes"] = input_tensor.element_size()
        return orig_cpd(input_tensor, output_tensor, **kw)

    monkeypatch.setattr(tilize_mod, "create_program_descriptor", cpd)
    orig_kd = ttnn.KernelDescriptor

    def kd(*a, **kw):
        src = str(kw.get("kernel_source", ""))
        ct = list(kw.get("compile_time_args", []))
        is_reader, is_writer = src.endswith("tilize_reader.cpp"), src.endswith("tilize_writer.cpp")
        if is_reader or is_writer:
            co_read = ct[28] if is_reader else ct[18]
            STATS["co_read"] = co_read
            if any(d[0] == "CO_READ_LISTED" for d in kw.get("defines", [])):
                STATS["listed"] = True  # grad: the geometric stick lists were sent
            if co_read and rewrite:
                entries = kw["runtime_args"].to_list()
                # reader RT: core_col_tiles at 4 (one column block per core on a one-position walk)
                seg_bytes = min(list(e[1])[4] for e in entries) * 32 * ctx["elem_bytes"]
                ctx_seg["flits"] = max(1, seg_bytes // 32)
                ctx_seg["cores"] = len(entries)
                STATS["seg_bytes"] = seg_bytes
                new = ttnn.RuntimeArgs()
                n_brisc, hop_sum, split_banks = [], [0, 0], []
                for core, args in entries:
                    args = list(args)
                    row_start = args[1]
                    rot = (args[6] if is_reader else args[8]) & 31
                    m = step_mask(mode, core, row_start, rot, co_read, dram=ctx["dram"])
                    if selected:
                        # kernels_listsel: K candidate physical rows (no harvested row assumed ->
                        # nominal[y], one or two harvested rows above -> nominal[y + 1], [y + 2]);
                        # the kernel picks by its NOC_NODE_ID. Blocks: [n, 8 packed words].
                        cands = _YS_NOMINAL[core.y : core.y + 3]
                        blocks = []
                        for cy in cands:
                            _FORCE_Y[(core.x, core.y)] = cy
                            true_y = tuple(_GEOM["cores"][f"{core.x},{core.y}"]["phys0"])[1]
                            # HAC_SEL_PROBE=1 (selection check): non-matching candidates get the INVERTED
                            # split, so a wrong pick shows up as the inv slowdown (+25..80 %)
                            cmode = "inv" if os.environ.get("HAC_SEL_PROBE") == "1" and cy != true_y else mode
                            mc = step_mask(cmode, core, row_start, rot, co_read, dram=ctx["dram"])
                            mine = [(s + rot) & 31 for s in range(32) if ((mc >> s) & 1) == int(is_writer)]
                            packed = [0] * 8
                            for i, j in enumerate(mine):
                                packed[i // 4] |= j << (8 * (i % 4))
                            blocks += [len(mine)] + packed
                            if cy == tuple(_GEOM["cores"][f"{core.x},{core.y}"]["phys0"])[1]:
                                m = mc  # stats: the list the kernel will select on this box
                        _FORCE_Y.pop((core.x, core.y))
                        new[core.x][core.y] = args + [len(cands)] + list(cands) + blocks
                    elif listed:
                        # kernels_list: this RISC-V's own sticks, in step order, one per byte
                        mine = [(s + rot) & 31 for s in range(32) if ((m >> s) & 1) == int(is_writer)]
                        packed = [0] * ((len(mine) + 3) // 4)
                        for i, j in enumerate(mine):
                            packed[i // 4] |= j << (8 * (i % 4))
                        # kernels_list16 (N<mode>) compiles the count in: balanced splits only
                        assert str(kw["kernel_source"]).find("kernels_list16") < 0 or len(mine) == (
                            co_read if is_writer else 32 - co_read
                        ), "N<mode> needs a balanced split"
                        new[core.x][core.y] = args + [len(mine)] + packed
                    else:
                        new[core.x][core.y] = args + [m]
                    n_brisc.append(bin(m).count("1"))
                    by_bank = {}
                    for s_ in range(32):
                        by_bank.setdefault((row_start * 32 + ((s_ + rot) & 31)) % 12, set()).add((m >> s_) & 1)
                    split_banks.append(sum(1 for v in by_bank.values() if len(v) == 2))
                    for s in range(32):
                        h0, h1 = resp_hops(core, (row_start * 32 + ((s + rot) & 31)) % 12)
                        hop_sum[0 if not (m >> s) & 1 else 1] += h0 if not (m >> s) & 1 else h1
                kw["runtime_args"] = new
                STATS["brisc_sticks"] = (min(n_brisc), max(n_brisc), sum(n_brisc) / len(n_brisc))
                STATS["mean_resp_hops"] = round(sum(hop_sum) / (32 * len(n_brisc)), 2)
                STATS["split_banks"] = round(sum(split_banks) / len(split_banks), 2)
        return orig_kd(*a, **kw)

    monkeypatch.setattr(ttnn, "KernelDescriptor", kd)


def _case(c):
    if c.startswith("w"):
        # w<W>[_l1][_f32][_h<H>]: [1,1,H,W] (H = 2048), interleaved; _l1 = L1 input; _f32 = float32 in/out
        parts = c.split("_")
        W = int(parts[0][1:])
        H = next((int(p[1:]) for p in parts[1:] if p.startswith("h")), 2048)
        dt = ttnn.float32 if "f32" in parts else ttnn.bfloat16
        src = _il(_L1) if "l1" in parts else _il(_DRAM)
        # _rs: output HEIGHT_SHARDED L1 on the 8x8 grid (resident: compute packs into the shard, no
        # NoC writes); _ol1: output interleaved L1 (tile writes on NoC1 to L1 banks)
        if "rs" in parts:
            out, api = _sh(_L1, _G64, (H // 64, W), _ROW, _HEIGHT), "legacy_2d"
        else:
            out, api = (_il(_L1) if "ol1" in parts else _il(_DRAM)), "none"
        scen = {"input_shape": [1, 1, H, W], "shard_api": api, "in": src, "out": out}
        th = next((int(p[1:]) for p in parts[1:] if p.startswith("t") and p[1:].isdigit()), None)
        if th:
            scen["tile_height"] = th  # _t<N>: output tile height N (tiny tiles)
        return {"inputs": (scen,), "dtype": dt, "output_dtype": dt}
    return LOOSE_CASES[int(c)]


def _axes(case):
    inputs = case["inputs"]
    dt = case.get("dtype", ttnn.bfloat16)
    odt = case.get("output_dtype", dt)
    return next(a for a in cartesian(TARGET, INPUT_TAGGERS, inputs) if a["dtype"] == dt and a["output_dtype"] == odt)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case_id", CASES)
def test_hop_aware_coread(device, monkeypatch, case_id, variant):
    device.clear_program_cache()
    STATS.clear()
    if variant == "off":
        monkeypatch.setattr(pd, "CO_READ_SHARE", 0.0)
    if _is_mask(variant):
        listed = variant[0] in "LNWS"
        mode = variant[1:] if listed else variant
        kdir = {"L": "kernels_list", "N": "kernels_list16", "W": "kernels_listw", "S": "kernels_listsel"}.get(
            variant[0], "kernels_mask"
        )
        monkeypatch.setattr(pd, "KERNEL_DIR", EXP / kdir)
        if mode != "pos128":
            monkeypatch.setattr(
                pd, "CO_READ_SEGMENT_BYTES", {ttnn.BufferType.DRAM: (0, None), ttnn.BufferType.L1: (0, None)}
            )
        _install(monkeypatch, mode, listed=listed, selected=variant[0] == "S")
    else:
        if variant == "grad":
            # the graduation candidate: graduate/tilize_program_descriptor.py + graduate/kernels
            import importlib.util

            if "tilize_pd_graduate" not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    "tilize_pd_graduate", EXP / "graduate/tilize_program_descriptor.py"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules["tilize_pd_graduate"] = mod
            monkeypatch.setattr(
                sys.modules["ttnn.operations.tilize.tilize"],
                "create_program_descriptor",
                sys.modules["tilize_pd_graduate"].create_program_descriptor,
            )
        _install(monkeypatch, "pos", rewrite=False)  # records co_read only; RT args untouched

    case = _case(case_id)
    helpers.run_tilize(case["inputs"], device=device, extras=case.get("extras"), **_axes(case))
    ttnn.synchronize_device(device)
    print(f"P2 case={case_id} variant={variant} done {STATS} geom={'approx' if _APPROX else 'exact'}")
