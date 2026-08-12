# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""I10 (combine_skew): CONTRIBUTOR SKEW of the cross-core combine.

The op's hidden split inside a reduction group is a ceil/floor ragged split
(`rms_norm_program_descriptor.create_program_descriptor`):

    w_floor = tensor_w_tiles // G ; w_rem = tensor_w_tiles % G
    slot < w_rem  ->  core_w = w_floor + 1     (the STRAGGLERS)
    slot >= w_rem ->  core_w = w_floor

so on the decode focus shape (224 hidden tiles, G = 110) four cores carry 3 tiles
and 106 carry 2, and the four stragglers are always slots 0..3 — i.e. the FIRST
four cores of grid row 0. The combine cannot start until the last partial lands, so
this bench asks two things:

  (a) does the choice of G (equal-vs-ragged C) matter — swept via the descriptor's
      MIN/MAX_W_GROUP_SIZE knobs, exactly as test_rms_norm_perf does;
  (b) does WHERE the ragged tiles sit matter — the straggler PLACEMENT.

ISOLATION. There is no new kernel and the op source is never edited. The candidate
placements are installed by taking `inspect.getsource(create_program_descriptor)`,
textually swapping ONLY the four-line slot->(core_w, w_start) block for a call into
a placement function, exec'ing the result against the descriptor module's globals,
and binding it over `ttnn.operations.rms_norm.rms_norm.create_program_descriptor`
for the duration of one dispatch. Everything else in the program build — CB sizes,
combine tree, mcast wiring, runtime args — is byte-identical to the op's, which is
what makes the measured delta attributable to the placement alone.

Placements (all produce a contiguous hidden cover, same max core_w, so `C` and
every CB size are unchanged):
  front   the op today — extras on slots 0..w_rem-1
  back    extras on the LAST slots, i.e. the group ROOT (members[-1]) and its
          neighbours: their extra tile overlaps the rendezvous they are waiting on
  rowend  extras on the level-1 row LEADERS (slot % gc == gc-1), one per grid row
  byrow   extras on the FIRST slot of each grid row — spread over rows, non-leader
  spread  extras evenly spaced over the slot order

Precision contract PINNED and never a lever: bf16 / TILE / gamma bf16 TILE /
fp32_dest_acc_en=False / MathFidelity.HiFi2. Every point is PCC-gated against a
torch fp32 reference at the focus case's soft threshold 0.9995.

Run:

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 3600 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    import sys; sys.path.insert(0, "<this dir>")
    import bench; bench.main()
    EOF

MEASURED — blackhole_p150b @1350MHz, DEVICE KERNEL DURATION ns, median of 3 fresh
runs (single run for the one-off rows). Every point pcc >= 0.99998.

  case            wt   G   floor/rem   front(=op)  byrow   spread  rowend
  decode7168      224 110    2 / 4         8014     8067    7774     7883
  decode5120      160 110    1 / 50        6919     6973    6743     7127
  decode2304       72  55    1 / 17        6110     6273    6086     6281
  decode1024       32  11    2 / 10        5067     5070    5017     5047
  rows128_5120    160  55    2 / 50       13525    13277   13419    13486   (1 run)
  prefill1024      32   2   16 / 0        89251    90056   89479    89338   (1 run, rem=0)

`spread` (extras evenly spaced over the slot order, i.e. over the whole group box
instead of packed into grid row 0) is strictly below EVERY front run at
decode7168 and decode5120 — 3/3 reps each — for 3.0% / 2.5%; flat elsewhere and
never a regression.

LEVER (a) — pick a G with an EQUAL C — is REJECTED by measurement. On an 11x10
grid G ∈ {1,2,5,10,11,22,55,110}, and 224 hidden tiles is divisible only by 2:
  G=110 front 8230 / spread 8111 ; G=55 8790 / 8419 ; G=22 8763 / 8923 ;
  G=2 (the only exact-equal C=112) 27511 / 27392 — 3.4x WORSE.
The lost parallelism dwarfs the straggler, so the op's occupancy-first pick stands.
"""

from __future__ import annotations

import contextlib
import inspect
import textwrap
import time

import torch
import ttnn

import importlib

import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

# `ttnn.operations.rms_norm.rms_norm` the ATTRIBUTE is the registered op function
# (it shadows the submodule of the same name), so the module itself comes from
# sys.modules via import_module.
rms_mod = importlib.import_module("ttnn.operations.rms_norm.rms_norm")
from ttnn.operations.rms_norm import rms_norm as rms_op

PCC_GATE = 0.9995

# --------------------------------------------------------------------------
# placement functions: (G, w_floor, w_rem, gc, gr) -> list of G widths
# --------------------------------------------------------------------------


def _apply(G, w_floor, w_rem, slots):
    widths = [w_floor] * G
    for s in slots:
        widths[s] += 1
    return widths


def place_front(G, w_floor, w_rem, gc, gr):
    return _apply(G, w_floor, w_rem, range(w_rem))


def place_back(G, w_floor, w_rem, gc, gr):
    return _apply(G, w_floor, w_rem, range(G - w_rem, G))


def place_rowend(G, w_floor, w_rem, gc, gr):
    """Leaders first (one extra each), then the remaining extras from the back."""
    leaders = [r * gc + gc - 1 for r in range(gr)]
    chosen = leaders[:w_rem]
    if len(chosen) < w_rem:
        rest = [s for s in range(G - 1, -1, -1) if s not in set(chosen)]
        chosen += rest[: w_rem - len(chosen)]
    return _apply(G, w_floor, w_rem, chosen)


def place_byrow(G, w_floor, w_rem, gc, gr):
    """Round-robin over grid ROWS, skipping the row leaders."""
    chosen, col = [], 0
    while len(chosen) < w_rem and col < gc:
        for r in range(gr):
            if len(chosen) == w_rem:
                break
            s = r * gc + col
            if gc > 1 and s % gc == gc - 1:
                continue
            chosen.append(s)
        col += 1
    if len(chosen) < w_rem:  # gc == 1, or every non-leader slot already taken
        chosen += [s for s in range(G) if s not in set(chosen)][: w_rem - len(chosen)]
    return _apply(G, w_floor, w_rem, chosen)


def place_spread(G, w_floor, w_rem, gc, gr):
    if w_rem == 0:
        return [w_floor] * G
    step = G / w_rem
    return _apply(G, w_floor, w_rem, sorted({min(G - 1, int(i * step)) for i in range(w_rem)}))


PLACEMENTS = {
    "front": place_front,
    "back": place_back,
    "rowend": place_rowend,
    "byrow": place_byrow,
    "spread": place_spread,
}

# the live placement, read by the patched descriptor
ACTIVE = [place_front]

_ORIG_BLOCK = """            for slot in range(G):
                if slot < w_rem:
                    core_w, w_start = w_floor + 1, slot * (w_floor + 1)
                else:
                    core_w, w_start = w_floor, w_rem * (w_floor + 1) + (slot - w_rem) * w_floor
"""

_NEW_BLOCK = """            _widths = _PLACE[0](G, w_floor, w_rem, w_group_cols, w_group_rows)
            assert len(_widths) == G and sum(_widths) == geo.tensor_w_tiles, (
                f"placement produced {_widths} for G={G} w_tiles={geo.tensor_w_tiles}")
            assert max(_widths) <= C, f"placement widened a core past C={C}"
            _starts, _acc = [], 0
            for _w in _widths:
                _starts.append(_acc)
                _acc += _w
            for slot in range(G):
                core_w, w_start = _widths[slot], _starts[slot]
"""


def _build_patched():
    src = textwrap.dedent(inspect.getsource(pd.create_program_descriptor))
    # the source is at module top level already (no dedent needed), but the block
    # match is indentation-sensitive, so assert we found it.
    assert _ORIG_BLOCK in src, "the ragged-split block moved; re-sync this bench with the op"
    src = src.replace(_ORIG_BLOCK, _NEW_BLOCK)
    g = dict(pd.__dict__)
    g["_PLACE"] = ACTIVE
    exec(compile(src, "<combine_skew patched descriptor>", "exec"), g)
    return g["create_program_descriptor"]


PATCHED = None


@contextlib.contextmanager
def placement(name):
    """Install a straggler placement for the duration of one dispatch."""
    global PATCHED
    if name is None:
        yield
        return
    if PATCHED is None:
        PATCHED = _build_patched()
    orig = rms_mod.create_program_descriptor
    ACTIVE[0] = PLACEMENTS[name]
    rms_mod.create_program_descriptor = PATCHED
    try:
        yield
    finally:
        rms_mod.create_program_descriptor = orig
        ACTIVE[0] = place_front


@contextlib.contextmanager
def knobs(**kw):
    old = {k: getattr(pd, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(pd, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(pd, k, v)


def pin_g(G):
    if G is None:
        return contextlib.nullcontext()
    return knobs(MIN_W_GROUP_SIZE=G, MAX_W_GROUP_SIZE=G)


# --------------------------------------------------------------------------
# selection recorder (so every measured row carries the geometry it ran)
# --------------------------------------------------------------------------
SEL = {}


def install_recorder():
    orig = pd._select_regime
    if getattr(orig, "_is_recorder", False):
        return

    def rec(geo, grid_x, grid_y, is_rm_out, budget):
        gc, gr, C, R = orig(geo, grid_x, grid_y, is_rm_out, budget)
        G = gc * gr
        s1, s2 = pd._tree_for_box(G, gc, gr)
        SEL.clear()
        SEL.update(
            G=G,
            gc=gc,
            gr=gr,
            C=C,
            R=R,
            s2=s2,
            wt=geo.tensor_w_tiles,
            rem=geo.tensor_w_tiles % G,
            floor=geo.tensor_w_tiles // G,
        )
        return gc, gr, C, R

    rec._is_recorder = True
    pd._select_regime = rec


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
CASES = {
    # name: shape                      (all INTERLEAVED, TILE, bf16)
    "decode7168": (1, 1, 32, 7168),  # focus: 224 tiles, G=110 -> 2/3 split, rem=4
    "decode5120": (1, 1, 32, 5120),  # 160 tiles, rem=50 -> half the cores doubled
    "decode2304": (1, 1, 32, 2304),  # 72 tiles
    "decode1024": (1, 1, 32, 1024),  # 32 tiles, small G
    "rows128_5120": (1, 1, 128, 5120),  # 4 tile-rows: occupancy key still choosing
    "prefill1024": (1, 1, 8192, 1024),  # G=2, rem=0 -> placement is a provable no-op
}


def make_case(device, name):
    shape = CASES[name]
    hidden = shape[-1]
    torch.manual_seed(0)
    ti = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tg = torch.randn((1, 1, 1, hidden), dtype=torch.float32).to(torch.bfloat16)
    x = ti.to(torch.float32)
    ref = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * tg.to(torch.float32)
    return dict(
        name=name,
        tt_input=ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        tt_gamma=ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ref=ref,
    )


def pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    n = a.numel()
    cov = float((a * b).sum()) - float(a.sum()) * float(b.sum()) / n
    va = float((a * a).sum()) - float(a.sum()) ** 2 / n
    vb = float((b * b).sum()) - float(b.sum()) ** 2 / n
    d = (va * vb) ** 0.5
    return 1.0 if d == 0 else cov / d


def run_point(device, case, check=True):
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    ttnn.ReadDeviceProfiler(device)
    out = rms_op(case["tt_input"], gamma=case["tt_gamma"], epsilon=1e-6, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    ns = None
    for programs in per_chip.values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get("DEVICE KERNEL DURATION [ns]")
            if entry is None:
                continue
            d = float(entry.duration)
            ns = d if ns is None else max(ns, d)
    p = pcc(ttnn.to_torch(out), case["ref"]) if check else None
    del out
    return ns, p


def main(names=None, places=None, gs=(None,), reps=1):
    install_recorder()
    names = list(names or ["decode7168", "decode5120", "decode2304", "decode1024", "rows128_5120", "prefill1024"])
    places = list(places or ["front", "back", "rowend", "byrow", "spread"])
    device = ttnn.open_device(device_id=0)
    try:
        for name in names:
            case = make_case(device, name)
            for G in gs:
                for rep in range(reps):
                    for pl in places:
                        with pin_g(G), placement(pl):
                            ns, p = run_point(device, case)
                        tag = "OK " if p >= PCC_GATE else "BAD"
                        print(
                            f"SKEW {name:12s} G_req={str(G):>4s} place={pl:7s} "
                            f"G={SEL.get('G')} gc={SEL.get('gc')} gr={SEL.get('gr')} C={SEL.get('C')} "
                            f"R={SEL.get('R')} s2={SEL.get('s2')} wt={SEL.get('wt')} "
                            f"floor={SEL.get('floor')} rem={SEL.get('rem')} "
                            f"ns={ns:>9.0f} pcc={p:.6f} {tag}",
                            flush=True,
                        )
                        assert p >= PCC_GATE, f"{name}/{pl}: pcc {p} < {PCC_GATE}"
                        time.sleep(0.15)
            del case
    finally:
        ttnn.close_device(device)
