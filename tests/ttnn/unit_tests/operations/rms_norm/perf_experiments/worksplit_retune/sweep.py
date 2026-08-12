# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""I7 (worksplit_retune): sweep the reduction-group size G of rms_norm's work split.

ISOLATED BAKE-OFF, host-side geometry variant. There is NO new kernel here: the
"variants" are the block geometries the host's `_select_regime` can already emit,
forced one at a time by monkeypatching the descriptor's module constants
(`MIN_W_GROUP_SIZE` / `MAX_W_GROUP_SIZE`) exactly the way
`test_rms_norm_perf.py::test_rms_norm_perf_wgroup_min` does. The op source is
never edited.

  baseline variant  = whatever `_select_regime` picks today (score = occupancy
                      first), i.e. G = 110 on the decode focus shape.
  candidate variants= every other achievable G on this grid.

Measurement follows `perf_zone_harness.py` (the known-good in-process profiler
recipe on this box): flush ReadDeviceProfiler -> dispatch -> synchronize ->
ReadDeviceProfiler -> get_latest_programs_perf_data(), reading
"DEVICE KERNEL DURATION [ns]". ONE fresh run per swept point.

Correctness is gated at EVERY swept point (PCC vs a torch fp32 reference,
threshold = the focus case's soft gate 0.9995). The precision contract is pinned
and never a lever: bf16 / TILE / gamma bf16 TILE / fp32_dest_acc_en=False /
MathFidelity.HiFi2.

Run (env vars must precede device init, hence on the command line):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 1800 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    import sys; sys.path.insert(0, "<this dir>")
    import sweep
    sweep.main(["decode7168"], gs=sweep.G_ALL)
    EOF
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time

import torch
import ttnn

import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm import rms_norm

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
_HARNESS_DIR = os.path.join(_REPO, "tests", "ttnn", "unit_tests", "operations", "rms_norm")
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)  # `eval.sharding`, imported by the harness's shard cases

import perf_zone_harness as H  # noqa: E402  (the known-good measurement recipe + CASES)

_ML = ttnn.TensorMemoryLayout

# Every reduction-group size an 11x10 compute grid can express: G = gc*gr over the
# divisor pairs. On this grid G uniquely determines (gc, gr), so pinning G pins the
# combine tree too:  G<=11 or gr==1 -> flat (s2=1); 22/55/110 -> two-stage (11, gr).
G_ALL = [1, 2, 5, 10, 11, 22, 55, 110]

PCC_GATE = 0.9995

ZONE_DIR = os.path.join(HERE, "zones")

# ---------------------------------------------------------------------------
# selection recorder: wrap `_select_regime` so every measured point carries the
# geometry that produced it (and so a pinned point that FELL BACK is visible).
# ---------------------------------------------------------------------------
SEL = {}

# `None` = the op's own `_select_regime`; set to a callable to measure a CANDIDATE
# SELECTION RULE instead (see `rule_two_stage_gain`). Lives beside the recorder so
# the two compose (the recorder always reports whichever rule ran).
RULE = [None]


def install_recorder():
    orig = pd._select_regime
    if getattr(orig, "_is_recorder", False):
        return

    def rec(geo, grid_x, grid_y, is_rm_out, budget):
        impl = RULE[0] or orig
        gc, gr, C, R = impl(geo, grid_x, grid_y, is_rm_out, budget)
        G = gc * gr
        num_groups = (grid_x // gc) * (grid_y // gr)
        active_groups = min(geo.tensor_row_tiles, num_groups)
        core_row_tiles = -(-geo.tensor_row_tiles // active_groups)
        s1, s2 = pd._tree_for_box(G, gc, gr)
        SEL.clear()
        SEL.update(
            G=G,
            gc=gc,
            gr=gr,
            C=C,
            R=R,
            s1=s1,
            s2=s2,
            nblocks=-(-core_row_tiles // R),
            core_row_tiles=core_row_tiles,
            active_cores=active_groups * G,
            max_tiles=core_row_tiles * C,
        )
        return gc, gr, C, R

    rec._is_recorder = True
    rec._orig = orig
    pd._select_regime = rec


@contextlib.contextmanager
def knobs(**kw):
    """Temporarily set descriptor module constants (never edit them)."""
    old = {k: getattr(pd, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(pd, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(pd, k, v)


# ---------------------------------------------------------------------------
# CANDIDATE SELECTION RULE (the thing that would be graduated into the op)
# ---------------------------------------------------------------------------
# A geometry predicate, not a shape whitelist. Adding a SECOND combine LEVEL
# (`stage2_span > 1`) costs one extra grid-wide rendezvous round; the only thing it
# buys is a smaller per-core tile count. So: pay for the second level only when the
# tiles it saves the busiest core are worth it. Measured turn on the decode profile
# (see the sweep table) — 1 tile saved LOSES 6.7%, 2 tiles LOSES, 5 tiles is FLAT,
# 7+ tiles WINS 4-13%. The threshold sits in the flat zone.
MIN_TWO_STAGE_TILE_GAIN = 4


def _s2_of(c):
    gc, gr = c[1][0], c[1][1]
    return pd._tree_for_box(gc * gr, gc, gr)[1]


def _admissible_by_tree_level(candidates):
    """Drop the two-stage-combine candidates when the extra LEVEL buys too few tiles.

    A no-op unless a two-stage candidate would actually WIN the score — which only
    happens when the occupancy key is choosing, i.e. when `tensor_row_tiles` is too
    small to fill the grid on the row axis alone (the decode regime). Every geometry
    whose winner is already flat (all four prefill shapes, every wide/ragged shape)
    is byte-identical.
    """
    if not candidates:
        return candidates
    flat = [c for c in candidates if _s2_of(c) == 1]
    two = [c for c in candidates if _s2_of(c) > 1]
    if not flat or not two:
        return candidates
    best_flat = max(flat, key=lambda c: c[0])
    best_two = max(two, key=lambda c: c[0])
    if best_two[0] <= best_flat[0]:
        return candidates  # flat already wins the score; nothing to arbitrate
    if best_flat[2] - best_two[2] < MIN_TWO_STAGE_TILE_GAIN:
        return flat
    return candidates


def rule_two_stage_gain(geo, grid_x, grid_y, is_rm_out, budget):
    """`_select_regime` + the tree-level band. Same body as the op's, one filter more."""
    candidates = pd._select_candidates(geo, grid_x, grid_y, is_rm_out, budget, pd.MAX_W_GROUP_SIZE, pd.MIN_W_GROUP_SIZE)
    if not candidates and (pd.MAX_W_GROUP_SIZE or pd.MIN_W_GROUP_SIZE > 1):
        candidates = pd._select_candidates(geo, grid_x, grid_y, is_rm_out, budget, 0, 1)
    candidates = pd._admissible_by_balance(candidates)
    candidates = _admissible_by_tree_level(candidates)
    if not candidates:
        raise RuntimeError(f"rms_norm: no work split fits L1 for shape {tuple(geo.shape)}")
    return max(candidates, key=lambda c: c[0])[1]


def pin_g(G):
    """Force the reduction-group size to exactly `G` (or fall back if infeasible)."""
    if G is None:
        return contextlib.nullcontext()
    return knobs(MIN_W_GROUP_SIZE=G, MAX_W_GROUP_SIZE=G)


# ---------------------------------------------------------------------------
# tensors + reference (built ONCE per shape and reused across the G points: the
# geometry is chosen per dispatch, so reuse cannot leak between points, and it
# keeps stray from_torch programs out of the profiled window)
# ---------------------------------------------------------------------------
# Refinement 4's collateral shapes (not in the harness's CASES table) — full shape,
# INTERLEAVED, TILE. `wide4096` is the wide-hidden guard, `ragged5119` the ragged one.
EXTRA_CASES = {
    "ragged5119": ((3, 1, 736, 5119), _ML.INTERLEAVED, None, None),
    "wide4096": ((1, 1, 4096, 4096), _ML.INTERLEAVED, None, None),
    # extra DECODE widths, to map where the flat-vs-two-stage turn actually is
    # (w_tiles = 40 / 48 / 96 against decode1024's 32 and decode2304's 72).
    "decode1280": ((1, 1, 32, 1280), _ML.INTERLEAVED, None, None),
    "decode1536": ((1, 1, 32, 1536), _ML.INTERLEAVED, None, None),
    "decode3072": ((1, 1, 32, 3072), _ML.INTERLEAVED, None, None),
    # the ROWS axis of the domain: at tensor_row_tiles 4 the occupancy key still
    # chooses (4 < 10 groups), at 10 it ties and `-G` already prefers the flat pick.
    "rows128_1024": ((1, 1, 128, 1024), _ML.INTERLEAVED, None, None),
    "rows320_1024": ((1, 1, 320, 1024), _ML.INTERLEAVED, None, None),
}


def make_case(device, name):
    if name in EXTRA_CASES:
        shape, memory_layout, shard_shape, core_grid = EXTRA_CASES[name]
        hidden = shape[-1]
    else:
        rows, hidden, memory_layout, shard_shape, core_grid = H.CASES[name]
        shape = (1, 1, rows, hidden)
    row_major = name in H.RM_CASES
    torch.manual_seed(0)
    layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, hidden), dtype=torch.float32).to(torch.bfloat16)
    memory_config = None
    if memory_layout != _ML.INTERLEAVED:
        from eval.sharding import shard_config

        memory_config = shard_config(
            shard_shape, core_grid, memory_layout, layout=layout, dtype=ttnn.bfloat16, device=device
        )
    kw = {} if memory_config is None else {"memory_config": memory_config}
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, **kw)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=layout, device=device)

    x = torch_input.to(torch.float32)
    ref = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * torch_gamma.to(torch.float32)
    return dict(
        tt_input=tt_input,
        tt_gamma=tt_gamma,
        ref=ref,
        memory_layout=memory_layout,
        name=name,
    )


def pcc(a, b):
    """Pearson correlation, accumulated in FLOAT64 in chunks.

    fp32 accumulation over a 59M-element prefill tensor drifts enough to report
    pcc > 1 (measured 1.0083), which would make the gate meaningless — so the
    moments are summed in double, a chunk at a time to keep host memory bounded.
    """
    a = a.flatten()
    b = b.flatten()
    n = a.numel()
    assert n == b.numel(), f"pcc shape mismatch {n} vs {b.numel()}"
    CH = 1 << 22
    sa = sb = saa = sbb = sab = 0.0
    for i in range(0, n, CH):
        x = a[i : i + CH].to(torch.float64)
        y = b[i : i + CH].to(torch.float64)
        sa += float(x.sum())
        sb += float(y.sum())
        saa += float((x * x).sum())
        sbb += float((y * y).sum())
        sab += float((x * y).sum())
    cov = sab - sa * sb / n
    va = saa - sa * sa / n
    vb = sbb - sb * sb / n
    denom = (va * vb) ** 0.5
    return 1.0 if denom == 0 else cov / denom


def run_point(device, case, check=True):
    """One fresh profiled dispatch. Returns (device_kernel_ns, pcc_or_None)."""
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    kw = {} if case["memory_layout"] == _ML.INTERLEAVED else {"memory_config": case["tt_input"].memory_config()}

    ttnn.ReadDeviceProfiler(device)  # flush anything queued before the measured window
    out = rms_norm(case["tt_input"], gamma=case["tt_gamma"], epsilon=1e-6, compute_kernel_config=cfg, **kw)
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
    p = None
    if check:
        p = pcc(ttnn.to_torch(out), case["ref"])
    del out
    return ns, p


def _row(name, greq, ns, p, note=""):
    # A SHARDED case never runs `_select_regime` (its shard spec supplies the
    # geometry), so SEL stays empty there -> print "-" rather than assume a pick.
    s = {
        k: ("-" if SEL.get(k) is None else SEL.get(k))
        for k in ("G", "gc", "gr", "C", "R", "s1", "s2", "nblocks", "active_cores", "max_tiles")
    }
    tag = "OK " if (p is None or p >= PCC_GATE) else "BAD"
    print(
        f"SWEEP {name:14s} G_req={str(greq):>7s} -> G={str(s['G']):>3} "
        f"(gc={s['gc']},gr={s['gr']}) C={str(s['C']):>3} R={str(s['R']):>3} "
        f"s1={str(s['s1']):>3} s2={str(s['s2']):>3} nblk={str(s['nblocks']):>3} "
        f"cores={str(s['active_cores']):>3} maxtiles={str(s['max_tiles']):>4} "
        f"ns={ns:>9.0f} pcc={('-' if p is None else f'{p:.6f}')} {tag} {note}",
        flush=True,
    )


_ZSEQ = [0]


def keep_zones(name, tag):
    _ZSEQ[0] += 1
    tag = f"{tag}_r{_ZSEQ[0]:02d}"
    src = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs", "profile_log_device.csv")
    if not os.path.exists(src):
        print(f"SWEEP zones: MISSING {src}", flush=True)
        return
    os.makedirs(ZONE_DIR, exist_ok=True)
    dst = os.path.join(ZONE_DIR, f"zones_{name}_{tag}.csv")
    shutil.copyfile(src, dst)
    print(f"SWEEP zones {name} {tag} -> {dst} ({os.path.getsize(dst)} bytes)", flush=True)


def main(names, gs=None, check=True, zones=False, default_first=True, rule=None, reps=1):
    """Sweep `gs` (plus the op's default pick) over each named case.

    `rule` (optional) = a candidate SELECTION RULE with `_select_regime`'s signature,
    monkeypatched in for the DEFAULT point only (so one call measures "op today" vs
    "op with the rule" back to back when `default_first` and `rule` are both set,
    and `gs=['rule']` requests the ruled point explicitly).
    """
    install_recorder()
    device = ttnn.open_device(device_id=0)
    try:
        for name in names:
            case = make_case(device, name)
            points = ([None] if default_first else []) + list(gs or [])
            for G in points * reps:
                SEL.clear()  # so a sharded case cannot show the previous pick
                RULE[0] = rule if G == "rule" else None
                try:
                    with pin_g(None if G == "rule" else G):
                        ns, p = run_point(device, case, check=check)
                finally:
                    RULE[0] = None
                _row(name, "default" if G is None else G, ns, p, note="(ruled)" if G == "rule" else "")
                if zones:
                    keep_zones(name, "default" if G is None else f"G{G}")
                if p is not None and p < PCC_GATE:
                    raise AssertionError(f"{name} G_req={G}: pcc {p} < {PCC_GATE}")
                time.sleep(0.15)
            del case
    finally:
        ttnn.close_device(device)
