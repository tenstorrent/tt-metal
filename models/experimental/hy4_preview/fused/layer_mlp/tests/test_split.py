# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MLP half with the "8 chips per group" expert layout (MlpLayer(split="tp8g4"), PLAN.md section 15) next to the
default one (tp32: 1/32 of every expert on every chip), on the same inputs.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k mock
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and solo"
    ... -k "rung and pair_row", "rung and pair_col", "rung and first"            (tp8g4: 1 / 2 / 2 / 32 chips, layer 1)
    pytest <this file> -x -k crafted                     (both layouts, crafted picks)
    pytest <this file> -x -k layers                       (both layouts, layers 1 to 5)
    pytest <this file> -x -k latency                     (both layouts, recorded picks)

test_mock      the 32 programs of either layout compile on the mock Galaxy; the largest set of circular buffers per
               core ROLE and kind of chip against L1; the kernel variants (distinct source + arguments + defines) of
               the 32 chips; the seconds of the first compile.
test_rung      tp8g4, checkpoint layer 1, one step of golden rows: the root alone, two chips of a row (two groups),
               two of a column (two slices of one group), all 32; test_mlp_layer's check_step (per-chip partials
               against exact arithmetic on the chip's own BFP4 share, F against the two-stage sum, the reference).
test_crafted   layer 1's expert weights, a crafted router (expert e reads hidden channel e) and streams that make
               the 8 rows pick what the case names:
                 full_g0   8 x 8 distinct picks, all in group 0 (the root's): 64 slots there, groups 1 and 2 have
                           nothing to do (U = 0: a zero partial), group 3 runs the shared expert alone
                 full_g3   all 64 in the shared expert's group: 65 slots (the buffers' limit), three idle groups
                 avoid_g2  44 distinct picks over groups 0, 1 and 3 (16, 16, 12); group 2 idle
               F against exact arithmetic on the BFP4 weights; the idle chips' partials zero bit for bit; words at
               rest. Both layouts run every case and leave x, the picks and F in the dump directory; the second
               one to run compares (same x, same picks, F against F).
test_layers    checkpoint layers 1 to 5, four steps each, both layouts: check_step, and the same comparison.
test_latency   traced replay (20 replays, one synchronize; 8 calls per trace and a trace of its own) of the half in
               the form a layer runs (leading post + pre), zero expert weights, the crafted router replaying RECORDED
               picks: 8 rows from three texts (mk11_picks.pt: HF's picks at every position, 77 MoE layers) whose
               union is 48 / 53 / 64 experts with the group imbalance that union has at its median; 8 rows of
               different texts never share enough for a union of 33 (38 is the least in 60,000 samples), so that
               case is 8 consecutive positions of one text. With stamps: x-to-partial per chip, by group.
Log lines start with HY4_TP8 (check_step's own with HY4_MK8)."""

import os
import random
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.layer_mlp.program import (
    E,
    HIDDEN,
    REDUCE,
    ROOT_REDUCE,
    SENDER,
    MlpLayer,
    is_mock,
)
from models.experimental.hy4_preview.fused.layer_mlp.tests import test_mlp_latency as lat
from models.experimental.hy4_preview.fused.layer_mlp.tests.test_mlp_layer import (
    FIRST_WAIT,
    RUNGS,
    Weights,
    build,
    check_step,
    gain,
    galaxy,
    golden_layer,
    site_params,
    two_stage_sum,
)
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe import prepare
from models.experimental.hy4_preview.tests.common import pcc

SPLITS = ["tp32", "tp8g4"]
GROUPED = "tp8g4"
# Both layouts are built with MlpLayer(arena=True) here (the transport and the hub's x / output as ranges of one L1
# tensor, as fused/model runs the half): the grouped layout does not fit the root without it (test_mock, form
# "tensors": 46,656 B over on a site core), and the two are compared in the same form.
ARENA = dict(arena=True)
DUMP = Path(os.environ.get("HY4_TP8_DUMP", "generated/hy4_preview/logs/tp8_dump"))
PICKS = Path(os.environ.get("HY4_TP8_PICKS", "generated/hy4_preview/logs/mk11_picks.pt"))
UNIONS = [33, 48, 53, 64]


def entries(split):
    """Stream entries of a chip: the routed experts it holds and the shared expert's two BFP4 terms."""
    groups = prepare.split_groups(split)
    return (E // groups[1] if groups else E) + 2


def zero_streams(mesh_device, split):
    share = prepare.split_share(split)
    rows = lambda stride: entries(split) * stride // L.BFP4_TILE_BYTES * L.TILE
    return tuple(
        L.to_device_stream(mesh_device, torch.zeros(rows(stride), 256))
        for stride in (share.gate_up_stride, share.down_stride)
    )


def group_counts(picks):
    """Distinct picked experts per group of 64."""
    union = {int(e) for row in picks for e in row}
    return [sum(1 for e in union if e // 64 == g) for g in range(4)]


def crafted_streams(picks, seed=0):
    """streams [8, 4, 6144] that make the crafted router (expert e reads hidden channel e) pick ``picks`` [8][8]:
    channels 0..255 are +4 for the row's experts and -4 elsewhere, the same in the four streams, so the normalised
    x has them whatever the pre-mix coefficients are (test_mlp_latency.crafted)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(L.ROWS, HIDDEN, generator=g)
    s[:, :E] = -4.0
    for r, ids in enumerate(picks):
        assert len(set(ids)) == 8
        s[r, list(ids)] = 4.0
    return s[:, None, :].repeat(1, 4, 1).contiguous()


# ---------------------------------------------------------------------------------------------------------------
# mock: L1 per core role, kernel variants, compile time


def roles(mlp, coord):
    root = tuple(coord) == mlp.topo.root
    op = mlp.op_root if root else mlp.op
    xy = lambda cores: [(c.x, c.y) for c in cores]
    out = {
        "hub": xy([op.hub]),
        "gate|up": xy(op.g_kernel_cores),
        "down": xy(op.d_kernel_cores),
        "finisher": xy(op.fin),
        "router": xy(op.r_cores),
        "sender": [SENDER],
    }
    if mlp.topo.is_head(coord):
        out["reduce"] = list(ROOT_REDUCE if root else REDUCE)
    if root:
        out["site"] = xy(mlp.site.cores)
    return out


def cb_bytes(desc):
    """{core: bytes of the program-allocated circular buffers} and {core: [(buffer id, bytes)]}."""
    per_core, detail = {}, {}
    for cb in desc.cbs:
        backed = cb.has_buffer() if callable(cb.has_buffer) else cb.has_buffer
        if backed:
            continue
        for c in ttnn.corerange_to_cores(cb.core_ranges):
            per_core[(c.x, c.y)] = per_core.get((c.x, c.y), 0) + cb.total_size
            detail.setdefault((c.x, c.y), []).append((cb.format_descriptors[0].buffer_index, cb.total_size))
    return per_core, detail


@galaxy
@pytest.mark.parametrize("form", ["arena", "tensors"])
@pytest.mark.parametrize("split", SPLITS)
def test_mock(mesh_device, split, form, expect_error):
    """form "tensors": the half with a transport and hub buffers of its own tensors (the default of tp32); the
    grouped layout is expected NOT to fit the root in it: the figures are logged and the compile's refusal is the
    pass."""
    if not is_mock():
        pytest.skip("the mock's compile and size check")
    g = torch.Generator().manual_seed(1)
    fits = not (split == GROUPED and form == "tensors")
    mlp = MlpLayer(mesh_device, split=split, timing=bool(os.environ.get("HY4_MLP_TIMING")), arena=form == "arena")
    split = f"{split} [{form}]"
    site = ((torch.randn(8, 4 * HIDDEN, generator=g) * 0.02), torch.randn(8, generator=g), torch.tensor([0.01, 0.005]))
    small = lambda: L.to_device_stream(mesh_device, torch.zeros(64, 256))
    mlp.set_layer(
        1,
        [site],
        0,
        torch.ones(HIDDEN),
        torch.randn(E, HIDDEN, generator=g) * 0.01,
        torch.zeros(E),
        streams=(small(), small()),
    )
    total = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_per_bank
    tensors = mlp.l1_bytes()
    share = mlp.op.share
    logger.info(
        f"HY4_TP8 mock {split}: share P {share.P} S {share.S}, piece {share.piece_bytes} B, expert {share.expert_bytes} B a chip; "
        f"max_slots {mlp.op.max_slots}, units {mlp.op.max_units}; L1 address space taken by persistent tensors {tensors} of {total} B per core"
    )
    worst_all = 0
    for name, coord in (("root", (0, 0)), ("head", (1, 0)), ("other", (0, 1)), ("other, shared group", (0, 3))):
        desc, _ = mlp._chip_program(coord, post=True)
        per_core, detail = cb_bytes(desc)
        by_role = {}
        for role, cores in roles(mlp, coord).items():
            worst = max(cores, key=lambda c: per_core.get(c, 0))
            by_role[role] = (per_core.get(worst, 0), worst)
        used = {c for cores in roles(mlp, coord).values() for c in cores}
        rest = [c for c in per_core if c not in used]
        if rest:
            worst = max(rest, key=per_core.get)
            by_role["no role"] = (per_core[worst], worst)
        top = max(by_role.values())
        worst_all = max(worst_all, top[0])
        logger.info(
            f"HY4_TP8 mock {split} L1 {name} chip {coord}: {len(desc.kernels)} kernels, {len(desc.cbs)} CBs; circular buffers by role (B, "
            f"with the tensors, free): "
            + "; ".join(f"{r} {v[0]} / {v[0] + tensors} / {total - v[0] - tensors}" for r, v in by_role.items())
        )
        if name == "root":
            for role in ("gate|up", "down", "finisher", "site", "hub"):
                core = by_role[role][1]
                logger.info(
                    f"HY4_TP8 mock {split} L1 root {role} core {core} buffers (id, B): {sorted(detail.get(core, []))}"
                )
    # the kernel variants over the 32 chips: distinct (source, compile-time arguments, defines)
    variants = {}
    for coord in mlp.topo.chips:
        desc, _ = mlp._chip_program(coord, post=True)
        for k in desc.kernels:
            key = (Path(str(k.kernel_source)).name, tuple(k.compile_time_args), tuple(tuple(d) for d in k.defines))
            variants.setdefault(Path(str(k.kernel_source)).name, set()).add(key)
    logger.info(
        f"HY4_TP8 mock {split} kernel variants over the 32 chips: {sum(len(v) for v in variants.values())} binaries of "
        f"{len(variants)} sources: " + ", ".join(f"{name} {len(v)}" for name, v in sorted(variants.items()))
    )
    if not fits:
        with expect_error(RuntimeError, "clash with L1 buffers"):
            mlp.run(post=True)
        logger.info(
            f"HY4_TP8 mock {split}: does not fit, as expected: largest buffer set {worst_all} B, with the tensors {worst_all + tensors} of {total} B"
        )
        return
    t0 = time.perf_counter()
    mlp.run(post=True)
    t1 = time.perf_counter()
    mlp.run(post=False)
    mlp.run_post()
    logger.info(
        f"HY4_TP8 mock {split}: the post + pre form compiled and checked in {t1 - t0:.1f} s (JIT cache as it was), the pre form and the "
        f"post program {time.perf_counter() - t1:.1f} s; largest buffer set {worst_all} B, with the tensors {worst_all + tensors} of {total} B"
    )


# ---------------------------------------------------------------------------------------------------------------
# board: the rungs, crafted picks, the checkpoint's layers; both layouts on the same inputs


def run_step(mlp, mesh_device, streams, post=True):
    mlp.set_streams(streams)
    ttnn.synchronize_device(mesh_device)
    mlp.run()
    ttnn.synchronize_device(mesh_device)
    status = mlp.status()
    assert not status, f"words not at rest: {status}"
    if post:
        mlp.run_post()
        ttnn.synchronize_device(mesh_device)


def leave(name, split, **tensors):
    """Leave a case's x, picks and F for the other layout's run, and compare with the other's when it is there."""
    DUMP.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, DUMP / f"{name}_{split}.pt")
    other = [s for s in SPLITS if s != split][0]
    path = DUMP / f"{name}_{other}.pt"
    if not path.is_file():
        logger.info(f"HY4_TP8 compare {name}: {split} saved; no {other} run to compare with yet")
        return None
    o = torch.load(path)
    a, b = (o, tensors) if split == GROUPED else (tensors, o)  # a: tp32, b: tp8g4
    same_x = torch.equal(a["x"], b["x"])
    same_picks = torch.equal(a["ids"], b["ids"]) and torch.equal(a["weights"], b["weights"])
    f32, f8 = a["f"], b["f"]
    line = (
        f"HY4_TP8 compare {name}: x bit-equal {same_x}, picks and weights equal {same_picks}; F tp8g4 vs F tp32: pcc={pcc(f32, f8):.6f} "
        f"gain={gain(f32, f8):.5f} max_abs={(f32 - f8).abs().max():.3e} (rms of F {f32.pow(2).mean().sqrt():.3e})"
    )
    if "ref" in a:
        line += (
            f"; against exact arithmetic on the BFP4 weights: tp32 pcc={pcc(a['ref'], f32):.6f} gain={gain(a['ref'], f32):.5f}, "
            f"tp8g4 pcc={pcc(b['ref'], f8):.6f} gain={gain(b['ref'], f8):.5f}"
        )
    logger.info(line)
    return same_x, same_picks, pcc(f32, f8)


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung):
    layer = 1
    hf = golden_layer(layer)
    mlp, w = build(mesh_device, layer, chips=RUNGS[rung], wait_limit=FIRST_WAIT, split=GROUPED, **ARENA)
    rows = slice(0, 8)
    streams = hf["streams_mid"][rows].float()
    if is_mock():
        mlp.set_streams(streams)
        mlp.run()
        mlp.run_post()
        return
    run_step(mlp, mesh_device, streams)
    logger.info(f"HY4_TP8 {GROUPED} {rung} L{layer}: all words at rest after the program")
    check_step(mlp, w, hf, rows, f"{GROUPED} {rung} L{layer} rows 0..7", full=RUNGS[rung] is None)


CRAFTED = {
    "full_g0": [[8 * r + j for j in range(8)] for r in range(8)],
    "full_g3": [[192 + 8 * r + j for j in range(8)] for r in range(8)],
    # rows 0..2 in group 0, 3..5 in group 1, 6..7 in group 3, neighbours sharing 4 experts: 16 + 16 + 12 distinct
    "avoid_g2": [[(0, 0, 0, 64, 64, 64, 192, 192)[r] + 4 * (r % 3) + j for j in range(8)] for r in range(8)],
}


@galaxy
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("case", list(CRAFTED))
def test_crafted(mesh_device, case, split):
    layer = 1
    w = Weights(layer)
    w.router_w, w.router_b = torch.eye(E, HIDDEN), torch.zeros(E)
    mlp = MlpLayer(mesh_device, split=split, wait_limit=FIRST_WAIT, **ARENA)
    mlp.set_layer(layer, [w.site], 0, torch.ones(HIDDEN), w.router_w, w.router_b)
    picks = CRAFTED[case]
    streams = crafted_streams(picks)
    if is_mock():
        mlp.set_streams(streams)
        mlp.run()
        return
    # a step whose picks reach every group first (rows r, r + 4 in group r): every chip's output buffer then holds
    # a partial that is not zero, so an idle chip's zero partial in the case's step is zeros that chip wrote
    warm = [[64 * (r % 4) + 8 * (r // 4) + j for j in range(8)] for r in range(L.ROWS)]
    run_step(mlp, mesh_device, crafted_streams(warm, seed=1), post=False)
    stale = sum(1 for c in mlp.topo.chips if mlp.read_chip(c)["out"].abs().max() > 0)
    run_step(mlp, mesh_device, streams, post=False)
    label = f"crafted {case} {split}"
    root = mlp.read_root()
    x, f = root["x"], root["f"]
    chips = {c: mlp.read_chip(c) for c in mlp.topo.chips}
    first = chips[mlp.topo.root]
    ids, weights = first["ids"].long(), first["weights"]
    failed = []
    x_ok = sum(torch.equal(v["x"], x) for v in chips.values())
    picks_ok = sum(
        torch.equal(v["ids"], first["ids"]) and torch.equal(v["weights"], first["weights"]) for v in chips.values()
    )
    got = [sorted(ids[r].tolist()) for r in range(L.ROWS)]
    if x_ok != 32 or picks_ok != 32:
        failed.append(f"x bit-equal on {x_ok} chips, picks equal on {picks_ok}")
    if got != [sorted(p) for p in picks]:
        failed.append(f"picks are not the crafted ones: {got}")
    partials = {c: v["out"] for c, v in chips.items()}
    wrong = int((two_stage_sum(mlp, partials) != f).sum())
    if wrong:
        failed.append(f"F differs from the two-stage sum of the partials in {wrong} elements")
    ref, parts = w.moe(x, ids, weights, rounded=True, per_chip=True, split=split, shared_group=mlp.shared_group)
    idle = [c for c in mlp.topo.chips if not parts[mlp.topo.chip(c)].any()]
    dirty = [c for c in idle if partials[c].abs().max() != 0]
    scores = {
        c: (pcc(parts[mlp.topo.chip(c)], partials[c]), gain(parts[mlp.topo.chip(c)], partials[c]))
        for c in mlp.topo.chips
        if c not in idle
    }
    pccs, gains = sorted(s[0] for s in scores.values()), sorted(s[1] for s in scores.values())
    logger.info(
        f"HY4_TP8 {label}: picks per group {group_counts(picks)}; after a step that left a partial that is not zero on {stale}/32 "
        f"chips: words at rest; x bit-equal on {x_ok}/32 chips, picks identical on "
        f"{picks_ok}/32 and the crafted ones; chips with nothing to do {len(idle)} (their partials not zero on {len(dirty)}); the other "
        f"{len(scores)} chips' partials vs exact arithmetic on their BFP4 share: pcc min {pccs[0]:.5f} median {pccs[len(pccs) // 2]:.5f}, "
        f"gain {gains[0]:.4f} to {gains[-1]:.4f}; F vs the two-stage sum of the partials: {wrong} elements differ; F vs exact arithmetic "
        f"on the BFP4 weights: pcc={pcc(ref, f):.6f} gain={gain(ref, f):.5f} max_abs={(ref - f).abs().max():.3e} (rms {ref.pow(2).mean().sqrt():.3e})"
    )
    if dirty:
        failed.append(f"idle chips with a partial that is not zero: {dirty}")
    if stale != 32:
        failed.append(f"the step before left a partial on {stale} chips only")
    if pccs[0] < 0.97 or pcc(ref, f) < 0.97:
        failed.append("a partial or F is off exact arithmetic on the BFP4 weights")
    expect_idle = {"tp32": 0, GROUPED: {"full_g0": 16, "full_g3": 24, "avoid_g2": 8}[case]}[split]
    if len(idle) != expect_idle:
        failed.append(f"{len(idle)} idle chips, expected {expect_idle}")
    leave(f"crafted_{case}", split, x=x, ids=first["ids"], weights=weights, f=f, ref=ref)
    assert not failed, f"{label}: " + "; ".join(failed)


@galaxy
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("layer", range(1, 6), ids=lambda layer: f"layer{layer}")
def test_layers(mesh_device, layer, split):
    """Four steps (golden rows 0..7, 40..47, 80..87, 120..127) of a checkpoint layer in one layout."""
    if is_mock():
        pytest.skip("needs the board")
    hf = golden_layer(layer)
    mlp, w = build(mesh_device, layer, wait_limit=FIRST_WAIT, split=split, **ARENA)
    results = []
    for start in (0, 40, 80, 120):
        rows = slice(start, start + 8)
        run_step(mlp, mesh_device, hf["streams_mid"][rows].float())
        res = check_step(mlp, w, hf, rows, f"{split} L{layer} rows {start}..{start + 7}", per_chip=start == 0)
        root, first = mlp.read_root(), mlp.read_chip(mlp.topo.root)
        ids = first["ids"].long()
        ref = w.moe(root["x"], ids, first["weights"], rounded=True)
        leave(
            f"L{layer}_rows{start}",
            split,
            x=root["x"],
            ids=first["ids"],
            weights=first["weights"],
            f=root["f"],
            ref=ref,
        )
        res["groups"] = group_counts(ids.tolist())
        results.append(res)
    a, b = "reference on the device's picks", "exact arithmetic on the BFP4 weights"
    logger.info(
        f"HY4_TP8 summary {split} L{layer}: unions {[r['union'] for r in results]} (per group {[r['groups'] for r in results]}); rows with "
        f"the host router's picks {[r['picks']['rows_exact'] for r in results]} of 8; F vs reference pcc {[round(r[a][0], 4) for r in results]} "
        f"gain {[round(r[a][1], 4) for r in results]}; F vs exact arithmetic on BFP4 pcc {[round(r[b][0], 5) for r in results]} gain "
        f"{[round(r[b][1], 4) for r in results]}; streams update pcc {[round(r['streams'][1], 4) for r in results]} gain "
        f"{[round(r['streams'][2], 4) for r in results]}"
    )


# ---------------------------------------------------------------------------------------------------------------
# latency on recorded picks


def recorded_cases(unions=UNIONS, samples=200000, seed=0):
    """{union: (picks [8][8], group counts, (text, position, MoE layer index) per row, how the case stands among the
    samples of that union)}: 8 rows of HF's recorded picks at one random MoE layer whose union has the wanted size
    and whose busiest group holds the median number of experts of all samples with that union. The rows are from
    the three texts in turn (3 + 3 + 2 rows) at random positions; a union no such sample comes near (33: rows of
    different texts share too little) is taken from 8 consecutive positions of one text. A wanted union without an
    exact sample is replaced by the nearest one that has samples (the key is the union the case really has).
    The search is kept in the dump directory (test_recorded_cases makes it on the host, before a board job)."""
    cache = DUMP / f"recorded_cases_{samples}_{seed}.pt"
    if cache.is_file():
        return torch.load(cache)
    if not Path(PICKS).is_file():
        pytest.skip(f"no recorded router picks at {PICKS} (a board run of the model records them)")
    d = torch.load(PICKS)
    names = list(d)
    P = {k: d[k]["picks"].long() for k in names}
    rng = random.Random(seed)
    found = {"mixed": {}, "consecutive": {}}
    for n in range(samples):
        layer = rng.randrange(P[names[0]].shape[1])
        mixed = [
            (names[i % len(names)], rng.randrange(32, P[names[i % len(names)]].shape[0]), layer) for i in range(L.ROWS)
        ]
        text = names[n % len(names)]
        start = rng.randrange(32, P[text].shape[0] - L.ROWS)
        for kind, where in (("mixed", mixed), ("consecutive", [(text, start + i, layer) for i in range(L.ROWS)])):
            picks = [P[k][p, layer].tolist() for k, p, _ in where]
            u = len({e for row in picks for e in row})
            if min(abs(u - t) for t in unions) <= 2:
                found[kind].setdefault(u, []).append((picks, where))
    out = {}
    for want in unions:
        near = lambda kind: min(found[kind], key=lambda u: abs(u - want), default=None)
        kind = "mixed" if near("mixed") is not None and abs(near("mixed") - want) <= 2 else "consecutive"
        u = near(kind)
        assert u is not None and abs(u - want) <= 2, f"no sample of 8 recorded rows with a union near {want}"
        cases = found[kind][u]
        busiest = sorted(max(group_counts(p)) for p, _ in cases)
        median = busiest[len(busiest) // 2]
        picks, where = next(c for c in cases if max(group_counts(c[0])) == median)
        rows = "three texts in turn" if kind == "mixed" else "8 consecutive positions of one text"
        out[u] = (
            picks,
            group_counts(picks),
            where,
            f"{rows}: {len(cases)} of {samples} samples, busiest group {busiest[0]} .. {busiest[-1]}, median {median}",
        )
    DUMP.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def test_recorded_cases():
    """Host only: the search for the latency cases, kept for the board job."""
    for u, (picks, groups, where, standing) in recorded_cases().items():
        assert len({e for row in picks for e in row}) == u and sum(groups) == u
        logger.info(f"HY4_TP8 recorded case union {u}: groups {groups}; {standing}; rows {where}")


@galaxy
@pytest.mark.parametrize("timing", [False, True], ids=["plain", "stamps"])
@pytest.mark.parametrize("split", SPLITS)
def test_latency(mesh_device, split, timing):
    mlp = MlpLayer(mesh_device, split=split, timing=timing, wait_limit=FIRST_WAIT, **ARENA)
    streams = (
        tuple(L.to_device_stream(mesh_device, torch.zeros(64, 256)) for _ in range(2))
        if is_mock()
        else zero_streams(mesh_device, split)
    )
    mlp.set_layer(1, [site_params(1)], 0, torch.ones(HIDDEN), torch.eye(E, HIDDEN), torch.zeros(E), streams=streams)
    if is_mock():
        mlp.run(post=True)
        return
    cases = dict(recorded_cases())
    # the layout's worst case next to them: all 64 picks in one group (crafted, not recorded: the recorded picks'
    # busiest group never held more than 29)
    for key in ("full_g0", "full_g3"):
        cases[key] = (CRAFTED[key], group_counts(CRAFTED[key]), "crafted", "crafted: every pick in one group")
    table = {}
    share = mlp.op.share
    for U, (picks, groups, where, standing) in cases.items():
        U = len({e for row in picks for e in row})
        name = f"U={U} groups {groups}"
        mlp.set_streams(crafted_streams(picks))
        mlp.set_f(torch.zeros(L.ROWS, HIDDEN))
        ttnn.synchronize_device(mesh_device)
        run = lambda: mlp.run(post=True)
        run()
        ttnn.synchronize_device(mesh_device)
        status = mlp.status()
        assert not status, f"{name}: words not at rest: {status}"
        ids = mlp.read_chip(mlp.topo.root)["ids"]
        assert all(sorted(ids[r].tolist()) == sorted(picks[r]) for r in range(L.ROWS)), f"{name}: picks {ids.tolist()}"
        one = tracing.traced_ms(mesh_device, run, calls=1)["replay_ms"] * 1e3
        eight = tracing.traced_ms(mesh_device, run, calls=8)["replay_ms"] * 1e3
        status = mlp.status()
        assert not status, f"{name} after the traces: words not at rest: {status}"
        ids = mlp.read_chip(mlp.topo.root)["ids"]
        assert all(
            sorted(ids[r].tolist()) == sorted(picks[r]) for r in range(L.ROWS)
        ), f"{name} after the traces: picks changed"
        # what the busiest chip streams: its slots (the shared expert's two terms where it has them) x its share
        if mlp.groups:
            slots = [g + (2 if i == mlp.shared_group else 0) for i, g in enumerate(groups)]
        else:
            slots = [U + 2] * 4
        mb = [s * share.expert_bytes / 1e6 for s in slots]
        table[name] = (eight, one, groups, max(mb))
        logger.info(
            f"HY4_TP8 latency {split} {name} stamps={timing}: {eight:.1f} us per call with 8 calls per trace, {one:.1f} us as a trace of "
            f"its own; stream entries per chip by group {slots} = {[round(v, 1) for v in mb]} MB; the rows: {where}; among the recorded "
            f"samples of this union: {standing}"
        )
        if timing:
            s = lat.stages(mlp)
            eo = s["experts_others"]
            grid = [[s["by_chip"][(r, c)] for c in range(4)] for r in range(8)]
            cols = [sorted(grid[r][c] for r in range(8)) for c in range(4)]
            logger.info(
                f"HY4_TP8 stages {split} {name} (us): root site {s['site']:.1f}; x on the root's hub at {s['x_root']:.1f}; router root "
                f"{s['router_root']:.1f} max {s['router_max']:.1f}; experts root {s['experts_root']:.1f}, other chips min {eo[0]:.1f} median "
                f"{eo[len(eo) // 2]:.1f} max {eo[-1]:.1f}; slowest chip {s['slowest'][0]} x-to-partial {s['slowest'][1]:.1f}; root reduce: "
                f"stage 1 landed {s['s1']:.1f}, stage 2 landed {s['s2']:.1f}, F written {s['f']:.1f}; X6 from the last partial to F "
                f"{s['x6']:.1f}; traced {eight:.1f} per call"
            )
            logger.info(
                f"HY4_TP8 x-to-partial {split} {name} by mesh column = group (us; min / median / max over its 8 chips): "
                + "; ".join(
                    f"g{c} [{groups[c]} picked] {v[0]:.0f} / {v[4]:.0f} / {v[-1]:.0f}" for c, v in enumerate(cols)
                )
            )
            logger.info(f"HY4_TP8 x-to-partial by chip {split} {name} (us, mesh rows x columns): {grid}")
    logger.info(
        f"HY4_TP8 latency {split} stamps={timing} summary, us per call (8 calls per trace / own trace; busiest chip MB): "
        + ", ".join(f"{u}: {v[0]:.1f} / {v[1]:.1f}; {v[3]:.1f} MB" for u, v in table.items())
    )
