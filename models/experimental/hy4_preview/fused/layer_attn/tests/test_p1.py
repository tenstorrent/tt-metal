# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program P1 of the attention half (fused/layer_attn/program.py): the root's iHC pass and norm, the broadcast X1,
the pre group, the all-gather Xa, the query group and the all-to-all X3, for 8 token rows.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k mock
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and solo"
    ... -k "rung and pair_row", "rung and pair_col", "rung and galaxy"    (each after the rung before it has passed)

A rung: checkpoint iHC site and norm of layer 1 on the golden's streams, random projection weights (the reference
attention with random parameters at a trained layer's scale, fused/projections/tests/harness.py). On a part of the
mesh the absent chips' Xa payloads (their q_a tiles and lane sums, from the host golden) are written into the
landings first, so every chip's query output is that of the whole layer. Checked on every chip that takes part
(every number is logged; lines start with HY4_MK8A):
  words      nothing left in any status word or counter (but X3's arrivals on the landing cores, which program P2
             takes: they must be the number of chips of the head group); the groups' status pages clean
  y, x       the site's pre-mix and norm against the reference; x bit-equal in every chip's pre hub
  pre        q_a, lane sums, gate against exact arithmetic on the chip's stored weights for the device's x (and
             against the reference weights)
  Xa         every chip's q_a tiles and lane sums as their chip produced them, on every chip (bit-equal)
  query      the absorbed queries of the chip's 2 heads against exact arithmetic on the stored weights for the q_a
             and the lane sums the chip holds
  X3         each head group's landing holds the 4 chips' query outputs (bit-equal)
"""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.ihc_site.op import golden_pre
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections.tests.harness import random_reference, tables
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import rms_norm
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
TRACE_REGION = 64 << 20
FIRST_WAIT = 100_000_000  # polls (about 1 s): an untraced first run launches the chips one after another

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)
RUNGS = {"solo": [(0, 0)], "pair_row": [(0, 0), (0, 1)], "pair_col": [(0, 0), (1, 0)], "galaxy": None}
ALL = [(r, c) for r in range(MESH[0]) for c in range(MESH[1])]


def gain(ref, dev):
    ref, dev = ref.double().flatten(), dev.double().flatten()
    return float((ref * dev).sum() / (ref * ref).sum())


def rel_rms(ref, dev):
    ref, dev = ref.double(), dev.double()
    return float((ref - dev).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())


def site_params(layer, site="hc_attn_layer"):
    prefix = f"model.layers.{layer}.{site}.hc_pre."
    return tuple(hy4_tensors([prefix + "hc_fn", prefix + "hc_base", prefix + "hc_scale"]))


def stored_share(layer, share):
    f = lambda m: L.stored(m.float().contiguous(), layer.fmt)  # the streams are packed on the host: one rounding
    return {k: [f(m) for m in v] if isinstance(v, list) else f(v) for k, v in share.items()}


class Case:
    """A layer with checkpoint site parameters of ``layer`` and random projection weights."""

    def __init__(self, mesh_device, layer=1, chips=None, seed=0, full=False, **kw):
        self.layer = layer
        self.site = site_params(layer)
        (self.gamma,) = hy4_tensors([f"model.layers.{layer}.input_layernorm.weight"])
        ref, _, indexer = random_reference(seed, full)
        if full:  # a full-indexer layer: its indexer, with a key LayerNorm that is not the identity
            g = torch.Generator().manual_seed(seed + 100)
            with torch.no_grad():
                indexer.k_norm_weight.copy_(1.0 + 0.25 * torch.randn(indexer.k_norm_weight.shape, generator=g))
                indexer.k_norm_bias.copy_(0.3 * torch.randn(indexer.k_norm_bias.shape, generator=g))
            kw = dict(kw, full=True)
        self.lw = L.layer_weights(ref.weights(), None, indexer.weights() if full else None)
        if os.environ.get("HY4_ATTN_SHARED") == "1":
            # the 8-row programs on the 16-row layout (PLAN.md 19.11): a 16-row layer holds the arena, the words and
            # the site; the layer under test is its second program set
            self.host = A.AttnLayer(mesh_device, chips=chips, rows=16, fidelity="recommended")
            kw = dict(kw, host=self.host)
        self.a = A.AttnLayer(mesh_device, chips=chips, **kw)
        self.a.set_site([self.site], 0, self.gamma)
        self.a.set_weights(self.lw)
        self._shares = {}
        _, _, self.cos, self.sin = tables(self.a.rows)
        self.a.set_rope(self.cos, self.sin)

    def share(self, chip):
        """(the chip's share, the same as its stream holds it)."""
        if chip not in self._shares:
            w = self.a.chip_share(self.lw, chip)
            self._shares[chip] = (w, stored_share(self.a, w))
        return self._shares[chip]

    def golden_x(self, streams):
        y, _, _, inv_rms = golden_pre(self.a.cfg, *self.site, streams)
        return y, inv_rms, rms_norm(y, self.gamma.float(), self.a.cfg.norm_eps)

    def prefill_absent(self, x):
        """Xa of the chips that do not take part, from the host golden on ``x`` (BF16), into the landings."""
        absent = [c for c in ALL if c not in self.a.topo.chips]
        if not absent:
            return
        updates = {}
        ones = torch.ones(self.a.rows)
        for coord in absent:
            g = L.pre_golden(x, ones, self.share(self.a.topo.chip(coord))[1])
            q_a = L.round7(g["q_a"])
            for key, items in self.a.xa_prefill(coord, q_a, L.lane_sums(q_a)).items():
                updates.setdefault(key, []).extend(items)
        self.a.write_arena(updates)


def check_p1(case, streams, label):
    a = case.a
    topo = a.topo
    failed = []
    status = a.status(p1_only=True)
    if status:
        failed.append(f"words not at rest: {status}")
    a.reset_words()
    images = {c: a.read_arena(c) for c in topo.chips}
    root = a.read_root(images[topo.root]) if a.rows != L.ROWS else a.read_root()
    r_y, r_inv, r_x = case.golden_x(streams)
    x = root["x"]
    logger.info(
        f"HY4_MK8A {label} root: y vs the reference pre-mix max_abs={(r_y - root['y']).abs().max():.3e}; 1/RMS max_rel="
        f"{((root['inv_rms'] - r_inv).abs() / r_inv).max():.3e}; x vs the reference norm max_abs={(r_x - x).abs().max():.3e} "
        f"(BF16 rounding alone {(r_x - r_x.to(torch.bfloat16).float()).abs().max():.3e})"
    )
    got = {c: a.read_p1(c, images[c]) for c in topo.chips}
    x_ok = sum(torch.equal(v["x"], x) for v in got.values())
    if x_ok != len(got):
        failed.append(f"x bit-equal on {x_ok} of {len(got)} chips")
    ones = torch.ones(a.rows)
    worst = {}
    lay = a.lay

    def note(name, ref, dev):
        p, g, r = pcc(ref.float(), dev.float()), gain(ref, dev), rel_rms(ref, dev)
        w = worst.setdefault(name, [1.0, g, g, 0.0])
        worst[name] = [min(w[0], p), min(w[1], g), max(w[2], g), max(w[3], r)]
        return p

    for coord in topo.chips:
        chip = topo.chip(coord)
        v = got[coord]
        for name, words in v["status"].items():
            if any(words[1:6]):
                failed.append(f"chip {coord} {name} group status {[hex(w) for w in words[:6]]}")
        w, ws = case.share(chip)
        exact = L.pre_golden(x, ones, ws)
        ref = L.pre_golden(x, ones, w)
        for key in ("q_a", "gate"):
            if note(f"pre {key} vs stored weights", exact[key], v[key]) < 0.999:
                failed.append(f"chip {coord} pre {key}")
            note(f"pre {key} vs the reference weights", ref[key], v[key])
        # the lane sums are those of the tiles the unit packed
        lanes = L.lane_sums(v["q_a"])
        err = ((v["lanes"] - lanes).abs() / lanes.abs().clamp_min(1e-30)).max().item()
        worst["lanes rel err"] = max(worst.get("lanes rel err", 0.0), err)
        if err > 1e-5:
            failed.append(f"chip {coord} lane sums off by {err:.2e}")
        # Xa: what every chip that takes part produced, as this chip holds it
        for other in topo.chips:
            o = topo.chip(other)
            if not torch.equal(v["q_a_all"][:, 64 * o : 64 * o + 64], got[other]["q_a"]):
                failed.append(f"chip {coord}: q_a tiles of chip {other} differ")
            if not torch.equal(v["lanes_all"][o], got[other]["lanes"]):
                failed.append(f"chip {coord}: lane sums of chip {other} differ")
        # query: against exact arithmetic for what the chip holds
        lanes_sum = v["lanes_all"].double().sum(dim=0).float()
        q_ref, _ = L.query_golden(v["q_a_all"], lanes_sum, case.cos, case.sin, ws)
        if note("query vs stored weights", q_ref, v["q_lat"]) < 0.999:
            failed.append(f"chip {coord} query")
        q_true, _ = L.query_golden(v["q_a_all"], lanes_sum, case.cos, case.sin, w)
        note("query vs the reference weights", q_true, v["q_lat"])
        # X3: the head group's landing on this chip
        for other in topo.chips:
            o = topo.chip(other)
            if o // 4 == chip // 4 and not torch.equal(v["q_landing"][o % 4], got[other]["q_lat"]):
                failed.append(f"chip {coord}: the landed queries of chip {other} differ")
        # other groups land on other merge cores of this chip
        for other in topo.chips:
            o = topo.chip(other)
            if o // 4 != chip // 4:
                raw = a.arena_bytes(
                    images[coord], A.MERGE[o // 4], lay.m_query + (o % 4) * lay.query_out, lay.query_out
                )
                landed = L.from_row_tiles(A.bytes_bf16(raw, (lay.query_out // lay.row_tile, lay.row_tile // 2)))
                landed = landed.reshape(a.rows, L.HEADS_PER_CHIP, L.LATENT).permute(1, 0, 2)
                if not torch.equal(landed, got[other]["q_lat"]):
                    failed.append(f"chip {coord}: the landed queries of chip {other} (group {o // 4}) differ")
    for name, w in worst.items():
        if isinstance(w, list):
            logger.info(
                f"HY4_MK8A {label} {name}: pcc min {w[0]:.6f}, gain {w[1]:.4f} to {w[2]:.4f}, rel rms max {w[3]:.3e}"
            )
        else:
            logger.info(f"HY4_MK8A {label} {name}: {w:.3e}")
    logger.info(
        f"HY4_MK8A {label}: {len(topo.chips)} chips, x bit-equal on {x_ok}; Xa and X3 landings "
        f"{'bit-equal everywhere' if not any('differ' in f for f in failed) else 'DIFFER'}"
    )
    assert not failed, f"{label}: " + "; ".join(failed[:12])


def golden_layer(layer):
    golden = hf_harness.find_golden()
    if golden is None:
        pytest.skip(f"no golden under {hf_harness.GOLDEN_ROOT}")
    return golden["layers"][layer]


def cb_report(mesh_device, a, build, names, program="p1"):
    total = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_per_bank
    for name, coord in names:
        desc, _ = build(coord)
        for kernel, buffer, waits, pushed in a.feed_audit(program, coord):
            logger.info(
                f"HY4_MK8A mock feed audit {name} chip: {kernel} waits for {waits} tiles of its {buffer}, {pushed} are pushed in every outcome"
            )
        per_core = {}
        for cb in desc.cbs:
            backed = cb.has_buffer() if callable(cb.has_buffer) else cb.has_buffer
            if backed:
                continue
            for c in ttnn.corerange_to_cores(cb.core_ranges):
                per_core[(c.x, c.y)] = per_core.get((c.x, c.y), 0) + cb.total_size
        worst = max(per_core, key=per_core.get)
        rows = A.cb_audit(desc)  # one producer and one consumer processor per circular buffer
        seen = {}
        for core, cb, push, pop in rows:
            seen.setdefault((cb, tuple(push), tuple(pop)), []).append(core)
        for (cb, push, pop), cores in sorted(seen.items()):
            logger.info(
                f"HY4_MK8A mock CB audit {name} chip: buffer {cb} on {len(cores)} core(s) from {cores[0]}: pushed by {list(push)}, popped by {list(pop)}"
            )
        logger.info(
            f"HY4_MK8A mock L1 {name} chip: {len(desc.kernels)} kernels, {len(desc.cbs)} CBs, {len(desc.semaphores)} "
            f"program semaphores; largest CB set {per_core[worst]} B on core {worst}; with the tensors "
            f"{per_core[worst] + a.l1_bytes()} of {total} B"
        )


@galaxy
def test_mock(mesh_device):
    """Compiles the 32 programs of P1 and checks every buffer against L1 (nothing runs)."""
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    case = Case(mesh_device, timing=bool(os.environ.get("HY4_ATTN_TIMING")))
    a = case.a
    logger.info(f"HY4_MK8A p1 mock: L1 address space taken by persistent tensors {a.l1_bytes()} B per core")
    for post in (False, True):
        a.run_p1(post=post)
    cb_report(mesh_device, a, lambda c: a._p1_chip(c, True), (("root", (0, 0)), ("other", (0, 1))))
    logger.info("HY4_MK8A p1 mock: P1 (pre and post + pre form of the site) compiled")


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung):
    """One step on the root alone (no fabric connection), on two chips, on all 32: the first runs of P1."""
    hf = golden_layer(1)
    streams = hf["streams_in"][slice(0, 8)].float()
    case = Case(mesh_device, chips=RUNGS[rung], wait_limit=FIRST_WAIT)
    a = case.a
    _, _, r_x = case.golden_x(streams)
    case.prefill_absent(r_x.to(torch.bfloat16).float())
    a.set_streams(streams)
    ttnn.synchronize_device(mesh_device)
    a.run_p1()
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    check_p1(case, streams, f"p1 {rung}")
    # a second step on other rows: the words and the landings are reused
    streams = hf["streams_in"][slice(8, 16)].float()
    _, _, r_x = case.golden_x(streams)
    case.prefill_absent(r_x.to(torch.bfloat16).float())
    a.set_streams(streams)
    ttnn.synchronize_device(mesh_device)
    a.run_p1()
    ttnn.synchronize_device(mesh_device)
    check_p1(case, streams, f"p1 {rung} step 2")


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_parse(mesh_device, rung):
    """The senders' argument tables of P1 as the kernel reads them on the chips of the rung, with no fabric
    connection opened and nothing sent (the collective sender's parse-only build): every chip's page must equal the
    table the host built. The other kernels run into their bounded waits; the words are zeroed afterwards."""
    hf = golden_layer(1)
    case = Case(mesh_device, chips=RUNGS[rung], wait_limit=2_000_000, parse_only=True)
    a = case.a
    a.set_streams(hf["streams_in"][slice(0, 8)].float())
    ttnn.synchronize_device(mesh_device)
    a.run_p1()
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    wrong = []
    for c in a.topo.chips:
        want = a.parsed[("p1", c)]
        page = a.read_parsed(c, len(want))
        logger.info(
            f"HY4_MK8A p1 parse {rung} chip {c}: {len(want)} words, {'equal to the host table' if page == want else 'DIFFERENT'}; "
            f"ops {want[4]}, connections {want[5]}, routes {want[6]}, fabric arguments from index {want[-2]}"
        )
        if page != want:
            wrong.append((c, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(page, want)) if x != y][:8]))
    a.reset()
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"


# ---- 16 token rows (PLAN.md section 19.15): two sites side by side on the root, [16, 32] tiles from X1 on ------------
KW16 = dict(rows=16, fidelity="recommended")


def streams16(hf, step):
    """16 token rows of the golden's streams for step 0 or 1 (other rows, or the same ones in another order)."""
    s = hf["streams_in"].float()
    if s.shape[0] >= 32:
        return s[16 * step : 16 * step + 16]
    return s[:16] if step == 0 else s[:16].flip(0)


@galaxy
def test_mock16(mesh_device):
    """Compiles the 32 programs of P1 at 16 rows and checks every buffer against L1 (nothing runs)."""
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    case = Case(mesh_device, timing=bool(os.environ.get("HY4_ATTN_TIMING")), **KW16)
    a = case.a
    logger.info(
        f"HY4_MK8A p1 16 rows mock: L1 address space taken by persistent tensors {a.l1_bytes()} B per core; arena {a.lay.arena} B, by role {a.lay.roles}"
    )
    a.run_p1()
    cb_report(mesh_device, a, lambda c: a._p1_chip(c, False), (("root", (0, 0)), ("other", (0, 1))))
    for name, coord in (("root", (0, 0)), ("other", (0, 1))):
        for row in A.noc_audit(a._p1_chip(coord, False)[0]):
            logger.info(f"HY4_MK8A p1 16 rows mock NoC {name} chip: {row}")
    logger.info("HY4_MK8A p1 16 rows mock: P1 compiled")


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung16(mesh_device, rung):
    """P1 at 16 rows on the root alone (no fabric connection; the absent chips' Xa host-written), on two chips, on
    all 32: two steps, every check of the 8-row rung on 16 rows."""
    hf = golden_layer(1)
    case = Case(mesh_device, chips=RUNGS[rung], wait_limit=FIRST_WAIT, **KW16)
    a = case.a
    for step in (0, 1):
        streams = streams16(hf, step)
        _, _, r_x = case.golden_x(streams)
        case.prefill_absent(r_x.to(torch.bfloat16).float())
        a.set_streams(streams)
        ttnn.synchronize_device(mesh_device)
        a.run_p1()
        if A.is_mock():
            return
        ttnn.synchronize_device(mesh_device)
        check_p1(case, streams, f"p1 16 rows {rung} step {step + 1}")


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_parse16(mesh_device, rung):
    """test_parse for P1 at 16 rows: the senders' tables as the kernel reads them, nothing sent."""
    hf = golden_layer(1)
    case = Case(mesh_device, chips=RUNGS[rung], wait_limit=2_000_000, parse_only=True, **KW16)
    a = case.a
    a.set_streams(streams16(hf, 0))
    ttnn.synchronize_device(mesh_device)
    a.run_p1()
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    wrong = []
    for c in a.topo.chips:
        want = a.parsed[("p1", c)]
        page = a.read_parsed(c, len(want))
        logger.info(
            f"HY4_MK8A p1 16 rows parse {rung} chip {c}: {len(want)} words, {'equal to the host table' if page == want else 'DIFFERENT'}; "
            f"ops {want[4]}, connections {want[5]}, routes {want[6]}, fabric arguments from index {want[-2]}"
        )
        if page != want:
            wrong.append((c, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(page, want)) if x != y][:8]))
    a.reset()
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"


@galaxy
def test_site_time16(mesh_device):
    """What the root's site stage costs at 16 rows (two sites side by side, 32 writers counting into one gather word)
    against the 8-row site, on ONE chip (no fabric; the absent chips' Xa host-written), from the stamps of site core
    0 (its writer's start; the broadcast's op made ready) and of the pre hub (its start; x arrived): 4 launches each,
    the first one aside."""
    hf = golden_layer(1)
    for rows, kw in ((8, dict(fidelity="recommended")), (16, KW16)):
        case = Case(mesh_device, chips=[(0, 0)], wait_limit=FIRST_WAIT, timing=True, **kw)
        a = case.a
        streams = streams16(hf, 0)[:rows]
        _, _, r_x = case.golden_x(streams)
        case.prefill_absent(r_x.to(torch.bfloat16).float())
        a.set_streams(streams)
        ttnn.synchronize_device(mesh_device)
        site, arrived = [], []
        for launch in range(4):
            a.run_p1()
            if A.is_mock():
                break
            ttnn.synchronize_device(mesh_device)
            words = a.t.read_words()
            five = lambda core: [int(v) for v in a.t.core_words(words[(0, 0)], core)[[8, 9, 10, 11, 15]]]
            s, p = five(A.SITE["origin"]), five(A.PRE_HUB)
            site.append(us(s[1] - s[0]))
            arrived.append(us(p[1] - p[0]))
            status = a.status(p1_only=True)
            assert not status, f"{rows} rows, launch {launch}: words not at rest: {status}"
            a.reset_words()
        if site:
            logger.info(
                f"HY4_MK8A p1 site time, one chip, {rows} rows: site core 0's writer start -> the broadcast's op ready "
                f"{[round(v, 1) for v in site]} us; the pre hub's start -> x in its buffer {[round(v, 1) for v in arrived]} us"
            )
        a.release()


@galaxy
def test_site_fetch(mesh_device):
    """Is the site's stage bound by its fetch of fn^T from the DRAM table (PLAN.md 19.15)? One chip, P1 at 8 rows
    and at 16 rows, the site's data-movement kernels built with HY4_IHC_FETCH_STAMPS: per site core the reader's
    start, its constants page landed, its fn^T and gamma pages landed; the writer's start, the projection's partial
    seen, the mix's sum seen, x seen, its shard written. 4 launches, the last one reported (us after the core's
    reader started; min / median / max over a site's 16 cores)."""
    from models.experimental.hy4_preview.fused.layer_common import transport as tp

    hf = golden_layer(1)
    names = [
        "constants landed",
        "fn^T and gamma landed",
        "writer started",
        "projection's partial seen",
        "mix's sum seen",
        "x seen (norm done)",
        "shard written",
    ]
    for rows, kw in ((8, dict(fidelity="recommended")), (16, KW16)):
        case = Case(mesh_device, chips=[(0, 0)], wait_limit=FIRST_WAIT, timing=True, **kw)
        a = case.a
        stamps = tp.core_buffer(mesh_device, tp.grid_set(mesh_device), 64, ttnn.uint32, width=16)
        sites = [a.site] + ([a.site2] if a.site2 is not None else [])
        for site in sites:
            site.fetch_stamps = tp.address(stamps)
        a._programs = {}
        streams = streams16(hf, 0)[:rows]
        _, _, r_x = case.golden_x(streams)
        case.prefill_absent(r_x.to(torch.bfloat16).float())
        a.set_streams(streams)
        ttnn.synchronize_device(mesh_device)
        for launch in range(4):
            a.run_p1()
            if A.is_mock():
                break
            ttnn.synchronize_device(mesh_device)
            status = a.status(p1_only=True)
            assert not status, f"{rows} rows, launch {launch}: words not at rest: {status}"
            a.reset_words()
        if not A.is_mock():
            words = ttnn.to_torch(ttnn.get_device_tensors(stamps)[0]).to(torch.int64).reshape(-1, 16) & 0xFFFFFFFF
            tw = a.t.read_words()
            s = [int(v) for v in a.t.core_words(tw[(0, 0)], A.SITE["origin"])[[8, 9]]]
            logger.info(
                f"HY4_MK8A p1 site fetch, one chip, {rows} rows: site core 0's writer start -> the broadcast's op ready {us(s[1] - s[0]):.1f} us"
            )
            for b, site in enumerate(sites):
                per = [words[c.y * a.grid[0] + c.x] for c in site.cores]
                for i, name in zip((1, 2, 3, 4, 5, 6, 7), names):
                    # (signed: a core's writer may start before its reader)
                    v = [((int(w[i]) - int(w[0]) + (1 << 31)) % (1 << 32) - (1 << 31)) * CYCLE_NS / 1e3 for w in per]
                    logger.info(
                        f"HY4_MK8A p1 site fetch, one chip, {rows} rows, site {b} ({len(per)} cores), us after the core's reader "
                        f"started: {name}: min {min(v):.1f} median {statistics.median(v):.1f} max {max(v):.1f}"
                    )
        ttnn.deallocate(stamps)
        a.release()


CYCLE_NS = (
    1e3 / 1350
)  # the cores' wall clock counts AICLK cycles (1350 MHz), as fused/layer_mlp/tests/test_transport.py


def us(ticks):
    return (ticks & 0xFFFFFFFF) * CYCLE_NS / 1e3


@galaxy
def test_latency(mesh_device):
    """P1 on 32 chips as a trace, 20 replays and one synchronize, 1 and 8 calls per trace, without and with stamps;
    with stamps the stages of the last replay. P1 runs alone here: nobody takes X3's arrivals on the root (its
    landing cores are its site cores in P1), so the words are zeroed between the measurements. The same chip's
    stamps are on one clock; stages of a chip are given from the moment x arrived on it."""
    hf = golden_layer(1)
    streams = hf["streams_in"][slice(0, 8)].float()
    fid = os.environ.get("HY4_ATTN_FIDELITY", "custom_mm")
    p1_latency(mesh_device, fid, streams, {})


@galaxy
def test_latency16(mesh_device):
    """test_latency for P1 at 8 rows and at 16 rows in one job, both at "recommended" (PLAN.md 19.15)."""
    hf = golden_layer(1)
    p1_latency(mesh_device, "recommended", hf["streams_in"][slice(0, 8)].float(), {}, release=True)
    p1_latency(mesh_device, "recommended, 16 rows", streams16(hf, 0), dict(rows=16), release=True)


def p1_latency(mesh_device, fid, streams, kw, release=False):
    out = {}
    for timing in (False, True):
        case = Case(mesh_device, timing=timing, fidelity=fid.split(",")[0], watch_x3=True, **kw)
        a = case.a
        a.set_streams(streams)
        ttnn.synchronize_device(mesh_device)
        if A.is_mock():
            a.run_p1()
            if release:
                a.release()
            continue
        a.run_p1()
        ttnn.synchronize_device(mesh_device)
        if not timing:
            check_p1(case, streams, f"p1 latency [{fid}] before the trace")
        for calls in (1, 8):
            a.reset_words()
            r = tracing.traced_ms(mesh_device, a.run_p1, calls=calls)
            out[(timing, calls)] = r["replay_ms"] * 1e3
            logger.info(
                f"HY4_MK8A p1 latency [{fid}] timing={timing} {calls} call(s) per trace: {r['replay_ms'] * 1e3:.1f} us per "
                f"call (rounds {[round(v * 1e3, 1) for v in r['rounds_ms']]})"
            )
        status = {k: v for k, v in a.t.status().items()}
        logger.info(
            f"HY4_MK8A p1 latency [{fid}] timing={timing}: transport words after the traces: {status or 'at rest'}"
        )
        if timing:
            topo = a.topo
            words = a.t.read_words()
            five = lambda c, core: [int(v) for v in a.t.core_words(words[c], core)[[8, 9, 10, 11, 15]]]
            pre = {c: five(c, A.PRE_HUB) for c in topo.chips}
            qry = {c: five(c, A.QUERY_HUB) for c in topo.chips}
            site = five(topo.root, A.SITE["origin"])
            root_pre = pre[topo.root]
            logger.info(
                f"HY4_MK8A p1 stages [{fid}] root (us from its kernels' start): x staged and the broadcast op ready "
                f"{us(site[1] - root_pre[0]):.1f}, x in the root's own pre hub {us(root_pre[1] - root_pre[0]):.1f}"
            )
            rows = {
                "pre released": [],
                "q_a units done": [],
                "Xa handed": [],
                "Xa complete": [],
                "query released": [],
                "query units done": [],
                "X3 handed": [],
                "last X3 landing": [],
            }
            for c in topo.chips:
                p, q = pre[c], qry[c]
                t0 = p[1]
                rows["pre released"].append(us(p[2] - t0))
                rows["q_a units done"].append(us(p[3] - t0))
                rows["Xa handed"].append(us(p[4] - t0))
                rows["Xa complete"].append(us(q[1] - t0))
                rows["query released"].append(us(q[2] - t0))
                rows["query units done"].append(us(q[3] - t0))
                rows["X3 handed"].append(us(q[4] - t0))
                if c != topo.root:
                    rows["last X3 landing"].append(max(us(five(c, core)[1] - t0) for core in A.MERGE))
            for name, v in rows.items():
                logger.info(
                    f"HY4_MK8A p1 stages [{fid}] per chip, us after x arrived on the chip: {name}: min {min(v):.1f} median "
                    f"{statistics.median(v):.1f} max {max(v):.1f}"
                )
            logger.info(
                f"HY4_MK8A p1 latency [{fid}]: traced {out[(False, 8)]:.1f} us per call without stamps (8 per trace), "
                f"{out[(False, 1)]:.1f} as a trace of its own; with stamps {out[(True, 8)]:.1f}"
            )
        a.reset()
        if release:  # (several layers in one process)
            a.release()
