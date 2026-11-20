# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program P2 of the attention half (fused/layer_attn/program.py) run alone: the sparse attention from the query
landings, the merge X4 (column stage, row stage, sink), the post group and the reduce X5, for 8 token rows.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k mock
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and attn_solo"
    ... -k "rung and solo", then (each after the rung before it has passed) "rung and pair_col", "pair_row", "galaxy"

What P1 would have left is written by the host: the absorbed queries of all 64 heads on the 8 landing cores with
their arrival counts, and each chip's gate in its pre hub. Random projection weights, a packed KV storage built
with the prefill stack's own pack ops (the same rows and the same selection on every chip: the merge over chips is
then a merge of equal partials, which is still a merge), a random sink.
Checked (every number is logged; lines start with HY4_MK8A):
  words      nothing left in any status word or counter; sparse_mla's status pages and the post group's clean
  partials   every attention core's final partial (O, max, sum) against the torch partial of the same rows
             (sparse_mla layout.partial), and bit-equal to what it wrote to the out tensor
  ctx        the chip's 2 heads against the log-sum-exp merge (layout.merge) of the DEVICE partials of every chip
             that lands on it, and of the sink
  post       the chip's partial of the sublayer output against exact arithmetic on its stored weights for the
             device's ctx and the gate
  F          against the host model of the two-stage sum of the chips' partials (bit-equal)
"""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, reconstruct_scaled_fp8_kv_cache
from models.experimental.hy4_preview.fused.collective.collective import to_bf16
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import (
    ALL,
    cb_report,
    gain,
    galaxy,
    site_params,
    stored_share,
    us,
)
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections.tests.harness import random_reference
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.op import STATUS_MAGIC
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc

RUNGS = {
    "attn_solo": ([(0, 0)], "attention"),
    "solo": ([(0, 0)], "all"),
    # one chip, no fabric connection, with the partners of chip (0, 0) on the whole mesh written by the host: 7 column
    # partials and 3 row partials per user to merge, 7 column ops through two staging slots and the row op as own copies
    "solo_x4": ([(0, 0)], "all"),
    "pair_col": ([(0, 0), (1, 0)], "all"),
    "pair_row": ([(0, 0), (0, 1)], "all"),
    "col4": ([(r, 0) for r in range(4)], "all"),  # 3 column ops a chip: a staging slot is reused after its served bit
    "square": ([(0, 0), (0, 1), (1, 0), (1, 1)], "all"),  # column stage, then row stage of the column results
    "col8": ([(r, 0) for r in range(8)], "all"),  # the full column: 7 column ops a chip, no row stage
    "galaxy": (None, "all"),
}
COUNTS = [64, 37, 128, 1, 200, 64, 0, 90]  # selected rows per user on a chip: one, two and no chunk
FID = os.environ.get("HY4_ATTN_FIDELITY", "recommended")
MERGE_FP32 = os.environ.get("HY4_ATTN_MERGE_FP32")  # "1" / "0": X4's merge kernel with a Float32 / 16-bit DEST
SINK_ARGS = os.environ.get("HY4_ATTN_SINK_ARGS")  # "1": the sink peer as runtime arguments of the merge cores
FIXED_ORDER = os.environ.get(
    "HY4_ATTN_FIXED_ORDER"
)  # "1": X4's merge cores take their partners in index order (PLAN.md 19.14)


class Case2:
    def __init__(self, mesh_device, chips=None, seed=0, counts=COUNTS, **kw):
        self.mesh = mesh_device
        ref, _, _ = random_reference(seed, False)
        # (lw: the weights object another case of the same layer already holds: a second program set then reads the
        # host's streams instead of packing its own)
        self.lw = kw.pop("lw", None) or L.layer_weights(ref.weights(), None)
        self.virtual = kw.get("virtual")
        keep_out = kw.pop("keep_out", True)  # the attention cores also write their partials to the out tensor
        if MERGE_FP32 is not None:
            kw["merge_fp32"] = MERGE_FP32 == "1"
        if SINK_ARGS is not None:
            kw.setdefault("sink_args", SINK_ARGS == "1")
        if FIXED_ORDER is not None:
            kw.setdefault("fixed_order", FIXED_ORDER == "1")
        self.n_rows = kw.get("rows", L.ROWS)  # token rows a pass (16: PLAN.md section 19)
        self.mask = kw.pop("mask", (1 << self.n_rows) - 1)  # 16 rows: the pass's live rows
        if self.n_rows != L.ROWS and len(counts) == L.ROWS:
            counts = list(counts) + list(counts[::-1])
        self.a = a = A.AttnLayer(mesh_device, chips=chips, fidelity=FID, **kw)
        a.set_site([site_params(1)], 0, hy4_tensors(["model.layers.1.input_layernorm.weight"])[0])
        a.set_weights(self.lw)
        self.rows = None
        self.make_storage(counts)
        cfg = self.cfg
        g = torch.Generator().manual_seed(7 + seed)
        self.q_lat = torch.randn(cfg.heads, cfg.users, cfg.width, generator=g).to(torch.bfloat16).float()
        self.sink = torch.randn(cfg.heads, generator=g)
        self.gates = {
            c: torch.sigmoid(torch.randn(self.n_rows, L.HEADS_PER_CHIP * L.V, generator=g)).to(torch.bfloat16).float()
            for c in ALL
        }
        a.set_cache(cfg, self.kv, self.sel, 0, keep_out=keep_out)
        a.set_sink(self.sink)
        self._shares = {}
        self.v_col, self.v_row = {}, {}
        if self.virtual is not None:  # the virtual partners' partials of this chip's head group (group 0)
            bf = lambda t: t.float().to(torch.bfloat16).float()
            for table, indices in ((self.v_col, self.virtual[0]), (self.v_row, self.virtual[1])):
                for index in indices:
                    users = []
                    for u in range(cfg.users):
                        n = (
                            int(torch.randint(0, 3, (1,), generator=g)) * 40
                        )  # 0, 40 or 80 rows: some partners are empty
                        rows = torch.randn(n, cfg.width, generator=g).to(torch.bfloat16).float()
                        o, m, sm_ = SL.partial(self.q_lat[:8, u], rows, cfg)
                        users.append((bf(o), bf(m), bf(sm_)))
                    table[index] = users

    def make_storage(self, counts):
        """Sets cfg, kv (the packed rows, the same on every chip), lists (every user's rows, the same on every
        chip) and sel (the selection table)."""
        self.cfg = cfg = SL.SparseMlaConfig(rows_per_slot=1024, num_layers=1, users=getattr(self, "n_rows", L.ROWS))
        latent, rope = hs.random_latents(cfg.users, cfg.rows_per_slot, cfg, seed=3)
        packed = hs.pack_rows(self.mesh, latent[:, None], rope[:, None])
        self.kv = ttnn.to_memory_config(packed, hs.kv_memory_config(self.mesh, cfg))
        ttnn.deallocate(packed)
        self.lists = [hs.spread_ids(cfg, n, seed=20 + u) for u, n in enumerate(counts)]
        self.sel = hs.make_selection(self.mesh, cfg, self.lists)

    # what check_p2 compares against (a case whose selection or queries are made on the chips overrides these)
    def selected(self, coord, user):
        """(ids, rows [n, 576]): the rows chip ``coord`` reads for ``user``, in the table's order."""
        ids = self.lists[user]
        return ids, self.stored_rows()[self.cfg.slot(user, 0)][ids]

    def queries(self, coord, image):
        """[64 heads, 8 users, 576]: the absorbed queries chip ``coord`` attends with."""
        return self.q_lat

    def gate(self, coord, image):
        """[8, 512]: the chip's gate as its pre hub holds it."""
        return self.gates[coord]

    def landed(self, coord):
        """bool [64]: the heads whose queries the step itself put into the chip's landings."""
        return torch.ones(64, dtype=torch.bool)

    def stored_rows(self):
        """The storage's logical rows [slots, R, 576] as chip 0 holds them (every chip holds the same)."""
        if self.rows is None:
            physical = ttnn.to_torch(self.kv, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
            geometry = MlaKvCacheGeometry(latent_dim=self.cfg.latent, rope_dim=self.cfg.rope)
            self.rows = reconstruct_scaled_fp8_kv_cache(physical[: self.cfg.users], geometry)[:, 0].float()
        return self.rows

    def prefill(self):
        updates, words = self.a.p2_prefill(self.q_lat, {c: self.gates[c] for c in self.a.topo.chips})
        if self.virtual is not None:
            more, more_words = self.a.p2_virtual_prefill(self.v_col, self.v_row, self.mask)
            for key, items in more.items():
                updates.setdefault(key, []).extend(items)
            for key, values in more_words.items():
                words.setdefault(key, {}).update(values)
        self.a.write_arena(updates)
        if words or not self.a.p2_alone:  # (P2 alone without partners' counts: the words stay as the device left them)
            self.a.write_words(words)
        if self.n_rows != L.ROWS:
            self.a.set_mask(self.mask)

    def live(self, user) -> bool:
        return bool((self.mask >> user) & 1)

    def share(self, chip):
        if chip not in self._shares:
            self._shares[chip] = stored_share(self.a, self.a.chip_share(self.lw, chip))
        return self._shares[chip]


def two_stage_sum(a, partials):
    """The host model of the reduce: per mesh row the Float32 sum of the chips' partials in column order, rounded to
    BFloat16 (ties away from zero), then the same over the rows (fused/layer_mlp/tests/test_mlp_layer.py)."""
    rows = []
    for r in a.topo.rows:
        total = None
        for c in a.topo.members[r]:
            total = partials[c].clone() if total is None else total + partials[c]
        rows.append(to_bf16(total, "away"))
    total = rows[0].clone()
    for other in rows[1:]:
        total = total + other
    return to_bf16(total, "away")


def check_p2(case, stage, label):
    a, cfg = case.a, case.cfg
    topo = a.topo
    failed = []
    assert a.streams_from is case.lw, "the layer's streams were packed from other weights than this check's reference"
    status = a.status()
    if case.n_rows != L.ROWS and stage == "all":
        # the sender's own count of the hand-overs it closed, per op (word 3 of an op's hand-over words), against
        # what the program asks of it: with the ready word at rest, every hand-over was taken exactly once
        words = a.read_words()
        sender = A.SENDER[1] * a.grid[0] + A.SENDER[0]
        for coord, asked in a.hand_overs.items():
            closed = [int(words[coord][sender][A.W_HAND + 4 * k + 3]) for k in range(len(asked))]
            asked = [n * getattr(case, "launches", 1) for n in asked]  # (launches since the words were last written)
            if closed != asked:
                failed.append(f"chip {coord}: the sender closed {closed} hand-overs per op, asked {asked}")
        logger.info(
            f"HY4_MK8A {label}: hand-overs asked per op on {len(a.hand_overs)} chips {sorted(set(map(tuple, a.hand_overs.values())))}; "
            f"chips whose sender closed another number: {sum(1 for f in failed if 'hand-overs per op' in f)}"
        )
    if stage != "all":  # nobody takes the partials' counts: one per merge core, 8 per other head group on the sender
        # (16 rows: byte p of a count word counts the live rows of sub-pass p)
        subs = [(case.mask >> (8 * p)) & 0xFF for p in range(case.n_rows // 8)]
        local = lambda i: hex(sum(((m >> i) & 1) << (8 * p) for p, m in enumerate(subs)))
        group = hex(sum(bin(m).count("1") << (8 * p) for p, m in enumerate(subs)))
        for coord in topo.chips:
            want = {(coord, core, "attn"): {A.W_LOCAL: local(i)} for i, core in enumerate(A.MERGE) if local(i) != "0x0"}
            want[(coord, A.SENDER, "attn")] = {A.W_CGROUP + g: group for g in range(8) if g != coord[0]}
            for key, value in want.items():
                if status.get(key) == value:
                    del status[key]
                else:
                    failed.append(f"counts at {key}: {status.get(key)} (wanted {value})")
        a.reset_words()
    if case.virtual is not None and stage == "all":  # the sends to the virtual partners were counted on core VIRT
        (coord,) = topo.chips
        payloads = sum(
            1 for p in range(case.n_rows // 8) if (case.mask >> (8 * p)) & 0xFF
        )  # per op: sub-passes with a live row
        want = {
            A.W_COLARR: hex(payloads * len(case.virtual[0])),
            A.W_ROWARR: hex(payloads * int(bool(case.virtual[1]))),
        }
        if status.pop((coord, A.VIRT, "attn"), None) != {k: v for k, v in want.items() if v != "0x0"}:
            failed.append(f"sends to the virtual partners: counts on {A.VIRT} are not {want}")
        a.reset_words()
    if status:
        failed.append(f"words not at rest: {status}")
    images = {c: a.read_arena(c) for c in topo.chips}
    got = {c: a.read_p2(c, images[c]) for c in topo.chips}
    worst_o, worst_ms, worst_other = 1.0, 0.0, 0.0
    for coord in topo.chips:
        v = got[coord]
        o, m, s, st = a.sm.pages_to_partials(ttnn.to_torch(ttnn.get_device_tensors(a.out)[topo.chip(coord)]))
        if not (st[:, :, 0] == STATUS_MAGIC).all():
            failed.append(f"chip {coord}: a root core did not write its status page")
        if not (st[:, :, 1] == 0).all():
            failed.append(f"chip {coord}: sparse_mla wait flags {st[:, :, 1].tolist()}")
        alive = [u for u in range(cfg.users) if case.live(u)]  # (a row that is not live leaves no partial anywhere)
        if not (
            torch.equal(o[alive], v["o"][alive])
            and torch.equal(m[alive], v["max"][alive])
            and torch.equal(s[alive], v["sum"][alive])
        ):
            failed.append(f"chip {coord}: the partials at the fixed address differ from the out tensor")
        q_lat = case.queries(coord, images[coord])
        for u in alive:
            ids, rows = case.selected(coord, u)
            ro, rm, rs = SL.partial(q_lat[:, u], rows, cfg)
            if ids.numel() == 0:
                if not (torch.equal(v["o"][u], torch.zeros_like(v["o"][u])) and float(v["sum"][u].abs().max()) == 0):
                    failed.append(f"chip {coord} user {u}: the empty partial is not empty")
                continue
            p = pcc(v["o"][u], ro.float())
            worst_o = min(worst_o, p)
            err = torch.maximum((v["max"][u].double() - rm).abs(), (v["sum"][u].double() - rs).abs() / rs)  # per head
            mine = case.landed(coord)
            worst_ms = max(worst_ms, float(err[mine].max()))
            if not bool(mine.all()):
                worst_other = max(worst_other, float(err[~mine].max()))
            if p < 0.99:
                failed.append(f"chip {coord} user {u}: partial O pcc {p:.5f}")
    logger.info(
        f"HY4_MK8A {label} partials on {len(topo.chips)} chips: O pcc min {worst_o:.6f} against the torch partial of the "
        f"same rows, max / sum error at most {worst_ms:.3e}"
        + (
            f" on the heads whose queries were landed in the step, {worst_other:.3e} on the others"
            if worst_other
            else ""
        )
        + "; equal to the out tensor's pages"
    )
    if stage != "all":
        assert not failed, f"{label}: " + "; ".join(failed[:10])
        return
    sink = case.a.sink_peer.double() * cfg.scale  # the logit of the BF16 peer the merge cores hold
    worst = {"ctx": 1.0, "post": 1.0}
    gains, ctx_want, ctx_mine = [], [], []
    partials = {}
    for coord in topo.chips:
        r, c = coord
        v = got[coord]
        if any(v["status"][1:6]):
            failed.append(f"chip {coord} post group status {[hex(w) for w in v['status'][:6]]}")
        srcs_r, srcs_c = ([], []) if case.virtual is not None else a.p2_sources(coord)
        peers = [(r, c)] + [(g, c) for g in srcs_r]
        for k in srcs_c:
            peers += [(r, k)] + [(g, k) for g in a.p2_sources((r, k))[0]]
        heads = [8 * r + 2 * c, 8 * r + 2 * c + 1]
        for u in range(cfg.users):
            if not case.live(u):  # (its ctx rows are whatever the buffer held: nothing reads them)
                continue
            parts = [(got[p]["o"][u][heads], got[p]["max"][u][heads], got[p]["sum"][u][heads]) for p in peers]
            if case.virtual is not None:
                own = [2 * c, 2 * c + 1]  # the chip's heads as rows of the group's partial
                for table in (case.v_col, case.v_row):
                    parts += [(users[u][0][own], users[u][1][own], users[u][2][own]) for users in table.values()]
            want = SL.merge(parts, sink[heads], cfg)
            mine = v["ctx"][:, u]
            if not torch.isfinite(mine).all():
                failed.append(f"chip {coord} user {u}: ctx is not finite")
                continue
            if float(want.abs().max()) == 0:  # no row on any peer: the sink takes everything
                if float(mine.abs().max()) != 0:
                    failed.append(
                        f"chip {coord} user {u}: ctx of a user without rows is not zero ({float(mine.abs().max()):.3e})"
                    )
                continue
            p = pcc(mine, want.float())
            ctx_want.append(want.float())
            ctx_mine.append(mine)
            worst["ctx"] = min(worst["ctx"], p)
            if p < 0.999:
                failed.append(f"chip {coord} user {u}: ctx pcc {p:.5f} over {len(parts)} peers")
        if not torch.equal(v["gate"], case.gate(coord, images[coord])):
            failed.append(f"chip {coord}: the gate in the post hub differs from the pre hub's")
        want, _ = L.post_golden(v["ctx"], v["gate"], case.share(topo.chip(coord)))
        p = pcc(want, v["out"])
        worst["post"] = min(worst["post"], p)
        gains.append(gain(want, v["out"]))
        if p < 0.999:
            failed.append(f"chip {coord}: post partial pcc {p:.5f}")
        partials[coord] = v["out"]
    if case.virtual is not None and case.virtual[0] and case.n_rows == L.ROWS:
        # the last column op's own copy on core VIRT: this chip's partials of that head group, as they were sent
        (coord,) = topo.chips
        g = case.virtual[0][-1]
        v = got[coord]
        sent = a.arena_bytes(images[coord], A.VIRT, 0, A.GROUP_BYTES)
        tiles = A.PART // A.ROW_TILE
        same = True
        for u in range(cfg.users):  # [max / sum tile | 16 O tiles] per user; the tile's other columns are the LLK's own
            part = L.from_row_tiles(A.bytes_bf16(sent[u * A.PART : (u + 1) * A.PART], (tiles, A.ROW_TILE // 2)))
            heads = slice(8 * g, 8 * g + 8)
            same &= torch.equal(part[:, 0], v["max"][u, heads]) and torch.equal(part[:, 1], v["sum"][u, heads])
            same &= torch.equal(part[:, L.TILE :], v["o"][u, heads])
        logger.info(
            f"HY4_MK8A {label}: the last column op's payload (head group {g}) {'equals' if same else 'DIFFERS from'} the chip's partials"
        )
        if not same:
            failed.append(f"the payload of the column op for group {g} differs from the partials")
    f = a.read_root()["f"]
    model = two_stage_sum(a, partials)
    wrong = int((model != f).sum())
    if ctx_want:
        logger.info(
            f"HY4_MK8A {label}: ctx against the host (float64) merge of the same device partials and the sink: gain "
            f"{gain(torch.stack(ctx_want), torch.stack(ctx_mine)):.4f} over {len(ctx_want)} chip-user results"
        )
    logger.info(
        f"HY4_MK8A {label} [{FID}]: ctx vs the merge of the device partials pcc min {worst['ctx']:.6f}; post partial vs "
        f"exact arithmetic on the stored weights pcc min {worst['post']:.6f}, gain {min(gains):.4f} to {max(gains):.4f}; F vs "
        f"the two-stage sum of the {len(partials)} partials: {wrong} of {f.numel()} elements differ"
    )
    if wrong:
        failed.append(f"F differs from the two-stage sum in {wrong} elements")
    assert not failed, f"{label}: " + "; ".join(failed[:10])


@galaxy
def test_mock(mesh_device):
    """Compiles the 32 programs of P2 (and of its attention stage alone) and checks every buffer against L1."""
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    case = Case2(mesh_device, timing=bool(os.environ.get("HY4_ATTN_TIMING")))
    a = case.a
    logger.info(f"HY4_MK8A p2 mock: L1 address space taken by persistent tensors {a.l1_bytes()} B per core")
    a.run_p2("attention")
    a.run_p2()
    a.run_p1(post=True)
    cb_report(
        mesh_device,
        a,
        lambda c: a._p2_chip(c, "all"),
        (("root", (0, 0)), ("head", (1, 0)), ("other", (0, 1))),
        program="p2",
    )
    for program, build in (("p1", lambda c: a._p1_chip(c, True)), ("p2", lambda c: a._p2_chip(c, "all"))):
        for coord in ((0, 0), (0, 1)):
            for name, proc, noc, cores in A.noc_audit(build(coord)[0]):
                logger.info(
                    f"HY4_MK8A {program} mock chip {coord} NoC: {name} on RISC {proc}, NoC {noc}, {cores} cores"
                )
    logger.info("HY4_MK8A p2 mock: P2 and its attention stage compiled")


@galaxy
def test_mock16(mesh_device):
    """P2 at 16 token rows (two sub-passes of 8, a row mask; PLAN.md section 19): compiles the 32 programs and
    checks every buffer against L1; the arena's use by role and the worst core."""
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    before = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank
    case = Case2(mesh_device, rows=16, sink_args=True)
    a = case.a
    logger.info(
        f"HY4_MK8A p2 mock 16 rows: arena {a.lay.arena} B (by role {a.lay.roles}), words {4 * a.n_words} B, L1 address "
        f"space taken by persistent tensors {a.l1_bytes()} B per core ({before} B before the layer was built)"
    )
    a.run_p2("attention")
    a.run_p2()
    cb_report(
        mesh_device,
        a,
        lambda c: a._p2_chip(c, "all"),
        (("root", (0, 0)), ("head", (1, 0)), ("other", (0, 1))),
        program="p2",
    )
    for coord in ((0, 0), (1, 0), (0, 1)):
        for name, proc, noc, cores in A.noc_audit(a._p2_chip(coord, "all")[0]):
            logger.info(f"HY4_MK8A p2 mock 16 rows chip {coord} NoC: {name} on RISC {proc}, NoC {noc}, {cores} cores")
        ops = a.parsed.get(("p2", coord))
        logger.info(f"HY4_MK8A p2 mock 16 rows chip {coord}: sender ops {ops[5] if ops else 0} of 16")
    logger.info("HY4_MK8A p2 mock 16 rows: P2 and its attention stage compiled")


@galaxy
def test_mock_shared(mesh_device):
    """The 8-row programs on the 16-row layout: a second AttnLayer(rows=8, host=the 16-row layer) with the host's
    arena (at the 16-row offsets), words, rope tiles, site and weight streams. Compiles its P1 and P2 for 32 chips
    beside the 16-row P2 and reports what L1 address space the second program set takes."""
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    l1 = lambda: ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank
    case16 = Case2(mesh_device, rows=16, sink_args=True)
    with16 = l1()
    case16.a.run_p2()
    case8 = Case2(mesh_device, host=case16.a, sink_args=True, lw=case16.lw)
    a = case8.a
    assert a.arena is case16.a.arena and a.words is case16.a.words and a.site is case16.a.site
    assert a.post.weights is case16.a.post.weights and a.lay.arena == case16.a.lay.arena
    logger.info(
        f"HY4_MK8A shared mock: persistent L1 per core with the 16-row layer {with16} B, with the 8-row program set "
        f"beside it {l1()} B (+ {l1() - with16} B); the 8-row set's offsets: column staging {a.lay.col}, row staging "
        f"{a.lay.row_stage}, column / row landing {a.lay.m_col} / {a.lay.m_row}, post out {a.lay.post_out}"
    )
    a.run_p2()
    a.run_p1(post=True)
    case16.a.run_p2()
    cb_report(mesh_device, a, lambda c: a._p2_chip(c, "all"), (("root", (0, 0)), ("other", (0, 1))), program="p2")
    cb_report(mesh_device, a, lambda c: a._p1_chip(c, True), (("root", (0, 0)), ("other", (0, 1))), program="p1")
    logger.info("HY4_MK8A shared mock: the 8-row P1 and P2 compiled on the 16-row layout beside the 16-row P2")


def stage_times(case, label):
    """Stamps of the last run (HY4L_TIMING builds), per chip on its own clock, us after its kernels started."""
    a = case.a
    words = a.t.read_words()
    five = lambda c, core: [int(v) for v in a.t.core_words(words[c], core)[[8, 9, 10, 11, 15]]]
    rows = {
        "local partial on the merge cores (queries, rows, attention)": [],
        "column partials merged in": [],
        "row partials merged in": [],
        "ctx written": [],
        "post hub: ctx complete": [],
        "post units released": [],
        "post partial complete": [],
        "partial handed to the reduce": [],
    }
    for c in a.topo.chips:
        hub = five(c, A.POST_HUB)
        t0 = hub[0]
        merges = [five(c, m) for m in A.MERGE]
        rows["local partial on the merge cores (queries, rows, attention)"].append(max(us(m[1] - m[0]) for m in merges))
        rows["column partials merged in"].append(max(us(m[2] - m[0]) for m in merges))
        rows["row partials merged in"].append(max(us(m[3] - m[0]) for m in merges))
        rows["ctx written"].append(max(us(m[4] - m[0]) for m in merges))
        for name, i in (
            ("post hub: ctx complete", 1),
            ("post units released", 2),
            ("post partial complete", 3),
            ("partial handed to the reduce", 4),
        ):
            rows[name].append(us(hub[i] - t0))
    for name, v in rows.items():
        logger.info(
            f"HY4_MK8A {label} stages, us after the chip's kernels started: {name}: min {min(v):.1f} median {statistics.median(v):.1f} max {max(v):.1f}"
        )
    red = five(a.topo.root, a.t.reduce_cores(a.topo.root)[0])
    logger.info(
        f"HY4_MK8A {label} stages, root reduce core (us after its start): stage 1 landed {us(red[1] - red[0]):.1f}, "
        f"stage 2 landed {us(red[2] - red[0]):.1f}, F written {us(red[3] - red[0]):.1f}"
    )


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung):
    """P2 alone on one chip (attention stage only, then whole), on two chips, on all 32: its first runs."""
    chips, stage = RUNGS[rung]
    timing = bool(os.environ.get("HY4_ATTN_TIMING"))
    kw = dict(virtual=(list(range(1, 8)), [1, 2, 3])) if rung == "solo_x4" else {}
    case = Case2(mesh_device, chips=chips, timing=timing, **kw)
    a = case.a
    case.prefill()
    ttnn.synchronize_device(mesh_device)
    a.run_p2(stage)
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    check_p2(case, stage, f"p2 {rung}")
    if timing and stage == "all":
        stage_times(case, f"p2 {rung}")
    # a second step: the same inputs again through the same words, landings and staging
    case.prefill()
    a.run_p2(stage)
    ttnn.synchronize_device(mesh_device)
    check_p2(case, stage, f"p2 {rung} step 2")


LIVE16 = {"16": 0xFFFF, "9": 0x01FF, "8 on one sub-pass": 0x00FF, "4": 0x000F, "1": 0x0001}
LIVE8 = {"8": 0xFF, "4": 0x0F, "1": 0x01}


def spread_selection(case, live):
    """Every LIVE row's 2,048 picks spread at random over the 32 chips (Binomial(2048, 1 / 32) rows per chip and
    row), no pick for a row that is not live: the case's selection table, set on the layer."""
    cfg, a = case.cfg, case.a
    g = torch.Generator().manual_seed(11)
    owner = torch.randint(0, A.CHIPS, (cfg.users, cfg.topk), generator=g)
    per_chip = [
        [int((owner[u] == c).sum()) if (live >> u) & 1 else 0 for u in range(cfg.users)] for c in range(A.CHIPS)
    ]
    tables = [
        SL.selection_table(cfg, [hs.spread_ids(cfg, n, seed=100 * c + u) for u, n in enumerate(per_chip[c])])
        for c in range(A.CHIPS)
    ]
    shards = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tables]
    ttnn.deallocate(case.sel)
    case.sel = ttnn.to_device(
        ttnn.from_host_shards(shards, case.mesh.shape), case.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    a.set_cache(cfg, case.kv, case.sel, 0, keep_out=False)
    flat = [n for row in per_chip for n in row if n]
    return (min(flat), max(flat), sum(flat) / len(flat)) if flat else (0, 0, 0.0)


@galaxy
def test_latency16(mesh_device):
    """P2 ALONE as a trace on 32 chips (AttnLayer(p2_alone=True): the relays do not wait for X3's arrival counts, so
    the program can be replayed on the queries and gates one prefill left), 8 launches per trace: the 16-row P2
    against the number of live rows, and the 8-row P2 in the same job; without and with stamps."""
    out = {}
    for rows, masks in ((16, LIVE16), (8, LIVE8)):
        for timing in (False, True):
            kw = dict(rows=rows) if rows != L.ROWS else {}
            case = Case2(
                mesh_device, counts=[64] * rows, timing=timing, keep_out=False, sink_args=True, p2_alone=True, **kw
            )
            a = case.a
            for name, live in masks.items():
                # 16 rows: the mask says which rows are live (the table keeps every row's picks); 8 rows: a row
                # that is not live has no picks
                lo, hi, mean = spread_selection(case, 0xFFFF if rows != L.ROWS else live)
                case.mask = live if rows != L.ROWS else case.mask
                case.prefill()
                ttnn.synchronize_device(mesh_device)
                a.run_p2()
                if A.is_mock():
                    continue
                ttnn.synchronize_device(mesh_device)
                status = a.status()
                assert not status, f"{rows} rows, {name} live: words not at rest after a launch: {status}"
                r = tracing.traced_ms(mesh_device, a.run_p2, calls=8)
                out[(rows, name, timing)] = r["replay_ms"] * 1e3
                logger.info(
                    f"HY4_MK8A p2 latency [{FID}] {rows}-row P2, {name} live row(s), timing={timing}: "
                    f"{r['replay_ms'] * 1e3:.1f} us per launch, 8 launches per trace (rounds "
                    f"{[round(v * 1e3, 1) for v in r['rounds_ms']]}); selected rows per chip and live row {lo} to {hi}, mean {mean:.1f}"
                )
                status = a.status()
                assert not status, f"{rows} rows, {name} live: words not at rest after the traces: {status}"
                if timing:
                    stage_times(case, f"p2 latency [{FID}] {rows}-row P2, {name} live")
            ttnn.deallocate(case.kv)
            ttnn.deallocate(case.sel)
            a.release()
    for (rows, name, timing), us in out.items():
        if not timing:
            n = int(name.split()[0])
            logger.info(
                f"HY4_MK8A p2 latency summary [{FID}]: {rows}-row P2 with {name} live: {us:.1f} us (with stamps "
                f"{out[(rows, name, True)]:.1f}) = {us / n:.1f} us per live row; {n / (78 * us * 1e-6):.0f} tokens/s if a pass "
                f"were 78 launches of this program alone"
            )


MASKS16 = {"all": 0xFFFF, "nine": 0x01FF, "sparse": 0xA5C3, "second": 0xFF00, "one": 0x0100}


@galaxy
@pytest.mark.parametrize("mask", list(MASKS16))
@pytest.mark.parametrize("rung", ["attn_solo", "solo", "solo_x4", "pair_col", "pair_row", "square", "galaxy"])
def test_rung16(mesh_device, rung, mask):
    """P2 at 16 token rows (two sub-passes of 8 and a row mask; PLAN.md section 19), the rungs of test_rung. Masks:
    every row; 9 rows (the least the host gives the 16-row programs when it packs live rows low); rows scattered
    over both sub-passes; the second sub-pass only and a single row of it (a first sub-pass without a live row: its
    ops are handed over with 0 bytes)."""
    chips, stage = RUNGS[rung]
    kw = dict(virtual=(list(range(1, 8)), [1, 2, 3])) if rung == "solo_x4" else {}
    case = Case2(mesh_device, chips=chips, rows=16, mask=MASKS16[mask], sink_args=True, **kw)
    a = case.a
    for step in ("", " step 2"):  # the second step: the same inputs through the same words, landings and stagings
        case.prefill()
        ttnn.synchronize_device(mesh_device)
        a.run_p2(stage)
        if A.is_mock():
            return
        ttnn.synchronize_device(mesh_device)
        check_p2(case, stage, f"p2 16 rows {rung} mask {mask}{step}")


@galaxy
@pytest.mark.parametrize("rung", ["solo", "solo_x4", "pair_col", "pair_row", "galaxy"])
def test_shared(mesh_device, rung):
    """The two program sets of one layer in turn on the same arena, words and site (PLAN.md 19.11): the 8-row P2 at
    the 16-row layout's offsets (AttnLayer(rows=8, host=)) and the 16-row P2: 8 rows, 16 rows with 9 live, 8 rows,
    16 rows with all live, each checked as in test_rung / test_rung16. Between the launches the host writes only
    what a P2 run alone has no P1 for (the set's queries and gates, an arena write; the mask); the WORDS are never
    written after the layer was built (P2 alone, AttnLayer(p2_alone=True), needs no arrival count), so every
    words-at-rest check is on what the device left. (solo_x4: the virtual partners' arrival counts are words, so
    there the words are written with every prefill.)"""
    chips, stage = RUNGS[rung]
    kw = dict(virtual=(list(range(1, 8)), [1, 2, 3])) if rung == "solo_x4" else dict(p2_alone=True)
    case16 = Case2(mesh_device, chips=chips, rows=16, mask=0x01FF, sink_args=True, **kw)
    case8 = Case2(mesh_device, chips=chips, host=case16.a, sink_args=True, **kw)
    assert case8.a.arena is case16.a.arena and case8.a.lay.cap == 16 and case8.a.lay.m_col == case16.a.lay.m_col
    for step, (case, mask) in enumerate(((case8, None), (case16, 0x01FF), (case8, None), (case16, 0xFFFF))):
        if mask is not None:
            case.mask = mask
            case.launches = 1 if rung == "solo_x4" else step // 2 + 1
        case.prefill()
        ttnn.synchronize_device(mesh_device)
        case.a.run_p2(stage)
        if A.is_mock():
            continue
        ttnn.synchronize_device(mesh_device)
        check_p2(
            case, stage, f"p2 shared {rung} step {step}: {case.n_rows} rows" + (f" mask {mask:#06x}" if mask else "")
        )


@galaxy
@pytest.mark.parametrize("rung", ["solo", "solo_x4"])
def test_chain2(mesh_device, rung):
    """The second chain of a merge core's launch against the same chain run alone: with every row live a merge core
    runs the chain of row i and then the chain of row 8 + i; with only rows 8..15 live it runs the latter alone. ctx
    of rows 8..15 (and the post partial's rows 8..15, which depend on nothing else) must be the same bit for bit:
    anything the first chain's normalising merge left in the merge kernel would show here."""
    chips, stage = RUNGS[rung]
    kw = dict(virtual=(list(range(1, 8)), [1, 2, 3])) if rung == "solo_x4" else {}
    case = Case2(mesh_device, chips=chips, rows=16, mask=0xFFFF, sink_args=True, **kw)
    a = case.a
    got = {}
    for name, mask in (("both", 0xFFFF), ("alone", 0xFF00), ("both again", 0xFFFF)):
        case.mask = mask
        case.prefill()
        ttnn.synchronize_device(mesh_device)
        a.run_p2(stage)
        if A.is_mock():
            return
        ttnn.synchronize_device(mesh_device)
        check_p2(case, stage, f"p2 16 rows {rung} chain2 {name}")
        got[name] = a.read_p2(chips[0])
    for name in ("alone", "both again"):
        ctx = (got[name]["ctx"][:, 8:] != got["both"]["ctx"][:, 8:]).sum().item()
        out = (got[name]["out"][8:] != got["both"]["out"][8:]).sum().item()
        worst = (got[name]["ctx"][:, 8:] - got["both"]["ctx"][:, 8:]).abs().max().item()
        logger.info(
            f"HY4_MK8A p2 16 rows {rung} chain2: rows 8..15, '{name}' against 'both': ctx differs in {ctx} of "
            f"{got['both']['ctx'][:, 8:].numel()} values (largest difference {worst:.3e}), the post partial in {out}"
        )
        assert ctx == 0 and out == 0, f"{rung}: rows 8..15 differ between '{name}' and 'both' (ctx {ctx}, post {out})"


@galaxy
@pytest.mark.parametrize("form", ["tensor", "args"])
def test_sets(mesh_device, form):
    """Two layers' sets on ONE AttnLayer (AttnLayer.layer_set; p2(layer=...)): other projection weights and another
    sink, the same programs. One chip; each set's P2 against its own golden, in the order A, B, A. form: the sink
    peer as an L1 tensor per set, or as runtime arguments of the merge cores (sink_args)."""
    case = Case2(mesh_device, chips=[(0, 0)], sink_args=form == "args")
    a = case.a
    first = (a.layer_set(), case.lw, case.sink)
    ref, _, _ = random_reference(1, False)
    lw = L.layer_weights(ref.weights(), None)
    sink = torch.randn(case.cfg.heads, generator=torch.Generator().manual_seed(99)) * 2.0
    a.set_weights(lw, free=False)
    a.set_sink(sink, new=True)
    second = (a.layer_set(), lw, sink)
    assert second[0].post is not first[0].post and (form == "args" or second[0].sink is not first[0].sink)
    if A.is_mock():
        for layer, _, _ in (first, second):
            a.run_p2(layer=layer)
            a.run_p1(post=True, layer=layer)
        return
    for name, (layer, weights, _) in (("A", first), ("B", second), ("A again", first)):
        case.lw, case._shares = weights, {}
        case.prefill()
        ttnn.synchronize_device(mesh_device)
        a.run_p2(layer=layer)
        ttnn.synchronize_device(mesh_device)
        with a._bound(layer):  # (the checks read the set's sink peer and weights through the layer's attributes)
            check_p2(case, "all", f"p2 sets [{form}] {name}")
    assert not torch.equal(first[0].sink_peer, second[0].sink_peer)


@galaxy
@pytest.mark.parametrize("rung", [r for r in RUNGS if RUNGS[r][1] == "all" and not r.startswith("solo")])
def test_parse(mesh_device, rung):
    """The senders' argument tables of P2 (with the served word) as the kernel reads them on the chips of the rung,
    no fabric connection opened and nothing sent: every chip's page must equal the table the host built. The other
    kernels run into their bounded waits; the words are zeroed afterwards."""
    case = Case2(mesh_device, chips=RUNGS[rung][0], wait_limit=2_000_000, parse_only=True)
    a = case.a
    case.prefill()
    a.run_p2()
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    wrong = []
    for c in a.topo.chips:
        want = a.parsed[("p2", c)]
        page = a.read_parsed(c, len(want))
        logger.info(
            f"HY4_MK8A p2 parse {rung} chip {c}: {len(want)} words, {'equal to the host table' if page == want else 'DIFFERENT'}; "
            f"ops {want[4]}, connections {want[5]}, routes {want[6]}, fabric arguments from index {want[-2]}"
        )
        if page != want:
            wrong.append((c, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(page, want)) if x != y][:8]))
    a.reset()
    a.sm.reset_semaphores()
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"
