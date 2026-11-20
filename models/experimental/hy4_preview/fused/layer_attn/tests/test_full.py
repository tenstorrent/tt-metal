# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention half of a FULL-indexer layer on the 8x4 mesh, first form: the selection made on the chips by the
pieces' own programs, unchanged, between P1 and P2 of fused/layer_attn/program.py:

    P1                     as on a shared layer (the index query of the scan comes from the host in this form)
    scan                   fused/indexer_scan Scan: every chip scores its stripe of each user's index keys and keeps
                           its top-2048 per user (the survivor, in DRAM)
    X2                     fused/merges TopkMergeHub: counts at thresholds over the fabric, 3 rounds; every chip
                           learns how many of its own entries are in the user's global top-2048 (the cut word)
    cut                    fused/indexer_scan Cut: the chip's winners as the selection table of fused/sparse_mla
    P2                     reads its rows by that table

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k mock
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and solo"
    ... -k "rung and pair", -k "rung and galaxy" (each after the rung before it has passed), -k latency

8 users at the AgentX contexts (45609 ... 535958 positions; stripes of 32,768 rows per chip and user, 1,408 to
16,768 visible), random index keys (different on every chip), random index queries and head weights from the host,
random cached rows (1024 distinct rows per user, repeated over the stripe).
Checked (lines start with HY4_MK8A):
  selection  every chip's table (counts and stripe rows, in order) against the table built on the host from the
             chips' own survivors and the global top-2048 by (score, chip, rank) (fused/merges golden): EXACT ids;
             the counts of a user sum to 2048; on two chips the survivor's rows against the host's exact scores
  attention  P2's checks on that selection (partials against the torch partial of the selected rows, ctx against the
             merge of the device partials, post, F bit-equal to the two-stage sum), words at rest
latency: the five programs as ONE trace, and the scan, the merge and the cut each alone.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as isc
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, reconstruct_scaled_fp8_kv_cache
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import galaxy, golden_layer
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p2 import Case2, check_p2
from models.experimental.hy4_preview.fused.merges import topk_merge as tm
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.projections.tests.harness import tables
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs

CONTEXTS = [45609, 71284, 92682, 115666, 144011, 175480, 215110, 535958]  # fused/README.md
ROWS = 32768  # stripe rows per chip and user: a cache of one million positions
SLAB = 17408  # cached rows per user on a chip: covers the 16,768 rows the longest context sees (lead: option B)
SMALL = 1024  # distinct cached rows per user (slab row r holds row r % SMALL)
CHIPS = 32
BITS = (6, 5, 5)  # the merge's rounds: 3 of 65 counts (84 KB of landing; (8, 8) takes 285 KB of L1 on every core)
WEIGHT_SCALE = 0.1
# one chip (it keeps its own top-2048: no merge); the pair the merge's hub form was first run on; all 32
RUNGS = {"solo": [(0, 0)], "pair": [(0, 0), (0, 1)], "galaxy": None}


def layer_streams():
    """The input streams of the 8 token rows, [8, 4, 6144] (layer 1's: the site's checkpoint parameters are that
    layer's): the prefill golden's first 8 rows, or, when the process runs the corrected pairing
    (HY4_ROPE_PAIRING=interleaved with HY4_GOLDEN_DIR, which has decode goldens only), the 6 decode rows of its
    6-layer golden of a 1,000-token prompt and the first two once more."""
    from models.experimental.hy4_preview.reference import hf_harness
    from models.experimental.hy4_preview.reference.attention import rope_pairing

    if rope_pairing() == "interleaved":
        golden = hf_harness.find_decode_golden(6, 1000)
        if golden is None:
            pytest.skip(f"no 6-layer decode golden of a 1000-token prompt under {hf_harness.GOLDEN_ROOT}")
        return golden["layers"][1]["streams_in"][[0, 1, 2, 3, 4, 5, 0, 1]].float()
    return golden_layer(1)["streams_in"][slice(0, 8)].float()


def make_slab(mesh):
    """The cached rows of every chip: (cfg, the packed storage [8, 1, SLAB, 656] as ONE output of the prefill
    stack's pack on every chip, the logical rows [8, SMALL, 576] of slab rows 0 .. SMALL - 1 as the host decodes
    them, seconds of the two packs). Slab row r holds row r % SMALL: the pack works row by row, so the host knows
    every stored row from a read-back of SMALL rows (tests/test_full.py test_setup checks that on two chips)."""
    cfg = SL.SparseMlaConfig(rows_per_slot=SLAB, num_layers=1)
    assert SLAB % SMALL == 0 and max(isc.visible_rows(c, 0, CHIPS, 128, ROWS) for c in CONTEXTS) <= SLAB
    latent, rope = hs.random_latents(cfg.users, SMALL, cfg, seed=3)
    latent, rope = latent.to(torch.bfloat16), rope.to(torch.bfloat16)
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    small = hs.pack_rows(mesh, latent[:, None], rope[:, None])
    physical = ttnn.to_torch(small, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[: cfg.users]
    ttnn.deallocate(small)
    t_small = time.perf_counter() - start
    geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
    small_rows = reconstruct_scaled_fp8_kv_cache(physical, geometry)[:, 0].float()
    times = SLAB // SMALL
    start = time.perf_counter()
    packed = hs.pack_rows(mesh, latent.repeat(1, times, 1)[:, None], rope.repeat(1, times, 1)[:, None])
    kv = ttnn.to_memory_config(packed, hs.kv_memory_config(mesh, cfg))
    ttnn.deallocate(packed)
    ttnn.synchronize_device(mesh)
    t_slab = time.perf_counter() - start
    return cfg, kv, small_rows, (t_small, t_slab)


class FullCase(Case2):
    """Case2 (P2's inputs; here P1 runs too) with a selection the chips make themselves."""

    scan_kw = {}  # more arguments of the ScanConfig (a case whose query P1 writes: query_rows)
    host_query = True  # the scan's query and multipliers are the host's (Scan.set_query)

    def __init__(self, mesh_device, chips=None, timing=False, keep_out=True, **kw):
        self.scan_coords = [(r, c) for r in range(8) for c in range(4)] if chips is None else list(chips)
        self.contexts = list(CONTEXTS)
        super().__init__(mesh_device, chips=chips, timing=timing, keep_out=keep_out, counts=None, **kw)
        a = self.a
        a.set_streams(layer_streams())
        _, _, cos, sin = tables()
        a.set_rope(cos, sin)
        ttnn.synchronize_device(mesh_device)

    def make_storage(self, counts):
        """The cached rows (a stripe of ROWS rows per user on every chip: SMALL distinct rows, repeated, so that
        the host knows every stored row from a read-back of SMALL of them), the index keys (every chip its own),
        the scan, the merge and the cut; the selection table is the cut's."""
        mesh = self.mesh
        self.cfg, self.kv, self.small_rows, _ = make_slab(mesh)
        cfg = self.cfg

        g = torch.Generator().manual_seed(21)
        visible = max(isc.visible_rows(c, 0, CHIPS, 128, ROWS) for c in CONTEXTS)
        base = torch.zeros(cfg.users, ROWS, isc.KEY_DIM)
        base[:, :visible] = torch.randn(cfg.users, visible, isc.KEY_DIM, generator=g).bfloat16().float()
        # every chip its own keys from one draw: the rows shifted by a chip-dependent number within the visible
        # part and the 128 key dimensions given a chip-dependent sign each (a row's score differs on every chip)
        signs = (torch.randint(0, 2, (CHIPS, isc.KEY_DIM), generator=g) * 2 - 1).float()
        shards = []
        for chip in range(CHIPS):
            keys = base.clone()
            keys[:, :visible] = torch.roll(base[:, :visible], shifts=37 * chip, dims=1) * signs[chip]
            self.edit_keys(chip, keys)
            shards.append(
                ttnn.from_torch(
                    keys.reshape(cfg.users, 1, ROWS, isc.KEY_DIM), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
                )
            )
        self.keys = ttnn.to_device(
            ttnn.from_host_shards(shards, mesh.shape), mesh, memory_config=isc.key_memory_config(mesh)
        )
        self.q = torch.randn(cfg.users, isc.HEADS, isc.KEY_DIM, generator=g).bfloat16().float()
        self.w = torch.randn(cfg.users, isc.HEADS, generator=g) * WEIGHT_SCALE
        self.scan_cfg = isc.ScanConfig(
            users=cfg.users, stripe_rows=ROWS, num_chips=CHIPS, chip_block=128, mode="tree", **self.scan_kw
        )
        self.scan = isc.Scan(mesh, self.scan_cfg, self.keys, list(range(cfg.users)), coords=self.scan_coords)
        self.qt, self.s = isc.device_query(self.q, self.w)
        if self.host_query:
            self.scan.set_query(self.qt, self.s)
        self.scan.set_contexts(self.contexts)
        self.cut = isc.Cut(self.scan)
        self.hub = None
        if len(self.scan_coords) > 1:
            self.hub = tm.TopkMergeHub(
                mesh, coords=self.scan_coords, users=cfg.users, bits=BITS, survivors=self.scan.out, cuts=self.cut.cuts
            )
        else:  # one chip: it keeps its own top-2048
            self.cut.set_cuts([isc.cut_by_count(isc.TOPK)] * cfg.users)
        self.sel = self.cut.table
        self.lists_of = {}
        self.q_mag, self.q_off = {}, []

    def edit_keys(self, chip, keys):
        """A chip's index keys [users, ROWS, 128] before they are packed (a case that plants rows)."""

    def exact_pairs(self):
        """[(coord, user)]: the survivors read_selection checks against exact scores."""
        chips = self.scan_coords
        return [
            (c, u) for c in chips[:1] + chips[len(chips) // 2 : len(chips) // 2 + 1] for u in (0, self.cfg.users - 1)
        ]

    def selected(self, coord, user):
        ids = self.lists_of[coord][user]
        return ids, self.small_rows[user][ids % SMALL]  # (ids are below SLAB: guard_table, read_selection)

    def queries(self, coord, image):
        """The absorbed queries as they landed on the chip's 8 landing cores (P1's X3): [64, 8 users, 576]. A head
        whose chip does not take part reads as whatever the landing holds, and P2 attends with exactly that."""
        a = self.a
        n = 4 * L.HEADS_PER_CHIP * P.LAT_TILES
        groups = []
        for core in A.MERGE:
            raw = A.bytes_bf16(a.arena_bytes(image, core, A.A_M_QUERY, n * A.ROW_TILE), (n, A.ROW_TILE // 2))
            groups.append(L.from_row_tiles(raw).reshape(L.ROWS, 8, L.LATENT).permute(1, 0, 2))
        q = torch.cat(groups)
        mine = self.landed(coord)
        # every landed head's query keeps the magnitude of the first step's (P1 computes the same queries from the
        # same streams every step; with ALL chips taking part a change means a stage of P1 read something stale)
        mag = q.abs().amax(dim=(1, 2))
        first = self.q_mag.setdefault(coord, mag)
        off = [
            h for h in range(64) if mine[h] and not (0.5 * float(first[h]) <= float(mag[h]) <= 2.0 * float(first[h]))
        ]
        if off:
            self.q_off.append(
                f"chip {coord}: heads {off[:8]} max |q| {[round(float(mag[h]), 3) for h in off[:8]]} against {[round(float(first[h]), 3) for h in off[:8]]} in the first step"
            )
        if (
            not mine.all()
        ):  # (a rung with absent chips: what their heads' queries are, since the max's error scales with them)
            logger.info(
                f"HY4_MK8A full chip {coord}: landed queries, largest magnitude over the {int(mine.sum())} heads of the chips "
                f"that take part {float(q[mine].abs().max()):.3e}, over the other heads {float(q[~mine].abs().max()):.3e}"
            )
        return q

    def gate(self, coord, image):
        return self.a.read_p1(coord, image)["gate"]

    def landed(self, coord):
        mine = torch.zeros(64, dtype=torch.bool)
        for r, c in self.a.topo.chips:
            mine[8 * r + 2 * c : 8 * r + 2 * c + 2] = True
        return mine

    def select(self):
        self.scan()
        if self.hub is not None:
            self.hub()
        self.cut()

    def step(self):
        self.a.run_p1()
        self.select()
        self.a.run_p2()

    def guard_table(self):
        """What must hold of every chip's table BEFORE P2 may read rows by it (the host between the cut and P2 of
        the first step): per user the count is at most the table's capacity, at most the rows the user sees on
        the chip and at most the chip's valid survivors; every id is a row the user sees, inside the slab."""
        failed = []
        cfg = self.cfg
        if self.hub is not None:  # the merge: no wait ran out, its site is idle
            flags = self.hub.read()["flags"]
            if bool((flags & tm.FLAG_TIMEOUT).any()):
                failed.append(f"the merge: a wait ran out, flags {flags.tolist()}")
            try:
                self.hub.site.check_idle()
            except AssertionError as e:
                failed.append(f"the merge's site is not idle: {str(e)[:300]}")
        for c in self.scan_coords:
            d = self.device_index(c)
            counts, rows = self.cut.read(d)
            valid = self.scan.read(d)["valid"].sum(-1)
            for u in range(cfg.users):
                n, vis = int(counts[u]), isc.visible_rows(self.contexts[u], d, CHIPS, 128, ROWS)
                if n > isc.TOPK or n > vis or n > int(valid[u]):
                    failed.append(
                        f"chip {c} user {u}: count {n} (capacity {isc.TOPK}, visible rows {vis}, valid survivors {int(valid[u])})"
                    )
                    continue
                ids = rows[u, :n]
                if n and (int(ids.min()) < 0 or int(ids.max()) >= min(vis, cfg.rows_per_slot)):
                    failed.append(
                        f"chip {c} user {u}: ids {int(ids.min())} .. {int(ids.max())} outside the {vis} visible rows of a {cfg.rows_per_slot}-row slab"
                    )
        return failed

    def guarded_step(self, label):
        """The step program by program with the host between; P2 is launched only if the cut's table passes
        guard_table (a table the row reader must not see is reported, not run)."""
        a, mesh = self.a, self.mesh
        a.run_p1()
        ttnn.synchronize_device(mesh)
        status = a.status(p1_only=True)
        assert not status, f"{label}: words after P1: {status}"
        self.scan()
        ttnn.synchronize_device(mesh)
        if self.hub is not None:
            self.hub()
            ttnn.synchronize_device(mesh)
        self.cut()
        ttnn.synchronize_device(mesh)
        failed = self.guard_table()
        logger.info(
            f"HY4_MK8A {label}: P1, the scan{', the merge' if self.hub is not None else ''} and the cut have ended; the tables before P2: {failed or 'pass the guard'}"
        )
        assert not failed, f"{label}: P2 NOT launched: " + "; ".join(failed[:8])
        a.run_p2()
        ttnn.synchronize_device(mesh)

    def compile(self):
        self.a.run_p1()
        self.scan.compile()
        if self.hub is not None:
            self.hub.compile()
        self.cut.compile()
        self.a.run_p2()

    def device_index(self, coord):
        return coord[0] * 4 + coord[1]

    def read_selection(self):
        """Every chip's survivors and table; sets self.lists_of[coord] for check_p2. Returns the failures."""
        failed = []
        cfg = self.cfg
        chips = self.scan_coords
        survivors = {c: self.scan.read(self.device_index(c)) for c in chips}
        words = torch.stack([survivors[c]["word"] for c in chips])  # [chips, users, 2048]
        gold = (
            tm.golden(words, isc.TOPK)
            if len(chips) > 1
            else {"keep": torch.stack([survivors[c]["valid"].sum(-1) for c in chips])}
        )
        self.lists_of, wrong, total = {}, 0, torch.zeros(cfg.users, dtype=torch.int64)
        for i, c in enumerate(chips):
            counts, rows = self.cut.read(self.device_index(c))
            self.lists_of[c] = [rows[u, : int(counts[u])] for u in range(cfg.users)]
            total += counts
            for u in range(cfg.users):
                keep = int(gold["keep"][i, u])
                want = survivors[c]["row"][u, :keep]
                if int(counts[u]) != keep or not torch.equal(rows[u, :keep], want):
                    wrong += 1
        want_total = [min(isc.TOPK, sum(int(survivors[c]["valid"][u].sum()) for c in chips)) for u in range(cfg.users)]
        per = [int(len(self.lists_of[c][u])) for c in chips for u in range(cfg.users)]
        logger.info(
            f"HY4_MK8A full selection: {len(chips)} chips x {cfg.users} users, tables differing from the host's "
            f"(the chips' survivors cut at the global top-{isc.TOPK} by score, chip, rank): {wrong}; picks per user "
            f"{total.tolist()} (wanted {want_total}); rows per chip and user {min(per)} to {max(per)}"
        )
        if wrong:
            failed.append(f"{wrong} selection tables differ from the host's")
        if self.hub is not None:
            got = self.hub.read()
            recount = int(got["recounts"].sum())
            logger.info(
                f"HY4_MK8A full merge: flags {sorted(set(got['flags'].flatten().tolist()))}; a later round counted again on {recount} of {len(chips) * cfg.users} user cores"
            )
            if bool((got["flags"] & tm.FLAG_TIMEOUT).any()):
                failed.append("the merge: a wait ran out")
        if total.tolist() != want_total:
            failed.append(f"picks per user {total.tolist()}, wanted {want_total}")
        # two chips' survivors against exact arithmetic on the device's operands: every kept row is visible and
        # scores at least the chip's 2048th best exact score, up to the scores' BF16 step
        pairs = self.exact_pairs()
        for c in dict.fromkeys(c for c, _ in pairs):
            d = self.device_index(c)
            keys = ttnn.to_torch(ttnn.get_device_tensors(self.keys)[d]).float()[:, 0]
            for u in [u for cc, u in pairs if cc == c]:
                vis = isc.visible_rows(self.contexts[u], d, CHIPS, 128, ROWS)
                exact = isc.golden_scores(self.qt[u], self.s[u], keys[u, :vis])
                sv = survivors[c]
                self.check_own(c, u, exact, sv, failed)
                rows = sv["row"][u][sv["valid"][u]]
                if rows.numel() != min(isc.TOPK, vis) or (rows.numel() and int(rows.max()) >= vis):
                    failed.append(f"chip {c} user {u}: {rows.numel()} survivors of {vis} visible rows")
                    continue
                kth = exact.topk(min(isc.TOPK, vis)).values[-1]
                worst = float((kth - exact[rows]).max())
                step = float(exact.abs().max()) / 128
                logger.info(
                    f"HY4_MK8A full scan chip {c} user {u}: {vis} visible rows, {rows.numel()} survivors, the worst "
                    f"survivor {worst:.3e} below the exact {min(isc.TOPK, vis)}th score (a BF16 step of the scores {step:.3e})"
                )
                if worst > 2 * step:
                    failed.append(f"chip {c} user {u}: a survivor scores {worst:.3e} below the exact cut")
        return failed

    def check_own(self, coord, user, exact, survivors, failed):
        """More checks of a (chip, user) pair's survivors against its exact scores (a case whose P1 writes keys)."""


class FullQueryCase(FullCase):
    """The full-indexer layer with NOTHING of the selection from the host (PLAN.md section 21.3): P1 is a full
    layer's (AttnLayer(full=True, append="truncate", index_query=True)): it writes the step's own latent row and
    index key into the stores and the scan's query tiles and multipliers into the scan's tensors; the scan runs with
    query_rows and the head weights as its multipliers. Per pass the host gives the positions (the append's, the RoPE
    tiles, the scan's context lengths = position + 1) and the streams.

    Step s (1, 2): user u is at position CONTEXTS[u] - 2 + s. Before step 1 the key rows of BOTH steps' positions hold
    a PLANTED key, built from the reference's estimate of the step's query so that it outscores every other row the
    user sees on that chip several times over: a scan that read a row before P1's key had replaced it has it as its
    best survivor. Step 1 must not see step 2's planted row at all (it is the first row beyond the visible ones)."""

    scan_kw = dict(query_rows=True)
    host_query = False
    PLANT = 3.0  # a planted key's estimated score, in units of the best other visible row's

    def __init__(self, mesh_device, chips=None, timing=False, keep_out=True):
        from models.experimental.hy4_preview.fused.projections.tests.harness import random_reference

        self.ref, _, self.indexer = random_reference(0, True)
        lw = L.layer_weights(self.ref.weights(), None, self.indexer.weights())
        self.step_no, self.appended, self.planted, self.own_lines = 1, {}, {}, []
        super().__init__(
            mesh_device,
            chips=chips,
            timing=timing,
            keep_out=keep_out,
            lw=lw,
            full=True,
            append="truncate",
            index_query=True,
        )
        a = self.a
        a.set_keys(self.keys, 0, 1)
        a.set_scan(self.scan)
        self.set_step(1)

    @staticmethod
    def positions_of(step):
        return [c - 2 + step for c in CONTEXTS]

    @staticmethod
    def owner(position):
        """(chip, stripe row) of a position."""
        return (position // 128) % CHIPS, (position // 4096) * 128 + position % 128

    def rope_of(self, step):
        from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin
        from models.experimental.hy4_preview.tt import frames

        cos, sin = rope_cos_sin(torch.tensor(self.positions_of(step)), AttentionConfig())
        return (cos, sin), frames.rope_tables(cos, sin)

    def set_step(self, step):
        self.step_no = step
        self.positions = self.positions_of(step)
        self.contexts = [p + 1 for p in self.positions]
        _, (cos, sin) = self.rope_of(step)
        self.a.set_rope(cos, sin)
        self.a.set_positions(self.positions)
        self.scan.set_contexts(self.contexts)
        ttnn.synchronize_device(self.mesh)
        self.prefill_absent()

    def prefill_absent(self):
        """A rung on a part of the mesh: what the chips that do not take part would have landed on the idx core,
        from the reference's estimate (their index heads' queries; the head weights when chip 22 is absent), so that
        the scan scores with all 32 heads. (A host write of the whole arena, mesh idle: P1 rewrites everything else
        P2 reads from it.)"""
        a = self.a
        est_q, est_w = self.estimate[self.step_no]
        bf = lambda t: t.to(torch.bfloat16).float()
        updates = {}
        for chip in range(CHIPS):
            coord = (chip // 4, chip % 4)
            if coord in a.topo.chips:
                continue
            head_w = bf(est_w) if (L.pool_of(chip, True) or ("", 0))[0] == "head_w" else None
            for key, items in a.idx_prefill(coord, bf(est_q[:, :, chip]), head_w).items():
                updates.setdefault(key, []).extend(items)
        if updates:
            a.write_arena(updates)

    def make_storage(self, counts):
        # the reference's estimate of each step's query and head weights on the golden's input (the device's query is
        # this up to its formats and a positive factor per token row): what the planted keys are built from
        from models.experimental.hy4_preview.fused.ihc_site.op import golden_pre
        from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import site_params
        from models.experimental.hy4_preview.reference.attention import rms_norm
        from models.experimental.hy4_preview.tests.common import hy4_tensors
        from models.experimental.hy4_preview.tt import frames

        streams = layer_streams()
        (gamma,) = hy4_tensors(["model.layers.1.input_layernorm.weight"])
        y, _, _, _ = golden_pre(self.a.cfg, *site_params(1), streams)
        x = rms_norm(y, gamma.float(), self.a.cfg.norm_eps).to(torch.bfloat16).float()
        self.estimate = {}
        with torch.no_grad():
            w = self.indexer.head_weights(x)
            # (the device's query is without q_a's norm scalar, one positive factor per token row)
            q_a = torch.nn.functional.linear(x, self.ref.q_a_proj).double()
            norm = torch.rsqrt(q_a.pow(2).mean(dim=1) + self.ref.cfg.norm_eps).float()
            for step in (1, 2):
                (cos, sin), _ = self.rope_of(step)
                q_resid, _, _ = self.ref.latent(x, cos, sin)
                q = (
                    frames.index_to_device(self.indexer.queries(q_resid, cos, sin), L.ROPE) / norm[:, None, None]
                )  # [8, 32, 128]
                self.estimate[step] = (q.transpose(1, 2).contiguous(), w)  # (device_query's form [8, 128, 32], [8, 32])
        super().make_storage(counts)

    def edit_keys(self, chip, keys):
        for step in (1, 2):
            qt, w = self.estimate[step]
            for u, position in enumerate(self.positions_of(step)):
                owner, row = self.owner(position)
                if owner != chip:
                    continue
                vis = isc.visible_rows(position + 1, chip, CHIPS, 128, ROWS)
                assert row == vis - 1, (position, row, vis)
                others = isc.golden_scores(qt[u], w[u], keys[u, : vis - 1 - (step - 1)])
                plant = (qt[u].double() * torch.relu(w[u].double())[None, :]).sum(dim=1)  # the heads of positive weight
                score = float(isc.golden_scores(qt[u], w[u], plant[None])[0])
                assert score > 0, (u, step, score)
                plant = (plant * (self.PLANT * float(others.max()) / score)).float()
                keys[u, row] = plant
                self.planted[(u, step)] = plant

    def exact_pairs(self):
        present = {self.device_index(c): c for c in self.scan_coords}
        pairs = super().exact_pairs()
        for step in range(1, self.step_no + 1):
            for u, position in enumerate(self.positions_of(step)):
                chip, _ = self.owner(position)
                if chip in present and (present[chip], u) not in pairs:
                    pairs.append((present[chip], u))
        return pairs

    def check_own(self, coord, user, exact, sv, failed):
        d = self.device_index(coord)
        valid = sv["valid"][user]
        rows, scores = sv["row"][user][valid], sv["score"][user][valid]
        step_bf16 = float(exact.abs().max()) / 128
        for step in range(1, self.step_no + 1):
            position = self.positions_of(step)[user]
            chip, row = self.owner(position)
            if chip != d:
                continue
            planted = float(
                isc.golden_scores(self.qt[user], self.s[user], self.planted[(user, step)][None].double())[0]
            )
            best = float(exact.max())
            at = (rows == row).nonzero().flatten()
            rank = int((exact > exact[row]).sum())
            seen = f"{float(scores[at[0]]):+.3f} at rank {int(at[0])}" if at.numel() else "not among the survivors"
            self.own_lines.append(
                f"user {user} step {step} position {position} (chip {d}, row {row}): exact score with the key P1 wrote "
                f"{float(exact[row]):+.3f} (rank {rank} of {exact.numel()} visible), the planted key would score {planted:+.3f} "
                f"(best visible row {best:+.3f}); the scan: {seen}"
            )
            if planted < 1.5 * best:
                failed.append(
                    f"user {user} step {step}: the planted key would not stand out ({planted:.3f} against {best:.3f})"
                )
            if at.numel() and abs(float(scores[at[0]]) - float(exact[row])) > 2 * step_bf16:
                failed.append(
                    f"chip {coord} user {user}: the score of position {position} is not the one of the key P1 wrote (a planted key read?)"
                )
            if not at.numel() and rank < isc.TOPK - 64:
                failed.append(
                    f"chip {coord} user {user}: position {position} (exact rank {rank}) is not among the survivors"
                )

    def read_query(self):
        """The query and multipliers the scan read (Scan.qt and the step pages, from DRAM): the same on every chip,
        and the host's assembly of the tiles as they landed on the first chip's idx core. Sets qt, s (what the
        exact scores are taken with). Returns the failures."""
        from models.experimental.hy4_preview.tests.common import pcc

        failed = []
        a, scan = self.a, self.scan
        qts, pages = [], []
        for c in self.scan_coords:
            d = self.device_index(c)
            qts.append(
                ttnn.to_torch(ttnn.get_device_tensors(scan.qt)[d]).float().reshape(L.ROWS, isc.HEADS, isc.KEY_DIM)
            )
            pages.append(
                (ttnn.to_torch(ttnn.get_device_tensors(scan.step)[d]).to(torch.int64) & 0xFFFFFFFF).reshape(8, 32)
            )
        same = sum(torch.equal(q, qts[0]) and torch.equal(p, pages[0]) for q, p in zip(qts, pages))
        got = a.read_idx(self.scan_coords[0])
        dev_q = got["landed"].permute(1, 0, 2).contiguous()
        bits = got["head_w"].to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
        want = torch.zeros(8, 32, dtype=torch.int64)
        want[:, 4] = torch.tensor(self.contexts)
        want[:, 16:] = bits[:, 0::2] | (bits[:, 1::2] << 16)
        ok_q, ok_w = torch.equal(qts[0], dev_q), torch.equal(pages[0], want)
        est_q, est_w = self.estimate[self.step_no]
        mine = self.landed(self.scan_coords[0])[
            0::2
        ]  # (index head h is chip h's: the heads of the chips that take part)
        w_pcc = pcc(est_w, got["head_w"]) if float(got["head_w"].abs().max()) > 0 else float("nan")
        logger.info(
            f"HY4_MK8A full query step {self.step_no}: Scan.qt and the step pages the same on {same} of {len(qts)} chips; "
            f"Scan.qt {'equals' if ok_q else 'DIFFERS from'} the assembly of the landed tiles, the step pages (context, "
            f"multipliers = the landed head weights, nothing else) {'as wanted' if ok_w else 'DIFFER'}; the device's query "
            f"against the reference's on the golden input (heads of the chips that take part) pcc "
            f"{pcc(est_q.transpose(1, 2)[:, mine], dev_q[:, mine]):.5f}, head weights {w_pcc:.5f}"
        )
        if same != len(qts) or not ok_q or not ok_w:
            failed.append(
                f"the scan's query tensors: the same on {same} of {len(qts)} chips, assembly {ok_q}, step pages {ok_w}"
            )
        self.qt, self.s = dev_q.transpose(1, 2).contiguous(), got["head_w"].clone()
        return failed

    def refresh_rows(self):
        """The latent rows P1's append has written so far (the own positions of the steps run), as their owner
        chips' stores hold them: what ``selected`` gives for those ids instead of the slab's repeated rows."""
        present = {self.device_index(c): c for c in self.scan_coords}
        geometry = MlaKvCacheGeometry(latent_dim=self.cfg.latent, rope_dim=self.cfg.rope)
        want = {}
        for step in range(1, self.step_no + 1):
            for u, position in enumerate(self.positions_of(step)):
                chip, row = self.owner(position)
                if chip in present:
                    want.setdefault(chip, []).append((u, row))
        changed = 0
        if not want:
            return
        # (an FP8 tensor is read through the mesh composer only: all 32 stores, 8 slots each)
        everything = ttnn.to_torch(self.kv, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        for chip, items in want.items():
            physical = everything[chip * self.cfg.users : (chip + 1) * self.cfg.users]
            for u, row in items:
                value = (
                    reconstruct_scaled_fp8_kv_cache(physical[u : u + 1, :, row : row + 1].clone(), geometry)
                    .float()
                    .reshape(-1)
                )
                changed += int(not torch.equal(value, self.small_rows[u][row % SMALL]))
                self.appended.setdefault((chip, u), {})[row] = value
        del everything
        logger.info(
            f"HY4_MK8A full query step {self.step_no}: latent rows of the own positions read back from their owner chips' "
            f"stores: {sum(len(v) for v in want.values())}, of which {changed} differ from the slab's row (written by P1's append)"
        )

    def selected(self, coord, user):
        ids, rows = super().selected(coord, user)
        mine = self.appended.get((self.device_index(coord), user))
        if mine:
            rows = rows.clone()
            for row, value in mine.items():
                rows[ids == row] = value
        return ids, rows

    def read_selection(self):
        self.own_lines = []
        failed = self.read_query()
        self.refresh_rows()
        failed += super().read_selection()
        for line in self.own_lines:
            logger.info(f"HY4_MK8A full query own position: {line}")
        picked = []
        for step in range(1, self.step_no + 1):
            for u, position in enumerate(self.positions_of(step)):
                chip, row = self.owner(position)
                for c in self.scan_coords:
                    if self.device_index(c) == chip:
                        picked.append((u, step, bool((self.lists_of[c][u] == row).any())))
        logger.info(
            f"HY4_MK8A full query step {self.step_no}: own positions in the final tables (user, the step that wrote the key, "
            f"picked): {picked}"
        )
        return failed


def run_and_check(case, label, step=True, guarded=False):
    if A.is_mock():
        case.step()
        return
    if guarded:
        case.guarded_step(label)
    elif step:
        case.step()
    ttnn.synchronize_device(case.mesh)
    failed = case.read_selection()
    assert not failed, f"{label}: " + "; ".join(failed[:8])
    check_p2(case, "all", label)
    if case.a.topo.whole_mesh:
        assert not case.q_off, f"{label}: landed queries changed magnitude between steps: " + "; ".join(case.q_off[:4])
    elif case.q_off:
        logger.info(
            f"HY4_MK8A {label}: landed queries whose magnitude left the first step's (absent chips: PLAN.md 15): {case.q_off[:2]}"
        )


def cb_peak(desc):
    """(bytes, core, semaphores): the largest sum of a program's own circular buffers on one core."""
    per_core = {}
    for cb in desc.cbs:
        backed = cb.has_buffer() if callable(cb.has_buffer) else cb.has_buffer
        if backed:
            continue
        for c in ttnn.corerange_to_cores(cb.core_ranges):
            per_core[(c.x, c.y)] = per_core.get((c.x, c.y), 0) + cb.total_size
    worst = max(per_core, key=per_core.get)
    return per_core[worst], worst, len(desc.semaphores)


@galaxy
def test_mock(mesh_device):
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    case = FullCase(mesh_device)
    a = case.a
    case.compile()
    total = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_per_bank
    held = a.l1_bytes()
    logger.info(
        f"HY4_MK8A full mock: L1 address space taken by persistent tensors {held} B per core (the layer's, the scan's and the merge site's)"
    )
    coord = (0, 1)
    empty = lambda: ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
    scan = case.scan.append_to(empty(), coord)
    hub = empty()
    frag = cl.Fragment(mesh_device, coord, hub, case.hub.sender_core, [case.hub.hub_core], case.hub.wait_limit)
    case.hub.append_to(hub, frag, coord)
    frag.finish()
    p1, _ = a._p1_chip(coord, False)
    p2, _ = a._p2_chip(coord, "all")
    for name, desc in (("P1", p1), ("scan", scan), ("merge", hub), ("P2", p2)):
        size, core, sems = cb_peak(desc)
        logger.info(
            f"HY4_MK8A full mock L1 {name}: {len(desc.kernels)} kernels, {len(desc.cbs)} CBs, {sems} program semaphores; "
            f"largest CB set {size} B on core {core}; with the tensors {size + held} of {total} B"
        )
        assert size + held <= total, f"{name} does not fit beside the persistent tensors"
        for kernel, proc, noc, cores in A.noc_audit(desc):
            logger.info(f"HY4_MK8A full mock {name} NoC: {kernel} on RISC {proc}, NoC {noc}, {cores} cores")
    logger.info(
        "HY4_MK8A full mock: the five programs compiled (the cut: 16,448 B of CBs on the users' root scan cores)"
    )


@galaxy
def test_setup(mesh_device):
    """The setup alone, before any of E's programs: the pack of 8 x SLAB rows on all 32 chips in one call, timed,
    and the stored rows of chip (0, 0) and of the last chip read back against the SMALL rows they repeat."""
    cfg, kv, small_rows, (t_small, t_slab) = make_slab(mesh_device)
    if A.is_mock():
        return
    start = time.perf_counter()
    physical = ttnn.to_torch(kv, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    t_read = time.perf_counter() - start
    geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
    want = small_rows.repeat(1, SLAB // SMALL, 1)
    same = {}
    for chip in (0, CHIPS - 1):
        rows = reconstruct_scaled_fp8_kv_cache(physical[chip * cfg.users : (chip + 1) * cfg.users], geometry)[
            :, 0
        ].float()
        same[chip] = int((rows != want).sum())
    logger.info(
        f"HY4_MK8A full setup: the pack of {cfg.users} x {SMALL} rows and its read-back {t_small:.2f} s; the pack of "
        f"{cfg.users} x {SLAB} rows on {CHIPS} chips in one call, to the storage's memory layout, synchronized: "
        f"{t_slab:.2f} s; read-back of the {CHIPS} storages {t_read:.2f} s; elements of chip 0 / chip {CHIPS - 1} that "
        f"differ from row r % {SMALL} of the small pack: {same[0]} / {same[CHIPS - 1]}"
    )
    assert not any(same.values()), same


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung):
    case = FullCase(mesh_device, chips=RUNGS[rung])
    if A.is_mock():
        case.compile()
        return
    # the first step program by program, the host checking the cut's table before P2 is launched; then back to back
    run_and_check(case, f"full {rung}", guarded=True)
    run_and_check(case, f"full {rung} step 2")
    # the five programs (four on one chip) as ONE trace, replayed; then the same checks on what the last replay left
    r = tracing.traced_ms(mesh_device, case.step, repeats=5, rounds=2, calls=1)
    run_and_check(case, f"full {rung} after the replays of the step as one trace", step=False)
    logger.info(f"HY4_MK8A full {rung}: the step as a trace of its own {r['replay_ms'] * 1e3:.1f} us per replay")


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_query_rung(mesh_device, rung):
    """The full-indexer half with nothing of the selection from the host (FullQueryCase): step 1 program by program
    with the host's guard before P2, step 2 one position further back to back (its scan sees the key step 1's P1
    wrote and the one its own P1 writes), then step 2 as ONE trace, replayed, and the checks on what the last replay
    left."""
    case = FullQueryCase(mesh_device, chips=RUNGS[rung])
    if A.is_mock():
        case.compile()
        return
    run_and_check(case, f"full query {rung} step 1", guarded=True)
    case.set_step(2)
    run_and_check(case, f"full query {rung} step 2")
    r = tracing.traced_ms(mesh_device, case.step, repeats=5, rounds=2, calls=1)
    run_and_check(case, f"full query {rung} after the replays of step 2 as one trace", step=False)
    logger.info(
        f"HY4_MK8A full query {rung}: P1 (latent append, key append, index query) | scan | "
        f"{'merge | ' if case.hub is not None else ''}cut | P2 as a trace of its own {r['replay_ms'] * 1e3:.1f} us per replay"
    )


@galaxy
def test_query_latency(mesh_device):
    """The full-indexer half with nothing of the selection from the host (FullQueryCase), 32 chips, 8 rows at the
    AgentX contexts, traced: the five programs per step, 8 steps per trace and as a trace of its own; and P1 + P2 of
    the same layer alone (the table as the last cut left it)."""
    case = FullQueryCase(mesh_device, keep_out=False)
    if A.is_mock():
        case.compile()
        return
    a = case.a
    case.step()
    ttnn.synchronize_device(mesh_device)
    assert not a.status(), f"words not at rest: {a.status()}"
    out = {}

    def layer_alone():
        a.run_p1()
        a.run_p2()

    for name, fn in (("P1 | scan | merge | cut | P2", case.step), ("P1 and P2 alone", layer_alone)):
        for calls in (1, 8):
            out[(name, calls)] = tracing.traced_ms(mesh_device, fn, calls=calls)["replay_ms"] * 1e3
        logger.info(
            f"HY4_MK8A full query latency {name}: {out[(name, 8)]:.1f} us per step with 8 per trace, {out[(name, 1)]:.1f} as a "
            f"trace of its own (32 chips; P1 of a full layer with the latent append, the key append and the index query)"
        )
    whole = out[("P1 | scan | merge | cut | P2", 8)]
    logger.info(
        f"HY4_MK8A full query latency: the attention half of a full-indexer layer with nothing from the host: {whole:.1f} us "
        f"per step = {8 / (78 * whole * 1e-6):.0f} tokens/s for 8 rows if all 78 layers were this half alone; the selection "
        f"(scan, merge, cut) {whole - out[('P1 and P2 alone', 8)]:.1f} us of it; words after the traces: {a.status() or 'at rest'}"
    )
    failed = case.read_selection()
    assert not failed, "after the traces: " + "; ".join(failed[:6])


@galaxy
def test_overlap(mesh_device):
    """Measurement for PLAN.md section 15 (the pair): the query hub's q_a buffer (arena offset A_Q_X = 0 on the query
    hub core, 64 tiles: 2 per chip) as P1 leaves it in step 1 and in step 2, against the final partial P2 leaves at
    the same offset (A_ATT_FINAL = 0: the core is an attention core in P2). Predicted: the tiles of the chips that
    take part are the same in both steps; the absent chips' tiles are zero in step 1; in step 2 the absent tiles
    among the first 17 hold step 1's partial, byte for byte, and the others are still zero."""
    case = FullCase(mesh_device, chips=RUNGS["pair"])
    if A.is_mock():
        case.compile()
        return
    a, mesh = case.a, mesh_device
    tile, n = A.ROW_TILE, 64
    grab = lambda c: bytes(a.arena_bytes(a.read_arena(c), A.QUERY_HUB, A.A_Q_X, n * tile))
    tiles = lambda raw: [raw[i * tile : (i + 1) * tile] for i in range(n)]
    seen = {}
    for step in (1, 2):
        a.run_p1()
        ttnn.synchronize_device(mesh)
        seen[("p1", step)] = {c: tiles(grab(c)) for c in a.topo.chips}
        case.select()
        ttnn.synchronize_device(mesh)
        failed = case.guard_table()
        assert not failed, f"overlap step {step}: P2 NOT launched: " + "; ".join(failed[:8])
        a.run_p2()
        ttnn.synchronize_device(mesh)
        seen[("p2", step)] = {c: tiles(grab(c)) for c in a.topo.chips}
    assert not a.status(), f"words not at rest: {a.status()}"
    zero = bytes(tile)
    present = sorted(a.topo.chip(c) for c in a.topo.chips)
    own = [2 * chip + k for chip in present for k in (0, 1)]
    part = A.PART // tile  # 17 tiles
    wrong = []
    for c in a.topo.chips:
        one, two, left = seen[("p1", 1)][c], seen[("p1", 2)][c], seen[("p2", 1)][c]
        nonzero1 = [i for i in range(n) if i not in own and one[i] != zero]
        nonzero2 = [i for i in range(n) if i not in own and two[i] != zero]
        own_same = all(one[i] == two[i] for i in own)
        stale = [i for i in nonzero2 if i < part and two[i] == left[i]]
        logger.info(
            f"HY4_MK8A full overlap chip {c}: q_a tiles of the chips that take part ({own}) equal in step 1 and step 2: "
            f"{own_same}; other tiles non-zero after P1 of step 1: {nonzero1}; after P1 of step 2: {nonzero2}, of which "
            f"equal to the partial P2 of step 1 left at the same offset: {stale}; P2's partial covers tiles 0..{part - 1}"
        )
        if not own_same or nonzero1 or nonzero2 != [i for i in range(part) if i not in own] or stale != nonzero2:
            wrong.append(str(c))
    assert not wrong, f"the q_a buffer is not as predicted on chips {wrong}"


@galaxy
def test_latency(mesh_device):
    case = FullCase(mesh_device, keep_out=False)
    if A.is_mock():
        case.compile()
        return
    a = case.a
    case.step()
    ttnn.synchronize_device(mesh_device)
    assert not a.status(), f"words not at rest: {a.status()}"
    out = {}

    def shared():
        a.run_p1()
        a.run_p2()

    for name, fn in (
        ("whole half (P1, scan, merge, cut, P2)", case.step),
        ("P1 and P2 alone (the table as the last cut left it)", shared),
        ("scan", case.scan),
        ("merge", case.hub),
        ("cut", case.cut),
    ):
        for calls in (1, 8):
            r = tracing.traced_ms(mesh_device, fn, calls=calls)
            out[(name, calls)] = r["replay_ms"] * 1e3
        logger.info(
            f"HY4_MK8A full latency {name}: {out[(name, 8)]:.1f} us per call with 8 per trace, {out[(name, 1)]:.1f} as a "
            f"trace of its own (32 chips; a replay ends when the slowest chip has)"
        )
    # the scan, one user at a time (the other users see no row): its time against that user's visible rows
    per_user = []
    for u in range(case.cfg.users):
        case.scan.set_contexts([c if v == u else 0 for v, c in enumerate(CONTEXTS)])
        ttnn.synchronize_device(mesh_device)
        r = tracing.traced_ms(mesh_device, case.scan, calls=8)
        vis = [isc.visible_rows(CONTEXTS[u], d, CHIPS, 128, ROWS) for d in range(CHIPS)]
        per_user.append(
            f"user {u} (context {CONTEXTS[u]}, {min(vis)} to {max(vis)} visible rows per chip): {r['replay_ms'] * 1e3:.1f}"
        )
    logger.info(f"HY4_MK8A full latency scan, one user at a time, us per call with 8 per trace: {'; '.join(per_user)}")
    case.scan.set_contexts(CONTEXTS)
    case.step()
    ttnn.synchronize_device(mesh_device)
    whole = out[("whole half (P1, scan, merge, cut, P2)", 8)]
    logger.info(
        f"HY4_MK8A full latency: the attention half of a full-indexer layer, 8 rows at the AgentX contexts: {whole:.1f} us "
        f"per step = {8 / (78 * whole * 1e-6):.0f} tokens/s for 8 rows if all 78 layers were this half alone; words after "
        f"the traces: {a.status() or 'at rest'}"
    )
