# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The index QUERY of a full-indexer layer on the chips (fused/layer_attn/program.py AttnLayer(index_query=True),
PLAN.md section 21.3): P1 writes the scan's own inputs. The pre hub gives the pre group the step's cos / sin tiles
and hands the chip's 4 index-query tiles to the sender; one more Xa op per chip and chip 22's pool op (the head
weights) land on the idx core of every chip, whose two data-movement processors assemble each token row's query as
the scan's 4 tiles and write them into ``Scan.qt`` and the row's 32 multipliers into the step page; the scan
(fused/indexer_scan, unchanged, ScanConfig(query_rows=True), unfolded multipliers) is the chip's next program.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "rung and solo"
    ... -k "parse and galaxy", "rung and pair_row", "rung and galaxy" (each first board run alone, after a note)

A rung: P1 of a full layer with the latent append, the key append and the index query (checkpoint site and norm of
layer 1 on the golden's streams, random projection and indexer weights), then the scan with NOTHING from the host
but the context lengths; two steps, the second one position further on other streams. On a part of the mesh the
absent chips' Xa payloads (q_a, lane sums, pool tiles, key tiles, index-query tiles, head weights: the host golden)
are written into the landings first. The key store: 16 slots (8 users x 2 layer slots, this layer's rank 1) of 4,096
stripe rows per chip, random keys; the scan reads the very tensor P1's key append writes.
Checked on every chip that takes part (lines start with HY4_MK8A):
  words     nothing left in any status word or counter (X3's arrivals aside)
  tiles     the 32 chips' index-query tiles and the head-weight tile as they landed on the idx core: the same on
            every chip; a chip's own tiles bit-equal to its pre group's output tiles 19..22, and against exact
            arithmetic on its stored weights for the device's x (rope part and pass part apart, in the frame before
            the Hadamard basis: the rope dims are rotated); against the reference (Hy4Indexer.queries and
            head_weights with the reference weights, tt/frames.py index_to_device), row by row up to q_a's norm scalar
  written   ``Scan.qt`` read back from DRAM: bit-equal to the host's assembly of the landed tiles ([token row,
            32 heads, 128]); the step pages: words 16..31 the row's head weights as landed, words 4, 5 the host's,
            nothing else
  key       every token row whose position a chip that takes part owns (and whose frame is mapped): the store's row
            holds the key core's packed key (values, as ttnn reads the store after the step)
  scan      launched behind P1: per user the chip's survivors against exact arithmetic on the device's own operands
            (the query and multipliers as read back, the key store as it is AFTER the step, own position included):
            every visible row when the user sees at most 2,048, else the top-2,048 up to the scores' BF16 step; the
            score of the step's own position against the exact score with the key P1 wrote in this step (and how far
            that is from the score with the key the row held before); then the same scan with the same query written
            by the host (Scan.set_query): the survivors' words bit-equal
latency: P1 of a full layer with both appends, without and with the index query, traced; the idx core's stamps.
"""

import os
import statistics

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as isc
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_attn.tests import key_bytes as KB
from models.experimental.hy4_preview.fused.layer_attn.tests.test_append import IDLE, RUNGS
from models.experimental.hy4_preview.fused.layer_attn.tests.test_key_append import KeyCase
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import ALL, galaxy, gain, golden_layer
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections.tests.harness import random_reference
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import frames

Q_LAYERS, Q_RANK = 2, 1  # layer slots a user in the key store, and this layer's
Q_ROWS = 4096  # stripe rows a slot (a stripe the scan's own tests run): 131,072 positions a user
CHIPS = 32
# positions of the 8 token rows in step 1 (step 2: one further): owners chip 0 (twice), chip 1, chip 4, chip 17,
# chip 31; one in a frame the user's table does not map (no key written; the scan sees the row as it was); one idle
# row. Users 3 and 5 see more than 2,048 rows on every chip (2,688 and 3,842 to 3,968): the chip's cut is a real one.
POSITIONS = [
    5,
    4096 + 3,
    128 + 9,
    20 * 4096 + 4 * 128 + 127,
    10 * 4096 + 17 * 128 + 64,
    30 * 4096 + 31 * 128 + 1,
    2 * 4096 + 11,
    IDLE,
]


class QueryCase(KeyCase):
    def __init__(self, mesh_device, chips=None, dry=False, **kw):
        # (the projection groups at the fidelity the model piece runs, unless the environment says otherwise)
        kw.setdefault("fidelity", os.environ.get("HY4_ATTN_FIDELITY", "recommended"))
        g = torch.Generator().manual_seed(17)
        slots = L.ROWS * Q_LAYERS
        keys_host = torch.randn(slots, Q_ROWS, L.IDX_DIM, generator=g).bfloat16().float()
        self.keys = isc.keys_to_device(mesh_device, keys_host)
        super().__init__(
            mesh_device,
            "single",
            chips=chips,
            dry=dry,
            store=(self.keys, Q_LAYERS, Q_RANK, Q_ROWS, None),
            index_query=True,
            **kw,
        )
        a = self.a
        self.ref, _, self.indexer = random_reference(0, True)  # (Case's own draw: the reference of self.lw)
        coords = list(ALL) if chips is None else list(chips)
        self.scan_cfg = isc.ScanConfig(
            users=L.ROWS, stripe_rows=Q_ROWS, num_chips=CHIPS, chip_block=128, mode="tree", query_rows=True
        )
        self.scan = isc.Scan(
            mesh_device, self.scan_cfg, self.keys, [self.slot(u) for u in range(L.ROWS)], coords=coords
        )
        a.set_scan(self.scan)
        self._scans = {}
        self.set_step(1)

    def set_step(self, step):
        """Step 1: the positions above; step 2: each one further. The contexts the scan is given: position + 1 (a
        query sees the keys up to its own position), 0 for the idle row."""
        self.positions = [p if p == IDLE else p + step - 1 for p in POSITIONS]
        self.contexts = [0 if p == IDLE else p + 1 for p in self.positions]
        real = torch.tensor([p if p != IDLE else 0 for p in self.positions])
        self.cos_hf, self.sin_hf = rope_cos_sin(real, AttentionConfig())
        self.cos, self.sin = frames.rope_tables(self.cos_hf, self.sin_hf)
        self.a.set_rope(self.cos, self.sin)
        self.a.set_positions(self.positions)
        self.scan.set_contexts(self.contexts)
        ttnn.synchronize_device(self.mesh)

    def prefill_absent(self, x):
        """Xa of the chips that do not take part, the index-query tiles and the head weights included."""
        super().prefill_absent(x)
        updates = {}
        ones = torch.ones(L.ROWS)
        bf = lambda t: t.to(torch.bfloat16).float()
        for coord in ALL:
            if coord in self.a.topo.chips:
                continue
            chip = self.a.topo.chip(coord)
            g = L.pre_golden(x, ones, self.share(chip)[1], self.cos, self.sin)
            head_w = bf(g["pool"]) if (L.pool_of(chip, True) or ("", 0))[0] == "head_w" else None
            for key, items in self.a.idx_prefill(coord, bf(g["idx_q_dev"]), head_w).items():
                updates.setdefault(key, []).extend(items)
        if updates:
            self.a.write_arena(updates)

    def run_scan(self):
        """The scan on the current rank's slots (a descriptor per rank over the one Scan's tensors)."""
        if self.rank not in self._scans:
            self._scans[self.rank] = self.scan.descriptor(slots=[self.slot(u) for u in range(L.ROWS)])
        return ttnn.generic_op(self.scan.io_tensors(), self._scans[self.rank])

    def chip_keys(self, coord):
        """[slots, stripe rows, 128]: a chip's key store as ttnn reads it."""
        t = ttnn.to_torch(ttnn.get_device_tensors(self.keys)[self.a.topo.chip(coord)]).float()
        return t.reshape(-1, Q_ROWS, L.IDX_DIM).clone()


def mult_words(head_w):
    """[rows, 32] BF16 head weights -> [rows, 16] the words 16..31 of the scan's step page (Scan.set_query)."""
    bits = head_w.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    return bits[:, 0::2] | (bits[:, 1::2] << 16)


def check_query(case, streams, label):
    """The module's "words", "tiles" and "written". Returns (the device's query [8 token rows, 32 heads, 128], the
    head weights [8, 32], the arena images)."""
    a, scan = case.a, case.scan
    topo = a.topo
    failed = []
    status = a.status(p1_only=True)
    if status:
        failed.append(f"words not at rest: {status}")
    images = {c: a.read_arena(c) for c in topo.chips}
    got = {c: a.read_idx(c, images[c]) for c in topo.chips}
    first = got[topo.chips[0]]
    same = sum(
        torch.equal(got[c]["landed"], first["landed"]) and torch.equal(got[c]["head_w"], first["head_w"])
        for c in topo.chips
    )
    if same != len(topo.chips):
        failed.append(
            f"the landed index-query tiles or head weights differ between chips ({same} of {len(topo.chips)} equal the first)"
        )
    x = a.read_root()["x"]
    ones = torch.ones(L.ROWS)
    h_mat = frames.hadamard(L.IDX_DIM).double()
    worst = {"rope": 1.0, "pass": 1.0, "all": 1.0}
    gains = []
    own_equal = 0
    w_pcc = None
    for c in topo.chips:
        chip = topo.chip(c)
        own = got[c]["own"]
        own_equal += int(torch.equal(first["landed"][chip], own))
        g = L.pre_golden(x, ones, case.share(chip)[1], case.cos, case.sin)
        want = g["idx_q_dev"].double()
        # (the Hadamard matrix is symmetric and orthonormal up to its entries' BF16 rounding: @ H leads back)
        back, want_back = own.double() @ h_mat, want @ h_mat
        worst["all"] = min(worst["all"], pcc(want.float(), own))
        gains.append(gain(want, own))
        worst["rope"] = min(worst["rope"], pcc(want_back[:, : L.ROPE].float(), back[:, : L.ROPE].float()))
        worst["pass"] = min(worst["pass"], pcc(want_back[:, L.ROPE :].float(), back[:, L.ROPE :].float()))
        if (L.pool_of(chip, True) or ("", 0))[0] == "head_w":
            w_pcc = pcc(g["pool"], first["head_w"])
    if own_equal != len(topo.chips):
        failed.append(f"a chip's landed tiles are not its pre group's output tiles ({own_equal} of {len(topo.chips)})")
    # the reference: Hy4Indexer.queries / head_weights with the reference weights on the device's x, in the device's
    # frame; the device's query lacks q_a's norm scalar, one positive factor per token row
    with torch.no_grad():
        q_resid, _, _ = case.ref.latent(x, case.cos_hf, case.sin_hf)
        ref_q = frames.index_to_device(case.indexer.queries(q_resid, case.cos_hf, case.sin_hf), L.ROPE)  # [8, 32, 128]
        ref_w = case.indexer.head_weights(x)
        q_a_raw = torch.nn.functional.linear(x, case.ref.q_a_proj)
    norm = torch.rsqrt(q_a_raw.double().pow(2).mean(dim=1) + AttentionConfig().norm_eps)
    present = sorted(topo.chip(c) for c in topo.chips)
    dev_q = first["landed"].permute(1, 0, 2).contiguous()  # [token row, head, 128]
    rows_pcc = [pcc(ref_q[u, present].float(), dev_q[u, present]) for u in range(L.ROWS)]
    rows_gain = [gain(ref_q[u, present], dev_q[u, present]) * float(norm[u]) for u in range(L.ROWS)]
    ref_w_pcc = pcc(ref_w.float(), first["head_w"])
    logger.info(
        f"HY4_MK8A {label}: status {status or 'empty'}; landed tiles the same on {same} of {len(topo.chips)} chips, own tiles "
        f"bit-equal to the pre group's output on {own_equal}; a chip's index query vs exact arithmetic on its stored weights: "
        f"pcc min {worst['all']:.6f}, gain {min(gains):.4f} to {max(gains):.4f} (before the Hadamard basis: rope dims {worst['rope']:.6f}, pass dims {worst['pass']:.6f}); "
        f"head weights vs exact arithmetic on chip 22's stored weights: {'chip 22 absent' if w_pcc is None else f'pcc {w_pcc:.6f}'}; "
        f"vs the reference (Hy4Indexer.queries, index_to_device; heads of the chips that take part): pcc per token row "
        f"{min(rows_pcc):.6f} to {max(rows_pcc):.6f}, gain x q_a's norm scalar {min(rows_gain):.4f} to {max(rows_gain):.4f}; "
        f"head weights vs the reference pcc {ref_w_pcc:.6f}"
    )
    if min(worst.values()) < 0.999 or (w_pcc is not None and w_pcc < 0.9995) or min(rows_pcc) < 0.995:
        failed.append(f"index query pcc {worst}, head weights {w_pcc}, vs the reference {min(rows_pcc):.5f}")
    # what the idx core wrote
    want_words = mult_words(first["head_w"])
    qt_ok = step_ok = 0
    for c in topo.chips:
        d = topo.chip(c)
        qt = ttnn.to_torch(ttnn.get_device_tensors(scan.qt)[d]).float().reshape(L.ROWS, isc.HEADS, L.IDX_DIM)
        qt_ok += int(torch.equal(qt, dev_q))
        page = (ttnn.to_torch(ttnn.get_device_tensors(scan.step)[d]).to(torch.int64) & 0xFFFFFFFF).reshape(8, 32)
        want = torch.zeros(8, 32, dtype=torch.int64)
        want[:, 4] = torch.tensor(case.contexts)
        want[:, 16:] = want_words
        step_ok += int(torch.equal(page, want))
        if not torch.equal(page, want):
            bad = (page != want).nonzero()[:4].tolist()
            failed.append(f"chip {c}: step page words differ at {bad}")
    logger.info(
        f"HY4_MK8A {label}: Scan.qt read back bit-equal to the host's assembly of the landed tiles on {qt_ok} of {len(topo.chips)} "
        f"chips; step pages (words 16..31 the landed head weights, words 4, 5 the host's, the rest zero) as wanted on {step_ok}"
    )
    if qt_ok != len(topo.chips):
        failed.append(f"Scan.qt differs from the assembly of the landed tiles on {len(topo.chips) - qt_ok} chips")
    assert not failed, f"{label}: " + "; ".join(failed[:8])
    return dev_q, first["head_w"].clone(), images


def check_scan(case, dev_q, head_w, images, before, label, rewritten=False):
    """The module's "key" and "scan". ``before``: {coord: the chip's key store before this step}. Returns the stores
    after it. rewritten: the launch before wrote the same rows from the same input (no row of the store changes)."""
    a, scan, cfg = case.a, case.scan, case.scan_cfg
    topo = a.topo
    mesh = case.mesh
    failed = []
    after = {c: case.chip_keys(c) for c in topo.chips}
    # key: the rows P1 must have written hold the packed keys
    packed = KB.values(a.read_key(topo.chips[0], images[topo.chips[0]])["packed"])[: L.ROWS]
    own = {}  # user -> (coord, stripe row) of its own position, when a chip that takes part wrote it
    written = 0
    for chip, blocks in case.writes().items():
        coord = next(c for c in topo.chips if topo.chip(c) == chip)
        for where, rows in blocks.items():
            slot, r0 = case.block_rows(where)
            for r, u in rows.items():
                own[u] = (coord, r0 + r)
                same = np.array_equal(after[coord][slot, r0 + r].double().numpy(), packed[u])
                written += int(same)
                if not same:
                    failed.append(f"chip {chip} slot {slot} row {r0 + r}: not token row {u}'s packed key")
    changed = {c: int((after[c] != before[c]).any(dim=-1).sum()) for c in topo.chips}
    logger.info(
        f"HY4_MK8A {label}: own positions written by chips that take part {sorted((u, topo.chip(c), r) for u, (c, r) in own.items())}: "
        f"{written} of {len(own)} rows hold the packed key; rows changed per chip {sorted(changed.values())}"
    )
    if sum(changed.values()) != (0 if rewritten else len(own)):
        failed.append(f"{sum(changed.values())} rows of the stores changed, {len(own)} wanted")
    # scan: launched with nothing from the host
    case.run_scan()
    ttnn.synchronize_device(mesh)
    qt = dev_q.transpose(1, 2).contiguous()  # [token row, 128, 32]: device_query's form
    got = {c: scan.read(topo.chip(c)) for c in topo.chips}
    worst_err = worst_cut = 0.0
    full_users = cut_users = 0
    own_lines = []
    for c in topo.chips:
        d = topo.chip(c)
        sv = got[c]
        for u in range(cfg.users):
            vis = isc.visible_rows(case.contexts[u], d, CHIPS, 128, Q_ROWS)
            valid = sv["valid"][u]
            rows, scores = sv["row"][u][valid], sv["score"][u][valid]
            if rows.numel() != min(isc.TOPK, vis) or (
                rows.numel() and (int(rows.max()) >= vis or rows.unique().numel() != rows.numel())
            ):
                failed.append(f"chip {c} user {u}: {rows.numel()} survivors of {vis} visible rows")
                continue
            if vis == 0:
                continue
            keys = after[c][case.slot(u), :vis].double()
            exact = isc.golden_scores(qt[u], head_w[u], keys)
            terms = (torch.relu(keys @ qt[u].double()) * head_w[u].double().abs()).sum(dim=1)
            scale = float(terms.max())
            err = float((scores.double() - exact[rows]).abs().max()) / scale
            worst_err = max(worst_err, err)
            if err > 1.5e-2:
                failed.append(
                    f"chip {c} user {u}: a survivor's score is {err:.3e} of the largest sum of terms from the exact one"
                )
            if vis <= isc.TOPK:
                full_users += 1
            else:
                cut_users += 1
                kth = exact.topk(isc.TOPK).values[-1]
                below = float((kth - exact[rows]).max()) / (float(exact.abs().max()) / 128)
                worst_cut = max(worst_cut, below)
                if below > 2.0:
                    failed.append(f"chip {c} user {u}: a survivor scores {below:.2f} BF16 steps below the exact cut")
            if u in own and own[u][0] == c:
                r = own[u][1]
                old = isc.golden_scores(qt[u], head_w[u], before[c][case.slot(u), r : r + 1].double())
                at = (rows == r).nonzero().flatten()
                seen = f"{float(scores[at[0]]):+.4f} at rank {int(at[0])}" if at.numel() else "not among the survivors"
                rank = int((exact > exact[r]).sum())
                own_lines.append(
                    f"user {u} (chip {d}, row {r} of {vis} visible): exact score with this step's key {float(exact[r]):+.4f} "
                    f"(rank {rank}), with the key the row held before {float(old[0]):+.4f}; the scan: {seen}"
                )
                if r >= vis:
                    failed.append(f"user {u}: its own position is not visible")
                elif at.numel() == 0 and rank < isc.TOPK - 64:
                    failed.append(f"user {u}: its own position (exact rank {rank}) is not among the survivors")
                elif at.numel() and abs(float(scores[at[0]]) - float(exact[r])) > 1.5e-2 * scale:
                    failed.append(
                        f"user {u}: the scan's score of its own position is not the exact one with this step's key"
                    )
    logger.info(
        f"HY4_MK8A {label}: the scan behind P1, nothing from the host: {len(topo.chips)} chips x {cfg.users} users; survivors "
        f"against exact arithmetic on the device's query and the key store after the step: largest score error "
        f"{worst_err:.3e} of the user's largest sum of terms; {full_users} (chip, user) pairs with every visible row kept, "
        f"{cut_users} with a cut: the worst survivor {worst_cut:.2f} BF16 steps of the scores below the exact 2,048th"
    )
    for line in own_lines:
        logger.info(f"HY4_MK8A {label}: own position, {line}")
    # the same scan on the same query written by the host
    words = {c: got[c]["word"].clone() for c in topo.chips}
    scan.set_query(qt, head_w)
    scan.set_contexts(case.contexts)
    ttnn.synchronize_device(mesh)
    case.run_scan()
    ttnn.synchronize_device(mesh)
    differ = sum(int((scan.read(topo.chip(c))["word"] != words[c]).sum()) for c in topo.chips)
    logger.info(
        f"HY4_MK8A {label}: the scan on the same query written by the host (Scan.set_query): {differ} survivor words differ"
    )
    if differ:
        failed.append(f"{differ} survivor words differ between P1's query and the host's")
    assert not failed, f"{label}: " + "; ".join(failed[:8])
    return after


@galaxy
def test_mock(mesh_device):
    if not A.is_mock():
        pytest.skip("the mock's compile and size check")
    from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import cb_report

    case = QueryCase(mesh_device)
    a = case.a
    a.run_p1()
    a.run_p1(post=True)
    case.scan.compile()
    names = (("root", (0, 0)), ("key tile", (4, 2)), ("head weights", (5, 2)), ("no pool tile", (7, 3)))
    cb_report(mesh_device, a, lambda c: a._p1_chip(c, True), names)
    for name, coord in names:
        for kernel, proc, noc, cores in A.noc_audit(a._p1_chip(coord, True)[0]):
            if kernel.startswith(("attn_idx", "attn_pre", "attn_key", "attn_append")):
                logger.info(
                    f"HY4_MK8A idx query mock NoC {name} chip: {kernel} processor {proc} NoC {noc} on {cores} core(s)"
                )
    ops = {c: a.parsed[("p1", c)][4] for c in ((0, 0), (0, 1), (4, 2), (5, 2), (7, 3))}
    lay = a.lay
    logger.info(
        f"HY4_MK8A idx query mock: sender ops per chip {ops}; persistent L1 {a.l1_bytes()} B per core (the layer's and the "
        f"scan's); idx cores {A.IDX_CORES}: on the first the landing at arena offset {lay.idx_land} and the head weights at "
        f"{lay.idx_w}; on each its two rows' query buffers at {lay.idx_work}, their multiplier words {lay.idx_mult}, end "
        f"{lay.idx_end} of {lay.arena} B"
    )


@galaxy
@pytest.mark.parametrize("rung", ["pair_row", "galaxy"])
def test_parse(mesh_device, rung):
    """The senders' argument tables of a full layer's P1 with the index query's two ops (every chip's index-query
    tiles and chip 22's head-weight tile, to the idx core), as the kernel reads them, with no fabric connection opened
    and nothing sent (the collective sender's parse-only build): every chip's page must equal the table the host
    built. The other kernels run into their bounded waits (the idx core then writes whatever its landing holds into
    the scan's tensors; nothing of them is checked here and no scan is launched); the key writer is built dry."""
    hf = golden_layer(1)
    case = QueryCase(mesh_device, chips=RUNGS[rung], dry=True, wait_limit=2_000_000, parse_only=True)
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
            f"HY4_MK8A idx query parse {rung} chip {c}: {len(want)} words, {'equal to the host table' if page == want else 'DIFFERENT'}; "
            f"ops {want[4]}, connections {want[5]}, routes {want[6]}, fabric arguments from index {want[-2]}"
        )
        if page != want:
            wrong.append((c, [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(page, want)) if x != y][:8]))
    a.reset()
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung(mesh_device, rung):
    hf = golden_layer(1)
    case = QueryCase(mesh_device, chips=RUNGS[rung])
    a = case.a
    if A.is_mock():
        a.set_streams(hf["streams_in"][slice(0, 8)].float())
        a.run_p1()
        case.scan.compile()
        return
    before = {c: case.chip_keys(c) for c in a.topo.chips}
    for step in (1, 2):
        case.set_step(step)
        streams = hf["streams_in"][slice(8 * (step - 1), 8 * step)].float()
        a.set_streams(streams)
        _, _, x = case.golden_x(streams)
        case.prefill_absent(x.to(torch.bfloat16).float())
        ttnn.synchronize_device(mesh_device)
        a.run_p1()
        ttnn.synchronize_device(mesh_device)
        label = f"idx query {rung} step {step}"
        dev_q, head_w, images = check_query(case, streams, label)
        before = check_scan(case, dev_q, head_w, images, before, label)
        a.reset_words()


@galaxy
def test_sets(mesh_device):
    """Two full-indexer layers on ONE AttnLayer and ONE Scan (what a chain of full layers is made of): LayerSet A
    (the case's weights, rank 1 of the key store's 2 layer slots) and LayerSet B (other projection and indexer
    weights, rank 0), run in the order A, B, A on one chip, each P1 with the scan of its rank behind it (a scan
    descriptor per rank over the same query, step and survivor tensors): every launch against its own layer's
    weights and its own slot of the key store, and the other rank's slots untouched by it."""
    hf = golden_layer(1)
    case = QueryCase(mesh_device, chips=RUNGS["solo"])
    a = case.a
    weights = {"A": (case.lw, case.ref, case.indexer, Q_RANK)}
    ref_b, _, indexer_b = random_reference(5, True)
    weights["B"] = (L.layer_weights(ref_b.weights(), None, indexer_b.weights()), ref_b, indexer_b, 0)
    sets = {"A": a.layer_set()}
    a.set_weights(weights["B"][0], free=False)
    a.set_key_rank(0)
    sets["B"] = a.layer_set()
    streams = hf["streams_in"][slice(0, 8)].float()
    a.set_streams(streams)
    if A.is_mock():
        for name in ("A", "B"):
            a.run_p1(layer=sets[name])
            ttnn.experimental.prepare_generic_op(
                case.scan.io_tensors(),
                case.scan.descriptor(slots=[u * Q_LAYERS + weights[name][3] for u in range(L.ROWS)]),
            )
        assert sets["A"].key_wb is not sets["B"].key_wb and (sets["A"].key_rank, sets["B"].key_rank) == (Q_RANK, 0)
        return
    _, _, x = case.golden_x(streams)
    before = {c: case.chip_keys(c) for c in a.topo.chips}
    for step, name in enumerate(("A", "B", "A"), start=1):
        case.lw, case.ref, case.indexer, case.rank = weights[name]
        case._shares = {}
        case.set_step(step)
        case.prefill_absent(x.to(torch.bfloat16).float())
        ttnn.synchronize_device(mesh_device)
        a.run_p1(layer=sets[name])
        ttnn.synchronize_device(mesh_device)
        label = f"idx query sets, launch {step}: set {name} (rank {case.rank})"
        dev_q, head_w, images = check_query(case, streams, label)
        other = [u * Q_LAYERS + (1 - case.rank) for u in range(L.ROWS)]
        after = check_scan(case, dev_q, head_w, images, before, label)
        untouched = all(torch.equal(after[c][other], before[c][other]) for c in a.topo.chips)
        logger.info(f"HY4_MK8A {label}: the other rank's slots of the key store unchanged: {untouched}")
        assert untouched, label
        before = after
        a.reset_words()


@galaxy
def test_missing(mesh_device):
    """The idx core's status path, which a clean run never takes: one chip, three launches of P1 with the scan
    behind each. (1) clean. (2) the idx core is told to wait for one arrival of index-query tiles more than can come
    (AttnLayer.idx_expect): the master's bounded wait must run out with status 0xDF << 24 | the arrivals it saw,
    the count is left in its word, ALL 8 processors must still be released and write their token row's query (Scan.qt read back equal
    to the host's assembly of the landing as it is, the step pages' multiplier words likewise), every other kernel of
    P1 must end (the program returns; the key core's packed keys and the store's own rows as in a clean step) and
    the scan after it must run and return. (3) the words zeroed, the count as it should be: a clean launch again."""
    hf = golden_layer(1)
    case = QueryCase(mesh_device, chips=RUNGS["solo"], wait_limit=2_000_000)
    a = case.a
    if A.is_mock():
        a.idx_expect, a._programs = 2, {}
        a.set_streams(hf["streams_in"][slice(0, 8)].float())
        a.run_p1()
        return
    (coord,) = a.topo.chips
    before = {c: case.chip_keys(c) for c in a.topo.chips}

    def launch(step, rows):
        case.set_step(step)
        streams = hf["streams_in"][rows].float()
        a.set_streams(streams)
        _, _, x = case.golden_x(streams)
        case.prefill_absent(x.to(torch.bfloat16).float())
        ttnn.synchronize_device(mesh_device)
        a.run_p1()
        ttnn.synchronize_device(mesh_device)
        return streams

    streams = launch(1, slice(0, 8))
    dev_q, head_w, images = check_query(case, streams, "idx query missing arrival, launch 1 (clean)")
    before = check_scan(case, dev_q, head_w, images, before, "idx query missing arrival, launch 1 (clean)")
    a.reset_words()

    a.idx_expect, a._programs = 2, {}
    launch(2, slice(8, 16))
    label = "idx query missing arrival, launch 2 (one arrival short)"
    status = a.status(p1_only=True)
    want = {(coord, A.IDX, "attn"): {A.W_IDX: hex(1), A.W_ASTATUS: hex(A.ST_IDX | 1)}}
    image = a.read_arena(coord)
    got = a.read_idx(coord, image)
    dev_q = got["landed"].permute(1, 0, 2).contiguous()
    d = a.topo.chip(coord)
    qt = ttnn.to_torch(ttnn.get_device_tensors(case.scan.qt)[d]).float().reshape(L.ROWS, isc.HEADS, L.IDX_DIM)
    page = (ttnn.to_torch(ttnn.get_device_tensors(case.scan.step)[d]).to(torch.int64) & 0xFFFFFFFF).reshape(8, 32)
    rows_ok = [bool(torch.equal(qt[u], dev_q[u])) for u in range(L.ROWS)]
    words_ok = bool(torch.equal(page[:, 16:], mult_words(got["head_w"])))
    own_ok = bool(torch.equal(got["landed"][d], got["own"]))
    after = {c: case.chip_keys(c) for c in a.topo.chips}
    packed = KB.values(a.read_key(coord, image)["packed"])[: L.ROWS]
    wrote = []
    for chip, blocks in case.writes().items():
        for where, rows in blocks.items():
            slot, r0 = case.block_rows(where)
            wrote += [
                bool(np.array_equal(after[coord][slot, r0 + r].double().numpy(), packed[u])) for r, u in rows.items()
            ]
    case.scan()
    ttnn.synchronize_device(mesh_device)
    sv = case.scan.read(d)
    counts = [int(sv["valid"][u].sum()) for u in range(L.ROWS)]
    wanted = [min(isc.TOPK, isc.visible_rows(case.contexts[u], d, CHIPS, 128, Q_ROWS)) for u in range(L.ROWS)]
    logger.info(
        f"HY4_MK8A {label}: status {status} (wanted {want}); P1 returned; this chip's own tiles in the landing: {own_ok}; "
        f"token rows whose query in Scan.qt is the assembly of the landing as it is: {rows_ok} (one "
        f"processor each); multiplier words as the landed head weights: {words_ok}; the key core's rows in the store "
        f"{wrote}; the scan after it returned: survivors per user {counts} (visible rows cut at 2,048: {wanted})"
    )
    assert status == want, f"{label}: status {status}, wanted {want}"
    assert all(rows_ok) and words_ok and own_ok and all(wrote) and wrote, label
    assert counts == wanted, f"{label}: the scan's survivor counts {counts}, wanted {wanted}"
    before = after

    a.reset()
    a.idx_expect, a._programs = None, {}
    streams = launch(2, slice(8, 16))
    label = "idx query missing arrival, launch 3 (clean again)"
    dev_q, head_w, images = check_query(case, streams, label)
    check_scan(case, dev_q, head_w, images, before, label, rewritten=True)
    a.reset_words()


@galaxy
def test_latency(mesh_device):
    """What the index query costs P1 of a full layer (PLAN.md 21.3, measured): P1 on 32 chips as a trace (8 per trace
    and as a trace of its own) with the latent and the key append, without and with the index query, in the same
    process; then the stamped build of the latter: on every chip, from the moment x arrived on it, when the first idx
    core had every chip's tiles, when the 8 processors of the idx cores had their token row's query and multipliers in
    DRAM (the writes drained), beside the key core's and the append core's last stamps, the chip's X3 hand-over and its last X3
    landing (where P1 ends without the appends and the query). The writers keep nothing (append_keep=False). P1 runs
    alone, so the words are zeroed between the measurements."""
    from models.experimental.hy4_preview.fused.common import tracing
    from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import us

    hf = golden_layer(1)
    streams = hf["streams_in"][slice(0, 8)].float()
    fid = os.environ.get("HY4_ATTN_FIDELITY", "recommended")
    out = {}
    for name, make in (
        (
            "with both appends",
            lambda timing: KeyCase(mesh_device, timing=timing, fidelity=fid, watch_x3=True, append_keep=False),
        ),
        (
            "with both appends and the index query",
            lambda timing: QueryCase(mesh_device, timing=timing, fidelity=fid, watch_x3=True, append_keep=False),
        ),
    ):
        for timing in (False, True) if "query" in name else (False,):
            case = make(timing)
            a = case.a
            a.set_streams(streams)
            ttnn.synchronize_device(mesh_device)
            a.run_p1()
            if A.is_mock():
                a.release()
                continue
            ttnn.synchronize_device(mesh_device)
            if not timing:  # (the stamped build's watch kernel takes X3's arrival counts itself)
                status = a.status(p1_only=True)
                assert not status, f"{name}: words after P1: {status}"
            for calls in (1, 8):
                a.reset_words()
                r = tracing.traced_ms(mesh_device, a.run_p1, calls=calls)
                out[(name, timing, calls)] = r["replay_ms"] * 1e3
            logger.info(
                f"HY4_MK8A idx query latency [{fid}] P1 of a full layer {name}, timing={timing}: {out[(name, timing, 8)]:.1f} us "
                f"per call with 8 per trace, {out[(name, timing, 1)]:.1f} as a trace of its own; transport words after: {a.t.status() or 'at rest'}"
            )
            if timing:
                topo = a.topo
                words = a.t.read_words()
                five = lambda c, core: [int(v) for v in a.t.core_words(words[c], core)[[8, 9, 10, 11, 15]]]
                rows = {
                    "Xa handed over (pre hub)": [],
                    "every chip's index tiles on the first idx core (the master's release)": [],
                    "the last of the 8 processors released": [],
                    "the first row's query in DRAM": [],
                    "the last row's query in DRAM": [],
                    "one processor: its row's piece reads and writes, drained": [],
                    "key blocks written": [],
                    "latent rows written": [],
                    "X3 handed over": [],
                    "last X3 landing": [],
                    "the last row's query in DRAM after the last X3 landing": [],
                }
                for c in topo.chips:
                    p, q, ap, ky = (five(c, core) for core in (A.PRE_HUB, A.QUERY_HUB, A.APPEND, A.KEY))
                    ix = [five(c, core) for core in A.IDX_CORES]
                    t0 = p[1]
                    released = [us(w[i] - t0) for w in ix for i in (1, 3)]
                    done = [us(w[i] - t0) for w in ix for i in (2, 4)]
                    rows["Xa handed over (pre hub)"].append(us(p[4] - t0))
                    rows["every chip's index tiles on the first idx core (the master's release)"].append(
                        us(ix[0][1] - t0)
                    )
                    rows["the last of the 8 processors released"].append(max(released))
                    rows["the first row's query in DRAM"].append(min(done))
                    rows["the last row's query in DRAM"].append(max(done))
                    rows["one processor: its row's piece reads and writes, drained"] += [
                        us(w[b] - w[a]) for w in ix for a, b in ((1, 2), (3, 4))
                    ]
                    rows["key blocks written"].append(us(ky[4] - t0))
                    rows["latent rows written"].append(us(ap[4] - t0))
                    rows["X3 handed over"].append(us(q[4] - t0))
                    if c != topo.root:
                        last = max(us(five(c, core)[1] - t0) for core in A.MERGE)
                        rows["last X3 landing"].append(last)
                        rows["the last row's query in DRAM after the last X3 landing"].append(max(done) - last)
                for key, v in rows.items():
                    logger.info(
                        f"HY4_MK8A idx query stages [{fid}] per chip, us after x arrived on the chip: {key}: min {min(v):.1f} "
                        f"median {statistics.median(v):.1f} max {max(v):.1f}"
                    )
            a.reset()
            for name_ in ("kv", "keys"):
                if hasattr(case, name_):
                    ttnn.deallocate(getattr(case, name_))
            a.release()
    if out:
        k, q = "with both appends", "with both appends and the index query"
        logger.info(
            f"HY4_MK8A idx query latency [{fid}]: P1 of a full layer {out[(k, False, 8)]:.1f} us with the latent and the key "
            f"append, {out[(q, False, 8)]:.1f} us with the index query too (8 per trace); {out[(k, False, 1)]:.1f} and "
            f"{out[(q, False, 1)]:.1f} as traces of their own"
        )
