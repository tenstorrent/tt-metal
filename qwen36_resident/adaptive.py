# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 2: up to 32 greedy users across two decode geometries in one process (AdaptiveEngine, host driven).

  spec   ResidentDFlashRows: 8 users x 4 verify rows + the DFlash2 drafter + the PP prefill with taps (the built first,
         l1_union: its persistent L1 scratch sized for the plain build too)
  plain  a plain rows ResidentModel (32 rows, 4 row groups) adopting the spec build's weights, KV pool and L1 scratch
         (ResidentModel(share=)); RowsEngine over it and the same PP prefill

A user lives in one of them. Its KV never moves (one pool; the block table goes with it); its recurrent state moves
in snapshot_slot's format (vrows_snapshot / restore_slot one way, snapshot_slot / vrows_restore the other). Returning
to the spec build its drafter window is refilled from its last `window` tokens re-prefilled as a fresh prompt
(option B: acceptance as with the exact context, tests/drafter_refill.py). Only one build runs at a time; every call
synchronizes. Greedy only (the host argmax of the plain build's logits / the spec build's device argmax).
"""
import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession
from qwen36_resident.model import ATTN_TAILS, TOP_K, Dims, ResidentModel, State, is_attn


class User:
    def __init__(self, uid, table, tokens):
        self.uid, self.table = uid, list(table)
        self.tokens = list(tokens)  # every token so far; tokens[pos] is the next one, not yet run
        self.geo, self.idx = None, None
        self.acc = 1  # tokens its last spec step committed (1 after a start / a move)
        self.drafts = None

    @property
    def pos(self):
        return len(self.tokens) - 1

    @property
    def next(self):
        return self.tokens[-1]


class AdaptiveEngine:
    SPEC_USERS, PLAIN_ROWS = 8, 32

    def __init__(self, mesh, ck, max_pos, kv_pool_blocks, R=4):
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        self.mesh, self.vocab = mesh, cfg["vocab_size"]
        d = self.d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=self.vocab)
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        self.spec = ResidentDFlashRows(
            mesh,
            ck,
            users=self.SPEC_USERS,
            max_pos=max_pos,
            R=R,
            prefill=True,
            kv_pool_blocks=kv_pool_blocks,
            l1_union=dict(cand_words=d.n * 2 * TOP_K * self.PLAIN_ROWS, attn_tails=ATTN_TAILS),
        )
        self.R, self.pp = R, self.spec.pp
        sts = [
            State(d, layers - n_attn, n_attn, 0, self.spec.max_pos, zero=True, kv=False) for _ in range(self.PLAIN_ROWS)
        ]
        self.plain = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            sts,
            layers,
            interval,
            lm_head=True,
            embed=self.pp.embed,
            kv_pool_blocks=kv_pool_blocks,
            block_tables=[[0]] * self.PLAIN_ROWS,
            rows=True,
            groups=self.PLAIN_ROWS // 8,
            share=self.spec.m,
        )
        self.plain.reset()
        self.plain.capture_trace()
        self.pp.capture()  # (after the second build: the prefill traces replay L1 placed around both builds' buffers)
        self.sess = SpecRowsSession(self.spec)
        self.emb = Q.embedding(ck).float()
        self.users = {}
        self.free_spec = list(range(self.SPEC_USERS))
        self.free_plain = [int(r) for r in self.plain.row_order()]
        self.spec_tables = [[0] for _ in range(self.SPEC_USERS * R)]
        self.plain_tables = [[0] for _ in range(self.PLAIN_ROWS)]
        self.switch_log = []  # (what, uid, seconds)

    # ------------------------------------------------------------------ helpers
    def sync(self):
        ttnn.synchronize_device(self.mesh)

    def _plain_logits(self, r):
        return self.plain.logits()[r, :, : self.d.vocab_chip].reshape(-1)[: self.vocab].float()

    def _set_spec_table(self, u, table):
        for r in range(u * self.R, (u + 1) * self.R):
            self.spec_tables[r] = list(table)
        self.spec.m.set_page_table(self.spec_tables)

    def _set_plain_table(self, j, table):
        self.plain_tables[j] = list(table)
        self.plain.set_page_table(self.plain_tables)

    def in_geo(self, geo):
        return [u for u in self.users.values() if u.geo == geo]

    # ------------------------------------------------------------------ users
    def add(self, items, geo):
        """new users, items = [(uid, prompt ids, block table)], all into `geo` ("spec" / "plain"): the prompts through the
        PP prefill and the first token (greedy) of each; returns {uid: first token}"""
        out = {}
        if geo == "spec":
            assert len(items) <= len(self.free_spec), "no free spec user"
            starts = []
            for uid, ids, table in items:
                us = self.users[uid] = User(uid, table, ids)
                us.geo, us.idx = "spec", self.free_spec.pop(0)
                self.spec_tables[us.idx * self.R : (us.idx + 1) * self.R] = [list(table)] * self.R
                starts.append((us.idx, list(ids), list(table)))
            for (uid, _, _), lg in zip(items, self.sess.start_many(starts)):
                us = self.users[uid]
                us.tokens.append(int(lg.argmax()))
                us.acc, out[uid], us.first_logits = 1, us.next, lg
            return out
        assert len(items) <= len(self.free_plain), "no free plain row"
        feed = {}
        for uid, ids, table in items:
            us = self.users[uid] = User(uid, table, ids)
            us.geo, us.idx = "plain", self.free_plain.pop(0)
            self._set_plain_table(us.idx, table)
            P = len(ids)
            if P > 1:
                self.pp.reset()
                self.pp.run(torch.as_tensor(list(ids[: P - 1])))
                self.pp.handoff(self.plain, P - 1, block_table=list(table), slot=us.idx)
            else:
                self.plain.clear_slot(us.idx)
            feed[us.idx] = (self.emb[int(ids[-1])], P - 1)
        # (preserve: the live plain users sitting this step out keep their conv history)
        self.plain.step(self.plain.token_state_rows(feed, preserve=True))
        for uid, _, _ in items:
            us = self.users[uid]
            us.first_logits = self._plain_logits(us.idx)
            us.tokens.append(int(us.first_logits.argmax()))
            out[uid] = us.next
        return out

    def remove(self, uid):
        us = self.users.pop(uid)
        if us.geo == "spec":
            self.free_spec.append(us.idx)
            self._set_spec_table(us.idx, [0])
        else:
            self.free_plain.append(us.idx)
            self._set_plain_table(us.idx, [0])

    # ------------------------------------------------------------------ steps
    def spec_step(self):
        """one speculative step of every spec user: propose (from its last commit) -> verify -> greedy accept; returns
        {uid: tokens committed}"""
        U, R = self.SPEC_USERS, self.R
        users = self.in_geo("spec")
        if not users:
            return {}
        anchors, pos, acc, act = [0] * U, [0] * U, [1] * U, [False] * U
        for us in users:
            anchors[us.idx], pos[us.idx], acc[us.idx], act[us.idx] = us.next, us.pos, us.acc, True
        drafts = self.sess.propose(anchors, pos, acc, act)
        tokens = [[anchors[u]] + list(drafts[u]) if act[u] else [0] * R for u in range(U)]
        g = self.sess.verify(tokens, pos, act)
        out = {}
        for us in users:
            u, k = us.idx, 0
            while k < R - 1 and drafts[u][k] == g[u][k]:
                k += 1
            new = list(drafts[u][:k]) + [g[u][k]]
            us.tokens += new
            us.acc, out[us.uid] = len(new), new
        return out

    def plain_step(self):
        """one step of every plain user (its next token at its position, greedy); returns {uid: [token]}"""
        users = self.in_geo("plain")
        if not users:
            return {}
        feed = {us.idx: (self.emb[us.next], us.pos) for us in users}
        self.plain.step(self.plain.token_state_rows(feed, preserve=True))
        out = {}
        for us in users:
            us.tokens.append(int(self._plain_logits(us.idx).argmax()))
            us.acc, out[us.uid] = 1, [us.next]
        return out

    # ------------------------------------------------------------------ moves
    def move_to_plain(self, u, j, table, acc):
        """spec user u's state into plain row j (the vLLM class's primitive): its last verify's commit (acc tokens)
        applied as its next verify would, vrows_snapshot -> restore_slot, the block table moves; u freed"""
        e, s = self.spec, self.sess
        if s.started[u]:  # (no verify since its start / return: the state is at the start's commit)
            ring, slot = e.ring[u], e.slot[u]
        else:
            a = int(acc) - 1
            ring, slot = e.ring[u] + 1 + a, (e.slot[u] + a) % self.R
        snap = e.m.vrows_snapshot(u, ring, slot)
        self._set_plain_table(j, table)
        self.plain.restore_slot(j, snap)
        for k in ("state", "hist"):
            ttnn.deallocate(snap[k])
        self._set_spec_table(u, [0])

    def move_to_spec(self, j, u, table, tokens):
        """plain row j's state into spec user u (ring 0, slot 0), its drafter window refilled from the last `window`
        of tokens[:-1] (the run tokens; tokens[-1] is the next, at position len(tokens) - 1) re-prefilled as a fresh
        prompt at their absolute positions (option B); plain row j's table cleared"""
        e, s = self.spec, self.sess
        snap = self.plain.snapshot_slot(j)
        self._set_spec_table(u, table)
        e.m.vrows_restore(u, snap)
        for k in ("state", "hist"):
            ttnn.deallocate(snap[k])
        n = len(tokens) - 1
        s0 = max(0, n - e.window) // 64 * 64
        self.pp.reset()
        self.pp.run(torch.as_tensor(list(tokens[s0:n])))
        e._fill_context(n - s0, phys=u, offset=s0)
        e.ring[u], e.slot[u], e.prev[u] = 0, 0, None
        s.pos[u], s.started[u], s.committed[u], s.last[u] = n, True, 1, None

    def to_plain(self, uid):
        """a spec user into a free plain row (its KV stays: the block table goes with it)"""
        import time

        t0 = time.perf_counter()
        us = self.users[uid]
        j = self.free_plain.pop(0)
        self.move_to_plain(us.idx, j, us.table, us.acc)
        self.free_spec.append(us.idx)
        us.geo, us.idx, us.acc = "plain", j, 1
        self.switch_log.append(("to_plain", uid, time.perf_counter() - t0))

    def to_spec(self, uid):
        """a plain user into a free spec user, its drafter window refilled (option B)"""
        import time

        t0 = time.perf_counter()
        us = self.users[uid]
        u = self.free_spec.pop(0)
        self.move_to_spec(us.idx, u, us.table, us.tokens)
        self._set_plain_table(us.idx, [0])
        self.free_plain.append(us.idx)
        us.geo, us.idx, us.acc = "spec", u, 1
        self.switch_log.append(("to_spec", uid, time.perf_counter() - t0))
