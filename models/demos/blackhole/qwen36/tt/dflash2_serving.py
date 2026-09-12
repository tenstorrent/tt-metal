# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative decode as a long-lived MULTI-SLOT serving session (vLLM block-output contract).

The demo decoder (DFlash2Decoder.generate) owns a closed batch of B prompts end to end. A server has
B_max SLOTS that requests join and leave at arbitrary steps, and cannot re-capture anything per
request. This decoder keeps every trace and buffer for the server's lifetime and expresses each
request as a slot of the batched substrate:

  alloc()     ONCE: tap bufs (B*T rows), the drafter's per-slot RING context KV + staging buffers.
  warm()      ONCE, eager, before any capture: GDN spec buffers, the per-slot seed programs, the
              drafter's draft/extend programs at B*block rows.
  capture()   ONCE: the verify trace at B users x T rows (dummy per-slot tables), then a dummy
              session so the drafter captures its draft/extend traces. Nothing spec-related
              compiles or allocates after this.
  ingest_prompt(u, taps, T, chunk_start)   per request: its prompt taps -> slot u's drafter ring.
  begin(u, first, T, page_table_row)       per request: seed slot u THROUGH the verify trace: slot u
              replays [first, pad x K] at T with its GDN state loaded from its prefilled decode row
              (seed_spec_state_user), every other slot replays its LAST inputs with a HOLD selector,
              which is a bit-exact no-op for them (the ring kernel reads its initial block before
              writing, the conv window is left as is). Row 0 of slot u gives its pending token and
              its tap row extends the drafter context by slot T.
  step()      ONE draft (all slots) + verify replay (live slots speculate, idle slots hold) + greedy
              accept + extend; returns {slot: committed ids} for the live slots.
  end(u)      slot u leaves (its rows keep holding until another request joins it).
  release()   shutdown.

Deferred commit: like the demo, a slot's accepted-prefix index mi is folded into the NEXT replay's
selectors; a slot's HOLD replays reuse the mi_prev, tokens and positions of its last real replay.
Page tables: the verify trace's per-user tables are restaged (host->device copy) whenever a slot's
vLLM row changes; idle slots point at block 0 (vLLM's never-allocated null block) so their held
rows write nowhere that matters.
"""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2_decode import _DRAFT_TRACED, DFlash2Decoder, _dbg, get_drafter
from models.tt_transformers.tt.common import get_block_size


class DFlash2ServingDecoder(DFlash2Decoder):
    """DFlash2Decoder whose captures and buffers live for the server's lifetime, with B_max slots."""

    def __init__(self, model, num_blocks, draft_len=None, weights_dir=None, stop_tokens=None, ring_tokens=None):
        B = int(model.args.max_batch_size)
        nb_v = int(num_blocks)  # width of vLLM's per-request page-table rows (the verify trace's table width)
        # Dummy per-slot tables for the capture: width nb_v, and DISJOINT block sets across slots (the
        # grouped verify KV write asserts that at capture). Slot u names blocks [u*nbu, (u+1)*nbu).
        nbu = max(1, nb_v // B)
        pt = torch.stack(
            [torch.tensor([u * nbu + (i % nbu) for i in range(nb_v)], dtype=torch.int32) for u in range(B)]
        )
        super().__init__(model, pt, draft_len=draft_len, stop_tokens=stop_tokens, weights_dir=weights_dir)
        assert self.tp, "the serving decoder needs the TP drafter (QWEN36_DFLASH_TP=1)"
        self.nb_v = nb_v
        self.tables = pt.clone()  # live per-slot verify tables (row u = slot u's vLLM row, fitted)
        self._ring_req = ring_tokens
        self.ring = None
        self.block_size = None
        self._alloc_done = False
        self._warm_done = False
        self._captured = False
        T = self.K + 1
        # Per-slot session state.
        self.active = [False] * B
        self.joined = [False] * B  # ever seeded: its hold inputs are real
        self.p = [0] * B
        self.pending = [0] * B
        self.mi = [0] * B  # accepted-prefix index the slot's NEXT replay commits
        self.tok_last = [[0] * T for _ in range(B)]  # inputs of the slot's last real replay (for holds)
        self.pos_last = [0] * B
        self.mi_last = [0] * B  # mi_prev that replay used
        self.iters = [0] * B
        self.accepted = [0] * B
        self.drafted = [0] * B
        self.hist = [[0] * (self.K + 1) for _ in range(B)]
        self._t_begin = [0.0] * B
        self.total_steps = 0

    # ------------------------------------------------------------------ one-time: buffers + programs
    def alloc(self):
        """Allocate every buffer the session traces will bake (before any capture, once)."""
        model = self.model
        assert not self._alloc_done
        T = self.K + 1
        B = self.B
        self.drafter = get_drafter(model, self._weights_dir, block=T)
        assert self.drafter.K == self.K and self.drafter.is_tp
        self.block_size = bs = get_block_size(model._paged_kv_caches)
        widest = max(self.drafter.windows) if all(self.drafter.windows) else 0
        assert widest, "the serving ring needs an all-windowed drafter (DFlash2); DFlash v1 has a full-attention layer"
        ring = (
            int(self._ring_req) if self._ring_req else -(-(widest + 2 * bs) // bs) * bs
        )  # window + >= 1 block of slack
        assert ring % bs == 0 and ring >= widest + T
        self.ring = ring
        nblk = ring // bs
        model._dflash_tap_layers = tuple(self.drafter.taps)
        rows = B * T
        if model._dflash_tap_bufs is not None and (
            len(model._dflash_tap_bufs) != len(model._dflash_tap_layers) or model._dflash_tap_bufs[0].shape[-2] != rows
        ):
            model.free_dflash_tap_bufs()
        if model._dflash_tap_bufs is None:
            model._dflash_tap_bufs = model.alloc_dflash_tap_bufs(rows)
        ring_pt = torch.stack([torch.arange(u * nblk, (u + 1) * nblk, dtype=torch.int32) for u in range(B)])
        self.drafter.alloc(ring_pt, bs, model._dflash_tap_bufs, ring_tokens=ring)
        self.ctx_len = [0] * B
        self._alloc_done = True
        logger.info(
            f"[dflash2-serve] allocated: slots={B} K={self.K} T={T} verify table width {self.nb_v}, "
            f"drafter ring {ring} positions ({nblk} blocks x {bs}) per slot"
        )

    def warm(self):
        """Compile every program the post-capture path needs, EAGERLY and before any capture: the GDN
        spec buffers + per-slot seed (each slice_write offset hashes separately), the B*block-row
        draft and the (all-scratch) extend."""
        assert self._alloc_done and not self._warm_done and not self._captured
        model = self.model
        T = self.K + 1
        for dn in self._gdn:
            dn.prepare_spec_verify(self.B, T)
        for u in range(self.B):
            for dn in self._gdn:
                dn.seed_spec_state_user(u)
        ttnn.synchronize_device(self.mesh)
        self.drafter.draft([0] * self.B, [1] * self.B, traced=False)
        self.drafter.extend_context([0] * self.B, [0] * self.B, traced=False)
        ttnn.synchronize_device(self.mesh)
        self._warm_done = True
        _dbg("warm done")

    def capture(self, warm_position=64):
        """Capture the verify trace against the dummy slot tables (then the drafter's own traces are
        captured by the first traced step). ``warm_position``: where the two throwaway passes write."""
        assert self._alloc_done and self._warm_done and not self._captured
        model = self.model
        model._dflash_tap = True
        try:
            model.capture_verify_trace(
                self.tables, self.K + 1, warm_positions=[int(warm_position)] * self.B, decode_cfg=True
            )
        finally:
            model._dflash_tap = False
        model._vfy_owner = self
        self._vfy_captured = True
        ttnn.synchronize_device(self.mesh)
        self._captured = True
        # From here on an IDLE slot points at block 0 (vLLM's never-allocated null block): its held
        # rows keep replaying, and their attention KV writes must land nowhere a request owns. The
        # dummy disjoint tables were only for the capture-time grouped-write check.
        self.tables[:] = 0
        model.refresh_verify_page_tables(self.tables)
        logger.info(f"[dflash2-serve] verify trace captured (B={self.B} x T={self.K + 1})")

    # ------------------------------------------------------------------ per-request context
    def ingest_prompt(self, u, taps, T, chunk_start=0):
        """Prompt taps of slot u (5 fractured [1,1,S,dim/tp] device tensors from an eager prefill chunk) ->
        the drafter's ring for slot u at positions chunk_start..; ``T`` = positions covered after this
        chunk. Frees the taps."""
        assert self._alloc_done
        _dbg(f"ingest slot={u} T={T} rows={taps[0].shape[-2]} start={chunk_start}")
        self.drafter.fill_context(taps, int(chunk_start), user=u, valid_len=int(T) - int(chunk_start))
        for t in taps:
            ttnn.deallocate(t)
        self.ctx_len[u] = int(T)

    def fit_row(self, row):
        """vLLM's page-table row (torch [nb] or [1, nb]) -> [nb_v] int32 (zero-padded / trimmed)."""
        row = torch.as_tensor(row).reshape(-1).to(torch.int32)
        n = row.shape[0]
        if n < self.nb_v:
            row = torch.cat([row, torch.zeros(self.nb_v - n, dtype=torch.int32)])
        elif n > self.nb_v:
            row = row[: self.nb_v]
        return row.contiguous()

    def set_table(self, u, row):
        """Point slot u's verify rows at this request's blocks (host->device restage, no capture)."""
        fitted = self.fit_row(row)
        if torch.equal(self.tables[u], fitted):
            return
        self.tables[u] = fitted
        if self._captured:
            self.model.refresh_verify_page_tables(self.tables)

    def _hold_inputs(self):
        T = self.K + 1
        tokens = [list(self.tok_last[u]) for u in range(self.B)]
        positions = list(self.pos_last)
        mi_prev = list(self.mi_last)
        assert all(len(t) == T for t in tokens)
        return tokens, positions, mi_prev

    def begin(self, u, first, T, page_table_row):
        """Start slot u's session after ingest_prompt(u, ...): seed through the verify trace."""
        assert self._captured, "begin() needs the captured session (warm-up must have run)"
        assert self.ctx_len[u] == int(T), f"slot {u}: ingest_prompt({T}) must precede begin (ctx_len={self.ctx_len[u]})"
        assert not self.active[u], f"slot {u} is still active; end() it first"
        self._t_begin[u] = time.perf_counter()
        self.set_table(u, page_table_row)
        # Slot u's durable GDN row (written by its prefill) -> ring slot 0 + window row u. Every other
        # slot's ring blocks / window rows are untouched.
        for dn in self._gdn:
            dn.seed_spec_state_user(u)
        tokens, positions, mi_prev = self._hold_inputs()
        tokens[u] = [int(first)] * (self.K + 1)  # row 0 = the seed; rows 1..K are junk the next verify overwrites
        positions[u] = int(T)
        mi_prev[u] = 0
        hold = [v for v in range(self.B) if v != u]
        _dbg(f"seed slot={u} T={T} hold={hold}")
        ids, _feed, _ = self.model.verify_traced(tokens, positions, mi_prev, read_logits=False, hold=hold)
        Tt = self.K + 1
        self.pending[u] = int(ids[u * Tt])
        # The verify trace copied its taps: slot u's row 0 (position T) -> its drafter ring slot T.
        self.drafter.extend_context(
            [int(T) if v == u else 0 for v in range(self.B)],
            [1 if v == u else 0 for v in range(self.B)],
            traced=_DRAFT_TRACED,
        )
        self.ctx_len[u] = int(T) + 1
        self.p[u] = int(T)
        self.mi[u] = 0
        self.tok_last[u], self.pos_last[u], self.mi_last[u] = tokens[u], int(T), 0
        self.active[u] = self.joined[u] = True
        self.iters[u] = self.accepted[u] = self.drafted[u] = 0
        self.hist[u] = [0] * (self.K + 1)
        _dbg(f"seed slot={u} -> pending {self.pending[u]}")

    # ------------------------------------------------------------------ the loop
    def step(self):
        """One speculative iteration over the live slots. Returns {slot: committed ids}."""
        assert self._captured
        B, T = self.B, self.K + 1
        live = [u for u in range(B) if self.active[u]]
        if not live:
            return {}
        anchors = [self.pending[u] if self.active[u] else 0 for u in range(B)]
        Cs = [self.p[u] + 1 if self.active[u] else 1 for u in range(B)]  # idle slots draft junk into their own ring
        for u in live:
            assert self.ctx_len[u] >= self.p[u] + 1, f"slot {u}: context {self.ctx_len[u]} < p+1 {self.p[u] + 1}"
        if not self._armed and _DRAFT_TRACED:
            self.drafter.arm_traces()
            self._armed = True
        drafts = self.drafter.draft(anchors, Cs, traced=_DRAFT_TRACED)
        tokens, positions, mi_prev = self._hold_inputs()
        for u in live:
            tokens[u] = [self.pending[u]] + [int(d) for d in drafts[u]]
            positions[u] = self.p[u] + 1
            mi_prev[u] = self.mi[u]
        hold = [u for u in range(B) if not self.active[u]]
        ids, _feed, _ = self.model.verify_traced(tokens, positions, mi_prev, read_logits=False, hold=hold)
        committed = {}
        slot0 = [0] * B
        nrows = [0] * B
        next_pending = dict()
        for u in live:
            row_ids = ids[u * T : (u + 1) * T]
            m = self._accept_greedy(drafts[u], row_ids)  # updates the shared histogram fields
            com = [self.pending[u]] + [int(d) for d in drafts[u][:m]]
            stop_i = next((i for i, t in enumerate(com) if t in self.stop_tokens), None)
            if stop_i is not None:
                com = com[: stop_i + 1]
            mi_u = len(com) - 1
            committed[u] = com
            next_pending[u] = int(row_ids[mi_u])
            self.tok_last[u], self.pos_last[u], self.mi_last[u] = tokens[u], positions[u], mi_prev[u]
            self.mi[u] = mi_u
            slot0[u], nrows[u] = self.p[u] + 1, len(com)
            self.iters[u] += 1
            self.accepted[u] += m
            self.drafted[u] += self.K
            self.hist[u][m] += 1
        self.drafter.extend_context(slot0, nrows, traced=_DRAFT_TRACED)
        for u in live:
            self.ctx_len[u] = slot0[u] + nrows[u]
            self.p[u] += nrows[u]
            self.pending[u] = next_pending[u]
        self.total_steps += 1
        return committed

    def end(self, u):
        """Slot u's request is done: its rows keep holding their last inputs until the next join."""
        if not self.active[u]:
            return
        self.active[u] = False
        # Its blocks go back to vLLM now; the slot's held rows must not touch them any more.
        self.set_table(u, torch.zeros(self.nb_v, dtype=torch.int32))
        if self.iters[u]:
            dt = time.perf_counter() - self._t_begin[u]
            n_tok = self.iters[u] + self.accepted[u]
            logger.info(
                f"[dflash2-serve] slot {u} session: {self.iters[u]} iters, {n_tok} committed tokens, "
                f"accept {self.accepted[u] / self.iters[u]:.2f}/{self.K} -> {n_tok / self.iters[u]:.2f} tok/iter, "
                f"{n_tok / max(dt, 1e-6):.1f} tok/s in-model (incl. seed); histogram {self.hist[u]}"
            )

    # ------------------------------------------------------------------ shutdown
    def release(self):
        model = self.model
        for u in range(self.B):
            self.active[u] = False
        try:
            ttnn.synchronize_device(self.mesh)
        except Exception:
            pass
        try:
            self.drafter.free()
        except Exception as e:
            logger.warning(f"[dflash2-serve] drafter free failed: {e!r}")
        try:
            model.release_verify_trace()
        except Exception as e:
            logger.warning(f"[dflash2-serve] verify trace release failed: {e!r}")
        self._vfy_captured = False
        self._captured = False
        model._free_dflash_eager_taps()
        model.free_dflash_tap_bufs()
        model._dflash_tap = False
        self._alloc_done = False
        self._warm_done = False


def dummy_prompt(S, seed=1234):
    """S plain text token ids for warmup prefills (no multimodal placeholder ids: they are > 200000)."""
    g = torch.Generator().manual_seed(seed + S)
    return torch.randint(100, 20000, (1, S), generator=g, dtype=torch.int32)


def serve_block_size():
    """Tokens one vLLM decode step commits in spec mode (QWEN36_DFLASH_SERVE_BLOCK; 1 = spec off)."""
    return max(1, int(os.environ.get("QWEN36_DFLASH_SERVE_BLOCK", "32")))
