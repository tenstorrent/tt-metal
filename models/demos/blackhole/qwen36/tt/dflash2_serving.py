# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative decode as a long-lived SERVING session (vLLM block-output contract).

The demo decoder (DFlash2Decoder.generate) owns one request end to end: it allocates the drafter's
buffers, captures the verify/commit/draft traces, runs the loop, and frees everything in a finally.
A server cannot afford that per request (capture is seconds, and every buffer the parked traces bake
must stay allocated while they can replay), so this decoder splits the same loop into a
server-lifetime part and a per-request part:

  arm()    ONCE at model warmup, after the plain decode trace is captured: allocate every persistent
           buffer (tap bufs, drafter KV + staging, anchor buffer), compile every program the loop
           needs (eager prefill taps for each mask bucket, seed, draft, extend), then capture the
           verify trace, the commit traces and the drafter's draft/extend traces. Nothing spec-related
           compiles or allocates after this.
  begin()  per request, at its first decode step: fill the drafter's context KV from the prompt taps
           the eager masked prefill left behind, re-point the verify trace at the request's vLLM page
           table, eager-seed the anchor token (position T) -> pending token; no capture.
  step()   ONE draft + traced verify + greedy accept + traced commit + extend; returns the committed
           tokens ([pending] + accepted drafts). The caller loops it to fill a block.
  end()    per request: leave the GDN taps live for whatever runs next.
  release() at shutdown: traces + buffers.

Page tables: the verify trace reads vLLM's page table for the request (refreshed in place through
Qwen36Model.refresh_verify_page_table whenever vLLM's row changes); the drafter's private context KV
uses its OWN identity table (position -> block = position // block_size) sized to the drafter's
context capacity, so it never depends on vLLM's block ids and never has to mirror the whole pool.
"""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2_decode import DFlash2Decoder, _dbg, get_drafter
from models.tt_transformers.tt.common import get_block_size


class DFlash2ServingDecoder(DFlash2Decoder):
    """DFlash2Decoder whose captures and buffers live for the server's lifetime (see module doc)."""

    def __init__(self, model, num_blocks, draft_len=None, weights_dir=None, ctx_blocks=None, stop_tokens=None):
        # num_blocks: width of vLLM's per-request page-table rows (the verify trace's table width).
        pt = torch.arange(int(num_blocks), dtype=torch.int32).reshape(1, int(num_blocks))
        super().__init__(model, pt, draft_len=draft_len, stop_tokens=stop_tokens, weights_dir=weights_dir)
        assert self.tp, "the serving decoder needs the TP drafter (QWEN36_DFLASH_TP=1)"
        self.nb_v = int(num_blocks)
        # Drafter context capacity in blocks (its private identity table). Multiple of 32 so the paged
        # SDPA's page-table stick stays tile-aligned; +32 spare blocks past the pool width because the
        # kernel scans whole 256-position chunks up to nearest_n(cur+1, 256).
        nb_d = int(ctx_blocks) if ctx_blocks else self.nb_v
        self.nb_d = (nb_d + 31) // 32 * 32 + 32
        self.block_size = None
        self.max_pos = None  # highest absolute position the loop may write (verify + draft rows)
        self._alloc_done = False
        self._captured = False
        self.active = False
        self.p = None
        self.pending = None
        self.Hp = None
        self._t_begin = 0.0

    # ------------------------------------------------------------------ persistent buffers (once)
    def _anchor_warmup(self, T, dim_frac, dtype):
        """The persistent anchor buffer is allocated ONCE (the base allocates one per seed): the commit
        traces bake its address, and a fresh per-request allocation could land on a parked trace's
        scratch."""
        if self._hp_buf is not None:
            return
        super()._anchor_warmup(T, dim_frac, dtype)

    def alloc(self):
        """Allocate every buffer the session traces will bake (call BEFORE any spec capture and only
        once): the model's tap bufs (verify -> drafter), the drafter's context KV + staging buffers."""
        model = self.model
        assert not self._alloc_done
        T = self.K + 1
        self.drafter = get_drafter(model, self._weights_dir, block=T)
        assert self.drafter.K == self.K and self.drafter.is_tp
        self.block_size = get_block_size(model._paged_kv_caches)
        model._dflash_tap_layers = tuple(self.drafter.taps)
        if model._dflash_tap_bufs is not None and (
            len(model._dflash_tap_bufs) != len(model._dflash_tap_layers) or model._dflash_tap_bufs[0].shape[-2] != T
        ):
            model.free_dflash_tap_bufs()
        if model._dflash_tap_bufs is None:
            model._dflash_tap_bufs = model.alloc_dflash_tap_bufs(T)
        ctx_pt = torch.arange(self.nb_d, dtype=torch.int32).reshape(1, self.nb_d)
        self.drafter.alloc(ctx_pt, self.block_size, model._dflash_tap_bufs)
        self.max_pos = min(self.nb_d, self.nb_v) * self.block_size - 256
        self._alloc_done = True
        logger.info(
            f"[dflash2-serve] allocated: K={self.K} T={T} verify table {self.nb_v} blocks, "
            f"drafter ctx {self.nb_d} blocks x {self.block_size} (max position {self.max_pos})"
        )

    # ------------------------------------------------------------------ per-request context
    def ingest_prompt(self, taps, T):
        """Prompt taps (list of 5 fractured [1,1,S,dim/tp] device tensors from the eager masked prefill,
        S = bucket rows >= T) -> the drafter's context KV at positions 0..S-1. Frees the taps."""
        assert self._alloc_done
        _dbg(f"ingest_prompt T={T} rows={taps[0].shape[-2]}")
        self.drafter.fill_context(taps, 0)
        for t in taps:
            ttnn.deallocate(t)
        self.ctx_len = int(T)

    def fit_page_table(self, row):
        """vLLM's page-table row (torch [nb] or [1, nb]) -> [1, nb_v] int32 (zero-padded / trimmed).
        Trailing entries index blocks past the sequence and are never read."""
        row = torch.as_tensor(row).reshape(1, -1).to(torch.int32)
        n = row.shape[1]
        if n < self.nb_v:
            row = torch.cat([row, torch.zeros(1, self.nb_v - n, dtype=torch.int32)], dim=1)
        elif n > self.nb_v:
            row = row[:, : self.nb_v]
        return row.contiguous()

    def set_page_table(self, row):
        """Point the seed (eager verify) and the verify trace at this request's blocks."""
        self.page_table = self.fit_page_table(row)
        if self._captured:
            self.model.refresh_verify_page_table(self.page_table)

    def begin(self, first, T, page_table_row):
        """Start a request's session after ingest_prompt: seed the anchor token ``first`` at position T
        (eager verify; also extends the drafter context by slot T) and derive the pending token."""
        assert self._alloc_done and self.ctx_len == T, f"ingest_prompt({T}) must precede begin (ctx_len={self.ctx_len})"
        self._t_begin = time.perf_counter()
        self.set_page_table(page_table_row)
        for dn in self._gdn:
            dn._capture_slots = False  # the eager seed must not write the trace's slot buffers
        self.model._dflash_tap = True  # the seed's eager verify clones its taps for the drafter
        try:
            Lp, Hp = self._seed(int(first), T - 1)
        finally:
            self.model._dflash_tap = False
            for dn in self._gdn:
                dn._capture_slots = True
        self.p = int(T)
        self.pending = int(Lp.argmax())
        self.Hp = Hp
        self.active = True
        self.iters = 0
        self.total_drafted = 0
        self.total_accepted = 0
        self.accept_hist = [0] * (self.K + 1)
        self.depth_hits = [0] * self.K
        self.zero_accept = 0

    # ------------------------------------------------------------------ one-time captures
    def capture(self):
        """After the FIRST begin(): compile the remaining draft programs and capture the verify + commit
        traces (the drafter's own draft/extend traces are captured by the first step())."""
        assert self.active and not self._captured
        model = self.model
        self._draft_warmup(self.pending, self.Hp, self.p)
        model._dflash_tap = True  # the trace body copies the taps into the (pre-allocated) tap bufs
        try:
            model.capture_verify_trace(
                self.page_table, self.K + 1, warm_start=self.p + 1, decode_cfg=True, commit_warmup=self.traced_commit
            )
        finally:
            model._dflash_tap = False
        model._vfy_owner = self
        self._vfy_captured = True
        self._commit_traced = bool(self.traced_commit and model.capture_commit_traces())
        logger.info(
            f"[dflash2-serve] verify trace captured (T={self.K + 1}); commit={'traced' if self._commit_traced else 'eager'}"
        )
        ttnn.synchronize_device(self.mesh)
        self._captured = True
        # The verify trace was captured against the identity table; the session's real table (if
        # different) is re-staged into the trace's buffers now that they exist.
        model.refresh_verify_page_table(self.page_table)

    # ------------------------------------------------------------------ the loop
    def step(self):
        """One speculative iteration. Returns the committed ids ([pending] + accepted drafts), or None
        when the next verify would write past the drafter's / pool's capacity (caller stops)."""
        assert self.active and self._captured
        p, pending = self.p, self.pending
        if p + 2 + self.K > self.max_pos:
            logger.warning(f"[dflash2-serve] position {p} near capacity {self.max_pos}: ending the session")
            return None
        drafts = self._draft(pending, self.Hp, p)
        vids, vhidden = self._verify([pending] + drafts, p)
        m = self._accept_greedy(drafts, vids)
        committed = [pending] + drafts[:m]
        mi = len(committed) - 1
        self._commit(mi)
        next_pending = vids[mi]
        self._set_anchor(vhidden, mi)
        self._reseed_mtp_batched(p + 1, vhidden, committed[1:])
        self.p = p + len(committed)
        self.pending = int(next_pending)
        self.iters += 1
        self.total_drafted += len(drafts)
        self.total_accepted += m
        return [int(t) for t in committed]

    def end(self):
        """Request done: leave the GDN tap buffers live (the verify only advances the conv window)."""
        if not self.active:
            return
        self.active = False
        for dn in self._gdn:
            dn.sync_conv_taps()
        if self.iters:
            dt = time.perf_counter() - self._t_begin
            n_tok = self.iters + self.total_accepted
            logger.info(
                f"[dflash2-serve] session: {self.iters} iters, {n_tok} committed tokens, "
                f"accept {self.total_accepted / self.iters:.2f}/{self.K} -> {n_tok / self.iters:.2f} tok/iter, "
                f"{n_tok / max(dt, 1e-6):.1f} tok/s in-model (incl. seed); accepted-drafts histogram {self.accept_hist}"
            )

    # ------------------------------------------------------------------ shutdown
    def release(self):
        """Release every trace and buffer this decoder pinned (best effort; shutdown only)."""
        model = self.model
        self.active = False
        try:
            ttnn.synchronize_device(self.mesh)
        except Exception:
            pass
        try:
            self.drafter.free()
        except Exception as e:
            logger.warning(f"[dflash2-serve] drafter free failed: {e!r}")
        try:
            model.release_commit_traces()
            tid = getattr(model, "_vfy_trace_id", None)
            if tid is not None:
                ttnn.release_trace(model.mesh_device, tid)
                model._vfy_trace_id = None
        except Exception as e:
            logger.warning(f"[dflash2-serve] verify trace release failed: {e!r}")
        self._vfy_captured = False
        self._captured = False
        model._free_dflash_eager_taps()
        model.free_dflash_tap_bufs()
        model._dflash_tap = False
        if self._hp_buf is not None:
            try:
                ttnn.deallocate(self._hp_buf)
            except Exception:
                pass
            self._hp_buf = None
        self._alloc_done = False


def dummy_prompt(S, seed=1234):
    """S plain text token ids for warmup prefills (no multimodal placeholder ids: they are > 200000)."""
    g = torch.Generator().manual_seed(seed + S)
    return torch.randint(100, 20000, (1, S), generator=g, dtype=torch.int32)


def serve_block_size():
    """Tokens one vLLM decode step commits in spec mode (QWEN36_DFLASH_SERVE_BLOCK; 1 = spec off)."""
    return max(1, int(os.environ.get("QWEN36_DFLASH_SERVE_BLOCK", "32")))
