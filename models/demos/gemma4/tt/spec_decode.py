# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding for Gemma4 (it-assistant MTP/EAGLE drafter), batch=1.

Loop (one iteration produces up to ``draft_len + 1`` committed tokens):

  1. Draft: the assistant proposes ``K = draft_len`` tokens autoregressively from
     a single fixed position, recurrently feeding its own projected hidden state
     and cross-attending into the target's last sliding / last full layer KV.
  2. Verify: the target runs ONE batched forward over ``[anchor, d1, ..., dK]``
     at consecutive positions (candidates in the batch dim, the user's page-table
     row replicated). This appends KV and yields per-position logits + hidden.
  3. Accept: greedy (argmax match) or speculative sampling. Committed = the
     matched prefix + one bonus/correction token. KV rollback at batch=1 is
     implicit — rejected positions are overwritten next iteration.

Correctness: the committed tokens are ALWAYS produced by the target verify, so
greedy speculative decode matches plain greedy decode and sampling matches the
target distribution — independent of the drafter's accuracy (which only affects
the acceptance rate / speed). The verify forward runs BATCHED (anchor + K
candidates), so its per-user RoPE + batched SDPA differ from batch=1 decode by
~1e-5; this flips only target near-ties (top-2 logit gap < ~1), so greedy
spec-decode is token-identical to plain greedy up to the first near-tie token
and produces an equally-valid greedy trajectory thereafter.
"""

import os
import time

import torch

import ttnn


def _to_probs(logits_row, temperature, top_p, top_k):
    """torch logits [vocab] -> probability vector [vocab] under temp/top-p/top-k.

    temperature<=0 returns a one-hot (greedy) distribution.
    """
    logits_row = logits_row.float()
    if not temperature or temperature <= 0:
        probs = torch.zeros_like(logits_row)
        probs[int(torch.argmax(logits_row))] = 1.0
        return probs
    logits_row = logits_row / temperature
    if top_k and top_k > 0 and top_k < logits_row.numel():
        kth = torch.topk(logits_row, top_k).values[-1]
        logits_row = torch.where(logits_row < kth, torch.full_like(logits_row, float("-inf")), logits_row)
    probs = torch.softmax(logits_row, dim=-1)
    if top_p and 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    s = probs.sum()
    return probs / s if s > 0 else probs


class SpeculativeDecoder:
    def __init__(
        self,
        target_model,
        assistant_model,
        mesh_device,
        tt_kv_cache,
        page_table_torch,
        stop_tokens=None,
        draft_len=None,
    ):
        self.target = target_model
        self.assistant = assistant_model
        self.mesh_device = mesh_device
        # ``Gemma4Generator.from_pretrained`` returns the cache wrapped per model
        # instance (``[model0_kv_cache]``); the generator unwraps ``kv_cache[0]``
        # before calling the model. We call the model forward directly, so peel a
        # single-model wrapper down to the per-layer ``[[k, v], ...]`` list. The
        # inner ``[0][0]`` type check disambiguates a wrapper from an already
        # per-layer cache (whose ``[0][0]`` is a ttnn.Tensor, not a list/tuple).
        if (
            isinstance(tt_kv_cache, (list, tuple))
            and tt_kv_cache
            and isinstance(tt_kv_cache[0], (list, tuple))
            and tt_kv_cache[0]
            and isinstance(tt_kv_cache[0][0], (list, tuple))
        ):
            tt_kv_cache = tt_kv_cache[0]
        self.tt_kv_cache = tt_kv_cache
        self.page_table_torch = page_table_torch
        self.stop_tokens = set(stop_tokens or [])
        self.draft_len = int(draft_len if draft_len is not None else os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
        if self.draft_len < 1:
            raise ValueError("draft_len must be >= 1")
        # Recurrent drafter-seed strategy across verify rounds (see generate()).
        self._seed_mode = os.environ.get("GEMMA4_SPEC_SEED_MODE", "reseed")
        if page_table_torch is None:
            raise ValueError("Speculative decoding requires paged attention (page_table_torch is None).")
        self._is_mesh = hasattr(mesh_device, "shape")
        self._mapper = target_model._replicate_to_mesh_mapper()
        self._tp = target_model.mesh_config.tp if target_model.mesh_config else 1
        # The drafter cross-attends to the target's last full / last sliding KV.
        self._shared_kv = target_model.get_shared_kv_caches()
        # Tracing: persistent I/O buffers + execute_trace replace per-op host
        # dispatch (the untraced loop is host-bound: ~77ms/decode vs a few ms
        # traced). Verify traces are keyed by batch (K+1 for verify, 1 for
        # seed/reseed). Captured lazily on first call at real inputs.
        self._use_trace = os.environ.get("GEMMA4_SPEC_TRACE", "0") == "1"
        self._verify_traces = {}
        # Single batch=1 drafter-step trace, replayed K times. The recurrent
        # hidden is kept ON DEVICE: between trace replays, ttnn.copy(next_hidden
        # -> h_in) refreshes the persistent recurrent buffer, so each replay
        # consumes the previous replay's hidden without a host round-trip. Only
        # the next draft *token* round-trips (host argmax -> tok_in), which is a
        # read + tiny scalar copy (no device allocation -> trace-safe).
        self._draft_trace = None
        # Single FUSED per-iteration trace (K drafter steps + verify in ONE
        # CCL-bearing trace). Replaying ONE trace per iter avoids interleaving
        # distinct CCL traces (the draft/verify-alternation deadlock). Requires
        # on-device argmax + re-embed for the draft recurrence and verify-input
        # assembly (see _fused_iter / _capture_fused_trace).
        self._fused_trace = None
        # Seed mode for the fused greedy trace. "shift" (default) is the fast
        # production mode: it reuses the previous verify hidden row as an
        # approximate drafter seed and repairs the anchor KV at the next verify.
        # "reseed" is the exact assistant contract: run a batch=1 target seed
        # inside the fused trace before drafting, so the drafter sees the current
        # anchor's exact target hidden/KV. It raises acceptance, but the extra
        # target forward currently costs more than it saves; keep it as an A/B
        # correctness/reference mode via GEMMA4_SPEC_FUSED_RESEED=1.
        self._fused_reseed = os.environ.get("GEMMA4_SPEC_FUSED_RESEED", "0") == "1"
        # Cheap approximate seed-row selection for fused shift mode. `next` is
        # the original position-aligned row p+m+1. `current` uses row m (usually
        # the last matched target row), and `last_accepted` uses row m-1 when at
        # least one draft was accepted. These are A/B knobs for acceptance vs
        # seed quality without paying a full target reseed.
        self._fused_shift_seed = os.environ.get("GEMMA4_SPEC_FUSED_SHIFT_SEED", "current")
        # Persistent anchor-hidden buffer for the traced loop (allocated once).
        # The traced loop MUST be allocation-free: any ttnn.clone/slice between
        # execute_trace calls can land in memory a later trace replay uses as
        # scratch and corrupt it (manifests as a hang on a *re*-replay). seed()
        # writes into this buffer via ttnn.copy instead of cloning.
        self._anchor_buf = None

    def _fused_shift_seed_row(self, accepted, K):
        if self._fused_shift_seed == "current":
            return min(accepted, K)
        if self._fused_shift_seed == "last_accepted":
            return max(min(accepted, K) - 1, 0)
        # Default: position-aligned next row. When all K drafts are accepted,
        # there is no row K+1, so fall back to row K.
        return min(accepted + 1, K)

    # ── device-tensor builders ────────────────────────────────────────────
    def _pos_tensors(self, positions):
        batch = len(positions)
        pu = torch.zeros((1, 32), dtype=torch.int32)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int32)
        pos_uint32 = ttnn.from_torch(
            pu, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper
        )
        pos_int32 = ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        return pos_uint32, pos_int32

    def _tokens_tensor(self, tokens):
        t = torch.tensor(tokens, dtype=torch.int32).reshape(1, len(tokens))
        return ttnn.from_torch(
            t, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper
        )

    def _page_table(self, batch):
        # Replicate the single user's block row across the `batch` pseudo-users so
        # every candidate/verify position indexes the SAME physical KV blocks.
        user_row = self.page_table_torch[0:1] if self.page_table_torch.dim() > 1 else self.page_table_torch.unsqueeze(0)
        pt = user_row.repeat(batch, 1).to(torch.int32)
        return ttnn.from_torch(
            pt, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=self._mapper
        )

    # ── host-side input tensors (for copy_host_to_device_tensor into traces) ──
    def _host_tokens(self, tokens):
        t = torch.tensor(tokens, dtype=torch.int32).reshape(1, len(tokens))
        return ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)

    def _host_pos(self, positions):
        batch = len(positions)
        pu = torch.zeros((1, 32), dtype=torch.int32)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int32)
        pos_uint32 = ttnn.from_torch(pu, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)
        pos_int32 = ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        return pos_uint32, pos_int32

    # ── traced verify ────────────────────────────────────────────────────────
    def _capture_verify_trace(self, tokens, positions):
        """Lazily capture a verify trace at the REAL first-call inputs.

        Persistent device inputs (x, pos_u, pos_i, page_table) are allocated once
        so trace capture binds stable addresses; later calls overwrite their
        contents via ``copy_host_to_device_tensor`` and ``execute_trace``. The
        compile + capture runs write the SAME KV positions with the SAME tokens
        as the first real call, so the KV writes are idempotent (no corruption).
        """
        from loguru import logger as _lg

        batch = len(tokens)
        _lg.info(f"[spec-trace] capture verify batch={batch}: building inputs")
        x_dev = self._tokens_tensor(tokens)
        pu_dev, pi_dev = self._pos_tensors(positions)
        pt_dev = self._page_table(batch)
        # Compile run (warm program cache before capture).
        _lg.info(f"[spec-trace] capture verify batch={batch}: compile run")
        logits, hidden = self.target.ttnn_verify_forward(
            x=x_dev, current_pos=pu_dev, current_pos_cache=pi_dev, page_table=pt_dev, kv_cache=self.tt_kv_cache
        )
        ttnn.synchronize_device(self.mesh_device)
        logits.deallocate(True)
        hidden.deallocate(True)
        _lg.info(f"[spec-trace] capture verify batch={batch}: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits, hidden = self.target.ttnn_verify_forward(
            x=x_dev, current_pos=pu_dev, current_pos_cache=pi_dev, page_table=pt_dev, kv_cache=self.tt_kv_cache
        )
        _lg.info(f"[spec-trace] capture verify batch={batch}: end_trace_capture")
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info(f"[spec-trace] capture verify batch={batch}: DONE")
        self._verify_traces[batch] = {
            "id": tid,
            "x": x_dev,
            "pu": pu_dev,
            "pi": pi_dev,
            "pt": pt_dev,
            "logits": logits,
            "hidden": hidden,
        }

    def _verify_traced(self, tokens, positions):
        batch = len(tokens)
        if batch not in self._verify_traces:
            self._capture_verify_trace(tokens, positions)
        from loguru import logger as _lg

        tr = self._verify_traces[batch]
        # Overwrite persistent inputs (page_table row is constant -> write once at
        # capture). x/positions change every call.
        h_x = self._host_tokens(tokens)
        h_pu, h_pi = self._host_pos(positions)
        ttnn.copy_host_to_device_tensor(h_x, tr["x"])
        ttnn.copy_host_to_device_tensor(h_pu, tr["pu"])
        ttnn.copy_host_to_device_tensor(h_pi, tr["pi"])
        _lg.info(f"[spec-trace] verify replay batch={batch}: execute")
        ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)
        _lg.info(f"[spec-trace] verify replay batch={batch}: read logits")
        lh = self._logits_to_host(tr["logits"]).reshape(batch, -1)
        # Return the PERSISTENT hidden (no clone). It is overwritten on the next
        # execute_trace, so the caller MUST consume it before the next verify and
        # MUST NOT deallocate it (allocation-free loop invariant). seed() copies
        # it into self._anchor_buf immediately; generate() ignores it in the
        # default reseed mode.
        return lh, tr["hidden"]

    def _logits_to_host(self, logits):
        if self._tp > 1:
            t = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            t = ttnn.to_torch(logits)
        return t[..., : self.target.vocab_size]

    # ── target forwards ───────────────────────────────────────────────────
    def _verify(self, tokens, positions):
        """Batched verify. Returns (logits_host [B,vocab], hidden_device [1,1,B,h])."""
        # Tracing: BOTH the batch=1 verify (seed/reseed) and the batch=K+1
        # speculative verify capture + replay correctly. The per-candidate
        # sequential_kv_write loop (the race fix for the shared paged block) is
        # fixed-count and fully trace-compatible — positions/tokens are bound as
        # data via copy_host_to_device_tensor, so a single trace per batch size
        # is reused across iterations as the anchor position grows. Traces are
        # keyed by batch (K+1 and 1), captured lazily on first call.
        if self._use_trace:
            return self._verify_traced(tokens, positions)
        x = self._tokens_tensor(tokens)
        pos_u, pos_i = self._pos_tensors(positions)
        pt = self._page_table(len(tokens))
        logits, hidden = self.target.ttnn_verify_forward(
            x=x, current_pos=pos_u, current_pos_cache=pos_i, page_table=pt, kv_cache=self.tt_kv_cache
        )
        lh = self._logits_to_host(logits).reshape(len(tokens), -1)
        logits.deallocate(True)
        for t in (x, pos_u, pos_i, pt):
            t.deallocate(True)
        return lh, hidden

    # ── traced drafting ──────────────────────────────────────────────────────
    def _capture_draft_trace(self, anchor_token, anchor_hidden, anchor_pos):
        """Capture ONE batch=1 drafter-step trace (replayed K times per iter).

        Persistent inputs ``tok_in`` (last token), ``h_in`` (recurrent hidden),
        ``pos_u/pos_i`` (fixed anchor position) and ``pt`` (page table) are
        allocated once. ``logits`` and ``h_next`` are persistent trace OUTPUTS.
        The recurrence copy ``h_next -> h_in`` is done BETWEEN replays (outside
        the trace) via ttnn.copy — an in-trace copy fails capture with "Writes
        not supported during trace capture".
        """
        from loguru import logger as _lg

        tok_in = self._tokens_tensor([anchor_token])
        h_in = ttnn.clone(anchor_hidden)
        pos_u, pos_i = self._pos_tensors([anchor_pos])
        pt = self._page_table(1)
        page_tables = {lt: pt for lt in self._shared_kv}
        _lg.info("[spec-trace] capture draft step: compile run")
        logits, h_next = self.assistant.step(tok_in, h_in, self._shared_kv, page_tables, pos_u, pos_i)
        ttnn.synchronize_device(self.mesh_device)
        logits.deallocate(True)
        h_next.deallocate(True)
        _lg.info("[spec-trace] capture draft step: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits, h_next = self.assistant.step(tok_in, h_in, self._shared_kv, page_tables, pos_u, pos_i)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info("[spec-trace] capture draft step: DONE")
        self._draft_trace = {
            "id": tid,
            "tok": tok_in,
            "h": h_in,
            "h_next": h_next,
            "pu": pos_u,
            "pi": pos_i,
            "pt": pt,
            "logits": logits,
        }

    def _draft_traced(self, anchor_token, anchor_hidden, anchor_pos, temperature=0.0, top_p=1.0, top_k=0):
        """Traced drafting: replay the batch=1 step trace K times. See _draft."""
        K = self.draft_len
        greedy = not temperature or temperature <= 0
        if self._draft_trace is None:
            self._capture_draft_trace(anchor_token, anchor_hidden, anchor_pos)
        tr = self._draft_trace
        # Per-iteration setup: position (anchor moves between iters), recurrent
        # hidden reset to this iter's seed, and the first token.
        h_pu, h_pi = self._host_pos([anchor_pos])
        ttnn.copy_host_to_device_tensor(h_pu, tr["pu"])
        ttnn.copy_host_to_device_tensor(h_pi, tr["pi"])
        ttnn.copy(anchor_hidden, tr["h"])  # device->device reset of recurrent buffer
        ttnn.copy_host_to_device_tensor(self._host_tokens([anchor_token]), tr["tok"])

        from loguru import logger as _lg

        drafts, draft_logits = [], []
        for step in range(K):
            _lg.info(f"[spec-trace] draft replay step {step}: execute")
            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)
            # Feed the recurrence: next step reads h_in, which the trace does NOT
            # update (in-trace copy is illegal). Copy the fresh output h_next into
            # h_in on-device (no allocation), queued after execute_trace.
            if step < K - 1:
                _lg.info(f"[spec-trace] draft replay step {step}: copy h_next->h")
                ttnn.copy(tr["h_next"], tr["h"])
            _lg.info(f"[spec-trace] draft replay step {step}: read logits")
            lh = self._logits_to_host(tr["logits"]).reshape(-1)
            draft_logits.append(lh)
            if greedy:
                tok = int(torch.argmax(lh))
            else:
                q = _to_probs(lh, temperature, top_p, top_k)
                tok = int(torch.multinomial(q, 1))
            drafts.append(tok)
            if step < K - 1:
                ttnn.copy_host_to_device_tensor(self._host_tokens([tok]), tr["tok"])
        return drafts, draft_logits

    # ── drafting ────────────────────────────────────────────────────────────
    def _draft(self, anchor_token, anchor_hidden, anchor_pos, temperature=0.0, top_p=1.0, top_k=0):
        """Produce K draft tokens + their logits (host). anchor_hidden stays on device.

        Greedy (temperature<=0): the next draft is argmax(q). Sampling: the next
        draft is SAMPLED from the drafter distribution q under the SAME
        temp/top_p/top_k as the target — required for speculative sampling
        correctness AND acceptance (proposing argmax(q) maximizes q(d), which
        drives the accept ratio min(1, p(d)/q(d)) to ~0)."""
        K = self.draft_len
        greedy = not temperature or temperature <= 0
        pos_u, pos_i = self._pos_tensors([anchor_pos])
        # Drafter page table: batch=1, same user's blocks for both layer types.
        pt = self._page_table(1)
        page_tables = {lt: pt for lt in self._shared_kv}

        drafts, draft_logits = [], []
        tok = anchor_token
        h_rec = anchor_hidden  # borrowed; do not deallocate the caller's anchor_hidden
        owns_h = False
        for _ in range(K):
            tok_tt = self._tokens_tensor([tok])
            logits_d, h_next = self.assistant.step(tok_tt, h_rec, self._shared_kv, page_tables, pos_u, pos_i)
            tok_tt.deallocate(True)
            lh = self._logits_to_host(logits_d).reshape(-1)
            logits_d.deallocate(True)
            draft_logits.append(lh)
            if greedy:
                tok = int(torch.argmax(lh))
            else:
                q = _to_probs(lh, temperature, top_p, top_k)
                tok = int(torch.multinomial(q, 1))
            drafts.append(tok)
            if owns_h:
                h_rec.deallocate(True)
            h_rec, owns_h = h_next, True
        if owns_h:
            h_rec.deallocate(True)
        pos_u.deallocate(True)
        pos_i.deallocate(True)
        pt.deallocate(True)
        return drafts, draft_logits

    # ── fully on-device fused iteration (greedy) ─────────────────────────────
    def _argmax_last(self, logits, rows):
        """argmax over the last (vocab) dim — fast path. Returns [1,1,rows] uint32.

        ``ttnn.argmax`` requires ROW_MAJOR input; passing a TILE tensor takes a
        single-core internal-untilize path that is catastrophically slow on a
        262144-wide vocab (~9 ms for 1 row, ~28 ms for 5). The multicore argmax
        is fast (~1.6 ms) but ROW-PARALLEL: it returns GARBAGE unless the row
        (batch) dim is tile-aligned (32) — verified by a correctness probe
        (1/5 rows -> wrong; padded to 32 -> exact). So pad the rows up to 32,
        run the multicore untilize + argmax, then slice back to ``rows``. Net
        ~1.6 ms vs ~9-28 ms for the bare path.
        """
        R32 = 32
        src = logits
        padded = None
        if rows < R32:
            padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, R32 - rows), (0, 0)], value=0.0)
            src = padded
        u = ttnn.untilize(src, use_multicore=True)
        if padded is not None:
            padded.deallocate(True)
        idx = ttnn.argmax(u, dim=-1, keepdim=False, use_multicore=True)  # [1,1,32 or rows] uint32 RM
        u.deallocate(True)
        if rows < R32:
            sliced = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])  # [1,1,rows]
            idx.deallocate(True)
            idx = sliced
        return idx

    def _id_to_host(self, id_tt):
        """[*,1] uint32 device id -> python int (TP: read device-0 replica)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(id_tt)[0]) if self._tp > 1 else ttnn.to_torch(id_tt)
        return int(t.reshape(-1)[0])

    def _ids_to_host(self, ids_tt, n):
        """[1,1,n] uint32 device ids -> list[int] (TP: read device-0 replica)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(ids_tt)[0]) if self._tp > 1 else ttnn.to_torch(ids_tt)
        flat = t.reshape(-1)
        return [int(flat[j]) for j in range(n)]

    def _fused_iter(self, anchor_tok_tt, anchor_hidden, anchor_pos):
        """ONE greedy speculative iteration, fully on-device.

        Chains K drafter steps with ON-DEVICE argmax + re-embed (no per-step host
        round-trip), assembles the verify batch ``[anchor, d0..d_{K-1}]`` on
        device, runs the batched verify, and argmaxes the verify logits on device.
        Only the ``2K+1`` token ids are read to host (acceptance is a host int
        compare). This is the eager twin of the fused single-iteration trace.

        Args:
            anchor_tok_tt: [1,1] uint32 device token id to draft from (= committed token).
            anchor_hidden: [1,1,1,backbone] device drafter seed.
            anchor_pos: int absolute position p of the anchor token.

        Returns:
            (drafts:list[int], target_ids:list[int] len K+1, verify_hidden device [1,1,K+1,backbone]).
        """
        K = self.draft_len
        # Drafter queries a single fixed position (HF SinglePositionMTP).
        d_pu, d_pi = self._pos_tensors([anchor_pos])
        d_pt = self._page_table(1)
        page_tables = {lt: d_pt for lt in self._shared_kv}

        draft_id_tts = []  # [1,1] uint32 each
        tok_tt = anchor_tok_tt
        h = anchor_hidden
        owns_h = False
        for _ in range(K):
            logits, h_next = self.assistant.step(tok_tt, h, self._shared_kv, page_tables, d_pu, d_pi)
            if owns_h:
                h.deallocate(True)
            h, owns_h = h_next, True
            idx = self._argmax_last(logits, rows=1)  # [1,1,1] uint32 RM (fast multicore argmax)
            logits.deallocate(True)
            tok_tt = ttnn.reshape(idx, (1, 1))  # [1,1] uint32 RM
            draft_id_tts.append(tok_tt)
        if owns_h:
            h.deallocate(True)

        # Verify input = [anchor, d0..d_{K-1}] at positions p..p+K.
        verify_x = ttnn.concat([anchor_tok_tt] + draft_id_tts, dim=1)  # [1, K+1] uint32 RM
        v_pos = [anchor_pos + j for j in range(K + 1)]
        v_pu, v_pi = self._pos_tensors(v_pos)
        v_pt = self._page_table(K + 1)
        vlogits, vhidden = self.target.ttnn_verify_forward(
            x=verify_x, current_pos=v_pu, current_pos_cache=v_pi, page_table=v_pt, kv_cache=self.tt_kv_cache
        )
        vidx = self._argmax_last(vlogits, rows=K + 1)  # [1,1,K+1] uint32 RM (fast multicore argmax)

        drafts = [self._id_to_host(t) for t in draft_id_tts]
        target_ids = self._ids_to_host(vidx, K + 1)

        for t in draft_id_tts:
            t.deallocate(True)
        for t in (verify_x, v_pu, v_pi, v_pt, d_pu, d_pi, d_pt, vlogits, vidx):
            t.deallocate(True)
        return drafts, target_ids, vhidden

    # ── acceptance ────────────────────────────────────────────────────────
    def _accept_greedy(self, drafts, verify_logits):
        """Return (m, committed_tokens). verify_logits[j] = logits at anchor+j."""
        g = [int(torch.argmax(verify_logits[j])) for j in range(len(drafts) + 1)]
        m = len(drafts)
        for i, d in enumerate(drafts):
            if d != g[i]:
                m = i
                break
        committed = list(drafts[:m]) + [g[m]]
        return m, committed

    def _accept_sampling(self, drafts, draft_logits, verify_logits, temperature, top_p, top_k):
        """Speculative-sampling acceptance. Matches the target distribution."""
        K = len(drafts)
        committed = []
        m = K
        for i in range(K):
            p = _to_probs(verify_logits[i], temperature, top_p, top_k)  # target dist at anchor+i
            q = _to_probs(draft_logits[i], temperature, top_p, top_k)  # draft dist for d_i
            d = drafts[i]
            qd = float(q[d])
            ratio = 1.0 if qd <= 0 else min(1.0, float(p[d]) / qd)
            if torch.rand(()) < ratio:
                committed.append(d)
            else:
                resid = torch.clamp(p - q, min=0.0)
                s = resid.sum()
                corr = int(torch.argmax(p)) if s <= 0 else int(torch.multinomial(resid / s, 1))
                committed.append(corr)
                m = i
                break
        if m == K:
            p = _to_probs(verify_logits[K], temperature, top_p, top_k)
            committed.append(
                int(torch.multinomial(p, 1)) if (temperature and temperature > 0) else int(torch.argmax(p))
            )
        return m, committed

    # ── main loop ───────────────────────────────────────────────────────────
    def seed(self, anchor_token, anchor_pos):
        """Compute the target hidden at the anchor (after prefill filled the KV).

        Runs one batch=1 verify of the last committed/prompt token at its
        position — idempotent on the KV cache (prefill already wrote it) — and
        returns the post-norm hidden ``[1,1,1,backbone]`` used to seed the first
        drafter step.
        """
        _, hidden = self._verify([anchor_token], [anchor_pos])
        if self._use_trace:
            # hidden is the persistent batch=1 verify output (do NOT deallocate).
            # Copy it into the persistent anchor buffer (one-time clone to alloc).
            if self._anchor_buf is None:
                self._anchor_buf = ttnn.clone(hidden)
            else:
                ttnn.copy(hidden, self._anchor_buf)
            return self._anchor_buf
        h = ttnn.clone(hidden[:, :, 0:1, :])
        hidden.deallocate(True)
        return h

    def generate_fused(self, anchor_token, anchor_pos, max_new_tokens):
        """Greedy speculative decode using the fully on-device fused iteration.

        Each iteration reads back only the ``2K+1`` token ids; the drafter
        recurrence (argmax + re-embed) and the verify-input assembly stay on
        device. The next drafter seed is SHIFT-mode: the verify hidden at the new
        anchor position ``p+m+1`` (slice ``vhidden[m+1]``) — no extra reseed
        forward, so the whole iteration is one device program (this is the eager
        twin of the fused trace). Returns ``(generated_ids, accepts_per_iter)``.
        """
        if self._use_trace:
            return self._generate_fused_traced(anchor_token, anchor_pos, max_new_tokens)
        self._last_fused_setup_s = 0.0
        K = self.draft_len
        out, accepts = [], []
        loop_t0 = time.perf_counter()
        anchor_hidden = self.seed(anchor_token, anchor_pos)  # [1,1,1,h] device
        anchor_tok_tt = self._tokens_tensor([anchor_token])
        while len(out) < max_new_tokens:
            if self._fused_reseed:
                new_anchor_hidden = self.seed(anchor_token, anchor_pos)
                anchor_hidden.deallocate(True)
                anchor_hidden = new_anchor_hidden
            drafts, target_ids, vhidden = self._fused_iter(anchor_tok_tt, anchor_hidden, anchor_pos)
            m = next((i for i in range(K) if drafts[i] != target_ids[i]), K)
            committed = drafts[:m] + [target_ids[m]]
            accepts.append(m)

            new_pos = anchor_pos + m + 1
            new_token = committed[-1]
            if self._fused_reseed:
                # The next iteration will run an exact batch=1 seed for
                # (new_token, new_pos), which also overwrites/repairs the target
                # KV at the new anchor before the drafter cross-attends to it.
                new_anchor_hidden = anchor_hidden
            else:
                # Fast approximate seed: hidden at position p+m+1 = vhidden row
                # (m+1); when all K accepted there is no row K+1, fall back to
                # the last available row (K). This is not the exact assistant
                # contract for correction/bonus anchors, but is useful for A/B.
                row = self._fused_shift_seed_row(m, K)
                new_anchor_hidden = ttnn.clone(vhidden[:, :, row : row + 1, :])
            vhidden.deallocate(True)
            if not self._fused_reseed:
                anchor_hidden.deallocate(True)
            anchor_hidden = new_anchor_hidden
            anchor_tok_tt.deallocate(True)
            anchor_tok_tt = self._tokens_tensor([new_token])
            anchor_pos = new_pos

            for tok in committed:
                out.append(tok)
                if tok in self.stop_tokens:
                    anchor_hidden.deallocate(True)
                    anchor_tok_tt.deallocate(True)
                    self._last_fused_replay_s = time.perf_counter() - loop_t0
                    return out, accepts
                if len(out) >= max_new_tokens:
                    break
        anchor_hidden.deallocate(True)
        anchor_tok_tt.deallocate(True)
        self._last_fused_replay_s = time.perf_counter() - loop_t0
        return out, accepts

    # ── fused single-iteration trace (greedy) ───────────────────────────────
    def _fused_body(self, tr):
        """The fused-iteration op graph over persistent buffers (capture + compile).

        Reads tr["anchor_tok"]/tr["h"] (persistent inputs) and the position /
        page-table buffers; returns the persistent OUTPUT handles
        (verify_x [1,K+1], vidx [1,1,K+1], vhidden [1,1,K+1,backbone]). Drafts are
        verify_x[1:]. All argmax/re-embed/concat are on device, so the K drafter
        steps chain in-graph (no inter-replay copy)."""
        K = self.draft_len
        page_tables = {lt: tr["d_pt"] for lt in self._shared_kv}
        tok = tr["anchor_tok"]
        if self._fused_reseed:
            seed_logits, h = self.target.ttnn_verify_forward(
                x=tr["anchor_tok"],
                current_pos=tr["d_pu"],
                current_pos_cache=tr["d_pi"],
                page_table=tr["d_pt"],
                kv_cache=self.tt_kv_cache,
            )
            seed_idx = self._argmax_last(seed_logits, rows=1)
            seed_logits.deallocate(True)
        else:
            h = tr["h"]
        draft_ids = []
        for _ in range(K):
            logits, h = self.assistant.step(tok, h, self._shared_kv, page_tables, tr["d_pu"], tr["d_pi"])
            idx = self._argmax_last(logits, rows=1)  # [1,1,1] uint32 RM (fast multicore argmax)
            logits.deallocate(True)
            tok = ttnn.reshape(idx, (1, 1))  # [1,1] uint32 RM
            draft_ids.append(tok)
        # In exact-reseed mode, the seed forward already computed row 0
        # (target logits for the anchor) and repaired the anchor KV before the
        # drafter reads shared KV. Verify only the draft tokens at p+1..p+K and
        # prepend seed_idx to form target_ids[0..K]. In shift mode, keep the old
        # single verify batch [anchor, d0..dK-1].
        verify_inputs = draft_ids if self._fused_reseed else [tr["anchor_tok"]] + draft_ids
        verify_x = ttnn.concat(verify_inputs, dim=1)  # reseed: [1,K], shift: [1,K+1]
        vlogits, vhidden = self.target.ttnn_verify_forward(
            x=verify_x,
            current_pos=tr["v_pu"],
            current_pos_cache=tr["v_pi"],
            page_table=tr["v_pt"],
            kv_cache=self.tt_kv_cache,
        )
        tail_rows = K if self._fused_reseed else K + 1
        tail_idx = self._argmax_last(vlogits, rows=tail_rows)  # [1,1,K or K+1] uint32 RM
        if self._fused_reseed:
            vidx = ttnn.concat([seed_idx, tail_idx], dim=2)  # [1,1,K+1]
            seed_idx.deallocate(True)
            tail_idx.deallocate(True)
        else:
            vidx = tail_idx
        vlogits.deallocate(True)
        return verify_x, vidx, vhidden

    def _capture_fused_trace(self, anchor_token, anchor_hidden, anchor_pos):
        """Capture ONE fused iteration at the real first-call inputs.

        Allocates persistent input buffers (anchor token, recurrent hidden, the
        drafter + verify position tensors, page tables), runs a compile pass, then
        captures the fused body. The compile + capture write the SAME KV positions
        with the SAME tokens (deterministic), so the writes are idempotent."""
        from loguru import logger as _lg

        K = self.draft_len
        v_pos = [anchor_pos + 1 + j for j in range(K)] if self._fused_reseed else [anchor_pos + j for j in range(K + 1)]
        d_pu, d_pi = self._pos_tensors([anchor_pos])
        v_pu, v_pi = self._pos_tensors(v_pos)
        tr = {
            "anchor_tok": self._tokens_tensor([anchor_token]),
            "h": ttnn.clone(anchor_hidden),
            "d_pu": d_pu,
            "d_pi": d_pi,
            "d_pt": self._page_table(1),
            "v_pu": v_pu,
            "v_pi": v_pi,
            "v_pt": self._page_table(K if self._fused_reseed else K + 1),
        }
        _lg.info("[spec-trace] capture fused: compile run")
        vx, vidx, vh = self._fused_body(tr)
        ttnn.synchronize_device(self.mesh_device)
        vx.deallocate(True)
        vidx.deallocate(True)
        vh.deallocate(True)
        _lg.info("[spec-trace] capture fused: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        vx, vidx, vh = self._fused_body(tr)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info("[spec-trace] capture fused: DONE")
        tr["id"] = tid
        tr["verify_x"] = vx
        tr["vidx"] = vidx
        tr["vhidden"] = vh
        self._fused_trace = tr

    def _hidden_row_to_device(self, row):
        """Read the verify hidden, slice row `row`, and copy it into tr["h"].

        Host round-trip for the recurrent seed: small ([1,1,1,backbone]) and
        allocation-free on device (copy_host_to_device into the persistent buffer),
        so it stays trace-safe (no device clone/slice that could alias trace
        scratch on a re-replay)."""
        tr = self._fused_trace
        vh = tr["vhidden"]
        vh_t = ttnn.to_torch(ttnn.get_device_tensors(vh)[0]) if self._tp > 1 else ttnn.to_torch(vh)
        sel = vh_t[:, :, row : row + 1, :].contiguous()  # [1,1,1,backbone]
        host_h = ttnn.from_torch(sel, layout=ttnn.TILE_LAYOUT, dtype=tr["h"].dtype, mesh_mapper=self._mapper)
        ttnn.copy_host_to_device_tensor(host_h, tr["h"])
        host_h.deallocate(True)

    def _generate_fused_traced(self, anchor_token, anchor_pos, max_new_tokens):
        """Greedy spec-decode via the single fused trace (one replay per iter)."""
        from loguru import logger as _lg

        K = self.draft_len
        # Eager seed (once, before the loop). Force the UNTRACED verify so we do
        # NOT capture a separate batch=1 verify trace — an active verify trace
        # would collide with the fused trace's eager compile-run allocations and
        # hang. The loop then replays ONLY the single fused trace (no interleaving
        # of distinct CCL traces).
        setup_t0 = time.perf_counter()
        self._use_trace = False
        anchor_hidden = self.seed(anchor_token, anchor_pos)
        self._use_trace = True
        self._capture_fused_trace(anchor_token, anchor_hidden, anchor_pos)
        anchor_hidden.deallocate(True)
        tr = self._fused_trace
        self._last_fused_setup_s = time.perf_counter() - setup_t0

        out, accepts = [], []
        cur_token, cur_pos = anchor_token, anchor_pos
        first = True  # capture already bound the first inputs (token/pos/hidden)
        replay_t0 = time.perf_counter()
        while len(out) < max_new_tokens:
            if not first:
                h_tok = self._host_tokens([cur_token])
                ttnn.copy_host_to_device_tensor(h_tok, tr["anchor_tok"])
                h_tok.deallocate(True)
                d_hpu, d_hpi = self._host_pos([cur_pos])
                ttnn.copy_host_to_device_tensor(d_hpu, tr["d_pu"])
                ttnn.copy_host_to_device_tensor(d_hpi, tr["d_pi"])
                d_hpu.deallocate(True)
                d_hpi.deallocate(True)
                v_pos = (
                    [cur_pos + 1 + j for j in range(K)] if self._fused_reseed else [cur_pos + j for j in range(K + 1)]
                )
                v_hpu, v_hpi = self._host_pos(v_pos)
                ttnn.copy_host_to_device_tensor(v_hpu, tr["v_pu"])
                ttnn.copy_host_to_device_tensor(v_hpi, tr["v_pi"])
                v_hpu.deallocate(True)
                v_hpi.deallocate(True)
                # In reseed mode the trace computes the exact seed internally.
                # Otherwise tr["h"] already holds this iter's approximate seed
                # (set at the end of the previous iteration).
            first = False

            _lg.debug(f"[spec-trace] fused replay pos={cur_pos} execute")
            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)

            vx = (
                ttnn.to_torch(ttnn.get_device_tensors(tr["verify_x"])[0])
                if self._tp > 1
                else ttnn.to_torch(tr["verify_x"])
            )
            vx = vx.reshape(-1)
            drafts = [int(vx[j if self._fused_reseed else 1 + j]) for j in range(K)]
            target_ids = self._ids_to_host(tr["vidx"], K + 1)
            m = next((i for i in range(K) if drafts[i] != target_ids[i]), K)
            committed = drafts[:m] + [target_ids[m]]
            accepts.append(m)

            if not self._fused_reseed:
                # Shift seed: choose one of the verified hidden rows as a cheap
                # approximate seed for the next anchor (see _fused_shift_seed).
                self._hidden_row_to_device(self._fused_shift_seed_row(m, K))
            cur_pos = cur_pos + m + 1
            cur_token = committed[-1]

            for tok in committed:
                out.append(tok)
                if tok in self.stop_tokens:
                    self._last_fused_replay_s = time.perf_counter() - replay_t0
                    return out, accepts
                if len(out) >= max_new_tokens:
                    break
        self._last_fused_replay_s = time.perf_counter() - replay_t0
        return out, accepts

    def generate(
        self, anchor_token, anchor_pos, max_new_tokens, anchor_hidden=None, temperature=0.0, top_p=1.0, top_k=0
    ):
        """Run speculative decode from a prompt anchor (prefill must have filled the KV).

        Args:
            anchor_token: last committed token id (e.g. last prompt token).
            anchor_pos: its absolute position p.
            max_new_tokens: max tokens to generate.
            anchor_hidden: optional pre-seeded target hidden at p; computed via
                ``seed`` when None.

        Returns:
            (generated_token_ids, num_accept_per_iter) — the latter for accept-rate stats.
        """
        greedy = not temperature or temperature <= 0
        if self._use_trace:
            if greedy:
                return self.generate_fused(anchor_token, anchor_pos, max_new_tokens)

            # Sampling mode cannot use the single fused greedy trace because the
            # draft/verify token selection is non-deterministic (it depends on the
            # sampled token), so draft and verify must run as two SEPARATE traces
            # that interleave each iteration. That interleaving is unsafe here:
            #   1. Deadlock: draft and verify each capture their own multi-device
            #      CCL (all-gather/reduce-scatter) trace. Replaying two distinct
            #      CCL traces back-to-back on the same command queue makes one
            #      trace's collective wait on buffers the other trace still owns,
            #      so the mesh hangs (the same failure the fused path avoids).
            #   2. Allocation collision: capturing a second (verify) trace while a
            #      draft trace is live reuses persistent I/O buffers the draft
            #      trace still references, corrupting in-flight state.
            # Greedy sidesteps both by fusing draft+verify into ONE trace; sampling
            # has no fused equivalent yet, so fall back to the untraced host loop.
            from loguru import logger as _lg

            _lg.warning(
                "Disabling speculative trace for sampling mode; separate draft/verify "
                "trace interleaving deadlocks the mesh (see generate() for details)"
            )
            self._use_trace = False
            try:
                return self.generate(
                    anchor_token,
                    anchor_pos,
                    max_new_tokens,
                    anchor_hidden=anchor_hidden,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            finally:
                self._use_trace = True

        traced = self._use_trace
        out = []
        accepts = []
        # Own (and free) only what we allocate: a caller-provided anchor_hidden
        # stays the caller's to manage; a seeded one is ours. In traced mode the
        # seeded anchor is the PERSISTENT self._anchor_buf (never deallocated).
        owns_anchor_hidden = (anchor_hidden is None) and not traced
        if anchor_hidden is None:
            anchor_hidden = self.seed(anchor_token, anchor_pos)
        draft_fn = self._draft_traced if self._use_trace else self._draft
        while len(out) < max_new_tokens:
            drafts, draft_logits = draft_fn(
                anchor_token, anchor_hidden, anchor_pos, temperature=temperature, top_p=top_p, top_k=top_k
            )

            verify_tokens = [anchor_token] + drafts
            verify_pos = [anchor_pos + j for j in range(len(verify_tokens))]
            verify_logits, hidden = self._verify(verify_tokens, verify_pos)

            if greedy:
                m, committed = self._accept_greedy(drafts, verify_logits)
            else:
                m, committed = self._accept_sampling(drafts, draft_logits, verify_logits, temperature, top_p, top_k)
            accepts.append(m)

            # New anchor token = committed[-1] = g[m] at position anchor_pos+m+1.
            # The drafter seed must be the target hidden for THIS token at THIS
            # position. The verify (tokens [anchor, d1..dK] at positions P..P+K)
            # only has hidden at index j = position P+j computed with the draft
            # token there as input — never g[m] at P+m+1. Seed quality drives the
            # acceptance rate (the first-step seed() is exact, PCC ~0.98):
            #   reseed: exact batch=1 verify of (g[m], P+m+1) — best quality, one
            #           extra small target forward (can be folded into the trace).
            #   shift : hidden[m+1] = position-aligned but input-approximate
            #           (computed with the rejected draft); no extra forward.
            #   cur   : hidden[m] = position P+m (input+position mismatched).
            new_pos = anchor_pos + m + 1
            new_token = committed[-1]
            # In traced mode `hidden` is the persistent verify output: never
            # deallocate it, and the default reseed path doesn't read it.
            if self._seed_mode == "cur":
                new_anchor_hidden = ttnn.clone(hidden[:, :, m : m + 1, :])
                if not traced:
                    hidden.deallocate(True)
            elif self._seed_mode == "shift" and m + 1 <= len(drafts):
                new_anchor_hidden = ttnn.clone(hidden[:, :, m + 1 : m + 2, :])
                if not traced:
                    hidden.deallocate(True)
            else:  # "reseed" (default) or shift fallback when m == draft_len
                if not traced:
                    hidden.deallocate(True)
                new_anchor_hidden = self.seed(new_token, new_pos)
            if owns_anchor_hidden:
                anchor_hidden.deallocate(True)
            anchor_hidden, owns_anchor_hidden = new_anchor_hidden, not traced
            anchor_pos = new_pos
            anchor_token = new_token

            for tok in committed:
                out.append(tok)
                if tok in self.stop_tokens:
                    if owns_anchor_hidden:
                        anchor_hidden.deallocate(True)
                    return out, accepts
                if len(out) >= max_new_tokens:
                    break

        if owns_anchor_hidden:
            anchor_hidden.deallocate(True)
        return out, accepts
