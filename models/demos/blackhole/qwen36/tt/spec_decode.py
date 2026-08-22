# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Qwen3.8 with the MTP drafter head (batch=1, greedy).

Verify = one masked-bucket chunk forward of [committed-tail, drafts] from the
block-aligned GDN state anchor `a`; per-row logits/hiddens via a one-hot
row-select + final norm + LM head. GDN state is NEVER committed by a verify —
every iteration snapshots the state (single device: the handles the chunk path
reassigns; TP: clones of the in-place carried buffers) and restores it; the
state advances only through periodic block-aligned commit chunks over committed
tokens. KV rollback is implicit: the next chunk always starts at `a` and
rewrites every polluted position before anything attends past it.

Runs single-device or TP (B=1); on TP the drafter head executes fully
replicated (see tt/mtp.py).

TP runs TRACED by default (TT_SPEC_TRACE=0 for eager): the verify chunk and the
drafter step are captured once and replayed per iteration, with every varying
quantity carried as data in persistent device buffers (tokens, RoPE window,
chunk page table, device chunk_start, the row-select one-hot, and the GDN
valid_len masks — the trace-safe form of the masked bucket). Capture ordering
is strict: prefill + the eager first verify (+ a 32-row extraction warm)
compile every program BEFORE any capture, the drafter trace captures first,
then the verify trace — so no compile ever happens after a trace is parked.
Commit chunks stay eager (amortized ~1/64 of iterations).

Committed tokens are ALWAYS produced by the target verify rows, so greedy
spec decode matches plain greedy decode up to bf16 chunk-vs-decode numerics.

See docs/mtp_design.md for the full design.
"""

import os

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import Mode

BLOCK = 64
_PREFILL_SEG = 2048


def block_aligned_prefill_len(prompt_len):
    """Largest block multiple strictly below prompt_len — the initial state anchor.

    Strictly below so the first (draft-less) verify chunk is non-empty: it
    processes the prompt tail and samples the first token.
    """
    assert prompt_len >= 1
    return BLOCK * ((prompt_len - 1) // BLOCK)


def commit_advance(pending_len):
    """Block-multiple of GDN state to commit, leaving >=1 committed token uncommitted.

    The target logits that accept draft 1 are the verify row AFTER processing
    the last committed token t_c, so t_c itself must be in every verify chunk:
    the anchor must never catch up to c+1 (row_start = c - a stays >= 0, and a
    fully-committed anchor would leave no row to accept from).
    """
    if pending_len <= BLOCK:
        return 0
    return BLOCK * ((pending_len - 1) // BLOCK)


def adaptive_draft_len(accept_ema, k_max):
    """Drafts worth proposing given the recent-accept EMA: one more than the
    expected accepts (each iteration commits accepts+1 tokens), clamped to
    [1, k_max]. Purely data-driven, so it is trace-compatible."""
    return max(1, min(int(k_max), int(accept_ema + 0.5) + 1))


def greedy_accept(drafts, target_ids):
    """Longest matching draft prefix + the target's correction/bonus token.

    target_ids[j] is the target argmax at the position of draft j (row j of the
    verify); target_ids has len(drafts)+1 entries (the last is the bonus row).

    Returns:
        (m, committed): m accepted drafts; committed = drafts[:m] + [target_ids[m]].
    """
    assert len(target_ids) == len(drafts) + 1
    m = len(drafts)
    for i, d in enumerate(drafts):
        if d != target_ids[i]:
            m = i
            break
    return m, list(drafts[:m]) + [target_ids[m]]


class Qwen36SpeculativeDecoder:
    """Draft (MTP head) -> verify (masked-bucket chunk) -> accept loop, B=1 greedy.

    Args:
        model: Qwen36Model (single device or TP, B=1) with paged KV caches allocated.
        mtp_head: Qwen36MTPHead with its KV cache allocated.
        page_table: torch [1, num_blocks] page table for the target's paged KV.
        draft_len: K draft tokens per iteration.
        stop_tokens: token ids that end generation.
        seed_window: max prompt positions to seed the drafter KV with
            (TT_SPEC_SEED_WINDOW overrides; earlier drafter slots stay zero).
    """

    # State the GDN chunk path reassigns (split_conv_state only exists after a
    # T=1 decode, which spec mode never runs — carried for safety).
    _GDN_STATE_ATTRS = (
        "recurrent_state",
        "fused_conv_state",
        "conv_state_q",
        "conv_state_k",
        "conv_state_v",
        "split_conv_state",
    )

    def __init__(self, model, mtp_head, page_table, draft_len=4, stop_tokens=None, seed_window=None):
        self.model = model
        self.mtp = mtp_head
        self.page_table = page_table
        self.draft_len = int(draft_len)
        assert 1 <= self.draft_len <= 31, "draft_len must fit the 32-row extraction"
        self.stop_tokens = set(stop_tokens or [])
        self.seed_window = int(seed_window if seed_window is not None else os.environ.get("TT_SPEC_SEED_WINDOW", 2048))
        self._dns = [layer.attention for layer in model.layers if not layer.is_full_attention]
        # TP (B=1): the TPGatedDeltaNet chunk path carries state IN PLACE
        # (_stable_state, set by the TP allocate_kv_caches) into rec_state /
        # conv_carry, so snapshots must clone+copy-back; the single-device chunk
        # path REASSIGNS state tensors, so snapshots are just the handles.
        self._tp = model.num_devices > 1
        if self._tp:
            for dn in self._dns:
                assert getattr(dn, "_stable_state", False), "TP spec decode needs in-place GDN state (allocate first)"
        # Traced loop (TT_SPEC_TRACE=0 falls back to eager). TP-only: the traced
        # verify needs the fused-chunk valid_masks path, which the single-device
        # masked bucket (seq adapter) does not have.
        self._use_trace = self._tp and os.environ.get("TT_SPEC_TRACE", "1") == "1"
        # Adaptive draft length (TT_SPEC_ADAPTIVE_K=1): shrink K toward the
        # recent accept EMA so rejected drafts stop paying drafter steps.
        self._adaptive_k = os.environ.get("TT_SPEC_ADAPTIVE_K", "0") == "1"
        self._k_ema = float(self.draft_len)
        self.k_used = []  # drafts proposed per iteration (stats)
        self._vt = None  # traced-verify persistent I/O (built lazily)
        self._snap_bufs = None  # persistent TP GDN snapshot buffers (allocation-free loop)

        self.committed = []  # token ids 0..c (prompt + generated)
        self.a = 0  # block-aligned state anchor: GDN state reflects tokens 0..a-1
        self._pending = []  # (input_token, hidden_row, drafter_pos) awaiting MTP catch-up
        self.accepts = []  # accepted drafts per iteration (stats)

    # ── GDN state bookkeeping ────────────────────────────────────────────────
    def _gdn_handles(self):
        return [{attr: getattr(dn, attr) for attr in self._GDN_STATE_ATTRS} for dn in self._dns]

    @staticmethod
    def _dealloc(t):
        if t is not None and not isinstance(t, list):
            ttnn.deallocate(t)

    def _gdn_snapshot(self):
        """State snapshot before a verify chunk (never-committed contract).

        TP clones the in-place carried buffers; single-device records handles.
        (TP note: the chunk's capture_state also refreshes the decode conv
        window conv_states, which is not restored — spec mode never runs GDN
        decode, so nothing reads it.)
        """
        if self._tp:
            # Persistent snapshot buffers: allocated once (before any trace is
            # captured), refreshed by ttnn.copy — the traced loop must stay
            # allocation-free between replays.
            if self._snap_bufs is None:
                self._snap_bufs = [
                    (
                        ttnn.clone(dn.rec_state, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                        ttnn.clone(dn.conv_carry, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    )
                    for dn in self._dns
                ]
            else:
                for dn, (rec, conv) in zip(self._dns, self._snap_bufs):
                    ttnn.copy(dn.rec_state, rec)
                    ttnn.copy(dn.conv_carry, conv)
            return self._snap_bufs
        return self._gdn_handles()

    def _gdn_restore(self, snap):
        """Roll GDN state back to the snapshot taken before the verify chunk."""
        if self._tp:
            for dn, (rec, conv) in zip(self._dns, snap):
                ttnn.copy(rec, dn.rec_state)
                ttnn.copy(conv, dn.conv_carry)
            return
        for dn, s in zip(self._dns, snap):
            for attr, old in s.items():
                cur = getattr(dn, attr)
                if cur is not old:
                    self._dealloc(cur)
                setattr(dn, attr, old)

    def _gdn_commit_ctx(self):
        """Pre-commit context: single-device must free the handles the chunk
        replaces; the TP chunk commits in place (nothing to free)."""
        return None if self._tp else self._gdn_handles()

    def _gdn_commit_done(self, ctx):
        if ctx is None:
            return
        for dn, s in zip(self._dns, ctx):
            for attr, prev in s.items():
                if getattr(dn, attr) is not prev:
                    self._dealloc(prev)

    # ── target chunk forwards ────────────────────────────────────────────────
    def _chunk_forward(self, tokens, chunk_start, valid_len):
        """Masked-bucket forward over `tokens` at absolute chunk_start.

        Advances GDN state through exactly valid_len tokens (by reassignment)
        and writes attn KV for the blocks covering chunk_start..+valid_len.
        Returns (hidden [1, bucket, dim] pre-final-norm, bucket).
        """
        model = self.model
        n = len(tokens)
        assert 1 <= valid_len <= n
        assert chunk_start % BLOCK == 0, f"chunk_start {chunk_start} must be block-aligned (paged_fill_cache)"
        bucket = model._mask_bucket_for(n)
        buf = torch.zeros(1, bucket, dtype=torch.int32)
        buf[0, :n] = torch.tensor(tokens, dtype=torch.int32)
        hidden = model._forward_prefill_chunk_masked(buf, valid_len, chunk_start, self.page_table, bucket)
        return hidden, bucket

    def _rows_to_host(self, t, n):
        """First n rows of a [.., R, D] device tensor as host float [n, D].

        TP tensors here are replicated (post DistributedNorm / lm-head gather),
        so one replica is the full value.
        """
        src = ttnn.get_device_tensors(t)[0] if self._tp else t
        return ttnn.to_torch(src).float().reshape(-1, t.shape[-1])[:n]

    def _extract_rows(self, hidden, bucket, row_start, n, want_logits=True):
        """Post-final-norm hiddens (+ logits) for rows row_start..row_start+n-1.

        One-hot row-select matmul keeps the program fixed per (bucket, 32-padded
        n); the row indices are data. Returns host float tensors
        (logits [n, vocab] or None, hidden [n, dim]).
        """
        model = self.model
        # A negative row_start would silently wrap the one-hot to the bucket's
        # padded tail and hand the accept path garbage logits.
        assert (
            0 <= row_start and row_start + n <= bucket
        ), f"row range [{row_start}, {row_start + n}) not in bucket {bucket}"
        rows_padded = ((n + 31) // 32) * 32
        # TP hidden is rank-4 [1,1,bucket,dim/tp] (fractured); single-device is
        # rank-3 [1,bucket,dim]. The replicated one-hot selects rows either way,
        # and the final norm (DistributedNorm on TP) hands back full-dim rows.
        sel_shape = (1, 1, rows_padded, bucket) if self._tp else (1, rows_padded, bucket)
        sel = torch.zeros(*sel_shape, dtype=torch.float32)
        for j in range(n):
            sel[..., j, row_start + j] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            dtype=hidden.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=model.device,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(model.device)) if self._tp else {}),
        )
        rows = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        rows = ttnn.to_memory_config(rows, ttnn.DRAM_MEMORY_CONFIG)
        normed = model.norm(rows, mode=Mode.PREFILL)
        ttnn.deallocate(rows)
        hid = self._rows_to_host(normed, n)
        logits_host = None
        if want_logits:
            logits = model._lm_head(normed)
            logits_host = self._rows_to_host(logits, n)
            ttnn.deallocate(logits)
        ttnn.deallocate(normed)
        return logits_host, hid

    # ── traced verify (TP) ───────────────────────────────────────────────────
    _VERIFY_BUCKET = 128
    _VERIFY_ROWS = 32

    def _init_verify_trace(self):
        """Persistent device I/O for the traced verify chunk (alloc once, pre-capture).

        Everything that varies per iteration is DATA in these buffers — token
        ids, RoPE tables for the window, the 2-block chunk page table, the
        device chunk_start, the row-select one-hot, and the GDN valid_len masks
        (conv one-hot + beta/g/qkv zero masks). Shapes are fixed by the bucket.
        """
        model = self.model
        mesh = model.device
        B = self._VERIFY_BUCKET
        rd = model.args.rope_head_dim
        conv_k = self._dns[0].K  # GDN conv kernel size
        rep = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def dev(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, **rep)

        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        self._vt = {
            "tok": dev(torch.zeros(1, B, dtype=torch.int64), ttnn.uint32, rm),
            "cos": dev(torch.zeros(1, 1, B, rd, dtype=torch.bfloat16), ttnn.bfloat16, tile),
            "sin": dev(torch.zeros(1, 1, B, rd, dtype=torch.bfloat16), ttnn.bfloat16, tile),
            "full_pt": dev(self.page_table.to(torch.int32), ttnn.int32, rm),  # constant
            "chunk_pt": dev(torch.zeros(1, 2, dtype=torch.int32), ttnn.int32, rm),
            "csi": dev(torch.zeros(1, dtype=torch.int32), ttnn.int32, rm),
            "sel": dev(torch.zeros(1, 1, self._VERIFY_ROWS, B, dtype=torch.float32), ttnn.bfloat16, tile),
            "conv_sel": dev(torch.zeros(1, conv_k - 1, (conv_k - 1) + B, dtype=torch.float32), ttnn.bfloat16, tile),
            "mask_f32": dev(torch.zeros(1, B, 1, dtype=torch.float32), ttnn.float32, tile),
            "mask_bf16": dev(torch.zeros(1, B, 1, dtype=torch.float32), ttnn.bfloat16, tile),
        }

    def _stage_verify_inputs(self, tokens, chunk_start, valid_len, row_start):
        """Refresh the persistent verify inputs (host build + copy, no allocs on device)."""
        model, vt = self.model, self._vt
        B = self._VERIFY_BUCKET
        n = len(tokens)
        assert n <= B and valid_len <= B
        conv_k = self._dns[0].K
        blk0 = chunk_start // BLOCK
        assert blk0 + 2 <= self.page_table.shape[1], "verify window exceeds the page-table block budget"

        tok = torch.zeros(1, B, dtype=torch.int64)
        tok[0, :n] = torch.tensor(tokens, dtype=torch.int64)
        cos_t, sin_t = model._rope_tp_cos_sin_torch(chunk_start, B)  # [1,1,B,rd] bf16
        sel = torch.zeros(1, 1, self._VERIFY_ROWS, B, dtype=torch.float32)
        for j in range(min(self._VERIFY_ROWS, valid_len - row_start)):
            sel[0, 0, j, row_start + j] = 1.0
        conv_sel = torch.zeros(1, conv_k - 1, (conv_k - 1) + B, dtype=torch.float32)
        for j in range(conv_k - 1):
            conv_sel[0, j, valid_len + j] = 1.0
        mask = torch.zeros(1, B, 1, dtype=torch.float32)
        mask[0, :valid_len, 0] = 1.0

        rep = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device))
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        for host_t, dtype, layout, dst in (
            (tok, ttnn.uint32, rm, "tok"),
            (cos_t, ttnn.bfloat16, tile, "cos"),
            (sin_t, ttnn.bfloat16, tile, "sin"),
            (self.page_table[:, blk0 : blk0 + 2].to(torch.int32).contiguous(), ttnn.int32, rm, "chunk_pt"),
            (torch.tensor([chunk_start], dtype=torch.int32), ttnn.int32, rm, "csi"),
            (sel, ttnn.bfloat16, tile, "sel"),
            (conv_sel, ttnn.bfloat16, tile, "conv_sel"),
            (mask, ttnn.float32, tile, "mask_f32"),
            (mask, ttnn.bfloat16, tile, "mask_bf16"),
        ):
            src = ttnn.from_torch(host_t, dtype=dtype, layout=layout, **rep)
            ttnn.copy_host_to_device_tensor(src, vt[dst])

    def _verify_trace_body(self):
        """The verify-chunk op graph over the persistent buffers (compile + capture).

        Mirrors _forward_prefill_chunk_masked_tp with the GDN valid_len masks fed
        as device tensors, plus the in-trace row-select + final norm + LM head.
        Returns the persistent output handles (logits, normed rows).
        """
        model, vt = self.model, self._vt
        B = self._VERIFY_BUCKET
        vm = {"conv_sel": vt["conv_sel"], "f32": vt["mask_f32"], "bf16": vt["mask_bf16"]}
        x = model.embd(vt["tok"])
        x = ttnn.reshape(x, (1, 1, B, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        for layer in model.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=vt["cos"],
                    sin=vt["sin"],
                    mode="prefill",
                    page_table=vt["full_pt"],
                    chunk_page_table=vt["chunk_pt"],
                    chunk_start_idx_tensor=vt["csi"],
                )
            else:
                x_new = layer.forward(
                    x, mode="prefill", chunk_size=model.args.gdn_chunk_size, valid_len=None, valid_masks=vm
                )
            ttnn.deallocate(x)
            x = x_new
        rows = ttnn.matmul(vt["sel"], x)  # [1,1,ROWS,dim/tp] fractured
        ttnn.deallocate(x)
        rows = ttnn.to_memory_config(rows, ttnn.DRAM_MEMORY_CONFIG)
        normed = model.norm(rows, mode=Mode.PREFILL)  # replicated [1,1,ROWS,dim]
        ttnn.deallocate(rows)
        logits = model._lm_head(normed)  # replicated [1,1,ROWS,vocab]
        return logits, normed

    def _verify_traced(self, tokens, chunk_start, valid_len, row_start, n_rows):
        """Traced verify: refresh inputs, replay, read the accept rows.

        Lazy capture on the first call — by then every program in the body is
        warm (prefill segments, the eager first verify, and _warm_extract32),
        so the compile run compiles nothing and cannot clobber the parked
        drafter trace (#48536 class).
        """
        mesh = self.model.device
        assert chunk_start % BLOCK == 0
        first = self._vt is None
        if first:
            self._init_verify_trace()
        self._stage_verify_inputs(tokens, chunk_start, valid_len, row_start)
        if first:
            # Compile + capture both advance GDN state (non-idempotent) — roll
            # back to the pre-verify snapshot between and after them so the
            # first real replay starts from the correct state.
            snap = self._gdn_snapshot()
            logits, normed = self._verify_trace_body()
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(logits)
            ttnn.deallocate(normed)
            self._gdn_restore(snap)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            logits, normed = self._verify_trace_body()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            self._gdn_restore(snap)
            self._vt.update({"id": tid, "logits": logits, "normed": normed})
            logger.info("spec verify trace captured (bucket 128)")
        ttnn.execute_trace(mesh, self._vt["id"], cq_id=0, blocking=False)
        lh = self._rows_to_host(self._vt["logits"], n_rows)
        hh = self._rows_to_host(self._vt["normed"], n_rows)
        return lh, hh

    def release_traces(self):
        """Release the verify + drafter traces (call before freeing KV caches)."""
        if self._vt is not None and "id" in self._vt:
            ttnn.release_trace(self.model.device, self._vt["id"])
        self._vt = None
        release = getattr(self.mtp, "release_step_trace", None)
        if release is not None:
            release()

    def _maybe_commit(self):
        """Advance the GDN anchor by whole blocks once enough tokens committed."""
        k = commit_advance(len(self.committed) - self.a)
        if k == 0:
            return
        ctx = self._gdn_commit_ctx()
        hidden, _ = self._chunk_forward(self.committed[self.a : self.a + k], self.a, k)
        ttnn.deallocate(hidden)
        self._gdn_commit_done(ctx)
        self.a += k

    # ── prefill + drafter seeding ────────────────────────────────────────────
    def prefill(self, token_ids):
        """Segmented masked-bucket prefill up to the block-aligned anchor.

        Processes prompt tokens 0..a0-1 (a0 = block_aligned_prefill_len(T)) in
        <=2048-token segments, capturing post-norm hiddens for the drafter seed
        window and seeding the drafter KV. The prompt tail t_a0..t_{T-1} is left
        for the first verify chunk (which samples the first token).
        """
        model = self.model
        T = token_ids.shape[1]
        assert T >= 2, "prompt too short for MTP"
        self.committed = [int(t) for t in token_ids[0]]
        self.a = a0 = block_aligned_prefill_len(T)
        self._pending = []
        self.accepts = []

        model._reset_gdn_state_for_new_sequence()
        model._build_request_rope(token_ids[:, :T], None)

        seed_from = max(0, a0 - self.seed_window)
        prompt_hidden = {}  # position -> torch [dim], only within the seed window
        for start in range(0, a0, _PREFILL_SEG):
            length = min(_PREFILL_SEG, a0 - start)
            ctx = self._gdn_commit_ctx()
            hidden, _bucket = self._chunk_forward(self.committed[start : start + length], start, length)
            if start + length > seed_from:
                normed = model.norm(hidden, mode=Mode.PREFILL)
                rows = self._rows_to_host(normed, length)
                if normed is not hidden:
                    ttnn.deallocate(normed)
                for i in range(max(seed_from, start), start + length):
                    prompt_hidden[i] = rows[i - start]
            ttnn.deallocate(hidden)
            self._gdn_commit_done(ctx)

        # Seed the drafter KV over the (windowed) prompt: pair (h_i, t_{i+1}) at
        # drafter position i. Always EAGER — this doubles as the drafter program
        # warmup, and no trace may be captured before the eager first verify has
        # compiled every target-side program (post-park compiles clobber traces).
        for i in range(seed_from, a0):
            self.mtp.step(self.committed[i + 1], prompt_hidden[i], i)
        logger.info(f"spec prefill: T={T} anchor a0={a0} drafter seeded on [{seed_from}, {a0})")

    # ── generation ───────────────────────────────────────────────────────────
    def _first_verify(self):
        """Draft-less verify over the prompt tail: samples the first token and
        yields the tail hiddens that arm the drafter catch-up pairs."""
        tail_len = len(self.committed) - self.a
        snap = self._gdn_snapshot()
        hidden, bucket = self._chunk_forward(self.committed[self.a :], self.a, tail_len)
        logits, hid = self._extract_rows(hidden, bucket, 0, tail_len)
        if self._use_trace and ((tail_len + 31) // 32) * 32 != self._VERIFY_ROWS:
            # Warm the 32-row extraction programs (row-select matmul, norm, LM
            # head at 32 rows) NOW, while no trace is parked: the traced verify's
            # compile run must not compile anything after the drafter trace is
            # captured (a post-park compile can clobber a parked trace).
            self._extract_rows(hidden, bucket, 0, 1)
        ttnn.deallocate(hidden)
        self._gdn_restore(snap)
        if self._use_trace and tail_len < BLOCK + 1:
            # Warm the 2-block paged_fill_cache program the traced verify bakes in
            # (its chunk page table is FIXED at 2 blocks; the eager tail above
            # spans only 1). Throwaway pass: state restored, and the garbage K/V
            # in the second block sits at future positions that every later chunk
            # rewrites before attending.
            pad = self.committed[self.a :] + [self.committed[-1]] * (BLOCK + 1 - tail_len)
            hidden, _ = self._chunk_forward(pad, self.a, BLOCK + 1)
            ttnn.deallocate(hidden)
            self._gdn_restore(snap)

        first_token = int(logits[tail_len - 1].argmax())
        c = len(self.committed) - 1  # last prompt position
        # Catch-up pairs (h_i, t_{i+1}) for the tail, including the new token.
        for i in range(self.a, c):
            self._pending.append((self.committed[i + 1], hid[i - self.a], i))
        self._pending.append((first_token, hid[c - self.a], c))
        self.committed.append(first_token)
        self._maybe_commit()
        return first_token

    def generate(self, max_new_tokens):
        """Greedy speculative generation. Returns (generated_ids, stats dict)."""
        K = self.draft_len
        out = []

        first_token = self._first_verify()
        out.append(first_token)
        if first_token in self.stop_tokens:
            return out, self._stats()

        while len(out) < max_new_tokens:
            # 1. Drafter catch-up over pending pairs; the last pair's logits are
            # draft 1, then chain K-1 steps on the drafter's own hidden. The
            # traced drafter keeps the chained hidden on device.
            k_t = adaptive_draft_len(self._k_ema, K) if self._adaptive_k else K
            self.k_used.append(k_t)
            last_pos = self._pending[-1][2]
            if self._use_trace:
                for tok, hid_row, pos in self._pending[:-1]:
                    self.mtp.step_traced(tok, hid_row, pos)
                d_logits = self.mtp.step_traced(*self._pending[-1])
            else:
                for tok, hid_row, pos in self._pending[:-1]:
                    self.mtp.step(tok, hid_row, pos)
                d_logits, g = self.mtp.step(*self._pending[-1])
            self._pending = []
            drafts = [int(d_logits.argmax())]
            for j in range(1, k_t):
                if self._use_trace:
                    d_logits = self.mtp.step_traced(drafts[-1], None, last_pos + j, chain_hidden=True)
                else:
                    d_logits, g = self.mtp.step(drafts[-1], g, last_pos + j)
                drafts.append(int(d_logits.argmax()))

            # 2. Verify chunk [t_a..t_c, drafts] at positions a..c+k_t.
            c = len(self.committed) - 1
            snap = self._gdn_snapshot()
            chunk_tokens = self.committed[self.a :] + drafts
            if self._use_trace:
                logits, hid = self._verify_traced(chunk_tokens, self.a, len(chunk_tokens), c - self.a, k_t + 1)
            else:
                hidden, bucket = self._chunk_forward(chunk_tokens, self.a, len(chunk_tokens))
                logits, hid = self._extract_rows(hidden, bucket, c - self.a, k_t + 1)
                ttnn.deallocate(hidden)

            # 3. Accept, then ALWAYS roll the GDN state back to the anchor.
            target_ids = [int(logits[j].argmax()) for j in range(k_t + 1)]
            m, new_tokens = greedy_accept(drafts, target_ids)
            self.accepts.append(m)
            self._k_ema = 0.7 * self._k_ema + 0.3 * m
            logger.debug(
                f"spec iter {len(self.accepts)}: c={c} a={self.a} K={k_t} drafts={drafts} targets={target_ids} m={m}"
            )
            self._gdn_restore(snap)

            # 4. Commit tokens + arm the next catch-up pairs (true target hiddens).
            stop = False
            for j, tok in enumerate(new_tokens):
                self.committed.append(tok)
                out.append(tok)
                self._pending.append((tok, hid[j], c + j))
                if tok in self.stop_tokens or len(out) >= max_new_tokens:
                    stop = True
                    break

            # 5. Block-aligned deferred GDN commit.
            self._maybe_commit()
            if stop:
                break
        return out, self._stats()

    def _stats(self):
        iters = len(self.accepts)
        total = sum(self.accepts)
        proposed = sum(self.k_used) if self.k_used else iters * self.draft_len
        return {
            "iterations": iters,
            "accepted_drafts": total,
            "accept_rate": (total / proposed) if proposed else 0.0,
            "tokens_per_iteration": ((total + iters) / iters) if iters else 0.0,
            "accepts": list(self.accepts),
            "k_used": list(self.k_used),
            "traced": self._use_trace,
        }
