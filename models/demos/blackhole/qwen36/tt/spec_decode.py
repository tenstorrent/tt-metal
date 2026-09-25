# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP speculative decode: draft K tokens, verify them in one chunk, commit the accepted prefix.
The drafter pairs (hidden_i, token_{i+1}). Verify buffers GDN state; commit points at the accepted slot.
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model_config import EAGER_RESEED_PROMPT_LEN
from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSampler


class SpeculativeDecoder:
    """Greedy path matches plain decode; sampling is optional."""

    # Field order of the QWEN36_SPEC_TIMING lines.
    _TPHASES = ("draft", "verify", "readback", "accept", "commit", "reseed", "other", "total")
    # Class-level call id so warmup and timed generate() logs stay distinct.
    _gen_calls = 0

    def __init__(self, model, page_table_torch, draft_len=None, stop_tokens=None, sampling=None):
        assert model.mtp is not None, "model has no MTP head (has_mtp / mtp.* weights?)"
        assert model.num_devices > 1, "SpeculativeDecoder is TP-only for now"
        self.model = model
        self.mesh = model.mesh_device
        self.args = model.args
        self.vocab = model.args.vocab_size
        self.page_table = page_table_torch  # torch [1, num_blocks]
        self.K = int(draft_len if draft_len is not None else os.environ.get("QWEN36_SPEC_DRAFT_LEN", 3))
        # QWEN36_SPEC_PROFILE=1: per-phase wall-clock.
        self._prof = bool(int(os.environ.get("QWEN36_SPEC_PROFILE", "0")))
        self._pt = {"draft": 0.0, "verify": 0.0, "commit": 0.0, "reseed": 0.0}
        # QWEN36_SPEC_TIMING=1: one log line per iteration.
        self._timing = bool(int(os.environ.get("QWEN36_SPEC_TIMING", "0")))
        self._tsum = {}  # phase -> summed seconds, warmup iterations excluded
        self._tn = 0  # iterations folded into _tsum
        self.stop_tokens = set(stop_tokens or [])
        self.mtp = model.mtp
        # V3 feeds final-norm output; V0 (QWEN36_SPEC_POSTNORM=0) feeds the pre-norm residual.
        self.spec_postnorm = bool(getattr(model, "spec_postnorm", False))
        assert self.spec_postnorm == bool(getattr(model.mtp, "spec_postnorm", False)), (
            f"spec feed contract mismatch: model spec_postnorm={self.spec_postnorm} but "
            f"mtp spec_postnorm={getattr(model.mtp, 'spec_postnorm', False)}"
        )
        # Flat python ints; model.py's trace capture still takes the torch page table.
        self._pt_row = [int(b) for b in page_table_torch.reshape(-1).tolist()]
        # MTP's own page table.
        self.mtp_pt = ttnn.Tensor(self._pt_row, [1, len(self._pt_row)], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.mesh)
        self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        assert model.gdn_fused_decode, (
            "SpeculativeDecoder needs model.set_gdn_fused_decode(True) before construction: the "
            "traced verify advances GDN with the fused recurrent op and decode must use the same math"
        )
        self._vfy_captured = False
        self._setup_traced = False  # commit/reseed/draft captures: once per decoder, not per generate
        # QWEN36_SPEC_TRACED_COMMIT=0: eager per-layer commit instead of one trace per prefix index.
        self.traced_commit = os.environ.get("QWEN36_SPEC_TRACED_COMMIT", "1") != "0"
        # QWEN36_SPEC_TRACED_DRAFT=0: eager per-leg draft chain.
        self.traced_draft = os.environ.get("QWEN36_SPEC_TRACED_DRAFT", "1") != "0"
        # QWEN36_SPEC_TRACED_RESEED=0: eager batched-reseed forward.
        self.traced_reseed = os.environ.get("QWEN36_SPEC_TRACED_RESEED", "1") != "0"
        self._reseed_traced = False
        self._draft_traced = False  # resolved in generate(): did the capture actually take?
        self._commit_traced = False  # resolved in generate(): did the capture actually take?
        # Persistent anchor-hidden buffer, allocated before any trace capture (see _anchor_warmup).
        self._hp_buf = None
        # None => greedy argmax-prefix; else host rejection sampling.
        self.sampler = SpecSampler(sampling, self.vocab) if sampling is not None else None
        # Greedy uses on-device argmax; sampling needs the verify logits.
        self.read_verify_logits = self.sampler is not None
        # QWEN36_SPEC_DEVICE_TOPK=1: device top-k. Off by default; per-iteration pad wedges live traces.
        self._logits_topk = (
            sampling.top_k
            if (
                sampling is not None
                and sampling.top_k > 0
                and sampling.presence_penalty <= 0
                and os.environ.get("QWEN36_SPEC_DEVICE_TOPK", "0") == "1"
            )
            else 0
        )
        # Mean target probability of the drafts the sampler evaluated (sampling path only).
        self._p_draft_sum = 0.0
        self._p_draft_n = 0
        # None: generate() picks batched vs eager from the prompt length.
        self.force_eager_reseed = None
        self._batched_reseed = True  # resolved in generate()
        # Scratch block is the MTP cache's extra block past the page table, not a sequence block.
        self._reseed_scratch_block = None  # filled in generate(), once the KV caches exist
        self._reseed_block_size = 0  # filled in generate(), once the KV caches exist
        self.total_drafted = 0
        self.total_accepted = 0  # accepted DRAFT tokens (excludes the mandatory correction/bonus)
        self.iters = 0
        self.accept_hist = [0] * (self.K + 1)  # how often exactly j drafts were accepted
        self.depth_hits = [0] * self.K  # depth_hits[j] = iterations that accepted draft j
        self.zero_accept = 0  # no draft accepted; the pending token is still committed
        self.mtp_extra_steps = 0  # drafter forwards spent on KV maintenance (reseed)
        self.prefill_time = 0.0  # set by generate(): TTFT (prefill + MTP warm + seed)
        self.decode_time = 0.0  # set by generate(): spec-loop wall-clock (excludes prefill)

    def _draft(self, pending_tok, anchor_hidden, p):
        """Draft K tokens on device; step j proposes position p+2+j from (hidden, token at slot+1)."""
        tok_tt = ttnn.Tensor([int(pending_tok)], [1, 1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh)
        if self._draft_traced:
            # One execute_trace per leg; positions for the window are staged once.
            self.mtp.stage_draft_window(p, self.K, rope_delta=self.model.rope.rope_delta)
            self.mtp.seed_draft_window(tok_tt, anchor_hidden)
            ttnn.deallocate(tok_tt)
            idxs = [self.mtp.draft_leg(k) for k in range(self.K)]
            return self._ids_to_host(idxs)  # ONE sync for all K ids

        owned_tok = [tok_tt]
        h = anchor_hidden
        for k in range(self.K):
            logits, h_next = self.model.ttnn_mtp_decode_forward(h, tok_tt, p + k, self.mtp_pt)
            idx = self._argmax_last(logits)  # [1,1,1] uint32 ROW_MAJOR
            ttnn.deallocate(logits)
            tok_tt = ttnn.reshape(idx, (1, 1))
            owned_tok.append(tok_tt)
            if h is not anchor_hidden:
                ttnn.deallocate(h)
            h = h_next
        if h is not anchor_hidden:
            ttnn.deallocate(h)
        drafts = self._ids_to_host(owned_tok[1:])  # ONE sync for all K ids
        for t in owned_tok:
            ttnn.deallocate(t)
        return drafts

    def _draft_warmup(self, pending, Hp, p):
        """Compile one logits-producing draft step before trace capture; its KV write matches draft step 0."""
        logits, h = self.model.ttnn_mtp_decode_forward(Hp, int(pending), p, self.mtp_pt)
        idx = self._argmax_last(logits)
        for t in (logits, idx, h):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)

    def _argmax_last(self, logits):
        """Device greedy pick for one row; same reduction as the traced chain."""
        return self.mtp._argmax_last(logits)

    def _ids_to_host(self, id_tts):
        """Read K device ids to host in one transfer, preserving order."""
        if not id_tts:
            return []
        if len(id_tts) == 1:
            return [self._id_to_host(id_tts[0])]
        cat = ttnn.concat([ttnn.reshape(t, (1, 1)) for t in id_tts], dim=0)  # [K, 1]
        flat = ttnn.get_device_tensors(cat)[0].to_list()
        ttnn.deallocate(cat)
        out = []

        def _walk(v):
            if isinstance(v, list):
                for e in v:
                    _walk(e)
            else:
                out.append(int(v))

        _walk(flat)
        assert len(out) == len(id_tts), f"batched id readback got {len(out)} ids, wanted {len(id_tts)}"
        return out

    def _id_to_host(self, id_tt):
        """Read one uint32 id from the device-0 replica; logits are replicated across the mesh."""
        flat = ttnn.get_device_tensors(id_tt)[0].to_list()
        while isinstance(flat, list):
            flat = flat[0]
        return int(flat)

    def _warm_mtp_chunk(self, hidden, chunk_start, valid_len, prompt_ids):
        """Warm MTP KV over one prompt chunk; the forward runs the full tile-aligned bucket, not only valid rows."""
        T = len(prompt_ids)
        if chunk_start >= T - 1:
            return
        bucket = hidden.shape[-2]
        # Slot i is fused with the token at i+1 (shift pairing); 0-pad past the prompt.
        n = min(bucket, T - 1 - chunk_start)
        toks = [int(t) for t in prompt_ids[chunk_start + 1 : chunk_start + 1 + n]] + [0] * (bucket - n)
        self.model.ttnn_mtp_prefill_forward(hidden, toks, chunk_start, self.page_table)

    def _warm_mtp_last(self, last_hidden, first_tok, slot):
        """Write slot T-1 MTP KV once the base's first predicted token exists."""
        _, h_next = self.model.ttnn_mtp_decode_forward(
            last_hidden, int(first_tok), slot, self.mtp_pt, need_logits=False
        )
        ttnn.deallocate(h_next)

    def _reseed_mtp(self, slot0, vhidden, tokens):
        """Rewrite committed slots' MTP KV from base hidden, one decode step per slot."""
        for i, tok in enumerate(tokens):
            row = ttnn.slice(vhidden, (0, 0, i, 0), (1, 1, i + 1, vhidden.shape[-1]))
            _, h_next = self.model.ttnn_mtp_decode_forward(row, int(tok), slot0 + i, self.mtp_pt, need_logits=False)
            ttnn.deallocate(row)
            ttnn.deallocate(h_next)
            self.mtp_extra_steps += 1

    def _reseed_mtp_batched(self, slot0, vhidden, tokens, scratch_only=False):
        """Reseed all verify rows in one forward; padding rows write a scratch block, not the sequence."""
        m = 0 if scratch_only else len(tokens)
        if m == 0 and not scratch_only:
            return
        T = vhidden.shape[-2]
        assert m <= T, f"reseed {m} slots into a {T}-row batch"
        mesh = self.mesh
        nb = len(self._pt_row)
        rm = ttnn.ROW_MAJOR_LAYOUT
        # Real rows alias the sequence; padding rows use the scratch block at position 0.
        tok = [int(t) for t in tokens[:m]] + [0] * (T - m)
        pos = list(range(slot0, slot0 + m)) + [0] * (T - m)
        pt = self._pt_row * m + [self._reseed_scratch_block] * (nb * (T - m))
        # cos/sin are gathered ON DEVICE off the resident rope table (no host trig, no upload).
        cos, sin = self.model._rope_tp_cos_sin_decode_rows(pos)
        if self._reseed_traced:
            self.mtp.stage_reseed_window(tok, pos, pt, cos, sin)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            self.mtp.reseed_replay()
            self.mtp_extra_steps += 1
            return
        tok_tt = ttnn.Tensor(tok, [T, 1], ttnn.uint32, rm, mesh)
        pos_tt = ttnn.Tensor(pos, [T], ttnn.int32, rm, mesh)
        pt_tt = ttnn.Tensor(pt, [T, nb], ttnn.int32, rm, mesh)
        _, h_next = self.mtp.forward_decode(
            vhidden, tok_tt, pos_tt, cos, sin, pt_tt, need_logits=False, alias_kv_write=True
        )
        for t in (tok_tt, pos_tt, pt_tt, cos, sin, h_next):
            ttnn.deallocate(t)
        self.mtp_extra_steps += 1

    def _reseed_warmup(self, T, dim_frac, dtype):
        """Compile the batched reseed before verify-trace capture; every row targets the scratch block."""
        z = ttnn.zeros(
            [1, 1, T, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # All rows are padding; the scratch block is past the page table, so no position reaches it.
        self._reseed_mtp_batched(0, z, [], scratch_only=True)
        self.mtp_extra_steps -= 1  # warmup is not a loop cost
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(z)

    def _accept_greedy(self, drafts, verify_ids):
        """Accept the matching draft-id prefix; verify_ids[j] is the base argmax at drafts[j]'s position."""
        K = len(drafts)
        m = K
        for j in range(K):
            if drafts[j] != verify_ids[j]:
                m = j
                break
        self.accept_hist[m] += 1
        for j in range(m):
            self.depth_hits[j] += 1
        if m == 0:
            self.zero_accept += 1
        return m

    def _accept_sample(self, drafts, vlogits):
        """Rejection-sample the accepted prefix; returns (m, next_token) for the next pending."""
        assert vlogits is not None, "sampling acceptance needs read_verify_logits; got None"
        # _verify hands back a device-selected (idx, vals) support when _logits_topk is on.
        if self._logits_topk:
            m, next_tok, p_draft = self.sampler.accept_support(vlogits, drafts)
        else:
            m, next_tok, p_draft = self.sampler.accept(vlogits, drafts)
        self.accept_hist[m] += 1
        for j in range(m):
            self.depth_hits[j] += 1
        if m == 0:
            self.zero_accept += 1
        self._p_draft_sum += sum(p_draft)
        self._p_draft_n += len(p_draft)
        return m, next_tok

    def _verify(self, tokens, p):
        """Replay the verify trace; returns (argmax ids, hidden rows, logits or None)."""
        lt, vhidden, ids = self.model.verify_traced(
            tokens,
            p + 1,
            read_logits=self.read_verify_logits,
            clone_rows=False,
            page_table=self._pt_row,
            logits_topk=self._logits_topk,
        )
        return ids, vhidden, lt

    def _commit(self, mi):
        """Point durable GDN state at accepted-prefix slot mi; mi == K is a host-only early-out."""
        if not self._commit_traced:
            for dn in self._gdn:
                dn.commit_verify_slot(mi)
            return
        self.model.replay_commit_trace(mi)
        for dn in self._gdn:
            dn.commit_verify_slot_host(mi)

    def _anchor_warmup(self, T, dim_frac, dtype):
        """Allocate the fixed-address anchor buffer and compile every refill slice before trace capture."""
        self._hp_buf = ttnn.zeros(
            [1, 1, 1, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.zeros(
            [1, 1, T, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for mi in range(T):
            self._set_anchor(z, mi)
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(z)

    def _set_anchor(self, vhidden, mi):
        """Refill the persistent anchor-hidden buffer from row `mi` of the verify window."""
        row = ttnn.slice(vhidden, (0, 0, mi, 0), (1, 1, mi + 1, vhidden.shape[-1]))
        ttnn.copy(row, self._hp_buf)
        ttnn.deallocate(row)

    def _seed(self, first, p):
        """Consume the prompt's first predicted token; returns (next id, persistent anchor hidden)."""
        tok, chidden = self.model.verify_forward([first], p + 1, self.page_table, gdn_recurrent=True, argmax=True)
        self._anchor_warmup(self.K + 1, chidden.shape[-1], chidden.dtype)
        self._set_anchor(chidden, 0)
        ttnn.deallocate(chidden)
        return tok, self._hp_buf

    def _phase(self, name, fn):
        """Run fn, accumulating its synchronize-bracketed wall time under `name` when profiling."""
        if not self._prof:
            return fn()
        ttnn.synchronize_device(self.mesh)
        t = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(self.mesh)
        self._pt[name] += time.perf_counter() - t
        return r

    def _tick(self):
        """Fence the device, then take a host timestamp."""
        ttnn.synchronize_device(self.mesh)
        return time.perf_counter()

    def _verify_split(self, tokens, p):
        """Split verify device time from the host readback; returns (ids, hidden, logits, device_s, readback_s)."""
        orig = ttnn.get_device_tensors
        mark = []

        def hooked(*a, **kw):
            if not mark:
                mark.append(time.perf_counter())
            return orig(*a, **kw)

        ttnn.get_device_tensors = hooked
        t0 = time.perf_counter()
        try:
            vids, vhidden, vlt = self._verify(tokens, p)
        finally:
            ttnn.get_device_tensors = orig
        ttnn.synchronize_device(self.mesh)
        t1 = time.perf_counter()
        t_mark = mark[0] if mark else t1
        return vids, vhidden, vlt, t_mark - t0, t1 - t_mark

    def _log_iter_timing(self, row):
        """Log one iteration and fold it into the mean, excluding the first two iterations."""
        row["other"] = row["total"] - sum(v for k, v in row.items() if k != "total")
        logger.info(
            f"[SPEC_TIMING] iter={self.iters} "
            + " ".join(f"{k}={row[k] * 1e3:.2f}" for k in self._TPHASES)
            + f" gen={self._gen_id}"
        )
        if self.iters >= 2:  # skip the 2 warmup iterations
            for k, v in row.items():
                self._tsum[k] = self._tsum.get(k, 0.0) + v
            self._tn += 1

    def _log_mean_timing(self):
        if not self._timing or not self._tn:
            return
        mean = {k: self._tsum.get(k, 0.0) / self._tn for k in self._TPHASES}
        cpi = self.accept_rate() + 1.0
        logger.info(
            f"[SPEC_TIMING] MEAN gen={self._gen_id} iters={self._tn} (excl 2 warmup) "
            + " ".join(f"{k}={mean[k] * 1e3:.2f}" for k in self._TPHASES)
            + f" committed_per_iter={cpi:.3f} tok_s={cpi / max(mean['total'], 1e-9):.2f}"
        )

    def log_profile(self, prefix="spec", tokens=None):
        if not self._prof:
            return
        total = sum(self._pt.values())
        logger.info(f"[{prefix}] phase profile ({total:.2f}s over {self.iters} iters):")
        for k, v in sorted(self._pt.items(), key=lambda kv: -kv[1]):
            per_tok = f", {v / tokens * 1e3:.1f} ms/tok" if tokens else ""
            logger.info(f"[{prefix}]   {k:8s}: {v:.3f}s ({v / max(total, 1e-9):.0%}){per_tok}")

    def generate(self, prompt_ids, max_new_tokens):
        """Returns generated token ids and records prefill_time and decode_time."""
        model = self.model
        T = len(prompt_ids)
        SpeculativeDecoder._gen_calls += 1
        self._gen_id = SpeculativeDecoder._gen_calls
        _t_start = time.perf_counter()

        # Past EAGER_RESEED_PROMPT_LEN, batched reseed's bf16 near-ties lose acceptance, so use the per-slot loop.
        eager_reseed = T > EAGER_RESEED_PROMPT_LEN if self.force_eager_reseed is None else self.force_eager_reseed
        self._batched_reseed = not eager_reseed
        logger.info(
            f"[spec] gen={self._gen_id} T={T} K={self.K} reseed={'eager' if eager_reseed else 'batched'} "
            f"feed={'V3' if self.spec_postnorm else 'V0'} max_new={max_new_tokens}"
        )

        # High-water includes verify's K+1 slots and the capture warmup, for both reseed modes.
        self._reseed_block_size = int(self.mtp.attention.paged_k.shape[-2])
        nb = self.page_table.shape[-1]
        _hi = T + self.K + max(1, max_new_tokens - 1)
        assert _hi < nb * self._reseed_block_size, (
            f"high-water slot {_hi} (T={T}, max_new={max_new_tokens}, K={self.K}) does not fit the "
            f"paged KV: {nb} blocks x {self._reseed_block_size} = {nb * self._reseed_block_size} slots"
        )

        # Chunked prefill; each chunk warms MTP KV before its hidden is freed.
        prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
        last_hidden = [None]  # the base hidden at slot T-1, kept for _warm_mtp_last

        def _on_chunk(hidden, chunk_start, valid_len):
            # Warm and the T-1 row must come from the same spec_feed tensor.
            feed = self.model.spec_feed_rows(hidden)
            self._warm_mtp_chunk(feed, chunk_start, valid_len, prompt_ids)
            if chunk_start + valid_len >= T:  # the chunk holding slot T-1
                i = T - 1 - chunk_start
                row = ttnn.slice(feed, (0, 0, i, 0), (1, 1, i + 1, feed.shape[-1]))
                last_hidden[0] = ttnn.clone(row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(row)
            if feed is not hidden:
                ttnn.deallocate(feed)

        logits_dev = model.prefill_for_spec(prompt, self.page_table, T, _on_chunk)
        first = model._argmax_device(logits_dev)
        out = [first]

        # Slot T-1 pairs the base hidden at T-1 with `first`, which only exists now.
        assert last_hidden[0] is not None, "prefill_for_spec never delivered the chunk holding T-1"
        self._warm_mtp_last(last_hidden[0], first, T - 1)
        ttnn.deallocate(last_hidden[0])

        # Seed must run before capture_verify_trace: compiling after capture clobbers the parked trace.
        for dn in self._gdn:
            dn._capture_slots = False  # the eager seed must not write the trace's slot buffers
        Lp, Hp = self._seed(first, T - 1)

        # Compile the B=K+1 reseed before any trace is parked.
        if self._batched_reseed:
            # Scratch block index comes from the cache length, not the page-table width.
            self._reseed_scratch_block = int(self.mtp.attention.paged_k.shape[0]) - 1
            assert max(self._pt_row) < self._reseed_scratch_block, (
                f"page table names block {max(self._pt_row)}, but the MTP cache's scratch block is "
                f"{self._reseed_scratch_block}; the cache needs one block past every block the table uses"
            )
            self._reseed_warmup(self.K + 1, Hp.shape[-1], Hp.dtype)

        p = T
        # Base's own next token: committed unconditionally and seeds the drafter.
        pending = Lp
        # Compile the logits drafter path before capture; its KV write is what draft step 0 repeats.
        self._draft_warmup(pending, Hp, p)

        # Compile every draft leg before the first trace capture; throwaway KV is rewritten by the real chain.
        if self.traced_draft:
            self.mtp.init_draft_window(self.K, Hp.shape[-1], Hp.dtype, self.mtp_pt)
            self.mtp.stage_draft_window(p, self.K, rope_delta=model.rope.rope_delta)
            self.mtp.compile_draft_window()
        if self._batched_reseed and self.traced_reseed:
            # Allocate now; compile against the verify rows buffer after capture_verify_trace.
            self.mtp.init_reseed_window(self.K + 1, self.page_table.shape[-1])

        # Capture once, after every pre-capture program has compiled; throwaway KV sits past the seed slot.
        if not self._vfy_captured:
            model.capture_verify_trace(
                self.page_table, self.K + 1, warm_start=T + 1, decode_cfg=True, commit_warmup=self.traced_commit
            )
            self._vfy_captured = True
        # Commit traces (mi in 0..K-1) once, after verify capture; programs were warmed before any capture.
        if not self._setup_traced:
            self._commit_traced = bool(self.traced_commit and model.capture_commit_traces())
        # Capture the draft window after the commit traces; capturing it earlier gets reclaimed.
        if (
            not self._setup_traced
            and self._batched_reseed
            and self.traced_reseed
            and getattr(model, "_vfy_rows_out", None) is not None
        ):
            # Padding rows target the scratch block; capture reads the verify trace's persistent rows buffer.
            _T = self.K + 1
            _zeros = [0] * _T
            _cos, _sin = self.model._rope_tp_cos_sin_decode_rows(_zeros)
            self.mtp.stage_reseed_window(
                _zeros,
                _zeros,
                [self._reseed_scratch_block] * (_T * len(self._pt_row)),
                _cos,
                _sin,
            )
            ttnn.deallocate(_cos)
            ttnn.deallocate(_sin)
            self.mtp.compile_reseed_window(model._vfy_rows_out)
            self.mtp.release_reseed_window()
            self.mtp.capture_reseed_window(model._vfy_rows_out)
            self._reseed_traced = True
        if not self._setup_traced and self.traced_draft:
            self.mtp.release_draft_window()
            self.mtp.capture_draft_window()
            self._draft_traced = True
        self._setup_traced = True
        logger.info(
            f"[spec] commit={'traced' if self._commit_traced else 'eager'} "
            f"draft={'traced' if self._draft_traced else 'eager'}"
        )
        ttnn.synchronize_device(self.mesh)
        self.prefill_time = time.perf_counter() - _t_start  # TTFT: prefill + MTP warm + seed
        _t_decode = time.perf_counter()

        # Prefill can emit a stop token directly (`first`); the loop must not extend past it.
        _prefill_stop = first in self.stop_tokens
        while not _prefill_stop and len(out) < max_new_tokens:
            _tm = self._tick() if self._timing else 0.0
            drafts = self._phase("draft", lambda: self._draft(pending, Hp, p))
            _t_draft = self._tick() if self._timing else 0.0

            # committed = [pending] + drafts[:m]; commit selects the accepted GDN slot.
            if self._timing:
                vids, vhidden, vlt, _s_verify, _s_read = self._verify_split([pending] + drafts, p)
                _t_verify = time.perf_counter()
            else:
                vids, vhidden, vlt = self._phase("verify", lambda: self._verify([pending] + drafts, p))
            # Sampling returns next_token; greedy reads it from vids[mi].
            sampled_next = None
            if self.sampler is not None:
                m, sampled_next = self._accept_sample(drafts, vlt)
            else:
                m = self._accept_greedy(drafts, vids)
            committed = [pending] + drafts[:m]
            # Stop emission at the first stop token; acceptance stats keep the full m.
            _stop_i = next((i for i, t in enumerate(committed) if t in self.stop_tokens), None)
            if _stop_i is not None:
                committed = committed[: _stop_i + 1]
            mi = len(committed) - 1  # accepted-prefix's last token index in the verify window
            _t_accept = time.perf_counter() if self._timing else 0.0  # host-only: no fence needed
            self._phase("commit", lambda: self._commit(mi))
            _t_commit = self._tick() if self._timing else 0.0
            prev_p = p
            # The next anchor's own next token: the base's argmax at the accepted-prefix's last row.
            next_pending = vids[mi] if sampled_next is None else sampled_next
            # Refill the same persistent anchor buffer; a fresh clone aliases commit-trace addresses.
            self._set_anchor(vhidden, mi)

            # Reseed committed slots from verify rows; vhidden is the trace buffer and is not freed.
            _rfn = self._reseed_mtp_batched if self._batched_reseed else self._reseed_mtp
            self._phase("reseed", lambda: _rfn(prev_p + 1, vhidden, committed[1:]))
            _t_reseed = self._tick() if self._timing else 0.0

            p += len(committed)
            pending = next_pending

            out.extend(committed)
            if self._timing:
                # `other` is anchor refill, deallocates, and the host pick of the next pending.
                self._log_iter_timing(
                    {
                        "draft": _t_draft - _tm,
                        "verify": _s_verify,
                        "readback": _s_read,
                        "accept": _t_accept - _t_verify,
                        "commit": _t_commit - _t_accept,
                        "reseed": _t_reseed - _t_commit,
                        "total": self._tick() - _tm,  # fenced, so nothing leaks into the next iter
                    }
                )
            self.iters += 1
            self.total_drafted += len(drafts)
            self.total_accepted += m
            assert p == prev_p + len(committed)

            if committed[-1] in self.stop_tokens:
                break

        ttnn.deallocate(self._hp_buf)  # == Hp; the persistent anchor buffer, one per generate
        self._hp_buf = None
        ttnn.synchronize_device(self.mesh)
        self.decode_time = time.perf_counter() - _t_decode  # spec loop wall-clock (excludes prefill)
        # Verify advances only the conv window; sync the per-tap buffers before anything else reads them.
        for dn in self._gdn:
            dn.sync_conv_taps()
        self._log_mean_timing()
        self.log_profile(tokens=len(out[:max_new_tokens]))
        return out[:max_new_tokens]

    def accept_rate(self):
        """Mean accepted DRAFT tokens per iteration (0..K); tokens/iter is this + 1."""
        return self.total_accepted / max(1, self.iters)

    def stats(self):
        n = max(1, self.iters)
        return {
            "iters": self.iters,
            "K": self.K,
            "accept_rate": self.accept_rate(),
            "committed_per_iter": self.accept_rate() + 1.0,
            # depth_rate[j] = P(draft j accepted), cumulative: [0] >= [1] >= ...
            "depth_rate": [h / n for h in self.depth_hits],
            # conditional[j] = P(draft j accepted | drafts 0..j-1 accepted)
            "conditional": [
                self.depth_hits[j] / max(1, self.depth_hits[j - 1] if j else self.iters) for j in range(self.K)
            ],
            "hist": list(self.accept_hist),
            "zero_accept_rate": self.zero_accept / n,
            "mtp_extra_steps": self.mtp_extra_steps,
        }

    def log_stats(self, prefix="spec"):
        s = self.stats()
        logger.info(
            f"[{prefix}] {s['iters']} iters, K={s['K']}: accept={s['accept_rate']:.2f}/{s['K']} "
            f"-> {s['committed_per_iter']:.2f} committed tokens/iter"
        )
        logger.info(f"[{prefix}] per-depth accept   : {[f'{x:.2f}' for x in s['depth_rate']]}")
        logger.info(f"[{prefix}] conditional accept : {[f'{x:.2f}' for x in s['conditional']]}")
        logger.info(f"[{prefix}] accepted histogram : {s['hist']} (index j = exactly j drafts accepted)")
        logger.info(f"[{prefix}] zero-accept iters  : {s['zero_accept_rate']:.1%} (still commit the pending token)")
        logger.info(f"[{prefix}] MTP reseed forwards: {s['mtp_extra_steps']}")
        return s
