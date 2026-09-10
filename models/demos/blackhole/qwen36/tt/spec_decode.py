# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Qwen3.6-27B using the built-in MTP drafter head.

The MTP head (models/demos/blackhole/qwen36/tt/mtp.py) drafts K tokens autoregressively; the base
model verifies them in one traced K+1-token chunk forward; accepted tokens are committed by pointing
the GDN recurrent state at the accepted slot.

Iteration shape (anchor p; the base has consumed through p and we hold its hidden h_p):

    pending = argmax(base logits at p)      # the base's OWN next token: free, already confirmed
    draft   : slot p+j fuses (hidden, token at slot+1) -> candidate for p+2+j, for j in 0..K-1
    verify  : ONE chunk over [pending, d_0..d_K-1] at positions p+1..p+K+1
    accept  : d_j's target is argmax(verify logits at p+1+j); commit [pending] + matching prefix

Two measured properties fix the shape of that loop:

* The head expects the DeepSeek-V3 / vLLM pairing (h_i, token_{i+1}) -> token_{i+2}, not the
  same-index pairing (h_i, token_i): 0.78 vs 0.64 depth-1 top-1 on the real weights, and e2e
  acceptance 2.82/3 vs 1.52/3.
* Because `pending` is known before drafting, all K drafter steps propose new tokens. Spending the
  first step re-predicting `pending` aborted 60% of iterations.

The base model's Gated DeltaNet (GDN) recurrent state is the hard part: a speculative verify advances
it through candidate positions, so a rejected draft must not leave it there. Verify buffers the state
after every token and commit points the durable state at the accepted-prefix slot — the reference's
`spec_state_indices` / `num_accepted_tokens` scheme, with no rollback and no commit forward. The
full-attention layers' paged KV is corrected implicitly (rejected positions past the frontier are
never attended and get overwritten next iteration).

TP (P150x4) only, B=1, greedy (temperature==0).
"""

import os
import time

import torch
from loguru import logger

import ttnn

# Prompt length above which the reseed goes back to the per-slot loop; see generate().
EAGER_RESEED_PROMPT_LEN = 131072


class SpeculativeDecoder:
    """Greedy MTP speculative decode. Reproduces the plain-decode greedy trajectory exactly.

    This is the demo's DEFAULT decode path (QWEN36_SPEC=0 opts out). Everything that is not a genuine
    tuning knob is fixed to its measured-best setting: the only env flags left are
    QWEN36_SPEC_DRAFT_LEN (K override) and QWEN36_SPEC_TIMING / QWEN36_SPEC_PROFILE (diagnostics).
    The A/B switches that used to be env vars are instance attributes instead — set them from python
    (force_eager_reseed, read_verify_logits here; use_fullbatch_verify on the GDN layers).
    """

    # Field order of the QWEN36_SPEC_TIMING lines.
    _TPHASES = ("draft", "verify", "readback", "accept", "commit", "reseed", "other", "total")
    # The demo runs a throwaway warmup generate() on a separate instance before the timed one, so
    # the timing lines are tagged with a class-level call id to tell the two apart in the log.
    _gen_calls = 0

    def __init__(self, model, page_table_torch, draft_len=None, stop_tokens=None):
        assert model.mtp is not None, "model has no MTP head (has_mtp / mtp.* weights?)"
        assert model.num_devices > 1, "SpeculativeDecoder is TP-only for now"
        self.model = model
        self.mesh = model.mesh_device
        self.args = model.args
        self.vocab = model.args.vocab_size
        self.page_table = page_table_torch  # torch [1, num_blocks]
        # K=3 is the conservative library default, kept for callers that pass no draft_len (the
        # correctness tests). The fully-batched GDN verify made verify cost ~flat in K (2.3 ms per
        # candidate — see use_fullbatch_verify in gdn/tp.py), so the demo passes the ISL-aware policy
        # instead: K=10 up to a 4k prompt, K=6 above it (_run_tp_spec_generation). QWEN36_SPEC_DRAFT_LEN
        # overrides both.
        self.K = int(draft_len if draft_len is not None else os.environ.get("QWEN36_SPEC_DRAFT_LEN", 3))
        # QWEN36_SPEC_PROFILE=1: per-phase wall-clock (synchronize-bracketed) to see where time goes.
        self._prof = bool(int(os.environ.get("QWEN36_SPEC_PROFILE", "0")))
        self._pt = {"draft": 0.0, "verify": 0.0, "commit": 0.0, "reseed": 0.0}
        # QWEN36_SPEC_TIMING=1: per-ITERATION breakdown (one log line per iteration + a mean at the
        # end). Off by default and every cost sits behind `self._timing`.
        self._timing = bool(int(os.environ.get("QWEN36_SPEC_TIMING", "0")))
        self._tsum = {}  # phase -> summed seconds, warmup iterations excluded
        self._tn = 0  # iterations folded into _tsum
        self.stop_tokens = set(stop_tokens or [])
        self.mtp = model.mtp
        # Feed contract (Qwen36Model.spec_feed_rows; env QWEN36_SPEC_POSTNORM). V3 (default) feeds
        # the drafter the final-norm output and chains mtp.norm's output; V0 (QWEN36_SPEC_POSTNORM=0)
        # feeds the pre-final-norm residual and chains the raw MTP block output. Both halves are
        # decided by the model at build time;
        # the drafter must agree or it would fuse mixed-scale hiddens across a chain.
        self.spec_postnorm = bool(getattr(model, "spec_postnorm", False))
        assert self.spec_postnorm == bool(getattr(model.mtp, "spec_postnorm", False)), (
            f"spec feed contract mismatch: model spec_postnorm={self.spec_postnorm} but "
            f"mtp spec_postnorm={getattr(model.mtp, 'spec_postnorm', False)}"
        )
        # The caller's page table as a flat python list, so every page table this class builds from
        # it is plain ints (the torch tensor itself is still what model.py's trace capture takes).
        self._pt_row = [int(b) for b in page_table_torch.reshape(-1).tolist()]
        # The MTP layer keeps its own paged KV cache with its own (identity) page table.
        self.mtp_pt = ttnn.Tensor(self._pt_row, [1, len(self._pt_row)], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, self.mesh)
        self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        for gdn in self._gdn:
            # verify and decode must share GDN math or near-tie argmax flips reduce acceptance
            gdn.use_fused_recurrent_decode = True
        self._vfy_captured = False
        # QWEN36_SPEC_TRACED_COMMIT=0 reverts the commit phase to the eager per-layer
        # commit_verify_slot loop; the default replays one pre-captured trace per accepted-prefix
        # index instead. Measured A/B at ISL 16k, K=7 (traced_16k demo case): commit 4.87 -> 3.21
        # ms/iteration, total 88.32 -> 86.55, 38.65 -> 39.44 tok/s, with acceptance byte-identical
        # (2.41/7 both ways) — the trace does the same copies, it just stops paying host dispatch
        # for 4 ops x 48 layers. The remaining 3.2 ms is the replay's own on-device dispatch of
        # those ~192 programs, which is why this is worth ~2% and not the whole 4.9 ms.
        # Kept as an env flag purely so the two can be A/B'd on the same build.
        self.traced_commit = os.environ.get("QWEN36_SPEC_TRACED_COMMIT", "1") != "0"
        # QWEN36_SPEC_TRACED_DRAFT=0 reverts the draft chain to the eager per-leg loop. On by
        # default: MEASURED on T3K/27B, an eager chain is ~97% host dispatch (K=6: enqueue 113 ms,
        # device drain 2.8 ms), so the traced chain is where the draft phase's cost goes.
        self.traced_draft = os.environ.get("QWEN36_SPEC_TRACED_DRAFT", "1") != "0"
        # QWEN36_SPEC_TRACED_RESEED=0 reverts the batched reseed to its eager forward. Same
        # dispatch-bound shape as the draft chain: MEASURED ~35 ms host enqueue vs 0.8 ms device.
        self.traced_reseed = os.environ.get("QWEN36_SPEC_TRACED_RESEED", "1") != "0"
        self._reseed_traced = False
        self._draft_traced = False  # resolved in generate(): did the capture actually take?
        self._commit_traced = False  # resolved in generate(): did the capture actually take?
        # Persistent anchor-hidden buffer, allocated before any trace capture (see _anchor_warmup).
        self._hp_buf = None
        # Read the full [T, vocab] verify logits back to host as well as the argmax ids. Greedy
        # acceptance does not need them (the trace argmaxes on device), so this is off; the device
        # path stays in model.verify_traced for whatever needs distributions (sampling, debug).
        self.read_verify_logits = False
        # None => generate() picks batched or eager from the prompt length (see the note there).
        # Set True/False to force one, for A/B work.
        self.force_eager_reseed = None
        self._batched_reseed = True  # resolved in generate()
        # The batched reseed's scratch block (its padding rows' KV sink) is the EXTRA block the MTP
        # cache carries past the page table's span — _allocate_mtp_kv_cache allocates num_blocks + 1.
        # It must not be one of the sequence's own blocks: stealing the last one caps the sequence at
        # (nb - 1) * block_size, which the 256k demo case overruns by 36 tokens. The page table stays
        # nb wide and identity, so nothing but the pad rows below ever names this block.
        self._reseed_scratch_block = len(self._pt_row)
        self._reseed_block_size = 0  # filled in generate(), once the KV caches exist
        self.total_drafted = 0
        self.total_accepted = 0  # accepted DRAFT tokens (excludes the mandatory correction/bonus)
        self.iters = 0
        # --- instrumentation (mean acceptance alone hides where the drafts die) ---
        self.accept_hist = [0] * (self.K + 1)  # how often exactly j drafts were accepted
        self.depth_hits = [0] * self.K  # depth_hits[j] = iterations that accepted draft j
        self.zero_accept = 0  # iterations where no draft was accepted (still commit the pending
        # token, so they are worth 1 token, not 0)
        self.mtp_extra_steps = 0  # drafter forwards spent on KV maintenance (reseed)
        self.prefill_time = 0.0  # set by generate(): TTFT (prefill + MTP warm + seed)
        self.decode_time = 0.0  # set by generate(): spec-loop wall-clock (excludes prefill)

    # --------------------------------------------------------------------- #
    # Draft
    # --------------------------------------------------------------------- #
    def _draft(self, pending_tok, anchor_hidden, p):
        """Autoregressively draft K tokens from the MTP head, starting at slot ``p``.

        The head is fused from (base hidden at slot s, embedding of the token at s+1) and predicts
        the token at s+2 — the DeepSeek-V3 / vLLM convention. So:

          step 0: (h_p, pending_tok) at slot p     -> candidate for position p+2
          step j: (own hidden, previous draft)     -> candidate for position p+2+j

        ``pending_tok`` is the base's OWN next token at p+1, already confirmed by the anchor logits.
        Feeding it here is what makes all K drafts genuinely new.

        The chain stays ON DEVICE: reading back the full 151k-vocab logits and argmaxing on host for
        every step forces a host round-trip between steps (~12 ms/step against ~3 ms of device work).
        Device argmax feeds the next step directly and defers the readback to K small ids at the end.

        Returns the K drafted ids; drafts[j] is the candidate for position p+2+j.
        """
        tok_tt = ttnn.Tensor([int(pending_tok)], [1, 1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh)
        if self._draft_traced:
            # TRACED chain: one execute_trace per leg, nothing per-leg on the host. See
            # Qwen36MTP.init_draft_window for why (an eager leg is ~19 ms of host against 0.5 ms of
            # device). Positions for the whole window go up in one staging call.
            self.mtp.stage_draft_window(p, self.K, rope_delta=self.model.rope.rope_delta)
            self.mtp.seed_draft_window(tok_tt, anchor_hidden)
            ttnn.deallocate(tok_tt)
            idxs = [self.mtp.draft_leg(k) for k in range(self.K)]
            return [self._id_to_host(i) for i in idxs]  # one sync, K ids

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
        drafts = [self._id_to_host(t) for t in owned_tok[1:]]  # one sync, K ids
        for t in owned_tok:
            ttnn.deallocate(t)
        return drafts

    def _draft_warmup(self, pending, Hp, p):
        """Run ONE full (need_logits=True) draft step eagerly, BEFORE any trace is captured.

        The first real ``_draft`` happens after capture_verify_trace, and the logits-producing drafter
        path has programs nothing earlier in generate() has run at B=1: head_norm in DECODE mode
        (gather-then-norm), the fp32 LM head and its vocab all-gather, the multicore untilize and the
        argmax — and, under V3, the mesh_partition that re-fractures mtp.norm's output. A program that
        first compiles while a trace is parked lands its kernel binaries in memory the replayed trace
        writes over (see capture_verify_trace), so they have to compile here. Runs under BOTH feed
        contracts.

        Side effect: writes the drafter's KV at slot ``p`` from (H_p, pending) — exactly the write
        draft step 0 repeats, from the same inputs, on the first real iteration — so it is inert. The
        drafted id is discarded.
        """
        logits, h = self.model.ttnn_mtp_decode_forward(Hp, int(pending), p, self.mtp_pt)
        idx = self._argmax_last(logits)
        for t in (logits, idx, h):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)

    def _argmax_last(self, logits):
        """argmax over the vocab dim for ONE row -> [1,1,1] uint32 ROW_MAJOR.

        ttnn.argmax needs ROW_MAJOR input: a TILE tensor takes a single-core internal-untilize path
        that is catastrophically slow on a 151k-wide vocab. So untilize multicore, then argmax.

        This used to pad 1 -> 32 rows first, on the belief that the multicore argmax is row-parallel
        and returns garbage below a full tile of rows. It does not: measured on the real drafter, the
        unpadded argmax returns byte-identical ids (K=3 lossless acceptance 2.57 either way) and the
        draft phase drops 35.1 -> 22.3 ms at K=10. The pad was pure traffic — [1,1,1,vocab] is
        ALREADY 32 rows physically, so padding it to 32 logical rows made untilize and argmax move
        ~32x the bytes they need, so the pad is gone.
        """
        u = ttnn.untilize(logits, use_multicore=True)
        out = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,1] uint32 RM
        ttnn.deallocate(u)
        return out

    def _id_to_host(self, id_tt):
        """[*,1] uint32 device id -> python int. Reads only the device-0 replica: the logits are
        replicated across the TP mesh, so a ConcatMeshToTensor would move 4x the bytes for nothing."""
        flat = ttnn.get_device_tensors(id_tt)[0].to_list()
        while isinstance(flat, list):
            flat = flat[0]
        return int(flat)

    # --------------------------------------------------------------------- #
    # MTP KV maintenance
    # --------------------------------------------------------------------- #
    def _warm_mtp_chunk(self, hidden, chunk_start, valid_len, prompt_ids):
        """Warm the MTP drafter's KV over ONE prompt chunk, in one forward.

        The drafter must see prompt context: with an empty cache at the first draft, acceptance
        collapses (2.82/3 -> 1.10/3 measured). Slot i is fused from (base_hidden_i, token_{i+1}) —
        the same shift pairing the draft loop uses.

        Slot T-1 is deliberately NOT written here: its token is the base's own prediction for
        position T, which is not known until the prefill logits exist. ``_warm_mtp_last`` writes it
        afterwards.

        The forward runs over the WHOLE bucket, not just the valid rows: the bucket is tile-aligned
        (128..2048) and the prefill matmuls require that, whereas an arbitrary valid_len (5, 2047)
        fails the matmul shape check. Rows past the prompt write junk MTP KV at slots >= T-1, which
        is harmless — slot T-1 is overwritten by _warm_mtp_last, and every slot above it is
        rewritten by the drafter before it is ever attended (draft step k writes slot p+k, then
        attends to slots <= p+k).
        """
        T = len(prompt_ids)
        if chunk_start >= T - 1:
            return
        bucket = hidden.shape[-2]
        # Slot i is fused with the token at i+1 (shift pairing); 0-pad past the prompt.
        n = min(bucket, T - 1 - chunk_start)
        toks = [int(t) for t in prompt_ids[chunk_start + 1 : chunk_start + 1 + n]] + [0] * (bucket - n)
        self.model.ttnn_mtp_prefill_forward(hidden, toks, chunk_start, self.page_table)

    def _warm_mtp_last(self, last_hidden, first_tok, slot):
        """Write the final prompt slot's MTP KV, whose token is the base's own first prediction.

        Load-bearing: the first draft happens at slot T and attends to slots <= T-1, so leaving T-1
        unwritten hands it stale KV. One decode step.
        """
        _, h_next = self.model.ttnn_mtp_decode_forward(
            last_hidden, int(first_tok), slot, self.mtp_pt, need_logits=False
        )
        ttnn.deallocate(h_next)

    def _reseed_mtp(self, slot0, vhidden, tokens):
        """Refresh the MTP KV of the committed slots with the BASE hidden, replacing the drafter's
        own chained hidden.

        The drafter wrote those slots from its own chained hidden while drafting; the prompt warming
        wrote base hiddens. Leaving the mismatch in place costs acceptance (2.00 -> 2.82 /3 measured),
        so every committed slot is rewritten from the base hidden the verify forward already produced.
        ``vhidden`` row i is the base hidden at slot0+i; tokens[i] is the token at slot0+i+1.

        One drafter DECODE step per slot. Superseded by _reseed_mtp_batched (one forward over all
        slots at once, ~11 ms/iteration cheaper at K=10) except past EAGER_RESEED_PROMPT_LEN prompt
        tokens, where generate() comes back here — see its reseed note.

        Batching it does NOT go through the drafter's prefill path — that needs a genuine prefill
        shape and neither candidate width works at an arbitrary mid-sequence slot0 (at one tile the
        stack silently picks DECODE matmuls while the norms stay in PREFILL mode, so the fused fc is
        handed a fractured activation; at 128 rows the SDPA rejects the unaligned chunk start). It
        goes through the DECODE path at B rows instead — see _reseed_mtp_batched.
        """
        for i, tok in enumerate(tokens):
            row = ttnn.slice(vhidden, (0, 0, i, 0), (1, 1, i + 1, vhidden.shape[-1]))
            _, h_next = self.model.ttnn_mtp_decode_forward(row, int(tok), slot0 + i, self.mtp_pt, need_logits=False)
            ttnn.deallocate(row)
            ttnn.deallocate(h_next)
            self.mtp_extra_steps += 1

    def _reseed_mtp_batched(self, slot0, vhidden, tokens, scratch_only=False):
        """``_reseed_mtp`` as ONE fixed-shape drafter forward over all T = K+1 verify rows.

        The per-slot loop above is m sequential decode forwards (~2.7 ms each, m ~ 6 at K=10). The
        rows are INDEPENDENT — only the KV write survives, and K/V come from the row's own
        (hidden, token, position) through the in-projection, never from the attention output — so
        running them as B=T pseudo-users of one sequence is exactly equivalent and costs one forward.

        This is the same hybrid trick the verify uses: per-row position tensor, page-table rows
        aliasing the sequence's blocks, alias_kv_write=True so the shared-block KV writes go row by
        row instead of racing several cores onto one 32-row tile.

        Rows m..T-1 are PADDING (m varies, the shape must not): their page-table rows point at a
        dedicated scratch block, so their KV write lands there and never touches the sequence, and
        their position is 0 so their (discarded) SDPA read is one slot deep instead of full-context.
        ``vhidden`` supplies their hidden rows unchanged — junk in, junk to the scratch block.

        ``scratch_only=True`` makes EVERY row padding (the warmup below): same shapes, same program,
        no real slot touched.
        """
        m = 0 if scratch_only else len(tokens)
        if m == 0 and not scratch_only:
            return
        T = vhidden.shape[-2]
        assert m <= T, f"reseed {m} slots into a {T}-row batch"
        mesh = self.mesh
        nb = len(self._pt_row)
        rm = ttnn.ROW_MAJOR_LAYOUT
        # Flat row-major lists: the real rows alias the sequence's own blocks, the padding rows aim
        # their whole page-table row at the scratch block and sit at position 0.
        tok = [int(t) for t in tokens[:m]] + [0] * (T - m)
        pos = list(range(slot0, slot0 + m)) + [0] * (T - m)
        pt = self._pt_row * m + [self._reseed_scratch_block] * (nb * (T - m))
        # cos/sin are gathered ON DEVICE off the resident rope table (no host trig, no upload).
        cos, sin = self.model._rope_tp_cos_sin_decode_rows(pos)
        if self._reseed_traced:
            # Same inputs, one execute_trace instead of ~35 ms of host enqueue.
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
        """Compile the batched-reseed program BEFORE the verify trace is captured.

        The batched reseed is a shape the loop has never run (B=T decode over the MTP layer), and a
        program that first compiles while the verify trace is parked lands its kernel binaries in
        memory the replayed trace writes over. Every row here targets the scratch block, so the
        throwaway forward touches no real KV.
        """
        z = ttnn.zeros(
            [1, 1, T, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # scratch_only: all T rows are padding, so they name the scratch block directly. (The scratch
        # block sits PAST the page table's span, so there is no position that reaches it by lookup.)
        self._reseed_mtp_batched(0, z, [], scratch_only=True)
        self.mtp_extra_steps -= 1  # warmup is not a loop cost
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(z)

    # --------------------------------------------------------------------- #
    # Accept (greedy)
    # --------------------------------------------------------------------- #
    def _accept_greedy(self, drafts, verify_ids):
        """Greedy acceptance of the matching prefix; returns the number of accepted drafts.

        The verify chunk ran p+1..p+K+1, so verify_ids[j] is the base model's own argmax at p+1+j,
        which predicts p+2+j — exactly drafts[j]'s position. No draft's target was known before
        drafting (that token is ``pending``, committed unconditionally), so every rejection is a
        genuine drafter miss and nothing extra needs committing: the correction arrives as the next
        iteration's ``pending``.

        Greedy compares IDS, so the verify trace argmaxes on device and this walks a [K+1] int list
        rather than a [K+1, 151936] host float tensor.
        """
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

    # --------------------------------------------------------------------- #
    # Verify / commit
    # --------------------------------------------------------------------- #
    def _verify(self, tokens, p):
        """Replay the captured verify trace over `tokens` = [pending] + drafts at positions p+1...

        Advances GDN recurrently token by token (the SAME kernel decode uses, so it is
        recurrent-faithful) while attention/MLP/norm/lm_head stay batched over the bucket. Buffers
        the per-token GDN state so commit_verify_slot(m) can roll the durable state to the accepted
        slot — no rollback, no commit forward.

        Returns (per-position argmax ids, per-position hidden rows [1,1,len,dim/tp]).
        """
        _lt, vhidden, ids = self.model.verify_traced(
            tokens, p + 1, read_logits=self.read_verify_logits, clone_rows=False
        )
        return ids, vhidden

    def _commit(self, mi):
        """Point the durable GDN state at the accepted prefix's last verify slot `mi`.

        Traced path (default): ONE execute_trace carries all 48 layers' commit device ops for this
        mi, and python only does the per-layer host bookkeeping (staleness marks + dropping the
        verify handle). Eager path: the original per-layer commit_verify_slot loop, unchanged.

        mi == K is FULL acceptance: the verify already left rec_state at the last token and
        _conv_win_buf at the last window, so there is nothing to copy and no trace was captured for
        it — replay_commit_trace returns False and only the host half runs. Same early-out the eager
        path takes, so the two agree token for token.
        """
        if not self._commit_traced:
            for dn in self._gdn:
                dn.commit_verify_slot(mi)
            return
        self.model.replay_commit_trace(mi)
        for dn in self._gdn:
            dn.commit_verify_slot_host(mi)

    def _anchor_warmup(self, T, dim_frac, dtype):
        """Allocate the persistent anchor-hidden buffer and compile every op the loop uses to refill
        it — BEFORE any trace is captured.

        Two reasons this has to happen here rather than lazily in the loop.

        1. ADDRESS STATIONARITY. The anchor hidden used to be a fresh ttnn.clone per iteration, and
           it is LIVE across the commit phase. A commit trace bakes its intermediates' addresses in
           at capture time, when no such per-iteration buffer exists, so the loop's clone could land
           on one of them and the replay would overwrite the anchor mid-flight — the drafter then
           chains from corrupted hidden and acceptance collapses (measured: 2.36 -> 0.09 on the third
           generate of test_spec_determinism). One fixed-address buffer, allocated before the
           captures, removes the whole class: at commit-replay time nothing but persistent buffers
           is allocated, exactly as at capture time.
        2. NO POST-CAPTURE COMPILE. The refill slices row `mi` of the verify window, and
           SliceDeviceOperation::compute_program_hash folds the slice offsets in, so every mi is its
           own program. Compiling one while a trace is parked writes kernel binaries over it.
        """
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
        """Consume the prompt's first predicted token at position p -> (next token id, hidden).

        One eager recurrent verify forward. Runs once per request, so it is not on the hot path.
        The greedy pick runs on device (argmax=True), so only the winning index comes back rather
        than the vocab row. The hidden is the persistent anchor buffer, not a fresh clone.
        """
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

    # --------------------------------------------------------------------- #
    # Per-iteration timing (QWEN36_SPEC_TIMING=1)
    # --------------------------------------------------------------------- #
    def _tick(self):
        """Fence the device, then take a host timestamp.

        Dispatch is async: without the fence a host timestamp bounds only the ENQUEUE of a phase, so
        every phase that does not itself read back to host would measure ~0 and the phase after it
        would absorb the device time.
        """
        ttnn.synchronize_device(self.mesh)
        return time.perf_counter()

    def _verify_split(self, tokens, p):
        """``_verify``, with the device->host readback split out of the device time.

        ``model.verify_traced`` runs execute_trace + synchronize_device and only THEN pulls the ids
        (and, when read_verify_logits is set, the logits) back via
        ``ttnn.to_torch(ttnn.get_device_tensors(...)[0])``, so hooking ``ttnn.get_device_tensors``
        for the duration of the call marks the device/host boundary without editing model.py. The
        trailing sync flushes the small hidden-rows clone that follows the readback, which lands in
        the readback bucket.

        Returns (vids, vhidden, device_seconds, readback_seconds).
        """
        orig = ttnn.get_device_tensors
        mark = []

        def hooked(*a, **kw):
            if not mark:
                mark.append(time.perf_counter())
            return orig(*a, **kw)

        ttnn.get_device_tensors = hooked
        t0 = time.perf_counter()
        try:
            vids, vhidden = self._verify(tokens, p)
        finally:
            ttnn.get_device_tensors = orig
        ttnn.synchronize_device(self.mesh)
        t1 = time.perf_counter()
        t_mark = mark[0] if mark else t1
        return vids, vhidden, t_mark - t0, t1 - t_mark

    def _log_iter_timing(self, row):
        """Log one iteration's breakdown and fold it into the mean (first 2 iterations excluded)."""
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
        """Log the mean-over-iterations breakdown gathered under QWEN36_SPEC_TIMING=1."""
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
        """Log the per-phase wall-clock breakdown gathered when QWEN36_SPEC_PROFILE=1."""
        if not self._prof:
            return
        total = sum(self._pt.values())
        logger.info(f"[{prefix}] phase profile ({total:.2f}s over {self.iters} iters):")
        for k, v in sorted(self._pt.items(), key=lambda kv: -kv[1]):
            per_tok = f", {v / tokens * 1e3:.1f} ms/tok" if tokens else ""
            logger.info(f"[{prefix}]   {k:8s}: {v:.3f}s ({v / max(total, 1e-9):.0%}){per_tok}")

    # --------------------------------------------------------------------- #
    # Generate (greedy)
    # --------------------------------------------------------------------- #
    def generate(self, prompt_ids, max_new_tokens):
        """Greedy speculative generation. Returns the list of generated token ids (excludes prompt).

        Records self.prefill_time (prompt prefill + MTP warm + seed, i.e. TTFT) and self.decode_time
        (the spec loop), both synchronize-bracketed, so callers can report ttft / decode tok/s.
        """
        model = self.model
        T = len(prompt_ids)
        SpeculativeDecoder._gen_calls += 1
        self._gen_id = SpeculativeDecoder._gen_calls
        _t_start = time.perf_counter()

        # Reseed shape, from the prompt length. Past EAGER_RESEED_PROMPT_LEN the batched reseed's B=K+1
        # in-projection drifts enough bf16 near-ties to cost ~0.3 accepted drafts/iter (256k: 19.7 vs
        # 20.9 tok/s), while its dispatch saving (~2 ms/iter) no longer covers that; the per-slot loop
        # keeps spec >= plain at every ISL. force_eager_reseed overrides for A/B work.
        eager_reseed = T > EAGER_RESEED_PROMPT_LEN if self.force_eager_reseed is None else self.force_eager_reseed
        self._batched_reseed = not eager_reseed
        logger.info(
            f"[spec] gen={self._gen_id} T={T} K={self.K} reseed={'eager' if eager_reseed else 'batched'} "
            f"feed={'V3' if self.spec_postnorm else 'V0'} max_new={max_new_tokens}"
        )

        # Chunked prompt prefill (2048-token chunks + masked tail — the same path the demo uses, so
        # long prompts work). Each chunk's hidden warms the MTP drafter's KV in ONE forward before it
        # is freed, so the drafter never sees an empty cache and TTFT stays flat in prompt length.
        prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
        last_hidden = [None]  # the base hidden at slot T-1, kept for _warm_mtp_last

        def _on_chunk(hidden, chunk_start, valid_len):
            # Drafter feed for the chunk (V0: `hidden` itself; V3: a new fractured post-norm tensor).
            # Both the warm and the slot-T-1 row must come from the SAME tensor, so the drafter is
            # never handed a mix of contracts. The caller still frees `hidden`.
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
        # Greedy pick on device (ttnn.argmax over the replicated row); only the index crosses the bus.
        first = model._argmax_device(logits_dev)
        out = [first]

        # Slot T-1 pairs the base hidden at T-1 with `first`, which only exists now.
        assert last_hidden[0] is not None, "prefill_for_spec never delivered the chunk holding T-1"
        self._warm_mtp_last(last_hidden[0], first, T - 1)
        ttnn.deallocate(last_hidden[0])

        # Seed: consume `first` at position T -> (L_T, H_T); anchor p=T.
        # MUST stay BEFORE capture_verify_trace. The seed is an EAGER verify at T=1, and under the
        # full-batch GDN verify none of its programs (conv1d over K-1+1 rows, the [1,1,*] slice /
        # reshape chain) appear in the T=K+1 trace, so running it after capture compiles them while
        # the trace is parked: their kernel-binary buffers land in memory the replayed trace writes
        # over, and the NEXT generate's seed (program-cache hit) dispatches corrupted binaries and
        # hangs the device. Every program the post-capture path needs must be compiled before capture.
        for dn in self._gdn:
            dn._capture_slots = False  # the eager seed must not write the trace's slot buffers
        Lp, Hp = self._seed(first, T - 1)

        # Batched-reseed warmup: compiles the B=K+1 drafter forward while nothing is traced yet.
        if self._batched_reseed:
            self._reseed_block_size = self.mtp.attention.paged_k.shape[-2]
            nb = self.page_table.shape[-1]
            # The reseed scratch is its own block PAST the page table (the MTP cache is nb + 1
            # blocks), so the sequence gets the whole page table: it only has to fit in nb blocks,
            # the same bound the base KV cache imposes.
            assert T + max_new_tokens <= nb * self._reseed_block_size, (
                f"sequence does not fit the paged KV: {nb} blocks x {self._reseed_block_size} "
                f"cannot hold {T} prompt + {max_new_tokens} generated tokens"
            )
            assert self.mtp.attention.paged_k.shape[0] > self._reseed_scratch_block, (
                f"MTP KV cache has {self.mtp.attention.paged_k.shape[0]} blocks; the batched reseed's "
                f"scratch block {self._reseed_scratch_block} needs one more than the page table's {nb}"
            )
            self._reseed_warmup(self.K + 1, Hp.shape[-1], Hp.dtype)

        p = T
        # The base's own next token, confirmed by the anchor logits. It is committed unconditionally
        # next iteration and is what seeds the drafter, so no drafter step re-predicts it.
        pending = Lp  # _seed already argmaxed on device
        # Draft warmup: the logits-producing drafter step (head norm, fp32 LM head, untilize, argmax
        # — plus V3's mesh_partition) has not run yet, and its first real run is AFTER the capture
        # below. Compile it now; its one KV write (slot p from (H_p, pending)) is what draft step 0
        # repeats.
        self._draft_warmup(pending, Hp, p)

        # Draft-window COMPILE. Every leg's programs must exist before the first begin_trace_capture
        # below -- a program that first compiles with a trace parked writes its kernel binaries over
        # that trace. Its K throwaway legs write junk drafter KV at exactly the slots [p, p+K) that
        # the first real chain rewrites in order, so it is inert (same argument as _draft_warmup).
        if self.traced_draft:
            self.mtp.init_draft_window(self.K, Hp.shape[-1], Hp.dtype, self.mtp_pt)
            self.mtp.stage_draft_window(p, self.K, rope_delta=model.rope.rope_delta)
            self.mtp.compile_draft_window()
        if self._batched_reseed and self.traced_reseed:
            # Compiled here (pre-capture) but bound to the verify trace's persistent rows buffer,
            # which only exists after capture_verify_trace -- so the compile pass runs just after it,
            # below, and only the CAPTURE waits until every other trace is parked.
            self.mtp.init_reseed_window(self.K + 1, self.page_table.shape[-1])

        # One-time verify-trace capture (replayed every iteration), done AFTER prefill + MTP warm +
        # seed so every program those paths need is already compiled: a compile that happens once the
        # trace is parked clobbers it. Its two throwaway passes write junk KV at [T+1, T+1+K] — past
        # the seed's slot at T, and overwritten by the first real verify — and restore the GDN state
        # they advance. Counts toward TTFT, not decode_time.
        if not self._vfy_captured:
            model.capture_verify_trace(
                self.page_table, self.K + 1, warm_start=T + 1, decode_cfg=True, commit_warmup=self.traced_commit
            )
            self._vfy_captured = True
        # Commit traces, one per accepted-prefix index mi in 0..K-1 (mi == K is full acceptance,
        # which commit_verify_slot early-outs to nothing). MUST come after the verify capture: the
        # commit ops read the per-token state buffer that capture allocates. Their programs were
        # compiled by capture_verify_trace's commit_warmup, before ANY begin_trace_capture, so
        # nothing compiles here with the verify trace parked.
        self._commit_traced = bool(self.traced_commit and model.capture_commit_traces())
        # Draft-window CAPTURE, last and EVERY generate. Two reasons it must be re-captured rather
        # than captured once: capture_commit_traces() above releases and re-captures its own traces
        # on every call, which reclaims trace memory the drafter traces would be sitting in, and the
        # drafter traces must stay the last thing captured. Captured once in gen=1 and left alone,
        # they come back corrupted in gen=2 -- observed as legs 3..5 replaying legs 0..2's ids
        # ([4752, 16666, 321, 4752, 16666, 321]) and acceptance falling 4.00 -> 2.25 of 6.
        # Nothing RECOMPILES here (compile_draft_window early-outs on its `compiled` flag), so this
        # cannot clobber the verify or commit traces it follows.
        if self._batched_reseed and self.traced_reseed and getattr(model, "_vfy_rows_out", None) is not None:
            # Aim every padding row at the scratch block so the compile pass's throwaway KV write
            # cannot touch the sequence, then compile + capture against the verify trace's own
            # persistent rows buffer (a fixed address the replay re-reads each iteration).
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
        if self.traced_draft:
            self.mtp.release_draft_window()
            self.mtp.capture_draft_window()
            self._draft_traced = True
        logger.info(
            f"[spec] commit={'traced' if self._commit_traced else 'eager'} "
            f"draft={'traced' if self._draft_traced else 'eager'}"
        )
        ttnn.synchronize_device(self.mesh)
        self.prefill_time = time.perf_counter() - _t_start  # TTFT: prefill + MTP warm + seed
        _t_decode = time.perf_counter()

        while len(out) < max_new_tokens:
            _tm = self._tick() if self._timing else 0.0
            drafts = self._phase("draft", lambda: self._draft(pending, Hp, p))
            _t_draft = self._tick() if self._timing else 0.0

            # Verify buffers per-token GDN state and keeps the hidden; commit = select the accepted
            # slot (no re-run forward). committed = [pending] + drafts[:m].
            if self._timing:
                vids, vhidden, _s_verify, _s_read = self._verify_split([pending] + drafts, p)
                _t_verify = time.perf_counter()
            else:
                vids, vhidden = self._phase("verify", lambda: self._verify([pending] + drafts, p))
            m = self._accept_greedy(drafts, vids)
            committed = [pending] + drafts[:m]
            mi = len(committed) - 1  # accepted-prefix's last token index in the verify window
            _t_accept = time.perf_counter() if self._timing else 0.0  # host-only: no fence needed
            self._phase("commit", lambda: self._commit(mi))
            _t_commit = self._tick() if self._timing else 0.0
            prev_p = p
            # The next anchor's own next token: the base's argmax at the accepted-prefix's last row.
            next_pending = vids[mi]
            # The new anchor hidden is the accepted prefix's last row of the verify window, refilled
            # into the SAME persistent buffer the drafter already read this iteration (see
            # _anchor_warmup: a fresh per-iteration clone here is what the commit traces aliased).
            self._set_anchor(vhidden, mi)

            # MTP KV maintenance over the slots just committed, in ONE drafter forward over the
            # verify window (row i is the base hidden at slot prev_p+1+i, paired with the token at
            # slot+1). vhidden is the verify trace's own persistent output row buffer, so it is not
            # freed here — the next replay overwrites it in place.
            _rfn = self._reseed_mtp_batched if self._batched_reseed else self._reseed_mtp
            self._phase("reseed", lambda: _rfn(prev_p + 1, vhidden, committed[1:]))
            _t_reseed = self._tick() if self._timing else 0.0

            p += len(committed)
            pending = next_pending

            out.extend(committed)
            if self._timing:
                # `other` = the anchor-hidden slice + copy, the deallocates, and the host argmax
                # that picks the next `pending`.
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
        # The verify advances the GDN conv window only; bring the K per-tap buffers back in step so
        # whatever runs next on this model (an eager decode, a state snapshot) reads a live shift
        # register. One-off, and outside the timed loop.
        for dn in self._gdn:
            dn.sync_conv_taps()
        self._log_mean_timing()
        self.log_profile(tokens=len(out[:max_new_tokens]))
        return out[:max_new_tokens]

    def accept_rate(self):
        """Mean accepted DRAFT tokens per iteration (0..K); tokens/iter is this + 1."""
        return self.total_accepted / max(1, self.iters)

    def stats(self):
        """Acceptance breakdown. Mean acceptance alone cannot distinguish 'the drafter is weak at
        depth 3' from 'the first draft keeps aborting the iteration', which need different fixes."""
        n = max(1, self.iters)
        return {
            "iters": self.iters,
            "K": self.K,
            "accept_rate": self.accept_rate(),
            "committed_per_iter": self.accept_rate() + 1.0,
            # depth_rate[j] = P(draft j accepted), cumulative: [0] >= [1] >= ...
            "depth_rate": [h / n for h in self.depth_hits],
            # conditional[j] = P(draft j accepted | drafts 0..j-1 accepted) — isolates per-depth
            # drafter quality from the compounding of earlier rejections.
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
