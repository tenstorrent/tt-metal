# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Qwen3.6-27B using the built-in MTP drafter head.
MTP (tt/mtp.py) drafts K tokens; the base verifies them in one traced K+1-token chunk; accepted
tokens are committed by pointing GDN at the accepted slot. Iteration (anchor p, hidden h_p):
pending = argmax(base logits at p) (already confirmed); draft slot p+j fuses (hidden, token at
slot+1) -> candidate for p+2+j; verify [pending, d_0..d_K-1] at p+1..p+K+1; commit [pending] +
accepted prefix. Head expects DeepSeek-V3 / vLLM pairing (h_i, token_{i+1}) -> token_{i+2}, not
(h_i, token_i); pending is known before drafting, so all K steps propose new tokens. GDN: verify
buffers state after every token and commit points durable state at the accepted-prefix slot
(spec_state_indices / num_accepted_tokens — no rollback). Full-attention paged KV is corrected
implicitly (rejected positions past the frontier never attended, overwritten next iteration). TP (P150x4) only, B=1. Greedy accepts the longest matching-argmax prefix (token-identical to plain greedy). Sampling (temp > 0, optional top-k/top-p) runs exact speculative rejection sampling (tt/spec_sampling.py), lossless in distribution; drafts stay the device ARGMAX so the accept test is u < p(d). Extra sampling cost is the [T, vocab] logits readback plus host accept math."""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSampler, SpecSamplingParams

# Prompt length above which the reseed goes back to the per-slot loop; see generate().
EAGER_RESEED_PROMPT_LEN = 131072


class SpeculativeDecoder:
    """MTP speculative decode, greedy or sampling.

    Greedy (``sampling=None``) reproduces the plain-decode greedy trajectory exactly. A
    ``SpecSamplingParams`` (temperature > 0, optional top-k / top-p) switches acceptance to exact
    speculative rejection sampling over the verify logits, lossless in distribution, and turns
    ``read_verify_logits`` on (needs [T, vocab] target rows, not the trace's argmax ids). Drafts are
    the drafter's argmax under sampling as well. This is the demo's DEFAULT decode path
    (QWEN36_SPEC=0 opts out). Remaining knobs: QWEN36_SPEC_DRAFT_LEN (K override),
    QWEN36_SPEC_TIMING (per-iteration timing), and the ``sampling`` constructor argument.
    """

    # Field order of the QWEN36_SPEC_TIMING lines.
    _TPHASES = ("draft", "verify", "readback", "accept", "commit", "reseed", "other", "total")
    # The demo runs a throwaway warmup generate() on a separate instance before the timed one, so
    # the timing lines are tagged with a class-level call id to tell the two apart in the log.
    _gen_calls = 0

    def __init__(
        self,
        model,
        page_table_torch,
        draft_len=None,
        stop_tokens=None,
        sampling: SpecSamplingParams | None = None,
    ):
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
        # QWEN36_SPEC_TIMING=1: per-ITERATION breakdown (one log line per iteration + a mean at the
        # end). Off by default and every cost sits behind `self._timing`.
        self._timing = bool(int(os.environ.get("QWEN36_SPEC_TIMING", "0")))
        self._tsum = {}  # phase -> summed seconds, warmup iterations excluded
        self._tn = 0  # iterations folded into _tsum
        self.stop_tokens = set(stop_tokens or [])
        self.mtp = model.mtp
        # The MTP layer keeps its own paged KV cache with its own (identity) page table.
        self.mtp_pt = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
        )
        self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        for gdn in self._gdn:
            # verify and decode must share GDN math or near-tie argmax flips reduce acceptance
            gdn.use_fused_recurrent_decode = True
        self._vfy_captured = False
        # The commit phase replays one pre-captured trace per accepted-prefix index, falling back to
        # the eager per-layer commit_verify_slot loop automatically when the capture is unavailable.
        # Measured A/B at ISL 16k, K=7 (traced_16k demo case): commit 4.87 -> 3.21 ms/iteration,
        # total 88.32 -> 86.55, 38.65 -> 39.44 tok/s, with acceptance byte-identical (2.41/7 both
        # ways) — the trace does the same copies, it just stops paying host dispatch for 4 ops x 48
        # layers. The remaining 3.2 ms is the replay's own on-device dispatch of those ~192
        # programs, which is why this is worth ~2% and not the whole 4.9 ms.
        self._commit_traced = False  # resolved in generate(): did the capture actually take?
        # Persistent anchor-hidden buffer, allocated before any trace capture (see _anchor_warmup).
        self._hp_buf = None
        # Acceptance mode. None => greedy (argmax-prefix) acceptance; a SpecSamplingParams =>
        # exact speculative rejection sampling on host over the verify logits (tt/spec_sampling.py).
        self.sampler = SpecSampler(sampling, self.vocab) if sampling is not None else None
        # Read the full [T, vocab] verify logits back to host as well as the argmax ids. Greedy
        # acceptance does not need them (the trace argmaxes on device), so it stays off there; the
        # sampling accept step needs the distributions, so the constructor turns it on for it.
        self.read_verify_logits = self.sampler is not None
        # Mean target probability of the drafts the sampler actually evaluated: the sampling path's
        # analogue of per-depth acceptance (see stats()).
        self._p_draft_sum = 0.0
        self._p_draft_n = 0
        # Batched or eager reseed, decided in generate() from the prompt length (see the note there).
        self._batched_reseed = True
        # The batched reseed's scratch block (its padding rows' KV sink) is the EXTRA block the MTP
        # cache carries past the page table's span — _allocate_mtp_kv_cache allocates num_blocks + 1.
        # It must not be one of the sequence's own blocks: stealing the last one caps the sequence at
        # (nb - 1) * block_size, which the 256k demo case overruns by 36 tokens. The page table stays
        # nb wide and identity, so nothing but the pad rows below ever names this block.
        self._reseed_scratch_block = int(page_table_torch.shape[-1])
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
        the token at s+2 — DeepSeek-V3 / vLLM. Step 0: (h_p, pending_tok) at slot p -> candidate for
        p+2; step j: (own hidden, previous draft) -> candidate for p+2+j. ``pending_tok`` is the
        base's OWN next token at p+1, already confirmed; feeding it here makes all K drafts new.
        The chain stays ON DEVICE: host argmax of 151k-vocab logits between steps is a round-trip;
        device argmax feeds the next step and defers readback to K small ids at the end. Each step
        is an fp32 LM head, then untilize + ttnn.argmax (``_argmax_last``); under sampling that
        argmax is the deterministic proposal (the delta at that id). Returns the K drafted ids."""
        tok_tt = ttnn.from_torch(
            torch.tensor([[int(pending_tok)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
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
        # ONE sync for the whole chain: the first readback blocks, and everything below is already
        # computed — K small ids.
        drafts = [self._id_to_host(t) for t in owned_tok[1:]]
        for t in owned_tok:
            ttnn.deallocate(t)
        return drafts

    def _draft_warmup(self, pending, Hp, p):
        """Run ONE full (need_logits=True) draft step eagerly, BEFORE any trace is captured.

        The first real ``_draft`` happens after capture_verify_trace, and the logits-producing
        drafter path has programs nothing earlier in generate() has run at B=1: head_norm in DECODE
        (gather-then-norm), the LM head and its vocab all-gather, the argmax pick, and the
        mesh_partition that re-fractures mtp.norm's output for the next chain step. A program that
        first compiles while a trace is parked lands its kernel binaries in memory the replayed
        trace writes over, so they must compile here. Side effect: writes the drafter's KV at slot
        ``p`` from (H_p, pending) — the same write draft step 0 repeats on the first real iteration,
        so it is inert. The drafted id is discarded."""
        logits, h = self.model.ttnn_mtp_decode_forward(Hp, int(pending), p, self.mtp_pt)
        idx = self._argmax_last(logits)
        for t in (logits, idx, h):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)

    def _argmax_last(self, logits):
        """argmax over the vocab dim for ONE row -> [1,1,1] uint32 ROW_MAJOR.

        ttnn.argmax needs ROW_MAJOR input: a TILE tensor takes a single-core internal-untilize path
        that is catastrophically slow on a 151k-wide vocab. So untilize multicore, then argmax.
        Used to pad 1 -> 32 rows on the belief that multicore argmax is row-parallel and returns
        garbage below a full tile. It does not: the unpadded argmax returns byte-identical ids, and
        [1,1,1,vocab] is ALREADY 32 rows physically, so padding to 32 logical rows made untilize
        and argmax move ~32x the bytes they need. The pad is gone.
        """
        u = ttnn.untilize(logits, use_multicore=True)
        out = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,1] uint32 RM
        ttnn.deallocate(u)
        return out

    def _id_to_host(self, id_tt):
        """[*,1] uint32 device id -> python int. Reads only the device-0 replica: the logits are
        replicated across the TP mesh, so a ConcatMeshToTensor would move 4x the bytes for nothing."""
        t = ttnn.to_torch(ttnn.get_device_tensors(id_tt)[0])
        return int(t.reshape(-1)[0])

    # --------------------------------------------------------------------- #
    # MTP KV maintenance
    # --------------------------------------------------------------------- #
    def _warm_mtp_chunk(self, hidden, chunk_start, valid_len, prompt_ids):
        """Warm the MTP drafter's KV over ONE prompt chunk, in one forward.

        The drafter must see prompt context: with an empty cache at the first draft, acceptance
        collapses. Slot i is fused from (base_hidden_i, token_{i+1}) — the same shift pairing the
        draft loop uses. Slot T-1 is deliberately NOT written here: its token is the base's own
        prediction for position T, unknown until the prefill logits exist; ``_warm_mtp_last``
        writes it afterwards. The forward runs over the WHOLE bucket, not just the valid rows: the
        bucket is tile-aligned (128..2048) and prefill matmuls require that, whereas an arbitrary
        valid_len fails the matmul shape check. Rows past the prompt write junk MTP KV at slots
        >= T-1, which is harmless — slot T-1 is overwritten by _warm_mtp_last, and every slot above it is rewritten by the drafter before it is ever attended.
        """
        T = len(prompt_ids)
        if chunk_start >= T - 1:
            return
        bucket = hidden.shape[-2]
        # Slot i is fused with the token at i+1 (shift pairing); 0-pad past the prompt.
        toks = torch.zeros(1, bucket, dtype=torch.int32)
        n = min(bucket, T - 1 - chunk_start)
        toks[0, :n] = torch.tensor(
            [int(t) for t in prompt_ids[chunk_start + 1 : chunk_start + 1 + n]], dtype=torch.int32
        )
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
        own chained hidden. The drafter wrote those slots from its own chained hidden while drafting;
        the prompt warming wrote base hiddens. Leaving the mismatch in place costs acceptance, so
        every committed slot is rewritten from the base hidden the verify forward already produced.
        ``vhidden`` row i is the base hidden at slot0+i; tokens[i] is the token at slot0+i+1. One
        drafter DECODE step per slot. Superseded by _reseed_mtp_batched (one forward over all slots)
        except past EAGER_RESEED_PROMPT_LEN, where generate() comes back here. Batching does NOT go
        through the drafter's prefill path — that needs a genuine prefill shape and neither candidate
        width works at an arbitrary mid-sequence slot0 (at one tile the stack silently picks DECODE
        matmuls while norms stay PREFILL; at 128 rows SDPA rejects the unaligned chunk start). Goes through the DECODE path at B rows instead — see _reseed_mtp_batched.
        """
        for i, tok in enumerate(tokens):
            row = ttnn.slice(vhidden, (0, 0, i, 0), (1, 1, i + 1, vhidden.shape[-1]))
            _, h_next = self.model.ttnn_mtp_decode_forward(row, int(tok), slot0 + i, self.mtp_pt, need_logits=False)
            ttnn.deallocate(row)
            ttnn.deallocate(h_next)
            self.mtp_extra_steps += 1

    def _reseed_mtp_batched(self, slot0, vhidden, tokens, scratch_only=False):
        """``_reseed_mtp`` as ONE fixed-shape drafter forward over all T = K+1 verify rows.

        The per-slot loop is m sequential decode forwards. The rows are INDEPENDENT — only the KV
        write survives, and K/V come from the row's own (hidden, token, position) through the
        in-projection, never from the attention output — so running them as B=T pseudo-users of one
        sequence is exactly equivalent and costs one forward. Same hybrid trick as verify: per-row
        position tensor, page-table rows aliasing the sequence's blocks, alias_kv_write=True so
        shared-block KV writes go row by row instead of racing several cores onto one 32-row tile.
        Rows m..T-1 are PADDING (m varies, the shape must not): their page-table rows point at a
        dedicated scratch block (KV write never touches the sequence) and position 0 so the discarded SDPA read is one slot deep. ``scratch_only=True`` makes EVERY row padding (warmup): same shapes/program, no real slot touched.
        """
        m = 0 if scratch_only else len(tokens)
        if m == 0 and not scratch_only:
            return
        T = vhidden.shape[-2]
        assert m <= T, f"reseed {m} slots into a {T}-row batch"
        mesh, rep = self.mesh, ttnn.ReplicateTensorToMesh(self.mesh)
        tok = torch.zeros(T, 1, dtype=torch.int32)
        tok[:m, 0] = torch.tensor([int(t) for t in tokens[:m]], dtype=torch.int32)
        pos = torch.zeros(T, dtype=torch.int32)
        pos[:m] = torch.arange(slot0, slot0 + m, dtype=torch.int32)
        pt = self.page_table.repeat(T, 1).contiguous()
        pt[m:, :] = self._reseed_scratch_block
        cos_t, sin_t = self.model._rope_tp_cos_sin_decode_torch(pos)
        tok_tt = ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pos_tt = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pt_tt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
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
    # Accept
    # --------------------------------------------------------------------- #
    def _accept_greedy(self, drafts, verify_ids):
        """Greedy acceptance of the matching prefix; returns the number of accepted drafts.

        The verify chunk ran p+1..p+K+1, so verify_ids[j] is the base model's own argmax at p+1+j,
        which predicts p+2+j — exactly drafts[j]'s position. No draft's target was known before
        drafting (that token is ``pending``, committed unconditionally), so every rejection is a
        genuine drafter miss and nothing extra needs committing: the correction arrives as the next
        iteration's ``pending``. Greedy compares IDS, so the verify trace argmaxes on device and
        this walks a [K+1] int list rather than a [K+1, 151936] host float tensor.
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

    def _accept_sample(self, drafts, vlogits, penalize_base=None):
        """Exact speculative rejection sampling over the verify logits; returns (m, next_token).

        The drafts are the drafter's argmax, so the proposal is the delta at ``d_j`` and the accept
        test is ``u_j < p_j(d_j)``. ``next_token`` is the recovered token from the rejection row
        (m < K) or the bonus token from the extra row (m == K), and becomes the next iteration's
        ``pending`` exactly like greedy's ``verify_ids[mi]`` (see spec_sampling.py). ``penalize_base``
        is this iteration's presence-penalty set (``generated_so_far ∪ {pending}``, None when the
        request has no presence penalty); the sampler adds ``drafts[:j]`` per row, so each verify
        row is penalized on exactly the output that precedes it. Same instrumentation as
        _accept_greedy, plus the mean target probability of the drafts the sampler evaluated."""
        m, next_tok, p_draft = self.sampler.accept(vlogits, drafts, penalize_base)
        self.accept_hist[m] += 1
        for j in range(m):
            self.depth_hits[j] += 1
        if m == 0:
            self.zero_accept += 1
        self._p_draft_sum += sum(p_draft)
        self._p_draft_n += len(p_draft)
        return m, next_tok

    def _pick_token(self, logits_row, penalize=None):
        """One token from a host 1-D logits row: argmax (greedy) or a sampler draw.

        Used at the two SEED sites (the prefill's first token and the anchor's `pending`), where
        there is no draft to accept and the token comes straight out of one distribution. The row
        may be padded past the vocabulary (the LM head's width), so it is sliced first.

        ``penalize`` is the presence-penalty set for the site: nothing at all for the prefill's
        `first` token (the output is empty there) and ``{first}`` for the anchor's `pending`, which
        follows it. Unused on the greedy path, which has no sampler and no penalty.
        """
        row = logits_row.reshape(-1)[: self.vocab]
        if self.sampler is None:
            return int(row.float().argmax())
        return self.sampler.pick(row.float(), penalize)

    # --------------------------------------------------------------------- #
    # Verify / commit
    # --------------------------------------------------------------------- #
    def _verify(self, tokens, p):
        """Replay the captured verify trace over `tokens` = [pending] + drafts at positions p+1...

        Advances GDN recurrently token by token (the SAME kernel decode uses, so it is
        recurrent-faithful) while attention/MLP/norm/lm_head stay batched over the bucket. Buffers
        the per-token GDN state so commit_verify_slot(m) can roll the durable state to the accepted
        slot — no rollback, no commit forward. Returns (per-position argmax ids, per-position hidden
        rows [1,1,len,dim/tp], per-position host logits [T, vocab] — None unless read_verify_logits
        is set, i.e. unless the sampling accept step needs the distributions).
        """
        lt, vhidden, ids = self.model.verify_traced(
            tokens, p + 1, read_logits=self.read_verify_logits, clone_rows=False
        )
        return ids, vhidden, lt

    def _commit(self, mi):
        """Point the durable GDN state at the accepted prefix's last verify slot `mi`.

        Traced path (default): ONE execute_trace carries all 48 layers' commit device ops for this
        mi; python only does the per-layer host bookkeeping (staleness marks + dropping the verify
        handle). Eager path: the original per-layer commit_verify_slot loop, unchanged. mi == K is
        FULL acceptance: the verify already left rec_state at the last token and _conv_win_buf at
        the last window, so there is nothing to copy and no trace was captured for it —
        replay_commit_trace returns False and only the host half runs. Same early-out the eager
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
        it — BEFORE any trace is captured. Two reasons this cannot happen lazily in the loop.
        1. ADDRESS STATIONARITY. The anchor hidden used to be a fresh ttnn.clone per iteration, and
        it is LIVE across the commit phase. A commit trace bakes its intermediates' addresses in at
        capture time, when no such per-iteration buffer exists, so the loop's clone could land on
        one of them and the replay would overwrite the anchor mid-flight — the drafter then chains
        from corrupted hidden and acceptance collapses. One fixed-address buffer, allocated before
        the captures, removes the whole class. 2. NO POST-CAPTURE COMPILE. The refill slices row
        `mi` of the verify window, and SliceDeviceOperation hashes the slice offsets, so every mi
        is its own program. Compiling one while a trace is parked writes kernel binaries over it."""
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
        """Consume the prompt's first predicted token at position p -> (logits, hidden).

        One eager recurrent verify forward. Runs once per request, so it is not on the hot path.
        The hidden is the persistent anchor buffer (_anchor_warmup), not a fresh clone.
        """
        clogits, chidden = self.model.verify_forward([first], p + 1, self.page_table, gdn_recurrent=True)
        self._anchor_warmup(self.K + 1, chidden.shape[-1], chidden.dtype)
        self._set_anchor(chidden, 0)
        ttnn.deallocate(chidden)
        return clogits[0], self._hp_buf

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
        the readback bucket. Returns (vids, vhidden, vlogits, device_seconds, readback_seconds).
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
            vids, vhidden, vlogits = self._verify(tokens, p)
        finally:
            ttnn.get_device_tensors = orig
        ttnn.synchronize_device(self.mesh)
        t1 = time.perf_counter()
        t_mark = mark[0] if mark else t1
        return vids, vhidden, vlogits, t_mark - t0, t1 - t_mark

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

    # --------------------------------------------------------------------- #
    # Generate
    # --------------------------------------------------------------------- #
    def generate(self, prompt_ids, max_new_tokens):
        """Speculative generation (greedy or sampling, per the constructor's ``sampling`` arg).
        Returns the list of generated token ids (excludes prompt).

        Records self.prefill_time (prompt prefill + MTP warm + seed, i.e. TTFT) and self.decode_time
        (the spec loop), both synchronize-bracketed, so callers can report ttft / decode tok/s.
        """
        model = self.model
        T = len(prompt_ids)
        SpeculativeDecoder._gen_calls += 1
        self._gen_id = SpeculativeDecoder._gen_calls
        _t_start = time.perf_counter()
        # The GENERATED ids so far (prompt excluded), which is the set the presence penalty is
        # defined over. Maintained unconditionally — a set add per committed token — and read only
        # when the request actually carries a penalty.
        self._out_set = set()

        # Reseed shape, from the prompt length. Past EAGER_RESEED_PROMPT_LEN the batched reseed's B=K+1
        # in-projection drifts enough bf16 near-ties to cost ~0.3 accepted drafts/iter (256k: 19.7 vs
        # 20.9 tok/s), while its dispatch saving (~2 ms/iter) no longer covers that; the per-slot loop
        # keeps spec >= plain at every ISL.
        eager_reseed = T > EAGER_RESEED_PROMPT_LEN
        self._batched_reseed = not eager_reseed
        if self.sampler is None:
            _samp = "sampling=greedy"
        else:
            _sp = self.sampler.params
            _samp = (
                f"sampling=temp={_sp.temperature} top_k={_sp.top_k} top_p={_sp.top_p} "
                f"presence={_sp.presence_penalty} seed={self.sampler.seed}"
            )
        logger.info(
            f"[spec] gen={self._gen_id} T={T} K={self.K} reseed={'eager' if eager_reseed else 'batched'} "
            f"max_new={max_new_tokens} {_samp}"
        )

        # Chunked prompt prefill (2048-token chunks + masked tail — the same path the demo uses, so
        # long prompts work). Each chunk's hidden warms the MTP drafter's KV in ONE forward before it
        # is freed, so the drafter never sees an empty cache and TTFT stays flat in prompt length.
        prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
        last_hidden = [None]  # the base hidden at slot T-1, kept for _warm_mtp_last

        def _on_chunk(hidden, chunk_start, valid_len):
            # Drafter feed for the chunk (a new fractured post-norm tensor). Both the warm and the
            # slot-T-1 row must come from the SAME tensor, so the drafter is never handed a mix of
            # scales. The caller still frees `hidden`.
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
        lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        # penalize=None: the presence penalty looks at the OUTPUT only (prompt tokens excluded, as
        # in vLLM), and at the prefill pick the output is still empty.
        first = self._pick_token(lt.reshape(-1)[: self.vocab], None)
        out = [first]
        self._out_set.add(first)

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
        # The base's own next token, taken from the anchor logits (argmax, or a sampler draw). It is
        # committed unconditionally next iteration and is what seeds the drafter, so no drafter step
        # re-predicts it. This position follows `first`, so `first` is the whole output set the
        # presence penalty sees here.
        pending = self._pick_token(Lp, torch.tensor([first], dtype=torch.int64))
        # Draft warmup: the logits-producing drafter step (head norm, fp32 LM head, untilize, argmax,
        # plus the chain's mesh_partition) has not run yet, and its first real run is AFTER the
        # capture below. Compile it now; its one KV write (slot p from (H_p, pending)) is what draft
        # step 0 repeats.
        self._draft_warmup(pending, Hp, p)

        # One-time verify-trace capture (replayed every iteration), done AFTER prefill + MTP warm +
        # seed so every program those paths need is already compiled: a compile that happens once the
        # trace is parked clobbers it. Its two throwaway passes write junk KV at [T+1, T+1+K] — past
        # the seed's slot at T, and overwritten by the first real verify — and restore the GDN state
        # they advance. Counts toward TTFT, not decode_time.
        if not self._vfy_captured:
            model.capture_verify_trace(
                self.page_table, self.K + 1, warm_start=T + 1, decode_cfg=True, commit_warmup=True
            )
            self._vfy_captured = True
        # Commit traces, one per accepted-prefix index mi in 0..K-1 (mi == K is full acceptance,
        # which commit_verify_slot early-outs to nothing). MUST come after the verify capture: the
        # commit ops read the per-token state buffer that capture allocates. Their programs were
        # compiled by capture_verify_trace's commit_warmup, before ANY begin_trace_capture, so
        # nothing compiles here with the verify trace parked.
        self._commit_traced = bool(model.capture_commit_traces())
        logger.info(f"[spec] commit={'traced' if self._commit_traced else 'eager'}")
        ttnn.synchronize_device(self.mesh)
        self.prefill_time = time.perf_counter() - _t_start  # TTFT: prefill + MTP warm + seed
        _t_decode = time.perf_counter()

        while len(out) < max_new_tokens:
            _tm = self._tick() if self._timing else 0.0
            drafts = self._draft(pending, Hp, p)
            _t_draft = self._tick() if self._timing else 0.0

            # Verify buffers per-token GDN state and keeps the hidden; commit = select the accepted
            # slot (no re-run forward). committed = [pending] + drafts[:m].
            if self._timing:
                vids, vhidden, vlogits, _s_verify, _s_read = self._verify_split([pending] + drafts, p)
                _t_verify = time.perf_counter()
            else:
                vids, vhidden, vlogits = self._verify([pending] + drafts, p)
            # Greedy compares IDS, so it walks the trace's on-device argmax. Sampling runs rejection
            # sampling over the [T, vocab] host logits and IGNORES vids (still produced by the trace),
            # drawing the emitted token itself — that token is the next iteration's `pending`.
            sampled_tok = None
            if self.sampler is None:
                m = self._accept_greedy(drafts, vids)
            else:
                # Presence-penalty set for this window's FIRST row: the output so far plus `pending`,
                # which the row follows (the sampler adds drafts[:j] for the deeper rows). Built only
                # when the request asks for a penalty; trivial for the <= 500 tokens a run generates.
                penalize_base = (
                    torch.tensor(sorted(self._out_set | {pending}), dtype=torch.int64)
                    if self.sampler.params.presence_penalty > 0
                    else None
                )
                m, sampled_tok = self._accept_sample(drafts, vlogits, penalize_base)
            committed = [pending] + drafts[:m]
            mi = len(committed) - 1  # accepted-prefix's last token index in the verify window
            _t_accept = time.perf_counter() if self._timing else 0.0  # host-only: no fence needed
            self._commit(mi)
            _t_commit = self._tick() if self._timing else 0.0
            prev_p = p
            # The next anchor's own next token: greedy takes the base's argmax at the accepted-
            # prefix's last row, sampling the token its accept step already drew from that row.
            next_pending = vids[mi] if self.sampler is None else sampled_tok
            # The new anchor hidden is the accepted prefix's last row of the verify window, refilled
            # into the SAME persistent buffer the drafter already read this iteration (see
            # _anchor_warmup: a fresh per-iteration clone here is what the commit traces aliased).
            self._set_anchor(vhidden, mi)

            # MTP KV maintenance over the slots just committed, in ONE drafter forward over the
            # verify window (row i is the base hidden at slot prev_p+1+i, paired with the token at
            # slot+1). vhidden is the verify trace's own persistent output row buffer, so it is not
            # freed here — the next replay overwrites it in place.
            _rfn = self._reseed_mtp_batched if self._batched_reseed else self._reseed_mtp
            _rfn(prev_p + 1, vhidden, committed[1:])
            _t_reseed = self._tick() if self._timing else 0.0

            p += len(committed)
            pending = next_pending

            out.extend(committed)
            self._out_set.update(committed)
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
            # Sampling only: mean target probability of the drafts the sampler evaluated. A draft is
            # accepted with exactly that probability, so it is the per-draft acceptance odds.
            "mean_draft_target_prob": (
                (self._p_draft_sum / max(1, self._p_draft_n)) if self.sampler is not None else None
            ),
        }

    def log_stats(self, prefix="spec"):
        s = self.stats()
        logger.info(
            f"[{prefix}] {s['iters']} iters, K={s['K']}: accept={s['accept_rate']:.2f}/{s['K']} "
            f"-> {s['committed_per_iter']:.2f} committed tokens/iter"
        )
        if self.sampler is not None:
            sp = self.sampler.params
            logger.info(
                f"[{prefix}] sampling: temp={sp.temperature} top_k={sp.top_k} top_p={sp.top_p} "
                f"presence={sp.presence_penalty} "
                f"seed={self.sampler.seed} mean p_target(draft)={s['mean_draft_target_prob']:.3f}"
            )
        logger.info(f"[{prefix}] per-depth accept   : {[f'{x:.2f}' for x in s['depth_rate']]}")
        logger.info(f"[{prefix}] conditional accept : {[f'{x:.2f}' for x in s['conditional']]}")
        logger.info(f"[{prefix}] accepted histogram : {s['hist']} (index j = exactly j drafts accepted)")
        logger.info(f"[{prefix}] zero-accept iters  : {s['zero_accept_rate']:.1%} (still commit the pending token)")
        logger.info(f"[{prefix}] MTP reseed forwards: {s['mtp_extra_steps']}")
        return s
