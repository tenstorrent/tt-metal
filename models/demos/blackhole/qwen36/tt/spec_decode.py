# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Qwen3.6-27B using the built-in MTP drafter head.
MTP (tt/mtp.py) drafts K tokens per user; the base verifies them in one traced chunk of B*(K+1)
rows; the accepted prefix is committed by pointing GDN at the accepted slot on the NEXT replay (a
deferred select — there is no commit phase). Iteration for user u (anchor p_u, hidden h_{p_u}):
pending_u = argmax(base logits at p_u) (already confirmed); draft slot p_u+j fuses (hidden, token at
slot+1) -> candidate for p_u+2+j; verify [pending, d_0..d_{K-1}] at p_u+1..p_u+K+1; commit
[pending] + accepted prefix. Head expects DeepSeek-V3 / vLLM pairing (h_i, token_{i+1}) ->
token_{i+2}, not (h_i, token_i); pending is known before drafting, so all K steps propose new
tokens. ROW LAYOUT (the invariant every batched piece shares): B users, T = K+1 rows per user, one
32-row decode tile carries all of them (B*T <= 32 asserted), rows are USER-MAJOR r = u*T + j, and
row r's position is p_u + 1 + j with page-table row page_tables[u]. GDN buffers state after every
token into a per-token ring and the next verify replay reads each user's initial state from its own
accepted slot (state_blk_idx / conv_sel — no rollback, no commit forward). Full-attention paged KV
is corrected implicitly (rejected positions past the frontier never attended, overwritten next
iteration). TP (P150x4) only; B == model.args.max_batch_size. Greedy accepts the longest
matching-argmax prefix per user (token-identical to plain greedy). Sampling (temp > 0, optional
top-k/top-p) runs exact speculative rejection sampling (tt/spec_sampling.py), lossless in
distribution; drafts stay the device ARGMAX so the accept test is u < p(d). Extra sampling cost is
the [B*T, vocab] logits readback plus host accept math."""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSampler, SpecSamplingParams

# Prompt length above which the reseed goes back to the per-slot loop; see generate().
EAGER_RESEED_PROMPT_LEN = 131072
# One decode tile carries every verify row, so B*(K+1) may not exceed it. Everything downstream
# (nlp_create_qkv_heads_decode, the 1-tile-row decode matmul configs, the fused GDN kernel's core
# budget) is built around a single 32-row tile — see "Beyond 32 rows" in the design note.
MAX_SPEC_ROWS = ttnn.TILE_SIZE


class SpeculativeDecoder:
    """MTP speculative decode over B users, greedy or sampling.

    Greedy (``sampling=None``) reproduces the plain-decode greedy trajectory exactly. A
    ``SpecSamplingParams`` (temperature > 0, optional top-k / top-p) switches acceptance to exact
    speculative rejection sampling over the verify logits, lossless in distribution, and turns
    ``read_verify_logits`` on (needs [B*T, vocab] target rows, not the trace's argmax ids). Drafts
    are the drafter's argmax under sampling as well. One shared ``SpecSampler`` serves every user,
    and the accept loop walks the LIVE users in order 0..B-1, so a seeded run is reproducible for a
    FIXED batch composition (the RNG stream is user-major within an iteration, and a user that has
    finished is frozen and draws NOTHING; changing B, which prompts share a batch, or when a user
    stops, changes which draws each user gets). This is the demo's DEFAULT decode
    path (QWEN36_SPEC=0 opts out). Remaining knobs: QWEN36_SPEC_DRAFT_LEN (K override),
    QWEN36_SPEC_TIMING (per-iteration timing), QWEN36_SPEC_CHECK_ANCHOR (validate the anchor
    matmul against the rows it selects), and the ``sampling`` constructor argument.
    """

    # Field order of the QWEN36_SPEC_TIMING lines. `commit` is gone (the commit is a host-side index
    # folded into the next replay's inputs); `anchor` is the one-hot anchor matmul.
    _TPHASES = ("draft", "verify", "readback", "accept", "anchor", "reseed", "other", "total")
    # The demo runs a throwaway warmup generate() on a separate instance before the timed one, so
    # the timing lines are tagged with a class-level call id to tell the two apart in the log.
    _gen_calls = 0

    def __init__(
        self,
        model,
        page_tables,
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
        # K=3 is the conservative library default, kept for callers that pass no draft_len (the
        # correctness tests). The fully-batched GDN verify made verify cost ~flat in K (2.3 ms per
        # candidate — see TTGatedDeltaNetTP._verify_fullbatch in gdn/tp.py), so the demo passes the
        # ISL-aware policy instead.
        # Demo policy (text_demo.py): greedy K=11 up to a 4k prompt and K=7 above it; K=7 under
        # sampling, then capped by the batch (B*T <= 32). T = K+1 must match an entry of
        # ``TPAttention._SPEC_SDPA_L1_FIT`` (T in {4, 8, 12}) to take the fused SDPA plan; any other T
        # falls back to the per-row SDPA path. QWEN36_SPEC_DRAFT_LEN overrides both.
        self.K = int(draft_len if draft_len is not None else os.environ.get("QWEN36_SPEC_DRAFT_LEN", 3))
        # Block table, one ROW PER USER: torch int32 [B, num_blocks]. A [1, nb] table (or a bare
        # [nb] vector) is the B == 1 case and is accepted unchanged.
        pt = page_tables if isinstance(page_tables, torch.Tensor) else torch.as_tensor(page_tables)
        if pt.dim() == 1:
            pt = pt.reshape(1, -1)
        assert pt.dim() == 2, f"page_tables must be [B, num_blocks], got {tuple(pt.shape)}"
        self.page_tables = pt.to(torch.int32).contiguous()
        self.B = int(self.page_tables.shape[0])
        assert self.B == self.args.max_batch_size, (
            f"page_tables has {self.B} rows but the model was built for max_batch_size="
            f"{self.args.max_batch_size}; the verify trace's row count is baked in at capture"
        )
        assert self.B * (self.K + 1) <= MAX_SPEC_ROWS, (
            f"B={self.B} users x T={self.K + 1} rows = {self.B * (self.K + 1)} > {MAX_SPEC_ROWS}: the verify "
            f"is one decode tile. Lower K (see the auto-K-by-batch policy) or lower the batch."
        )
        # QWEN36_SPEC_TIMING=1: per-ITERATION breakdown (one log line per iteration + a mean at the
        # end). Off by default and every cost sits behind `self._timing`.
        self._timing = bool(int(os.environ.get("QWEN36_SPEC_TIMING", "0")))
        self._tsum = {}  # phase -> summed seconds, warmup iterations excluded
        self._tn = 0  # iterations folded into _tsum
        self.stop_tokens = set(stop_tokens or [])
        self.mtp = model.mtp
        # The MTP layer keeps its own paged KV cache with its own page table; same [B, nb] rows as
        # the base. `mtp_pt` is the batched form the B-row draft and the B*T-row reseed consume;
        # `mtp_pt_rows[u]` is user u's single row, for the eager per-user calls (prompt warm, the
        # slot-T-1 write, the eager reseed) that run one row at a time.
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        self.mtp_pt = ttnn.from_torch(
            self.page_tables, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=rep
        )
        self.mtp_pt_rows = [
            ttnn.from_torch(
                self.page_tables[u : u + 1].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
            )
            for u in range(self.B)
        ]
        self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        # Spec verify advances GDN with the fused recurrent op, so plain decode must use the SAME op or
        # every greedy near-tie flips between the two paths (measured 2.82 -> 2.00 / 3 accepted when they
        # disagreed at ~1e-5). The switch is explicit and model-scoped (model.set_gdn_fused_decode(True)),
        # never a side effect of constructing this object: on a shared model it changes every later
        # plain decode too, so the caller must make that choice visibly.
        assert model.gdn_fused_decode, (
            "SpeculativeDecoder needs model.set_gdn_fused_decode(True) before construction: verify runs "
            "the fused GDN op, and decode must use the same math"
        )
        self._vfy_captured = False
        # Persistent anchor-hidden buffer [1,1,B,dim/tp] and the one-hot selector [1,1,B,B*T] that
        # refills it; both allocated before any trace capture (see _anchor_warmup).
        self._hp_buf = None
        self._anchor_sel = None
        # The anchor select is a matmul, so it must be EXACT: HiFi4 + fp32 accumulate makes a bf16
        # one-hot row selection bit-exact (packer_l1_acc off — L1 accumulation would round the
        # partials in bf16). Same init_device_compute_kernel_config idiom as gdn/tp.py's conv1d.
        self._anchor_cc = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # QWEN36_SPEC_CHECK_ANCHOR=1: after every anchor refill, read Hp and the verify feed rows
        # back and assert bit-equality with the rows the one-hot names. Two full readbacks per
        # iteration, so it is a debug switch, not a default.
        self._check_anchor = bool(int(os.environ.get("QWEN36_SPEC_CHECK_ANCHOR", "0")))
        # Acceptance mode. None => greedy (argmax-prefix) acceptance; a SpecSamplingParams =>
        # exact speculative rejection sampling on host over the verify logits (tt/spec_sampling.py).
        self.sampler = SpecSampler(sampling, self.vocab) if sampling is not None else None
        # Read the full [B*T, vocab] verify logits back to host as well as the argmax ids. Greedy
        # acceptance does not need them (the trace argmaxes on device), so it stays off there; the
        # sampling accept step needs the distributions, so the constructor turns it on for it.
        self.read_verify_logits = self.sampler is not None
        # Mean target probability of the drafts the sampler actually evaluated: the sampling path's
        # analogue of per-depth acceptance (see stats()).
        self._p_draft_sum = 0.0
        self._p_draft_n = 0
        # Batched or eager reseed, decided in generate() from the prompt length (see the note there).
        self._batched_reseed = True
        # The batched reseed's scratch block (its padding rows' KV sink) is the MTP cache's extra LAST
        # block, index = cache block count - 1 (_allocate_mtp_kv_cache allocates one block past the
        # sequence's own). It is derived from the CACHE in generate(), never from the page table's
        # width: a vLLM block table is [B, blocks_per_request] of PHYSICAL block ids, so its width is
        # a per-request capacity and not the block count, and a width-derived index would alias a live
        # block belonging to some other sequence.
        self._reseed_scratch_block = None  # set in generate() from the MTP cache, once the KV caches exist
        self._reseed_block_size = 0  # filled in generate(), once the KV caches exist
        self.total_drafted = 0
        self.total_accepted = 0  # accepted DRAFT tokens (excludes the mandatory correction/bonus)
        self.iters = 0  # loop iterations
        # LIVE user-iterations: summed over iterations, the users that were not yet FROZEN (see
        # generate()'s loop). It equals iters * B while the whole batch is still generating and
        # grows more slowly once users finish. Denominator of every accept rate.
        self.user_iters = 0
        # --- instrumentation (mean acceptance alone hides where the drafts die) ---
        # Aggregated over the LIVE users: one accept decision per live user per iteration, so the
        # histogram sums to user_iters, and to iters * B only while no user has finished.
        self.accept_hist = [0] * (self.K + 1)  # how often exactly j drafts were accepted
        self.depth_hits = [0] * self.K  # depth_hits[j] = user-iterations that accepted draft j
        self.zero_accept = 0  # user-iterations where no draft was accepted (still commit the pending
        # token, so they are worth 1 token, not 0)
        self.mtp_extra_steps = 0  # drafter forwards spent on KV maintenance (reseed)
        self.prefill_time = 0.0  # set by generate(): TTFT (prefill + MTP warm + seed)
        self.decode_time = 0.0  # set by generate(): spec-loop wall-clock (excludes prefill)

    # --------------------------------------------------------------------- #
    # Draft
    # --------------------------------------------------------------------- #
    def _draft(self, pending, anchor_hidden, p):
        """Autoregressively draft K tokens PER USER from the MTP head, from each user's slot p_u.

        The head is fused from (base hidden at slot s, embedding of the token at s+1) and predicts
        the token at s+2 — DeepSeek-V3 / vLLM. Step 0: (h_{p_u}, pending_u) at slot p_u -> candidate
        for p_u+2; step k: (own hidden, previous draft) -> candidate for p_u+2+k. ``pending`` is
        each user's OWN next token at p_u+1, already confirmed; feeding it here makes all K drafts
        new. The B users ride the SAME K steps as B decode rows (they are independent: only the KV
        write survives, and K/V come from the row's own (hidden, token, position)), so batching
        costs nothing beyond the wider tile. The chain stays ON DEVICE: host argmax of 151k-vocab
        logits between steps is a round-trip; device argmax feeds the next step and defers readback
        to K*B small ids at the end. Each step is an fp32 LM head, then untilize + ttnn.argmax
        (``_argmax_last``); under sampling that argmax is the deterministic proposal (the delta at
        that id). Returns ``drafts[u][k]``, B lists of K ids."""
        B = self.B
        tok_tt = ttnn.from_torch(
            torch.tensor([[int(t)] for t in pending], dtype=torch.int32),  # [B, 1]
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        owned_tok = [tok_tt]
        h = anchor_hidden
        for k in range(self.K):
            logits, h_next = self.model.ttnn_mtp_decode_forward(h, tok_tt, [pu + k for pu in p], self.mtp_pt)
            idx = self._argmax_last(logits)  # [1,1,B] uint32 ROW_MAJOR
            ttnn.deallocate(logits)
            tok_tt = ttnn.reshape(idx, (B, 1))
            owned_tok.append(tok_tt)
            if h is not anchor_hidden:
                ttnn.deallocate(h)
            h = h_next
        if h is not anchor_hidden:
            ttnn.deallocate(h)
        # ONE sync for the whole chain: the first readback blocks, and everything below is already
        # computed — K reads of B small ids.
        steps = [self._id_to_host(t) for t in owned_tok[1:]]  # steps[k][u]
        for t in owned_tok:
            ttnn.deallocate(t)
        return [[steps[k][u] for k in range(self.K)] for u in range(B)]

    def _draft_warmup(self, pending, Hp, p):
        """Run ONE full (need_logits=True) draft step eagerly, BEFORE any trace is captured.

        The first real ``_draft`` happens after capture_verify_trace, and the logits-producing
        drafter path has programs nothing earlier in generate() has run: head_norm in DECODE
        (gather-then-norm), the LM head and its vocab all-gather, the argmax pick, and the
        mesh_partition that re-fractures mtp.norm's output for the next chain step. A program that
        first compiles while a trace is parked lands its kernel binaries in memory the replayed
        trace writes over, so they must compile here. Side effect: writes each user's drafter KV at
        slot p_u from (H_{p_u}, pending_u) — the same write draft step 0 repeats on the first real
        iteration, so it is inert. The drafted ids are discarded."""
        logits, h = self.model.ttnn_mtp_decode_forward(Hp, [int(t) for t in pending], list(p), self.mtp_pt)
        idx = self._argmax_last(logits)
        for t in (logits, idx, h):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)

    def _argmax_last(self, logits):
        """argmax over the vocab dim for B rows -> [1,1,B] uint32 ROW_MAJOR.

        ttnn.argmax needs ROW_MAJOR input: a TILE tensor takes a single-core internal-untilize path
        that is catastrophically slow on a 151k-wide vocab. So untilize multicore, then argmax.
        Used to pad B -> 32 rows on the belief that multicore argmax is row-parallel and returns
        garbage below a full tile. It does not: the unpadded argmax returns byte-identical ids, and
        [1,1,B,vocab] is ALREADY a full tile physically, so padding to 32 logical rows made untilize
        and argmax move ~32/B the bytes they need. The pad is gone.
        """
        u = ttnn.untilize(logits, use_multicore=True)
        out = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,B] uint32 RM
        ttnn.deallocate(u)
        return out

    def _id_to_host(self, id_tt):
        """[*, B] uint32 device ids -> list of B python ints. Reads only the device-0 replica: the
        logits are replicated across the TP mesh, so a ConcatMeshToTensor would move 4x the bytes
        for nothing."""
        t = ttnn.to_torch(ttnn.get_device_tensors(id_tt)[0])
        return [int(v) for v in t.reshape(-1)[: self.B]]

    # --------------------------------------------------------------------- #
    # MTP KV maintenance
    # --------------------------------------------------------------------- #
    def _warm_mtp_chunk(self, hidden, chunk_start, valid_len, prompt_ids, page_table_torch):
        """Warm ONE user's MTP drafter KV over ONE prompt chunk, in one forward.

        The drafter must see prompt context: with an empty cache at the first draft, acceptance
        collapses. Slot i is fused from (base_hidden_i, token_{i+1}) — the same shift pairing the
        draft loop uses. Slot T-1 is deliberately NOT written here: its token is the base's own
        prediction for position T, unknown until the prefill logits exist; ``_warm_mtp_last``
        writes it afterwards. The forward runs over the WHOLE bucket, not just the valid rows: the
        bucket is tile-aligned (128..2048) and prefill matmuls require that, whereas an arbitrary
        valid_len fails the matmul shape check. Rows past the prompt write junk MTP KV at slots
        >= T-1, which is harmless — slot T-1 is overwritten by _warm_mtp_last, and every slot above
        it is rewritten by the drafter before it is ever attended. ``page_table_torch`` is THIS
        user's row of the block table, torch [1, nb]: users are prefilled one at a time, so the
        drafter prefill stays a B=1 shape.
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
        self.model.ttnn_mtp_prefill_forward(hidden, toks, chunk_start, page_table_torch)

    def _warm_mtp_last(self, last_hidden, first_tok, slot, page_table_tt):
        """Write ONE user's final prompt slot's MTP KV, whose token is the base's own first
        prediction.

        Load-bearing: the first draft happens at slot T and attends to slots <= T-1, so leaving T-1
        unwritten hands it stale KV. One decode step at that user's page-table row.
        """
        _, h_next = self.model.ttnn_mtp_decode_forward(
            last_hidden, int(first_tok), slot, page_table_tt, need_logits=False
        )
        ttnn.deallocate(h_next)

    def _reseed_mtp(self, prev_p, vfeed, committed):
        """Refresh the MTP KV of the committed slots with the BASE hidden, replacing the drafter's
        own chained hidden. The drafter wrote those slots from its own chained hidden while drafting;
        the prompt warming wrote base hiddens. Leaving the mismatch in place costs acceptance, so
        every committed slot is rewritten from the base hidden the verify forward already produced.
        ``vfeed`` row u*T+i is user u's base hidden at slot prev_p[u]+1+i; committed[u][1+i] is the
        token at that slot + 1. One drafter DECODE step per (user, slot), at that user's page-table
        row. Superseded by _reseed_mtp_batched (one forward over every user's slots) except past
        EAGER_RESEED_PROMPT_LEN, where generate() comes back here. Batching does NOT go through the
        drafter's prefill path — that needs a genuine prefill shape and neither candidate width works
        at an arbitrary mid-sequence slot0 (at one tile the stack silently picks DECODE matmuls while
        norms stay PREFILL; at 128 rows SDPA rejects the unaligned chunk start). Goes through the
        DECODE path at B*T rows instead — see _reseed_mtp_batched.
        """
        T = self.K + 1
        W = vfeed.shape[-1]
        for u, com in enumerate(committed):
            for i, tok in enumerate(com[1:]):
                r = u * T + i
                row = ttnn.slice(vfeed, (0, 0, r, 0), (1, 1, r + 1, W))
                _, h_next = self.model.ttnn_mtp_decode_forward(
                    row, int(tok), prev_p[u] + 1 + i, self.mtp_pt_rows[u], need_logits=False
                )
                ttnn.deallocate(row)
                ttnn.deallocate(h_next)
                self.mtp_extra_steps += 1

    def _reseed_mtp_batched(self, prev_p, vfeed, committed, scratch_only=False):
        """``_reseed_mtp`` as ONE fixed-shape drafter forward over all B*T verify rows.

        The per-slot loop is sum(m_u) sequential decode forwards. The rows are INDEPENDENT — only the
        KV write survives, and K/V come from the row's own (hidden, token, position) through the
        in-projection, never from the attention output — so running them as B*T pseudo-users of the
        batch is exactly equivalent and costs one forward. Same hybrid trick as verify: per-row
        position tensor, page-table rows aliasing each user's own blocks, alias_kv_write=True so
        shared-block KV writes go row by row instead of racing several cores onto one 32-row tile.
        Row u*T+i is REAL for i < m_u (= len(committed[u]) - 1) and PADDING otherwise (m_u varies,
        the shape must not): padding rows point at a dedicated scratch block (the KV write never
        touches a sequence) and position 0 so the discarded SDPA read is one slot deep.
        ``scratch_only=True`` makes EVERY row padding (warmup): same shapes/program, no real slot
        touched. ``vfeed`` is consumed AS IS — its row layout already matches.
        """
        T = self.K + 1
        R = self.B * T
        assert vfeed.shape[-2] == R, f"reseed wants {R} rows (B={self.B} x T={T}), got {vfeed.shape[-2]}"
        lens = [0] * self.B if scratch_only else [len(c) - 1 for c in committed]
        if not scratch_only and not any(lens):
            return
        mesh, rep = self.mesh, ttnn.ReplicateTensorToMesh(self.mesh)
        nb = self.page_tables.shape[-1]
        tok = torch.zeros(R, 1, dtype=torch.int32)
        pos = torch.zeros(R, dtype=torch.int32)
        pt = torch.full((R, nb), int(self._reseed_scratch_block), dtype=torch.int32)
        for u, m in enumerate(lens):
            assert m <= T, f"reseed {m} slots into a {T}-row group"
            if m == 0:
                continue
            r0 = u * T
            tok[r0 : r0 + m, 0] = torch.tensor([int(t) for t in committed[u][1 : 1 + m]], dtype=torch.int32)
            pos[r0 : r0 + m] = torch.arange(prev_p[u] + 1, prev_p[u] + 1 + m, dtype=torch.int32)
            pt[r0 : r0 + m, :] = self.page_tables[u]
        cos_t, sin_t = self.model._rope_tp_cos_sin_decode_torch(pos)
        tok_tt = ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pos_tt = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pt_tt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        # spec_n_users=B: the rows are USER-MAJOR (u*T + i), one row per user per candidate index,
        # so the KV write goes out as T calls of B rows instead of B*T single-row calls
        # (TPAttention._write_kv_aliased). The page table is NOT the per-user one — PADDING rows
        # point at the scratch block — so the grouped write slices each group's table out of pt_tt.
        # Padding rows of different users do collide there (same scratch block, position 0); that
        # block is write-only garbage whose SDPA read is discarded, exactly as with the per-row
        # write, which also left one arbitrary padding row's bytes behind.
        _, h_next = self.mtp.forward_decode(
            vfeed, tok_tt, pos_tt, cos, sin, pt_tt, need_logits=False, alias_kv_write=True, spec_n_users=self.B
        )
        for t in (tok_tt, pos_tt, pt_tt, cos, sin, h_next):
            ttnn.deallocate(t)
        self.mtp_extra_steps += 1

    def _reseed_warmup(self, rows, dim_frac, dtype):
        """Compile the batched-reseed program BEFORE the verify trace is captured.

        The batched reseed is a shape the loop has never run (B*T decode rows over the MTP layer),
        and a program that first compiles while the verify trace is parked lands its kernel binaries
        in memory the replayed trace writes over. Every row here targets the scratch block, so the
        throwaway forward touches no real KV.
        """
        z = ttnn.zeros(
            [1, 1, rows, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # scratch_only: all rows are padding, so they name the scratch block directly. (The scratch
        # block sits PAST the page tables' span, so there is no position that reaches it by lookup.)
        self._reseed_mtp_batched([0] * self.B, z, [], scratch_only=True)
        self.mtp_extra_steps -= 1  # warmup is not a loop cost
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(z)

    # --------------------------------------------------------------------- #
    # Accept
    # --------------------------------------------------------------------- #
    def _accept_greedy(self, drafts, verify_ids):
        """Greedy acceptance of one user's matching prefix; returns the number of accepted drafts.

        ``verify_ids`` is that user's T-row slice of the verify ids (rows u*T .. u*T+T-1). The verify
        chunk ran p+1..p+K+1, so verify_ids[j] is the base model's own argmax at p+1+j, which
        predicts p+2+j — exactly drafts[j]'s position. No draft's target was known before drafting
        (that token is ``pending``, committed unconditionally), so every rejection is a genuine
        drafter miss and nothing extra needs committing: the correction arrives as the next
        iteration's ``pending``. Greedy compares IDS, so the verify trace argmaxes on device and
        this walks a [T] int list rather than a [T, 151936] host float tensor.
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
        """Exact speculative rejection sampling over ONE user's verify logits; returns (m, token).

        ``vlogits`` is that user's [T, vocab] slice of the verify logits (rows u*T .. u*T+T-1). The
        drafts are the drafter's argmax, so the proposal is the delta at ``d_j`` and the accept test
        is ``u_j < p_j(d_j)``. ``next_token`` is the recovered token from the rejection row (m < K)
        or the bonus token from the extra row (m == K), and becomes that user's next ``pending``
        exactly like greedy's ``verify_ids[mi]`` (see spec_sampling.py). ``penalize_base`` is this
        user's presence-penalty set (``generated_so_far ∪ {pending}``, None when the request has no
        presence penalty); the sampler adds ``drafts[:j]`` per row, so each verify row is penalized
        on exactly the output that precedes it. One shared sampler serves every user and the caller
        walks them in order, so the RNG stream is user-major within an iteration. Same
        instrumentation as _accept_greedy, plus the mean target probability of the drafts the
        sampler evaluated."""
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
    # Verify / anchor
    # --------------------------------------------------------------------- #
    def _verify(self, tokens, positions, mi_prev):
        """Replay the captured verify trace over every user's [pending] + drafts.

        ``tokens`` is B lists of T ids, ``positions`` is each user's ABSOLUTE first verify slot
        (p_u + 1 — the same convention capture_verify_trace's warm_positions uses), and ``mi_prev``
        is the PREVIOUS iteration's accepted-prefix index per user. mi_prev is the whole commit: the
        replay reads each user's GDN initial state from its own accepted slot of the per-token ring
        and rebuilds its conv window from the same index, so there is no commit phase between
        replays. Advances GDN recurrently token by token (the SAME kernel decode uses, so it is
        recurrent-faithful) while attention/MLP/norm/lm_head stay batched over the B*T rows. Returns
        (per-row argmax ids (B*T), per-row feed hidden [1,1,B*T,dim/tp], per-row host logits
        [B*T, vocab] — None unless read_verify_logits is set, i.e. unless the sampling accept step
        needs the distributions).
        """
        return self.model.verify_traced(tokens, positions, mi_prev, read_logits=self.read_verify_logits)

    def _anchor_warmup(self, dim_frac, dtype):
        """Allocate the persistent anchor buffers and compile the refill matmul — BEFORE any trace
        is captured.

        ADDRESS STATIONARITY: the anchor hidden used to be a fresh ttnn.clone per iteration, and it
        is LIVE across the whole iteration. A parked trace bakes its intermediates' addresses in at
        capture time, when no such per-iteration buffer exists, so the loop's clone could land on one
        of them and the replay would overwrite the anchor mid-flight — the drafter then chains from
        corrupted hidden and acceptance collapses. Two fixed-address buffers, allocated before the
        capture, remove the whole class. NO POST-CAPTURE COMPILE: a program that first compiles while
        a trace is parked writes kernel binaries over it, so the matmul runs once here. It is ONE
        program for every accepted-prefix index (the selector is data, not shape) — the per-mi slice
        it replaces hashed its offsets and needed a warmup pass per mi.
        """
        T = self.K + 1
        self._hp_buf = ttnn.zeros(
            [1, 1, self.B, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._anchor_sel = ttnn.from_torch(
            torch.zeros(1, 1, self.B, self.B * T, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.zeros(
            [1, 1, self.B * T, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._set_anchor(z, [0] * self.B)
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(z)

    def _set_anchor(self, feed_rows, mi):
        """Refill the persistent anchor buffer with each user's accepted-prefix row of the verify feed.

        Row u of the result must be verify feed row u*T + mi[u]. With B users that is a GATHER, not a
        slice, so it is done as a one-hot matmul: ``anchor_sel`` [1,1,B,B*T] has a single 1.0 at
        (u, u*T+mi[u]) and ``anchor_sel @ feed_rows`` selects those rows. bf16 one-hot under HiFi4 +
        fp32 accumulate is EXACT (every product is either 0 or the value itself, and the fp32
        accumulator holds a single bf16 addend), so the anchor is bit-identical to the row it names —
        QWEN36_SPEC_CHECK_ANCHOR=1 asserts exactly that. One program for all mi, and the selector is
        re-staged into a FIXED-address persistent buffer, so nothing allocates per iteration.
        """
        T = self.K + 1
        sel = torch.zeros(1, 1, self.B, self.B * T, dtype=torch.bfloat16)
        for u, m in enumerate(mi):
            sel[0, 0, u, u * T + int(m)] = 1.0
        _h = ttnn.from_torch(
            sel,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(_h, self._anchor_sel)
        Hp = ttnn.matmul(self._anchor_sel, feed_rows, compute_kernel_config=self._anchor_cc)
        ttnn.copy(Hp, self._hp_buf)
        if self._check_anchor:
            self._check_anchor_rows(Hp, feed_rows, mi)
        ttnn.deallocate(Hp)

    def _check_anchor_rows(self, Hp, feed_rows, mi):
        """QWEN36_SPEC_CHECK_ANCHOR=1: assert the one-hot matmul is bit-identical to the rows it names.

        Two full readbacks per iteration ([1,1,B,dim/tp] and [1,1,B*T,dim/tp] off the device-0
        replica), so this is a debug switch. It is the only thing standing between "the matmul is
        exact" and a silent 1-ulp anchor drift that would show up as a slow acceptance loss.
        """
        T = self.K + 1
        got = ttnn.to_torch(ttnn.get_device_tensors(Hp)[0]).reshape(-1, Hp.shape[-1])
        ref = ttnn.to_torch(ttnn.get_device_tensors(feed_rows)[0]).reshape(-1, feed_rows.shape[-1])
        for u, m in enumerate(mi):
            assert torch.equal(got[u], ref[u * T + int(m)]), (
                f"anchor matmul mismatch for user {u} (mi={int(m)}, feed row {u * T + int(m)}): "
                f"max |delta| {float((got[u].float() - ref[u * T + int(m)].float()).abs().max()):.3e}"
            )

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

    def _verify_split(self, tokens, positions, mi_prev):
        """``_verify``, with the device->host readback split out of the device time.

        ``model.verify_traced`` runs execute_trace + synchronize_device and only THEN pulls the ids
        (and, when read_verify_logits is set, the logits) back via
        ``ttnn.to_torch(ttnn.get_device_tensors(...)[0])``, so hooking ``ttnn.get_device_tensors``
        for the duration of the call marks the device/host boundary without editing model.py. The
        trailing sync flushes anything that follows the readback into the readback bucket. Returns
        (vids, vfeed, vlogits, device_seconds, readback_seconds).
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
            vids, vfeed, vlogits = self._verify(tokens, positions, mi_prev)
        finally:
            ttnn.get_device_tensors = orig
        ttnn.synchronize_device(self.mesh)
        t1 = time.perf_counter()
        t_mark = mark[0] if mark else t1
        return vids, vfeed, vlogits, t_mark - t0, t1 - t_mark

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
        cpi = self.accept_rate() + 1.0  # per LIVE user
        tot = cpi * self.B  # tokens the batch commits per iteration with every user still live
        # A finished user is frozen and commits nothing, while the iteration costs the same (the
        # replay's row count is fixed), so the DELIVERED rate scales the same cpi by the MEAN number
        # of live users instead of B. The two lines agree until the first user stops.
        live_mean = self.user_iters / max(1, self.iters)
        dlv = cpi * live_mean
        logger.info(
            f"[SPEC_TIMING] MEAN gen={self._gen_id} iters={self._tn} (excl 2 warmup) users={self.B} "
            + " ".join(f"{k}={mean[k] * 1e3:.2f}" for k in self._TPHASES)
            + f" committed_per_iter={cpi:.3f} tokens_per_iter_total={tot:.3f} "
            + f"tok_s={tot / max(mean['total'], 1e-9):.2f} "
            + f"live_users={live_mean:.3f} tokens_per_iter_delivered={dlv:.3f} "
            + f"delivered_tok_s={dlv / max(mean['total'], 1e-9):.2f}"
        )

    # --------------------------------------------------------------------- #
    # Generate
    # --------------------------------------------------------------------- #
    def generate(self, prompts, max_new_tokens):
        """Speculative generation over the batch (greedy or sampling, per the constructor's
        ``sampling`` arg).

        ``prompts`` is either ONE prompt as a ``list[int]`` (only when B == 1; the return is then a
        flat ``list[int]``) or B prompts as a ``list[list[int]]`` (the return is B lists). Prompt
        lengths may differ per user — every per-user position is carried in host state. Returns the
        generated token ids, prompt excluded.

        Records self.prefill_time (prompt prefill + MTP warm + seed, i.e. TTFT) and self.decode_time
        (the spec loop), both synchronize-bracketed, so callers can report ttft / decode tok/s.
        """
        model = self.model
        B, T = self.B, self.K + 1
        assert len(prompts) > 0, "generate() needs at least one prompt"
        _p0 = prompts[0]
        flat_in = not (isinstance(_p0, (list, tuple)) or (isinstance(_p0, torch.Tensor) and _p0.ndim >= 1))
        if flat_in:
            assert B == 1, f"a flat list[int] prompt needs B == 1, but this decoder carries {B} users"
            prompt_lists = [[int(t) for t in prompts]]
        else:
            prompt_lists = [[int(t) for t in pr] for pr in prompts]
        assert len(prompt_lists) == B, f"expected {B} prompts (one per page-table row), got {len(prompt_lists)}"
        Tp = [len(pr) for pr in prompt_lists]  # per-user prompt length
        SpeculativeDecoder._gen_calls += 1
        self._gen_id = SpeculativeDecoder._gen_calls
        _t_start = time.perf_counter()

        # Per-user host state. p[u] is the ANCHOR slot (the verify window starts at p[u]+1); mi[u]
        # is the accepted-prefix index the NEXT replay commits; out_sets[u] is the GENERATED ids so
        # far (prompt excluded), which is the set the presence penalty is defined over — maintained
        # unconditionally (a set add per committed token) and read only when the request carries one.
        p = list(Tp)  # rewritten after the seed; Tp is where each user's anchor lands
        pending = [0] * B
        mi = [0] * B
        out = [[] for _ in range(B)]
        done = [False] * B
        self._out_sets = [set() for _ in range(B)]
        out_sets = self._out_sets

        # Reseed shape. Past EAGER_RESEED_PROMPT_LEN the batched reseed's wide in-projection drifts
        # enough bf16 near-ties to cost ~0.3 accepted drafts/iter (256k: 19.7 vs 20.9 tok/s), while
        # its dispatch saving (~2 ms/iter) no longer covers that; the per-slot loop keeps spec >=
        # plain at every ISL. That policy is B == 1 ONLY: the eager loop is a 1-ROW drafter decode,
        # a shape the B-row warmups never compiled, so at B > 1 its first call would compile a
        # program with the verify trace parked and write kernel binaries over it. B > 1 therefore
        # always takes the batched reseed (already warmed at B*T rows), whatever the prompt length.
        eager_reseed = B == 1 and max(Tp) > EAGER_RESEED_PROMPT_LEN
        self._batched_reseed = not eager_reseed
        if B > 1 and max(Tp) > EAGER_RESEED_PROMPT_LEN:
            logger.info(
                f"[spec] prompt {max(Tp)} > {EAGER_RESEED_PROMPT_LEN} but users={B}: keeping the BATCHED reseed "
                f"(the eager per-slot reseed is a B=1-only policy; its 1-row program is not warmed at B>1)"
            )
        if self.sampler is None:
            _samp = "sampling=greedy"
        else:
            _sp = self.sampler.params
            _samp = (
                f"sampling=temp={_sp.temperature} top_k={_sp.top_k} top_p={_sp.top_p} "
                f"presence={_sp.presence_penalty} seed={self.sampler.seed}"
            )
        logger.info(
            f"[spec] gen={self._gen_id} users={B} T_prompt={Tp} K={self.K} rows={B * T} "
            f"reseed={'eager' if eager_reseed else 'batched'} max_new={max_new_tokens} {_samp}"
        )

        # Capacity check, per user: bounded by the SPECULATIVE high-water slot, not the returned-token
        # count. Verify writes T candidate slots per iteration no matter how many are later accepted,
        # and capture_verify_trace's throwaway warmup writes them even when the loop body never runs
        # (max_new_tokens == 1). A slot past the cache reaches paged_update_cache, which indexes
        # page_table[slot / block_size] with no bounds check. Must run before prefill/seed/warmup/
        # capture, and for both reseed modes (eager reseed used to skip this check entirely).
        # The bound is the SAME at every batch size because a finished user is FROZEN (see the loop):
        # its p and its pending stop moving the moment it is done, so every later iteration replays
        # the same T tokens at the same T positions and re-writes the slots its last LIVE iteration
        # already wrote. Nothing climbs. So user u's last live iteration alone sets its high-water
        # mark: it starts from anchor p = Tp[u] + len(out[u]) - 1 with len(out[u]) <= max_new_tokens
        # - 1 (a longer output would already have frozen it) and writes through p + T, which is
        # Tp[u] + K + max_new_tokens - 1. The max(1, ...) covers max_new_tokens <= 1, where the loop
        # body never runs and only capture_verify_trace's warmup writes Tp[u]+1 .. Tp[u]+T.
        # Before freezing this had to be Tp[u] + T * (max_new_tokens + 1) at B > 1, because a
        # stopped user kept marching T slots up the cache every iteration until the slowest user
        # finished.
        self._reseed_block_size = int(self.mtp.attention.paged_k.shape[-2])
        nb = self.page_tables.shape[-1]
        _cap = nb * self._reseed_block_size
        for u in range(B):
            _hi = Tp[u] + self.K + max(1, max_new_tokens - 1)
            assert _hi < _cap, (
                f"user {u}: high-water slot {_hi} (prompt={Tp[u]}, max_new={max_new_tokens}, K={self.K}, "
                f"users={B}) does not fit the paged KV: {nb} blocks x {self._reseed_block_size} = {_cap} slots"
            )

        # Chunked prompt prefill, ONE USER AT A TIME (the prefill path is a single-sequence chunked
        # forward and each user lands its GDN state in its own slot). 2048-token chunks + masked tail
        # — the same path the demo uses, so long prompts work. Each chunk's hidden warms that user's
        # MTP drafter KV in ONE forward before it is freed, so the drafter never sees an empty cache
        # and TTFT stays flat in prompt length.
        first = []
        for u in range(B):
            prompt = torch.tensor([prompt_lists[u]], dtype=torch.int32)
            pt_u = self.page_tables[u : u + 1].contiguous()
            last_hidden = [None]  # the base hidden at slot Tp[u]-1, kept for _warm_mtp_last

            def _on_chunk(hidden, chunk_start, valid_len, _u=u, _Tu=Tp[u], _pt=pt_u, _lh=last_hidden):
                # Drafter feed for the chunk (a new fractured post-norm tensor). Both the warm and the
                # slot-T-1 row must come from the SAME tensor, so the drafter is never handed a mix of
                # scales. The caller still frees `hidden`.
                feed = model.spec_feed_rows(hidden)
                self._warm_mtp_chunk(feed, chunk_start, valid_len, prompt_lists[_u], _pt)
                if chunk_start + valid_len >= _Tu:  # the chunk holding slot Tp[u]-1
                    i = _Tu - 1 - chunk_start
                    row = ttnn.slice(feed, (0, 0, i, 0), (1, 1, i + 1, feed.shape[-1]))
                    _lh[0] = ttnn.clone(row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(row)
                if feed is not hidden:
                    ttnn.deallocate(feed)

            logits_dev = model.prefill_for_spec(prompt, pt_u, Tp[u], _on_chunk, slot=u)
            lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
            # penalize=None: the presence penalty looks at the OUTPUT only (prompt tokens excluded, as
            # in vLLM), and at the prefill pick the output is still empty.
            f = self._pick_token(lt.reshape(-1)[: self.vocab], None)
            first.append(f)
            out[u].append(f)
            out_sets[u].add(f)
            # Prefill can emit a stop token directly, and `first` alone already satisfies
            # max_new_tokens == 1; either way that user is FROZEN before the loop starts.
            done[u] = f in self.stop_tokens or len(out[u]) >= max_new_tokens
            # Slot Tp[u]-1 pairs the base hidden at Tp[u]-1 with `f`, which only exists now.
            assert last_hidden[0] is not None, f"prefill_for_spec never delivered user {u}'s chunk holding T-1"
            self._warm_mtp_last(last_hidden[0], f, Tp[u] - 1, self.mtp_pt_rows[u])
            ttnn.deallocate(last_hidden[0])

        # Seed: ONE eager VERIFY-STYLE forward (B users x T = 1 row) that consumes each user's
        # `first` at its own position Tp[u] -> (L_{Tp[u]}, H_{Tp[u]}); anchor p[u] = Tp[u]. It is
        # the verify body, not a plain decode step, because the GDN state it leaves is what every
        # replay resumes from and the decode path's shift-register conv is not bit-equal to the
        # verify's conv1d — seeding with the other formulation forks the greedy trajectory at a
        # near tie (see model.seed_spec_step). MUST stay BEFORE capture_verify_trace: it compiles
        # the B-row seed programs (conv1d at (B, K), the T = 1 fused recurrent, sync_conv_taps),
        # and a program that first compiles while a trace is parked lands its kernel binaries in
        # memory the replayed trace writes over — the NEXT generate's seed (program-cache hit)
        # would dispatch corrupted binaries and hang the device. Every program the post-capture
        # path needs compiles first.
        Lp, Hp_rows = model.seed_spec_step(first, list(Tp), self.page_tables)
        # Each user's own next token, taken from its anchor logits (argmax, or a sampler draw). It is
        # committed unconditionally next iteration and is what seeds the drafter, so no drafter step
        # re-predicts it. This position follows `first`, so `first` is the whole output set the
        # presence penalty sees here.
        for u in range(B):
            pending[u] = self._pick_token(Lp[u], torch.tensor([first[u]], dtype=torch.int64))

        # Anchor buffers + the one-hot matmul that refills them, compiled while nothing is traced.
        # The warmup writes zeros into _hp_buf, so the real seed rows are copied in AFTERWARDS.
        self._anchor_warmup(Hp_rows.shape[-1], Hp_rows.dtype)
        ttnn.copy(Hp_rows, self._hp_buf)
        ttnn.deallocate(Hp_rows)  # seed_spec_step hands the caller a fresh spec_feed_rows tensor
        Hp = self._hp_buf

        # Reseed scratch = the MTP cache's extra LAST block, derived from the cache (D1). Never the
        # page tables' width: a vLLM block table is [B, blocks_per_request] of physical block ids,
        # so its width is not the block count and a width-derived index aliases a live block. No
        # user's table may name the scratch block, so the largest entry over the WHOLE table stays
        # below it.
        self._reseed_scratch_block = int(self.mtp.attention.paged_k.shape[0]) - 1
        assert int(self.page_tables.max()) < self._reseed_scratch_block, (
            f"page tables name block {int(self.page_tables.max())}, but the MTP cache's scratch block is "
            f"{self._reseed_scratch_block}; the MTP cache needs one block past every block the tables use"
        )

        # Batched-reseed warmup: compiles the B*T-row drafter forward while nothing is traced yet.
        if self._batched_reseed:
            self._reseed_warmup(B * T, Hp.shape[-1], Hp.dtype)
        # Draft warmup: the logits-producing drafter step (head norm, fp32 LM head, untilize, argmax,
        # plus the chain's mesh_partition) has not run yet, and its first real run is AFTER the
        # capture below. Compile it now; its one KV write per user (slot p[u] from (H_p, pending)) is
        # what draft step 0 repeats.
        self._draft_warmup(pending, Hp, p)

        # One-time verify-trace capture (replayed every iteration), done AFTER prefill + MTP warm +
        # seed so every program those paths need is already compiled: a compile that happens once the
        # trace is parked clobbers it. ``warm_positions`` is each user's ABSOLUTE first verify slot
        # (p[u]+1 — the same convention verify_traced's ``positions`` uses), so the two throwaway
        # passes write junk KV past the seed's slot, where the first real verify overwrites it.
        # Counts toward TTFT, not decode_time.
        if not self._vfy_captured:
            model.capture_verify_trace(self.page_tables, T, warm_positions=[pu + 1 for pu in p], decode_cfg=True)
            self._vfy_captured = True
        # Hand the GDN layers' live state to the spec ring (token slot 0) and their live conv window
        # to E_prev, so the first replay's mi_prev = 0 names exactly the seeded state. MUST come
        # AFTER the capture, both ways round: the ring and E_prev are allocated by the
        # prepare_spec_verify that capture_verify_trace runs, and capture's two throwaway passes
        # scribble both (there is no snapshot/restore any more — this IS the restore). The live
        # rec_state and conv window the seed forward left behind are untouched by those passes, so
        # they are still the truth to seed from; the seed also leaves the K taps in step with the
        # window, which is what keeps this call on the same sync_conv_win branch the throwaway
        # seed_spec_state inside capture_verify_trace already compiled.
        for dn in self._gdn:
            dn.seed_spec_state()
        ttnn.synchronize_device(self.mesh)
        self.prefill_time = time.perf_counter() - _t_start  # TTFT: prefill + MTP warm + seed
        _t_decode = time.perf_counter()

        # Every user runs every iteration: the verify is one fixed B*T-row replay, so a user that has
        # already stopped still occupies its T rows. Those rows are FROZEN. A done user keeps its p
        # and its pending for ever, so each later iteration replays the very same T tokens at the
        # very same T positions; its accept result is thrown away (forced m = 0, committed =
        # [pending], mi = 0), it contributes no reseed row, and it feeds no statistic. Its attention
        # KV, its drafter KV and its GDN ring writes therefore keep landing on slots it already
        # owns instead of marching up the cache — which is what makes the capacity bound above
        # tight and batch-size independent. Those repeated writes do fill the finished user's own
        # drafter KV and GDN ring with repeat-token junk (the reseed no longer repairs the slots
        # the draft step overwrites); nothing reads either again. The loop ends when every user is
        # done. At B == 1 this is unchanged: the single user freezing ends the loop.
        while any(not d for d in done):
            # LIVE = not yet frozen at the START of this iteration. A user that finishes INSIDE this
            # iteration is live here, and its accept decision and committed tokens are real work.
            live = [not d for d in done]
            n_live = sum(live)
            _tm = self._tick() if self._timing else 0.0
            drafts = self._draft(pending, Hp, p)
            _t_draft = self._tick() if self._timing else 0.0

            # The replay commits the PREVIOUS iteration's accepted prefix (mi) as it goes: GDN reads
            # each user's initial state from its own accepted ring slot. committed_u = [pending_u] +
            # drafts_u[:m_u].
            vtok = [[pending[u]] + drafts[u] for u in range(B)]
            vpos = [pu + 1 for pu in p]
            if self._timing:
                vids, vfeed, vlogits, _s_verify, _s_read = self._verify_split(vtok, vpos, mi)
                _t_verify = time.perf_counter()
            else:
                vids, vfeed, vlogits = self._verify(vtok, vpos, mi)

            # Per-user acceptance over that user's T-row slice. Greedy compares IDS, so it walks the
            # trace's on-device argmax. Sampling runs rejection sampling over the [T, vocab] host
            # logits and IGNORES vids (still produced by the trace), drawing the emitted token
            # itself — that token is the user's next `pending`.
            prev_p = list(p)
            committed = []
            next_pending = list(pending)
            for u in range(B):
                if not live[u]:
                    # FROZEN. Its T rows ran (the row count is baked into the trace) but nothing
                    # they produced is used: no accept test at all, so `_accept_sample` is never
                    # called for it and the shared RNG stream advances for LIVE users only; no
                    # histogram entry; m = 0; and a commit of just its unchanging `pending`, whose
                    # len - 1 == 0 makes it contribute NO reseed row (the batched reseed pads its
                    # group with scratch-block rows). mi = 0 keeps its GDN state-block index and
                    # conv-window offset in range on the next replay, and keeps
                    # QWEN36_SPEC_CHECK_ANCHOR's one-hot assertion pointed at a real feed row.
                    committed.append([pending[u]])
                    mi[u] = 0
                    continue
                row_ids = vids[u * T : (u + 1) * T]
                if self.sampler is None:
                    m = self._accept_greedy(drafts[u], row_ids)
                    sampled_tok = None
                else:
                    # Presence-penalty set for this user's FIRST row: its output so far plus its
                    # `pending`, which the row follows (the sampler adds drafts[:j] for the deeper
                    # rows). Built only when the request asks for a penalty; trivial for the <= 500
                    # tokens a run generates.
                    penalize_base = (
                        torch.tensor(sorted(out_sets[u] | {pending[u]}), dtype=torch.int64)
                        if self.sampler.params.presence_penalty > 0
                        else None
                    )
                    m, sampled_tok = self._accept_sample(drafts[u], vlogits[u * T : (u + 1) * T], penalize_base)
                com = [pending[u]] + drafts[u][:m]
                # The accept test can accept drafts PAST a stop token, so emit only through the first
                # one. anchor/reseed/p all follow the shortened prefix, since they derive from `com` /
                # `mi[u]` below; acceptance stats deliberately keep the full `m`.
                _stop_i = next((i for i, t in enumerate(com) if t in self.stop_tokens), None)
                if _stop_i is not None:
                    com = com[: _stop_i + 1]
                committed.append(com)
                mi[u] = len(com) - 1  # accepted-prefix's last token index in this user's window
                # The next anchor's own next token: greedy takes the base's argmax at the accepted-
                # prefix's last row, sampling the token its accept step already drew from that row.
                next_pending[u] = row_ids[mi[u]] if self.sampler is None else sampled_tok
                self.total_accepted += m
            _t_accept = time.perf_counter() if self._timing else 0.0  # host-only: no fence needed

            # The new anchor hidden is each user's accepted-prefix row of the verify feed, gathered
            # into the SAME persistent buffer the drafter already read this iteration (see
            # _anchor_warmup: a fresh per-iteration clone here is what a parked trace can alias).
            self._set_anchor(vfeed, mi)
            _t_anchor = self._tick() if self._timing else 0.0

            # MTP KV maintenance over the slots just committed, in ONE drafter forward over all B*T
            # verify rows (row u*T+i is user u's base hidden at slot prev_p[u]+1+i, paired with the
            # token at slot+1). vfeed is the verify trace's own persistent output row buffer, so it is
            # not freed here — the next replay overwrites it in place.
            _rfn = self._reseed_mtp_batched if self._batched_reseed else self._reseed_mtp
            _rfn(prev_p, vfeed, committed)
            _t_reseed = self._tick() if self._timing else 0.0

            for u in range(B):
                if not live[u]:
                    continue  # frozen before this iteration: p, pending and out are already final
                out[u].extend(committed[u])
                out_sets[u].update(committed[u])
                if committed[u][-1] in self.stop_tokens or len(out[u]) >= max_new_tokens:
                    # FREEZE, leaving p and pending at their PRE-commit values. The rows this user
                    # replays from now on then re-cover exactly the slots this iteration just wrote,
                    # so its high-water slot stays p + T and the tight capacity bound above holds.
                    # The tokens are already in `out`; nothing reads this user's KV, its drafter
                    # cache or its GDN state again.
                    done[u] = True
                    continue
                p[u] += len(committed[u])
                pending[u] = next_pending[u]
                assert p[u] == prev_p[u] + len(committed[u])
            if self._timing:
                # `other` = the deallocates and the host bookkeeping around the phases above.
                self._log_iter_timing(
                    {
                        "draft": _t_draft - _tm,
                        "verify": _s_verify,
                        "readback": _s_read,
                        "accept": _t_accept - _t_verify,
                        "anchor": _t_anchor - _t_accept,
                        "reseed": _t_reseed - _t_anchor,
                        "total": self._tick() - _tm,  # fenced, so nothing leaks into the next iter
                    }
                )
            self.iters += 1
            # LIVE users only. A frozen user's drafts and accept result are discarded, so counting
            # it would dilute every rate with work that produced no token. `iters` stays the raw
            # loop-iteration count (each one costs a full replay however many users are live).
            self.user_iters += n_live
            self.total_drafted += n_live * self.K

        # RELEASE THE VERIFY TRACE FIRST. materialize_spec_state slices the ring and the conv window
        # at offsets that depend on mi, and SliceDeviceOperation hashes slice_start/slice_end — so
        # every distinct mi is its OWN program, and there is no way to warm them all before the
        # capture (sync_conv_taps' concat/copy/reshape chain is the same story). A program that
        # first compiles while a trace is parked lands its kernel binaries in memory the trace
        # occupies. The loop is over and the trace is never replayed again in this generate(), so
        # dropping it makes those compiles safe; _vfy_captured goes back to False so a later
        # generate() on this decoder re-captures instead of replaying a released id.
        self.model.release_verify_trace()
        self._vfy_captured = False
        # Roll each user's durable GDN state out of the spec ring at its last accepted index, and
        # bring the K per-tap conv buffers back in step, so whatever runs next on this model (an
        # eager decode, a state snapshot) reads live state. One-off, outside the timed loop.
        # It runs for EVERY user, FROZEN ones included: a frozen user's mi is 0 and its ring slot 0
        # holds the drifted state its discarded rows produced, so what lands in its rec_state is
        # junk. That is fine here — its output is already final, GDN state is per-user so the junk
        # cannot reach a live user, and the next generate() re-prefills every user before anything
        # reads GDN state — but a finished user's post-loop GDN state is NOT resumable.
        for dn in self._gdn:
            dn.materialize_spec_state(mi)
            dn.sync_conv_taps()
        ttnn.deallocate(self._hp_buf)  # == Hp; the persistent anchor buffer, one per generate
        self._hp_buf = None
        ttnn.deallocate(self._anchor_sel)
        self._anchor_sel = None
        ttnn.synchronize_device(self.mesh)
        self.decode_time = time.perf_counter() - _t_decode  # spec loop wall-clock (excludes prefill)
        self._log_mean_timing()
        res = [o[:max_new_tokens] for o in out]
        return res[0] if flat_in else res

    def accept_rate(self):
        """Mean accepted DRAFT tokens per USER per iteration (0..K); tokens/iter/user is this + 1."""
        return self.total_accepted / max(1, self.user_iters)

    def stats(self):
        """Acceptance breakdown. Mean acceptance alone cannot distinguish 'the drafter is weak at
        depth 3' from 'the first draft keeps aborting the iteration', which need different fixes.

        Every rate is per USER-ITERATION (there is one accept decision per user per loop iteration),
        so the numbers are comparable across batch sizes; ``tokens_per_iter_total`` is what the batch
        commits per iteration and is the number throughput scales with.

        FINISHED USERS DO NOT COUNT. A user that has stopped is FROZEN: its T rows still replay
        (the row count is baked into the trace) but its accept result is discarded, so it feeds
        neither ``user_iters`` nor the histograms nor ``total_drafted`` / ``total_accepted``. Every
        rate below is therefore over USEFUL work only, whatever order the users finish in.
        ``iters`` stays the raw loop-iteration count, and ``mean_live_users`` (= user_iters / iters)
        says how full the batch was on average. So ``tokens_per_iter_total`` (= B x
        committed_per_iter) is the rate with the batch FULL — the number throughput scales with —
        while ``tokens_per_iter_delivered`` (= mean_live_users x committed_per_iter) is what the run
        actually emitted per iteration. The two are equal when no user finishes early."""
        n = max(1, self.user_iters)
        return {
            "iters": self.iters,
            "n_users": self.B,
            "K": self.K,
            "accept_rate": self.accept_rate(),
            "committed_per_iter": self.accept_rate() + 1.0,
            "tokens_per_iter_total": self.B * (self.accept_rate() + 1.0),
            # Mean users still generating per iteration (== B until the first user finishes) and
            # the delivered rate that follows from it; see the docstring.
            "mean_live_users": self.user_iters / max(1, self.iters),
            "tokens_per_iter_delivered": (self.user_iters / max(1, self.iters)) * (self.accept_rate() + 1.0),
            # depth_rate[j] = P(draft j accepted), cumulative: [0] >= [1] >= ...
            "depth_rate": [h / n for h in self.depth_hits],
            # conditional[j] = P(draft j accepted | drafts 0..j-1 accepted) — isolates per-depth
            # drafter quality from the compounding of earlier rejections.
            "conditional": [self.depth_hits[j] / max(1, self.depth_hits[j - 1] if j else n) for j in range(self.K)],
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
            f"[{prefix}] {s['iters']} iters, users={s['n_users']}, K={s['K']}: "
            f"accept={s['accept_rate']:.2f}/{s['K']} -> {s['committed_per_iter']:.2f} committed tokens/iter/user, "
            f"{s['tokens_per_iter_total']:.2f} tokens/iter total"
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
