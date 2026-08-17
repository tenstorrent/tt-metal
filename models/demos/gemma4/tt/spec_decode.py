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
        # Single FUSED batched (B>1) per-iteration trace (batched drafter chain +
        # batched packed verify). Captured once, replayed per iter (prefill never
        # traced). See _capture_fused_trace_batched / _generate_fused_traced_batched.
        self._fused_trace_batched = None
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
        # Packed-query verify: all K+1 candidates in the query-heads dim of ONE
        # batch=1 forward (one QKV/norm/RoPE over K+1 rows, one masked SDPA per
        # layer, loop-free staging KV write) instead of K+1 pseudo-users with
        # sequential per-candidate KV writes. This is the default multi-token
        # verify; single-token calls (seed/reseed) keep the plain batch=1 path.
        # (The fused single-trace paths keep the batch-dim verify.)
        # S_k (mask key length) is padded to this bucket so the verify trace
        # shape stays stable as the context grows; a new trace is captured when
        # the bucket rolls.
        self._pv_sk_bucket = 1024
        self._pv_ready = False
        self._pv_a_prev = -1  # last hot block index (-1 ⇒ staging unseeded)
        self._pv_traces = {}  # (P, S_k) -> persistent trace inputs/outputs

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
        # int64 host source for the uint32 tensor: ttnn downcasts it host-side, avoiding
        # the int32->uint32 C++ conversion that emits the #18536 row-major warning.
        pu = torch.zeros((1, 32), dtype=torch.int64)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int64)
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
        # int64 host source for the uint32 tensor (see _pos_tensors): avoids the
        # int32->uint32 conversion that triggers the #18536 row-major warning.
        t = torch.tensor(tokens, dtype=torch.int64).reshape(1, len(tokens))
        return ttnn.from_torch(
            t, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper
        )

    def _page_table(self, batch, user_idx=0):
        # Replicate ONE user's block row across the `batch` pseudo-users so every
        # candidate/verify position indexes the SAME physical KV blocks (the
        # batch-alias / single-user verify trick). ``user_idx`` selects which
        # real user's row to replicate (default 0).
        row = user_idx
        user_row = (
            self.page_table_torch[row : row + 1]
            if self.page_table_torch.dim() > 1
            else self.page_table_torch.unsqueeze(0)
        )
        pt = user_row.repeat(batch, 1).to(torch.int32)
        return ttnn.from_torch(
            pt, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=self._mapper
        )

    def _page_table_users(self, B):
        # Distinct per-user page-table rows [B, blocks] for a true B-user batched
        # forward (each user attends to its OWN physical KV blocks). Requires
        # page_table_torch to have >= B rows.
        pt = self.page_table_torch[:B].to(torch.int32)
        return ttnn.from_torch(
            pt, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=self._mapper
        )

    def _from_b(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, device=self.mesh_device, layout=layout, dtype=dtype, mesh_mapper=self._mapper)

    # ── host-side input tensors (for copy_host_to_device_tensor into traces) ──
    def _host_tokens(self, tokens):
        # int64 host source for the uint32 tensor (see _pos_tensors): avoids the
        # int32->uint32 conversion that triggers the #18536 row-major warning.
        t = torch.tensor(tokens, dtype=torch.int64).reshape(1, len(tokens))
        return ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)

    def _host_pos(self, positions):
        batch = len(positions)
        # int64 host source for the uint32 tensor (see _pos_tensors): avoids the
        # int32->uint32 conversion that triggers the #18536 row-major warning.
        pu = torch.zeros((1, 32), dtype=torch.int64)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int64)
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

    # ── packed-query verify ──────────────────────────────────────────────
    def _pv_setup(self):
        """Lazy one-time setup for the packed verify: allocate per-layer hot-
        block staging buffers (non-KV-shared layers only) and stash the shape
        constants the host-side index/mask builders need."""
        if self._pv_ready:
            return
        from models.demos.gemma4.tt.attention.kv_cache import PV_HOT_BLOCKS, init_kv_staging

        target = self.target
        # Paged cache shape: [max_blocks, nkv_local, block_size, head_dim].
        self._pv_bs = int(self.tt_kv_cache[0][0].padded_shape[2])
        self._pv_blk = PV_HOT_BLOCKS
        self._pv_s2 = self._pv_blk * self._pv_bs
        tp = self._tp
        first_cfg = target.layers[0].self_attn.config
        self._pv_h_local = first_cfg.num_attention_heads // tp
        self._pv_window = target.hf_config.sliding_window
        self._pv_nkv = {}  # layer_type -> nkv_local (staging/cache head count)
        for i, layer in enumerate(target.layers):
            cfg = layer.self_attn.config
            if cfg.cache_position_modulo is not None:
                raise NotImplementedError("packed verify does not support bounded sliding KV caches")
            if i in target.kv_shared_layer_map:
                continue  # shares source layer's cache+staging; skips KV writes
            layer.self_attn.kv_staging = init_kv_staging(
                self.mesh_device, cfg, max_batch_size=1, block_size=self._pv_bs, blk=self._pv_blk
            )
            lt = target.hf_config.layer_types[i]
            self._pv_nkv[lt] = int(layer.self_attn.kv_staging[0].shape[1])
        self._pv_pages = (self.page_table_torch[0] if self.page_table_torch.dim() > 1 else self.page_table_torch).to(
            torch.int64
        )
        self._pv_ready = True

    def _pv_seed_staging(self, c):
        """Seed every layer's staging block-slot 0 with the committed content of
        the hot block at position ``c`` (read from the cache — the only
        committed-cache read in the design, once per generation)."""
        bs, S2 = self._pv_bs, self._pv_s2
        a = c // bs
        self._pv_a_prev = a
        if c % bs == 0:
            return  # fresh block — zeros are fine
        blk_phys = int(self._pv_pages[a])
        for i, layer in enumerate(self.target.layers):
            if i in self.target.kv_shared_layer_map:
                continue
            staging = layer.self_attn.kv_staging
            cache = self.tt_kv_cache[i]
            for kv in (0, 1):
                stg = staging[kv]
                nkv, hd = stg.shape[1], stg.shape[3]
                block = ttnn.slice(cache[kv], [blk_phys, 0, 0, 0], [blk_phys + 1, nkv, bs, hd])
                tail = ttnn.slice(stg, [0, 0, bs, 0], [1, nkv, S2, hd])
                rebuilt = ttnn.concat([block, tail], dim=2)
                ttnn.assign(rebuilt, stg)
                for t in (rebuilt, block, tail):
                    ttnn.deallocate(t)

    def _pv_host_inputs(self, c, P):
        """Host tensors for one packed verify at anchor position ``c``.

        Returns dict of torch tensors: pos [1,P]u32, masks [1,1,H*P,S_k] bf16
        (full + sliding), embed_idx per layer type [1,nkv*S2]u32, hot_pt
        [1,BLK]i32, and S_k.
        """
        bs, S2, BLK = self._pv_bs, self._pv_s2, self._pv_blk
        a, off = c // bs, c % bs
        roll = 1 if a == self._pv_a_prev + 1 else 0
        H, W = self._pv_h_local, self._pv_window

        pos = torch.arange(c, c + P, dtype=torch.int32).reshape(1, P)

        # Additive masks, head-major rows h*P+p: causal upper bound c+p; sliding
        # adds the window lower bound. S_k is bucket-padded for trace stability.
        NEG = -1e9
        S_k = ((c + P + self._pv_sk_bucket - 1) // self._pv_sk_bucket) * self._pv_sk_bucket
        j = torch.arange(S_k)
        rows_full = torch.empty(P, S_k)
        rows_slide = torch.empty(P, S_k)
        for p in range(P):
            upper = c + p
            rows_full[p] = torch.where(j <= upper, 0.0, NEG)
            rows_slide[p] = torch.where((j <= upper) & (j > upper - W), 0.0, NEG)
        mask_full = rows_full.repeat(H, 1).reshape(1, 1, H * P, S_k).to(torch.bfloat16)
        mask_slide = rows_slide.repeat(H, 1).reshape(1, 1, H * P, S_k).to(torch.bfloat16)

        # merge_idx over staging positions: committed prefix from staging
        # (identity, or +bs on a rollover — the prefix came from the spill
        # block), the P new rows from new_seq (concat index S2+p), stale tail
        # identity. embed_idx bakes the per-head flattened row offset in
        # (new_seq is padded to 32 rows in packed_decode_forward).
        m = torch.arange(S2, dtype=torch.int64)
        if roll:
            m[:off] += bs
        m[off : off + P] = S2 + torch.arange(P)
        p_pad = ((P + 31) // 32) * 32
        src_seq = S2 + p_pad
        embed = {}
        for lt, nkv in self._pv_nkv.items():
            off_h = (torch.arange(nkv, dtype=torch.int64) * src_seq).unsqueeze(1)
            embed[lt] = (m.unsqueeze(0) + off_h).reshape(1, nkv * S2).to(torch.int32)

        hot = torch.full((1, BLK), -1, dtype=torch.int32)
        hot[0, 0] = int(self._pv_pages[a])
        if off + P > bs:
            hot[0, 1] = int(self._pv_pages[a + 1])

        return {"pos": pos, "mask_full": mask_full, "mask_slide": mask_slide, "embed": embed, "hot": hot, "S_k": S_k}

    def _pv_from_torch(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
        return ttnn.from_torch(
            t,
            device=self.mesh_device if device else None,
            layout=layout,
            dtype=dtype,
            mesh_mapper=self._mapper,
        )

    def _pv_device_inputs(self, tokens, h):
        """Device tensors for one packed verify from host dict ``h``."""
        return {
            "x": self._tokens_tensor(tokens),
            "pos": self._pv_from_torch(h["pos"], ttnn.uint32),
            "mask_full": self._pv_from_torch(h["mask_full"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "mask_slide": self._pv_from_torch(h["mask_slide"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "embed": {lt: self._pv_from_torch(e, ttnn.uint32) for lt, e in h["embed"].items()},
            "hot": self._pv_from_torch(h["hot"], ttnn.int32),
        }

    def _pv_call(self, dev, P):
        return self.target.ttnn_packed_verify_forward(
            x=dev["x"],
            position_idx=dev["pos"],
            attn_mask_full=dev["mask_full"],
            attn_mask_sliding=dev["mask_slide"],
            packed_p=P,
            page_table=dev.get("pt"),
            kv_cache=self.tt_kv_cache,
            embed_idx_full=dev["embed"].get("full_attention"),
            embed_idx_sliding=dev["embed"].get("sliding_attention"),
            hot_pt=dev["hot"],
        )

    def _verify_packed(self, tokens, positions):
        """Packed verify (eager). Same contract as ``_verify``."""
        self._pv_setup()
        P = len(tokens)
        c = positions[0]
        if self._pv_a_prev < 0:
            self._pv_seed_staging(c)
        h = self._pv_host_inputs(c, P)
        dev = self._pv_device_inputs(tokens, h)
        dev["pt"] = self._page_table(1)
        logits, hidden = self._pv_call(dev, P)
        self._pv_a_prev = c // self._pv_bs
        lh = self._logits_to_host(logits).reshape(P, -1)
        logits.deallocate(True)
        for t in (dev["x"], dev["pos"], dev["mask_full"], dev["mask_slide"], dev["hot"], dev["pt"]):
            t.deallocate(True)
        for e in dev["embed"].values():
            e.deallocate(True)
        return lh, hidden

    def _verify_packed_traced(self, tokens, positions):
        """Packed verify via trace, keyed by (P, S_k bucket). Persistent device
        inputs are refreshed per step via copy_host_to_device_tensor; a new
        trace is captured lazily when P or the S_k bucket changes."""
        self._pv_setup()
        P = len(tokens)
        c = positions[0]
        if self._pv_a_prev < 0:
            self._pv_seed_staging(c)
        h = self._pv_host_inputs(c, P)
        key = (P, h["S_k"])
        tr = self._pv_traces.get(key)
        if tr is None:
            dev = self._pv_device_inputs(tokens, h)
            dev["pt"] = self._page_table(1)
            # Compile run (warm program cache), then capture. Both runs write
            # the SAME tokens to the SAME positions, so KV writes are idempotent.
            logits, hidden = self._pv_call(dev, P)
            ttnn.synchronize_device(self.mesh_device)
            logits.deallocate(True)
            hidden.deallocate(True)
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            logits, hidden = self._pv_call(dev, P)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            dev.update({"id": tid, "logits": logits, "hidden": hidden})
            self._pv_traces[key] = dev
        else:
            ttnn.copy_host_to_device_tensor(self._host_tokens(tokens), tr["x"])
            for src, dst in (
                (self._pv_from_torch(h["pos"], ttnn.uint32, device=False), tr["pos"]),
                (
                    self._pv_from_torch(h["mask_full"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=False),
                    tr["mask_full"],
                ),
                (
                    self._pv_from_torch(h["mask_slide"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=False),
                    tr["mask_slide"],
                ),
                (self._pv_from_torch(h["hot"], ttnn.int32, device=False), tr["hot"]),
            ):
                ttnn.copy_host_to_device_tensor(src, dst)
            for lt, e in h["embed"].items():
                ttnn.copy_host_to_device_tensor(self._pv_from_torch(e, ttnn.uint32, device=False), tr["embed"][lt])
            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)
        tr = self._pv_traces[key]
        self._pv_a_prev = c // self._pv_bs
        lh = self._logits_to_host(tr["logits"]).reshape(P, -1)
        # Persistent hidden — caller must consume before the next replay.
        return lh, tr["hidden"]

    # ── target forwards ───────────────────────────────────────────────────
    def _verify(self, tokens, positions):
        """Batched verify. Returns (logits_host [B,vocab], hidden_device [1,1,B,h])."""
        # Packed-query verify: all K+1 candidates in one batch=1 pass (positions
        # packed into the query-heads dim, loop-free staging KV write).
        # Single-token calls (seed/reseed) keep the plain verify.
        if len(tokens) > 1:
            if self._use_trace:
                return self._verify_packed_traced(tokens, positions)
            return self._verify_packed(tokens, positions)
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
    def _draft(self, anchor_token, anchor_hidden, anchor_pos, temperature=0.0, top_p=1.0, top_k=0, user_idx=0):
        """Produce K draft tokens + their logits (host). anchor_hidden stays on device.

        Greedy (temperature<=0): the next draft is argmax(q). Sampling: the next
        draft is SAMPLED from the drafter distribution q under the SAME
        temp/top_p/top_k as the target — required for speculative sampling
        correctness AND acceptance (proposing argmax(q) maximizes q(d), which
        drives the accept ratio min(1, p(d)/q(d)) to ~0).

        ``user_idx`` selects which user's KV blocks the drafter cross-attends to
        (batched generate loops the drafter at batch=1 over each user)."""
        K = self.draft_len
        greedy = not temperature or temperature <= 0
        pos_u, pos_i = self._pos_tensors([anchor_pos])
        # Drafter page table: batch=1, this user's blocks for both layer types.
        pt = self._page_table(1, user_idx=user_idx)
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
        (batch) dim is EXACTLY one tile (32) — verified by a correctness probe
        (1/5 rows -> wrong; padded to 32 -> exact; and >32 rows in one call also
        returns garbage beyond the first tile). So process the rows in 32-row
        chunks: pad each ≤32-row chunk up to 32, run the multicore untilize +
        argmax, slice back, and concat. Net ~1.6 ms/chunk vs ~9-28 ms bare.
        """
        R32 = 32
        if rows > R32:
            # Batched packed verify (B*P > 32): argmax each 32-row tile separately
            # (offsets are multiples of 32, so the TILE slices are tile-aligned)
            # and concat — a single >32-row multicore argmax garbles later tiles.
            vocab = logits.shape[-1]
            chunks = []
            off = 0
            while off < rows:
                n = min(R32, rows - off)
                part = ttnn.slice(logits, [0, 0, off, 0], [1, 1, off + n, vocab])
                chunks.append(self._argmax_last(part, n))  # recurse (≤32 rows)
                part.deallocate(True)
                off += n
            out = ttnn.concat(chunks, dim=2)  # [1,1,rows]
            for c in chunks:
                c.deallocate(True)
            return out
        src = logits
        padded = None
        if rows < R32:
            padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, R32 - rows), (0, 0)], value=0.0)
            src = padded
        u = ttnn.untilize(src, use_multicore=True)
        if padded is not None:
            padded.deallocate(True)
        idx = ttnn.argmax(
            u, dim=-1, keepdim=False
        )  # [1,1,32 or rows] uint32 RM (multicore by default on ROW_MAJOR last-dim)
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
        self._pv_a_prev = -1  # re-seed packed-verify staging for the new anchor/request
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

    # ── batched (B>1) speculative decode ─────────────────────────────────────
    def _seed_batched(self, tokens, positions):
        """Batched drafter seed: one batch=B target verify of each user's anchor
        token at its position (idempotent re-write of the prompt's last KV).
        Returns the post-norm hidden [1,1,B,backbone]; slice row b for user b's
        recurrent drafter seed."""
        x = self._tokens_tensor(tokens)  # [1,B]
        pu, pi = self._pos_tensors(positions)  # pu [1,32] (B filled), pi [B]
        pt = self._page_table_users(len(tokens))  # [B, blocks] distinct
        logits, hidden = self.target.ttnn_verify_forward(
            x=x, current_pos=pu, current_pos_cache=pi, page_table=pt, kv_cache=self.tt_kv_cache
        )
        logits.deallocate(True)
        for t in (x, pu, pi, pt):
            t.deallocate(True)
        return hidden  # [1,1,B,backbone]

    def _packed_H(self):
        """Local (per-device) query-head count for the packed additive mask."""
        return self.target.layers[0].self_attn.config.num_attention_heads // self._tp

    def _packed_mask_host(self, cs, B, P, H, window, S_k):
        """Host bf16 additive masks [B,1,H*P,S_k] for the packed verify (head-major
        rows h*P+p; per-user causal bound c_b+p, + sliding-window lower bound)."""
        NEG = -1e9
        j = torch.arange(S_k)
        mf = torch.empty(B, 1, H * P, S_k)
        ms = torch.empty(B, 1, H * P, S_k)
        for b in range(B):
            rf = torch.empty(P, S_k)
            rs = torch.empty(P, S_k)
            for p in range(P):
                upper = cs[b] + p
                rf[p] = torch.where(j <= upper, 0.0, NEG)
                rs[p] = torch.where((j <= upper) & (j > upper - window), 0.0, NEG)
            mf[b, 0] = rf.repeat(H, 1)
            ms[b, 0] = rs.repeat(H, 1)
        return mf.to(torch.bfloat16), ms.to(torch.bfloat16)

    def _verify_packed_batched(self, tokens_b, cs):
        """Packed verify for B users at PER-USER anchor positions (ragged).

        Folds each user's P=K+1 candidates into the query-head dim of a batch=B
        forward — B KV reads total (vs B*(K+1) for the batch-alias verify). Uses
        the fallback per-position ``paged_update_cache`` write (no staging), so
        no per-user staging is needed.

        Args:
            tokens_b: list[B] of P-length token-id lists ([anchor, d1..dK]).
            cs:       list[B] of anchor positions c_b (user b verifies c_b..c_b+K).

        Returns:
            (logits_host [B*P, vocab], vhidden device [1,1,B*P,backbone]); row
            u*P+p is user u's p-th packed position.
        """
        target = self.target
        B = len(tokens_b)
        P = len(tokens_b[0])
        H = self._packed_H()
        window = target.hf_config.sliding_window
        max_c = max(cs)
        # S_k (mask key length): cover positions 0..max_c+K, multiple of 64.
        S_k = ((max_c + P + 63) // 64) * 64

        x = self._from_b(
            torch.tensor([tok for toks in tokens_b for tok in toks], dtype=torch.int64).reshape(1, B * P), ttnn.uint32
        )
        position_idx = self._from_b(
            torch.tensor([[cs[b] + p for b in range(B) for p in range(P)]], dtype=torch.int64), ttnn.uint32
        )
        mf, ms = self._packed_mask_host(cs, B, P, H, window, S_k)
        mask_full = self._from_b(mf, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        mask_slide = self._from_b(ms, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        write_idxs = [
            self._from_b(torch.tensor([cs[b] + p for b in range(B)], dtype=torch.int32), ttnn.int32) for p in range(P)
        ]
        pt = self._page_table_users(B)
        logits, vhidden = target.ttnn_packed_verify_forward(
            x=x,
            position_idx=position_idx,
            attn_mask_full=mask_full,
            attn_mask_sliding=mask_slide,
            packed_p=P,
            page_table=pt,
            kv_cache=self.tt_kv_cache,
            kv_write_idxs=write_idxs,
            embed_idx_full=None,
            embed_idx_sliding=None,
            hot_pt=None,
        )
        lh = self._logits_to_host(logits).reshape(B * P, -1)
        logits.deallocate(True)
        for t in (x, position_idx, mask_full, mask_slide, pt, *write_idxs):
            t.deallocate(True)
        return lh, vhidden

    def _draft_batched(self, anchor_tokens, anchor_hidden_b, anchor_positions):
        """Batched drafter: propose K tokens for ALL B users in ONE batch=B chain.

        Runs K batched drafter steps (each ``assistant.step`` at batch=B, the
        drafter cross-attending into every user's own KV via distinct page-table
        rows). The recurrent hidden [1,1,B,backbone] is carried across steps; the
        per-step argmax over B rows is read to host to feed the next step. This
        amortizes the (tiny) drafter over B users instead of B sequential batch=1
        loops — the key to batched throughput.

        Args:
            anchor_tokens: list[B] committed token ids to draft from.
            anchor_hidden_b: [1,1,B,backbone] batched drafter seed (borrowed).
            anchor_positions: list[B] anchor positions (fixed across the K steps).

        Returns:
            drafts_b: list[B] of K-length draft-token lists.
        """
        K = self.draft_len
        B = len(anchor_tokens)
        pos_u, pos_i = self._pos_tensors(anchor_positions)  # pu [1,32] (B filled), pi [B]
        pt = self._page_table_users(B)
        page_tables = {lt: pt for lt in self._shared_kv}

        drafts_b = [[] for _ in range(B)]
        tok_tt = self._tokens_tensor(anchor_tokens)  # [1,B]
        h = anchor_hidden_b
        owns_h = False
        for _ in range(K):
            logits, h_next = self.assistant.step(tok_tt, h, self._shared_kv, page_tables, pos_u, pos_i)
            if owns_h:
                h.deallocate(True)
            h, owns_h = h_next, True
            lh = self._logits_to_host(logits).reshape(B, -1)  # [B, vocab]
            logits.deallocate(True)
            toks = [int(torch.argmax(lh[b])) for b in range(B)]
            for b in range(B):
                drafts_b[b].append(toks[b])
            tok_tt.deallocate(True)
            tok_tt = self._tokens_tensor(toks)  # [1,B]
        if owns_h:
            h.deallocate(True)
        tok_tt.deallocate(True)
        pos_u.deallocate(True)
        pos_i.deallocate(True)
        pt.deallocate(True)
        return drafts_b

    # ── fused batched trace (greedy) ─────────────────────────────────────────
    def _fused_body_batched(self, tr):
        """Fused batched iteration op-graph: K batched drafter steps chained on
        device (argmax + re-embed in-graph), user-major packed-verify input
        assembled on device, ONE batched packed verify, on-device argmax. Returns
        the persistent output handles (verify_x [1,B*P], vidx [1,1,B*P], vhidden
        [1,1,B*P,backbone])."""
        K = self.draft_len
        B = tr["B"]
        P = K + 1
        page_tables = {lt: tr["d_pt"] for lt in self._shared_kv}
        tok = tr["anchor_tok"]  # [1,B]
        h = tr["h"]  # [1,1,B,backbone]
        draft_cols = []
        for _ in range(K):
            logits, h = self.assistant.step(tok, h, self._shared_kv, page_tables, tr["d_pu"], tr["d_pi"])
            idx = self._argmax_last(logits, rows=B)  # [1,1,B] uint32 RM
            logits.deallocate(True)
            tok = ttnn.reshape(idx, (1, B))
            draft_cols.append(tok)
        # Assemble user-major x [1,B*P] (row u*P+p) from the P columns [1,B]:
        # stack -> [1,1,P,B] -> permute -> [1,1,B,P] -> flatten.
        cols4d = [ttnn.reshape(tr["anchor_tok"], (1, 1, 1, B))] + [ttnn.reshape(c, (1, 1, 1, B)) for c in draft_cols]
        stacked = ttnn.concat(cols4d, dim=2)  # [1,1,P,B]
        um = ttnn.permute(stacked, (0, 1, 3, 2))  # [1,1,B,P]
        stacked.deallocate(True)
        verify_x = ttnn.reshape(um, (1, B * P))
        um.deallocate(True)
        vlogits, vhidden = self.target.ttnn_packed_verify_forward(
            x=verify_x,
            position_idx=tr["v_pos"],
            attn_mask_full=tr["mask_full"],
            attn_mask_sliding=tr["mask_slide"],
            packed_p=P,
            page_table=tr["v_pt"],
            kv_cache=self.tt_kv_cache,
            kv_write_idxs=tr["write_idxs"],
            embed_idx_full=None,
            embed_idx_sliding=None,
            hot_pt=None,
        )
        vidx = self._argmax_last(vlogits, rows=B * P)  # [1,1,B*P] uint32 RM
        vlogits.deallocate(True)
        return verify_x, vidx, vhidden

    def _capture_fused_trace_batched(self, anchor_tokens, seed_hidden_b, anchor_positions, S_k):
        """Capture ONE fused batched iteration over persistent buffers."""
        from loguru import logger as _lg

        K = self.draft_len
        B = len(anchor_tokens)
        P = K + 1
        H = self._packed_H()
        window = self.target.hf_config.sliding_window
        cs = list(anchor_positions)
        d_pu, d_pi = self._pos_tensors(cs)
        v_pos_t = torch.tensor([[cs[b] + p for b in range(B) for p in range(P)]], dtype=torch.int64)
        mf, ms = self._packed_mask_host(cs, B, P, H, window, S_k)
        tr = {
            "B": B,
            "P": P,
            "H": H,
            "window": window,
            "S_k": S_k,
            "anchor_tok": self._tokens_tensor(list(anchor_tokens)),  # [1,B]
            "h": ttnn.clone(seed_hidden_b),  # [1,1,B,backbone]
            "d_pu": d_pu,
            "d_pi": d_pi,
            "d_pt": self._page_table_users(B),
            "v_pos": self._from_b(v_pos_t, ttnn.uint32),
            "mask_full": self._from_b(mf, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "mask_slide": self._from_b(ms, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "v_pt": self._page_table_users(B),
            "write_idxs": [
                self._from_b(torch.tensor([cs[b] + p for b in range(B)], dtype=torch.int32), ttnn.int32)
                for p in range(P)
            ],
        }
        _lg.info(f"[spec-trace] capture batched fused: compile run (B={B}, S_k={S_k})")
        vx, vidx, vh = self._fused_body_batched(tr)
        ttnn.synchronize_device(self.mesh_device)
        vx.deallocate(True)
        vidx.deallocate(True)
        vh.deallocate(True)
        _lg.info("[spec-trace] capture batched fused: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        vx, vidx, vh = self._fused_body_batched(tr)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info("[spec-trace] capture batched fused: DONE")
        tr["id"] = tid
        tr["verify_x"] = vx
        tr["vidx"] = vidx
        tr["vhidden"] = vh
        self._fused_trace_batched = tr

    def _hidden_rows_to_device_batched(self, rows_b):
        """Shift-seed: gather packed-verify hidden row ``b*P+rows_b[b]`` per user
        into the persistent batched recurrent buffer tr["h"] (host round-trip)."""
        tr = self._fused_trace_batched
        B, P = tr["B"], tr["P"]
        vh = tr["vhidden"]
        vh_t = ttnn.to_torch(ttnn.get_device_tensors(vh)[0]) if self._tp > 1 else ttnn.to_torch(vh)
        sel = torch.cat([vh_t[:, :, b * P + rows_b[b] : b * P + rows_b[b] + 1, :] for b in range(B)], dim=2)
        host_h = ttnn.from_torch(
            sel.contiguous(), layout=ttnn.TILE_LAYOUT, dtype=tr["h"].dtype, mesh_mapper=self._mapper
        )
        ttnn.copy_host_to_device_tensor(host_h, tr["h"])
        host_h.deallocate(True)

    def _generate_fused_traced_batched(self, anchor_tokens, anchor_positions, max_new_tokens, max_seq_len):
        """Traced batched greedy spec-decode: capture one fused batched iteration
        (drafter chain + packed verify), then replay once per iteration, updating
        only host->device inputs (tokens, positions, per-user masks, KV-write
        indices, shift seed). Prefill is NOT traced (stays eager)."""
        B = len(anchor_tokens)
        K = self.draft_len
        P = K + 1
        S_k = ((max_seq_len + 63) // 64) * 64

        setup_t0 = time.perf_counter()
        # Eager (untraced) batched seed, then capture (mirrors the single-user path:
        # an active verify trace would collide with the fused capture allocations).
        self._use_trace = False
        seed_h = self._seed_batched(list(anchor_tokens), list(anchor_positions))
        self._use_trace = True
        self._capture_fused_trace_batched(anchor_tokens, seed_h, anchor_positions, S_k)
        seed_h.deallocate(True)
        tr = self._fused_trace_batched
        self._last_fused_setup_s = time.perf_counter() - setup_t0

        toks = list(anchor_tokens)
        pos = list(anchor_positions)
        outs = [[] for _ in range(B)]
        accepts = [[] for _ in range(B)]
        done = [False] * B
        pos_cap = max_seq_len - P
        H = tr["H"]
        window = tr["window"]

        first = True
        replay_t0 = time.perf_counter()
        while not all(done):
            if not first:
                cs = list(pos)
                h_tok = self._host_tokens(toks)
                ttnn.copy_host_to_device_tensor(h_tok, tr["anchor_tok"])
                h_tok.deallocate(True)
                d_hpu, d_hpi = self._host_pos(cs)
                ttnn.copy_host_to_device_tensor(d_hpu, tr["d_pu"])
                ttnn.copy_host_to_device_tensor(d_hpi, tr["d_pi"])
                d_hpu.deallocate(True)
                d_hpi.deallocate(True)
                vpos = ttnn.from_torch(
                    torch.tensor([[cs[b] + p for b in range(B) for p in range(P)]], dtype=torch.int64),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    mesh_mapper=self._mapper,
                )
                ttnn.copy_host_to_device_tensor(vpos, tr["v_pos"])
                vpos.deallocate(True)
                mf, ms = self._packed_mask_host(cs, B, P, H, window, tr["S_k"])
                hmf = ttnn.from_torch(mf, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=self._mapper)
                hms = ttnn.from_torch(ms, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=self._mapper)
                ttnn.copy_host_to_device_tensor(hmf, tr["mask_full"])
                ttnn.copy_host_to_device_tensor(hms, tr["mask_slide"])
                hmf.deallocate(True)
                hms.deallocate(True)
                for p in range(P):
                    wi = ttnn.from_torch(
                        torch.tensor([cs[b] + p for b in range(B)], dtype=torch.int32),
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        dtype=ttnn.int32,
                        mesh_mapper=self._mapper,
                    )
                    ttnn.copy_host_to_device_tensor(wi, tr["write_idxs"][p])
                    wi.deallocate(True)
            first = False

            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)

            vx = (
                ttnn.to_torch(ttnn.get_device_tensors(tr["verify_x"])[0])
                if self._tp > 1
                else ttnn.to_torch(tr["verify_x"])
            ).reshape(-1)
            gids = self._ids_to_host(tr["vidx"], B * P)

            rows_b = []
            for b in range(B):
                drafts = [int(vx[b * P + 1 + j]) for j in range(K)]
                g = [gids[b * P + j] for j in range(P)]
                m = next((i for i in range(K) if drafts[i] != g[i]), K)
                committed = drafts[:m] + [g[m]]
                rows_b.append(min(m + 1, K))
                if done[b]:
                    continue
                accepts[b].append(m)
                for tok in committed:
                    outs[b].append(tok)
                    if tok in self.stop_tokens or len(outs[b]) >= max_new_tokens:
                        done[b] = True
                        break
                pos[b] = pos[b] + m + 1
                toks[b] = committed[-1]
                if pos[b] >= pos_cap:
                    done[b] = True

            if not all(done):
                self._hidden_rows_to_device_batched(rows_b)

        self._last_fused_replay_s = time.perf_counter() - replay_t0
        return outs, accepts

    def generate_batched(self, anchor_tokens, anchor_positions, max_new_tokens, max_seq_len, temperature=0.0):
        """Greedy batched speculative decode for B independent users (ragged).

        Each iteration drafts ALL B users in one batch=B drafter chain, runs ONE
        batched packed verify over all B users (the KV-bandwidth win), and accepts
        per user — so users advance by their own matched-prefix+1 and their
        positions diverge. With ``self._use_trace`` the whole iteration is ONE
        replayed metal trace (prefill is never traced); otherwise it runs eager.

        Returns (outs: list[B] of generated token-id lists, accepts: list[B] of
        per-iteration accept counts).
        """
        if temperature and temperature > 0:
            raise NotImplementedError("batched spec-decode supports greedy only (temperature<=0)")
        if self._use_trace:
            return self._generate_fused_traced_batched(anchor_tokens, anchor_positions, max_new_tokens, max_seq_len)
        B = len(anchor_tokens)
        K = self.draft_len
        P = K + 1
        toks = list(anchor_tokens)
        pos = list(anchor_positions)
        outs = [[] for _ in range(B)]
        accepts = [[] for _ in range(B)]
        done = [False] * B
        # furthest position a verify touches is pos[b]+K; keep it in range.
        pos_cap = max_seq_len - P

        # Per-user recurrent drafter seed (clone immediately: _seed_batched's
        # hidden may alias a reused verify-output buffer).
        seed_h = self._seed_batched(toks, pos)  # [1,1,B,backbone]
        backbone = seed_h.shape[-1]
        anchor_h = [ttnn.clone(ttnn.slice(seed_h, [0, 0, b, 0], [1, 1, b + 1, backbone])) for b in range(B)]
        seed_h.deallocate(True)

        _draft_mode = os.environ.get("GEMMA4_SPEC_DRAFT_MODE", "batched")

        while not all(done):
            if _draft_mode == "loop":
                drafts_b = [self._draft(toks[b], anchor_h[b], pos[b], temperature=0.0, user_idx=b)[0] for b in range(B)]
            else:
                anchor_hb = ttnn.concat(anchor_h, dim=2)  # [1,1,B,backbone]
                drafts_b = self._draft_batched(toks, anchor_hb, pos)
                anchor_hb.deallocate(True)

            tokens_b = [[toks[b]] + drafts_b[b] for b in range(B)]
            vlogits, vhidden = self._verify_packed_batched(tokens_b, [pos[b] for b in range(B)])

            for b in range(B):
                row_logits = [vlogits[b * P + j] for j in range(P)]
                m, committed = self._accept_greedy(drafts_b[b], row_logits)
                # next drafter seed (shift): position-aligned hidden row min(m+1,K).
                row = min(m + 1, K)
                new_h = ttnn.clone(ttnn.slice(vhidden, [0, 0, b * P + row, 0], [1, 1, b * P + row + 1, backbone]))
                anchor_h[b].deallocate(True)
                anchor_h[b] = new_h
                if done[b]:
                    continue
                accepts[b].append(m)
                for tok in committed:
                    outs[b].append(tok)
                    if tok in self.stop_tokens or len(outs[b]) >= max_new_tokens:
                        done[b] = True
                        break
                pos[b] = pos[b] + m + 1
                toks[b] = committed[-1]
                if pos[b] >= pos_cap:
                    done[b] = True
            vhidden.deallocate(True)

        for b in range(B):
            anchor_h[b].deallocate(True)
        return outs, accepts

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
        self._pv_a_prev = -1  # re-seed packed-verify staging for the new anchor/request
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
