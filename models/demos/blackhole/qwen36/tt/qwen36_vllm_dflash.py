# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.x on Blackhole served by vLLM WITH the DFlash2 speculative drafter, for up to B_max concurrent
requests (one multi-user speculative step per vLLM decode step).

vLLM's own speculative_config stays unset (the TT plugin rejects it); speculation is model-internal
through the plugin's ADAPTIVE BLOCK-OUTPUT contract, extended by ``tt_adaptive_block_batched``:

  * prefill emits one host-sampled anchor token per request (width 1), as before;
  * EVERY decode step runs the multi-user DFlash2 draft/verify loop INSIDE decode_forward until every
    live request has QWEN36_DFLASH_SERVE_BLOCK committed tokens, and returns them as one host [B, W]
    block (EOS-filled at a genuine stop; vLLM trims at the first stop token; surplus committed tokens
    are carried into the next step);
  * every text prompt speculates (no prompt-length frontier): a long prompt takes the eager chunked
    spec prefill, so the plain decode / chunk-prefill TRACES never run once the spec traces are
    parked (a spec replay after them hangs the device -- see DFlash2ServingDecoder);
  * the scheduler reserves W placeholders + KV lookahead for every request of a decode step.

Greedy: each request's tokens are the target's greedy trajectory from its sampled anchor (lossless,
tested per request in tests/test_dflash2_serving.py). max_num_seqs may be 1..4 (B_max x (K+1) verify
rows must fit one 32-row decode tile; K = 7 for DFlash2). QWEN36_DRAFTER=mtp (or
QWEN36_DFLASH_SERVE_BLOCK=1) turns speculation off and this class serves exactly like
Qwen36ForCausalLM (plain batched decode), so one bundle covers both profiles.

Slots: the plugin keys a request's device state by a "state slot" and may permute decode rows
(``slot_remap``: row i reads slot remap[i]); the base class moves its GDN state accordingly. The
speculative session state (GDN ring, drafter ring, tables) is NEVER moved: this class keeps a
row -> physical-session indirection and composes the plugin's remaps into it.

Warmup (server start): after the plain traces are captured, every spec buffer is allocated and every
spec program compiled (eager tap-capturing prefill for each mask bucket into a slot, the chunked spec
prefill shapes, the per-slot seed, the B*block-row draft and extend), then the verify trace is
captured ONCE and a dummy session per slot captures the drafter's draft/extend traces.
"""
import json
import math
import os
import time

import torch
from loguru import logger

# Speculation needs the EAGER masked-bucket prefill (its python-side tap clones do not run inside the
# traced bucket replay), so the bucket-trace gate must be off before model_config's defaults apply.
_SPEC_ON = (
    os.environ.get("QWEN36_DRAFTER", "dflash2") == "dflash2"
    and int(os.environ.get("QWEN36_DFLASH_SERVE_BLOCK", "32")) > 1
)
if _SPEC_ON:
    os.environ["QWEN36_PREFILL_BUCKET_TRACE"] = "0"

from vllm.model_executor.models.qwen3_5 import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor  # noqa: E402
from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.tt.dflash2_decode import default_draft_len  # noqa: E402
from models.demos.blackhole.qwen36.tt.dflash2_serving import (  # noqa: E402
    DFlash2ServingDecoder,
    dummy_prompt,
    serve_block_size,
)
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, TT_Qwen3_5ProcessingInfo  # noqa: E402

_W = serve_block_size() if _SPEC_ON else 1
# Prompts up to one chunk minus one token take the single eager MASKED-BUCKET prefill; longer prompts
# the eager CHUNKED spec prefill. Both go through prefill_for_spec and both capture the drafter taps.
_PREFILL_CHUNK = 2048
# Every DFlash checkpoint drafts at most block-1 <= 15 tokens; the KV lookahead only has to bound the
# verify's reach past the block (W committed positions + K+1 candidate rows).
_MAX_DRAFT = 15
_DEBUG = os.environ.get("QWEN36_DFLASH_DEBUG", "0") == "1"


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36DFlashForCausalLM(Qwen36ForCausalLM):
    """Qwen36ForCausalLM + model-internal multi-user DFlash2 speculation on decode steps (see module doc)."""

    model_capabilities = {
        **Qwen36ForCausalLM.model_capabilities,
        # A decode step commits exactly _W tokens per request (EOS-filled at a stop).
        "output_tokens_per_step": _W,
        # Block on decode steps; prefill anchors are plain width-1 rows.
        "tt_adaptive_block_output": _W > 1,
        # ...for EVERY request of the step (one multi-user speculative step), not only when solo.
        "tt_adaptive_block_batched": _W > 1,
        # EVERY text prompt speculates (0 = no prompt-length frontier): the plain decode / chunk-prefill
        # traces never run on the request path (a spec replay after them hangs the device).
        "tt_adaptive_block_max_prompt_tokens": 0,
        # The block step writes the W committed positions AND the last verify's K+1 candidate rows into
        # the paged KV inside one step: have the scheduler allocate that reach up front.
        "tt_block_output_kv_lookahead_tokens": (_W + _MAX_DRAFT + 1) if _W > 1 else 0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec = None  # DFlash2ServingDecoder, armed at warmup (phase 2)
        self._spec_pre = None  # prepared (allocated + compiled) decoder between warmup phases
        self._in_warmup = False
        model = self.model[0]
        self._eos = self._load_eos_ids(model)
        self._eos_fill = min(self._eos)
        B = int(model.args.max_batch_size)
        self._B = B
        # row -> physical session slot (the plugin's slot_remap is composed into it; nothing moves on device)
        self._phys = list(range(B))
        self._pending = [None] * B  # (prompt_len, page_table_row) from a slot's prefill, until its first decode
        self._carry = [[] for _ in range(B)]  # committed-but-unemitted tokens per physical slot
        self._stopped = [False] * B  # a stop token was committed; the row is EOS-filled from there
        self._prev_tail = [None] * B
        self._anchor_warned = False
        self._oov_warned = False
        self._nosession_warned = False
        if _W > 1:
            if model.num_devices <= 1:
                raise RuntimeError("Qwen36DFlash speculative serving needs the TP mesh (MESH_DEVICE=P150x4)")
            K = default_draft_len()
            if B * (K + 1) > 32:
                raise RuntimeError(
                    f"Qwen36DFlash speculative serving: max_num_seqs={B} x (K+1)={K + 1} verify rows exceed one "
                    f"32-row decode tile; launch with --max-num-seqs <= {32 // (K + 1)} (or QWEN36_DRAFTER=mtp)"
                )
            logger.info(
                f"Qwen36DFlash serving: slots={B} block W={_W} tokens/step, K={K}, eos={sorted(self._eos)}, "
                f"drafter={os.environ.get('DFLASH_WEIGHTS')}"
            )
        else:
            logger.info("Qwen36DFlash serving: speculation OFF (plain Qwen36ForCausalLM behaviour)")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_eos_ids(model):
        ids = set()
        ckpt = model.args.CKPT_DIR
        for name in ("generation_config.json", "config.json"):
            try:
                with open(os.path.join(ckpt, name)) as f:
                    v = json.load(f).get("eos_token_id")
            except Exception:
                v = None
            if isinstance(v, int):
                ids.add(int(v))
            elif isinstance(v, (list, tuple)):
                ids.update(int(x) for x in v)
        hf_eos = getattr(getattr(model.args, "hf_config", None), "eos_token_id", None)
        if isinstance(hf_eos, int):
            ids.add(int(hf_eos))
        elif isinstance(hf_eos, (list, tuple)):
            ids.update(int(x) for x in hf_eos)
        if not ids:
            raise RuntimeError("no eos_token_id found for the served checkpoint")
        return ids

    def _fill_block(self, block):
        """Exactly _W ids: a short block happens only at a genuine stop -> EOS-fill (the scheduler trims at
        the first stop token)."""
        block = list(block)[:_W]
        vocab = self.model[0].vocab_size
        oov = [t for t in block if not 0 <= t < vocab]
        if oov:
            if not self._oov_warned:
                self._oov_warned = True
                logger.warning(
                    f"Qwen36DFlash: {len(oov)} out-of-vocab committed id(s) (first={oov[0]}); substituting EOS"
                )
            block = [t if 0 <= t < vocab else self._eos_fill for t in block]
        out = torch.full((_W,), self._eos_fill, dtype=torch.int32)
        if block:
            out[: len(block)] = torch.tensor(block, dtype=torch.int32)
        return out

    @staticmethod
    def _spec_pref_pt(model, pt):
        """A page-table row padded/trimmed to the width the (batched, B=1-scratch) prefill programs were
        compiled with: the chunk buffer's width when the chunk-prefill trace exists, else as given."""
        pt = torch.as_tensor(pt).reshape(1, -1).to(torch.int32)
        buf = getattr(model, "_chunk_full_page_table_buf", None)
        if buf is None:
            return pt
        nb = int(buf.shape[-1])
        if pt.shape[1] >= nb:
            return pt[:, :nb].contiguous()
        return torch.cat([pt, torch.zeros(1, nb - pt.shape[1], dtype=torch.int32)], dim=1)

    def _spec_prefill(self, model, dec, phys, prompt, T, pt_row):
        """Eager tap-capturing prefill of ONE request into physical slot ``phys`` (masked bucket for a
        short prompt, 2048-token chunks + masked tail for a long one), each chunk's taps ingested into the
        drafter's ring for that slot. Returns host logits [1, vocab] (float)."""

        def on_chunk(hidden, chunk_start, valid_len):
            taps = model.take_dflash_eager_taps()
            if taps is None:
                raise RuntimeError("Qwen36DFlash: eager prefill captured no drafter taps (bucket trace gate on?)")
            dec.ingest_prompt(phys, taps, chunk_start + valid_len, chunk_start=chunk_start)

        dec.ctx_len[phys] = 0
        model._dflash_tap = True
        try:
            logits_dev = model.prefill_for_spec(prompt, self._spec_pref_pt(model, pt_row), T, on_chunk, slot=phys)
        finally:
            model._dflash_tap = False
        lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
        ttnn.deallocate(logits_dev)
        if dec.ctx_len[phys] != T:
            raise RuntimeError(f"Qwen36DFlash: spec prefill covered {dec.ctx_len[phys]} of {T} positions (slot {phys})")
        return lt.reshape(-1, model.vocab_size)[:1].float()

    # ------------------------------------------------------------------ warmup: arm the session once
    def warmup_model_decode(self, *args, **kwargs):
        """Two-phase like the runner's own warmup: phase 1 (enable_trace=False) ALLOCATES every spec
        buffer and COMPILES every spec program while no trace is parked yet; phase 2 (enable_trace=True)
        only CAPTURES the spec traces, after the plain decode traces. Buffers the parked traces bake must
        exist before any capture, and nothing spec-related may compile once one is parked."""
        self._in_warmup = True
        try:
            out = super().warmup_model_decode(*args, **kwargs)
            if _W <= 1:
                return out
            self._spec_warmup_phases(kwargs)
        finally:
            self._in_warmup = False
        return out

    def warmup_model_prefill(self, *args, **kwargs):
        self._in_warmup = True
        try:
            return super().warmup_model_prefill(*args, **kwargs)
        finally:
            self._in_warmup = False

    def _spec_warmup_phases(self, kwargs):
        num_blocks = kwargs.get("num_blocks")
        if not num_blocks:
            kv = kwargs.get("kv_cache")
            num_blocks = int(kv[0][0].shape[0]) if kv else 4096
        if not kwargs.get("enable_trace"):
            if self._spec_pre is None:
                self._spec_pre = self._spec_prepare(int(num_blocks), kwargs.get("kv_cache"))
        elif self._spec is None:
            if self._spec_pre is None:  # a runner that skipped phase 1 (no compile warmup)
                self._spec_pre = self._spec_prepare(int(num_blocks), kwargs.get("kv_cache"))
            self._spec_capture()

    def _spec_prepare(self, num_blocks, kv_cache):
        """Phase 1: allocate + compile (no capture). Returns the prepared decoder."""
        model = self.model[0]
        t0 = time.perf_counter()
        # Spec verify runs the fused GDN op, so decode must use the same math (model-scoped, explicit).
        model.set_gdn_fused_decode(True)
        # The verify trace's table width is vLLM's per-request row width, which the runner pads to the
        # whole pool: use the pool block count.
        dec = DFlash2ServingDecoder(model, num_blocks, stop_tokens=self._eos)
        if dec.K > _MAX_DRAFT:
            raise RuntimeError(f"drafter K={dec.K} exceeds the KV lookahead bound {_MAX_DRAFT}")
        dec.alloc()
        dec.warm()
        B = self._B
        # Dummy per-slot page-table rows for the warm-up prefills: disjoint block ranges of the pool.
        nbu = max(1, num_blocks // B)
        rows = [torch.tensor([u * nbu + (i % nbu) for i in range(num_blocks)], dtype=torch.int32) for u in range(B)]
        # 1) The eager tap-capturing prefill + drafter context fill for EVERY mask bucket a prompt can take
        #    (each bucket is its own program set), into slot 0; the largest bucket with a (chunk-1)-token
        #    prompt. Then the chunked shapes (one full chunk + a masked tail, and an exact chunk multiple).
        top = model._mask_bucket_for(_PREFILL_CHUNK - 1)
        buckets = [b for b in model._PREFILL_MASK_BUCKETS if b <= top] or [top]
        lens = [min(S, _PREFILL_CHUNK - 1) for S in buckets] + [_PREFILL_CHUNK + buckets[0], _PREFILL_CHUNK]
        for T in lens:
            self._spec_prefill(model, dec, 0, dummy_prompt(T, seed=T), T, rows[0])
        # 2) Every OTHER slot once (the GDN slot-write and the drafter's per-slot fill are per-slot programs).
        S0 = min(buckets[0], _PREFILL_CHUNK - 1)
        for u in range(1, B):
            self._spec_prefill(model, dec, u, dummy_prompt(S0, seed=100 + u), S0, rows[u])
        ttnn.synchronize_device(model.mesh_device)
        logger.info(f"Qwen36DFlash phase-1 warmup (alloc + compile) done in {time.perf_counter() - t0:.1f}s")
        self._warm_rows = rows
        return dec

    def _spec_capture(self):
        """Phase 2: capture the verify trace, then a dummy session per slot (the drafter's draft/extend
        traces are captured by the first traced step) that also proves nothing compiles any more."""
        model = self.model[0]
        dec = self._spec_pre
        B = self._B
        t0 = time.perf_counter()
        S0 = min(model._PREFILL_MASK_BUCKETS[0], _PREFILL_CHUNK - 1)
        dec.capture(warm_position=S0 + 1)
        rows = self._warm_rows
        for u in range(B):
            lt = self._spec_prefill(model, dec, u, dummy_prompt(S0, seed=7 + u), S0, rows[u])
            dec.begin(u, int(lt.argmax()), S0, rows[u])
        ttnn.synchronize_device(model.mesh_device)
        t1 = time.perf_counter()
        n = 0
        for _ in range(4):
            n += sum(len(v) for v in dec.step().values())
        ttnn.synchronize_device(model.mesh_device)
        t2 = time.perf_counter()
        for u in range(B):
            dec.end(u)
        self._spec = dec
        self._spec_pre = None
        logger.info(
            f"Qwen36DFlash phase-2 warmup (captures) done in {time.perf_counter() - t0:.1f}s: "
            f"{(t2 - t1) / 4 * 1e3:.1f} ms/step, {n / 4:.1f} tok/step over {B} slot(s) (K={dec.K}, W={_W})"
        )

    # ------------------------------------------------------------------ prefill: eager, taps -> drafter rings
    def _spec_ready(self):
        """True once the session is armed; False for the base warmup's own forwards (they must pass
        through to the plain path); raises if a request arrives without a session, since the declared
        block-output contract could not be honoured (a width mismatch would kill the engine later)."""
        if self._spec is not None:
            return True
        if self._in_warmup:
            return False
        raise RuntimeError(
            "Qwen36DFlash: speculative session not armed -- the plugin's decode warmup (enable_model_warmup "
            "with decode traces) did not run, but the block-output contract is declared; serve with "
            "warmup on, or QWEN36_DRAFTER=mtp for plain decode"
        )

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        if _W <= 1 or not self._spec_ready():
            return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        model = self.model[0]
        dec = self._spec
        if self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs, "pixel_values_videos"):
            raise RuntimeError(
                "Qwen36DFlash speculative serving is text-only (the platform rejects multimodal prompts)"
            )
        N = int(tokens.shape[0])
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        empty_slots = kwargs.get("empty_slots")
        logical = [int(s) for s in empty_slots] if empty_slots is not None else list(range(N))
        pt = torch.as_tensor(page_table)
        out = []
        for u in range(N):
            phys = self._phys[logical[u]]
            if dec.active[phys]:
                # vLLM released the previous occupant before reusing its slot; close its session if not.
                dec.end(phys)
            self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None
            T = plens[u]
            prompt = torch.as_tensor(tokens)[u : u + 1, :T].to(torch.int32)
            row = pt[u].reshape(-1).clone()
            logger.info(f"Prefilling slot {phys} up to {T} tokens (TP eager spec prefill)")
            lt = self._spec_prefill(model, dec, phys, prompt, T, row)
            out.append(lt.view(1, 1, -1))
            self._pending[phys] = (T, row)
        logger.info(f"Finished prefill of {N} request(s), starting decode...")
        return torch.cat(out, dim=0), torch.zeros(N, dtype=torch.long)

    # ------------------------------------------------------------------ decode: one block per step, all live slots
    def decode_forward(self, *args, **kwargs):
        if _W <= 1 or not self._spec_ready():
            return super().decode_forward(*args, **kwargs)

        def _read(name, pos):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if pos < len(args) else None

        tokens = _read("tokens", 0)
        start_pos = _read("start_pos", 1)
        page_table = _read("page_table", 2)
        slot_remap = _read("slot_remap", 10)
        if tokens is None:
            raise RuntimeError("Qwen36DFlash decode expects token input")
        dec = self._spec
        B = self._B
        if slot_remap is not None:
            # Row i now reads the state that was at slot remap[i]; compose into our indirection so the
            # sessions stay where they are on device.
            remap = [int(r) for r in torch.as_tensor(slot_remap).reshape(-1).tolist()]
            if sorted(remap) != list(range(B)):
                raise RuntimeError(f"Qwen36DFlash: slot_remap {remap} is not a permutation of {B} slots")
            self._phys = [self._phys[r] for r in remap]
        Bp = int(tokens.shape[0])
        if Bp > B:
            raise RuntimeError(f"Qwen36DFlash: {Bp} decode rows for {B} slots")
        toks = torch.as_tensor(tokens).reshape(Bp, -1)[:, 0].tolist()
        poss = torch.as_tensor(start_pos).reshape(-1).tolist() if start_pos is not None else [0] * Bp
        pts = torch.as_tensor(page_table) if page_table is not None else None
        live_rows = []
        for i in range(Bp):
            if int(poss[i]) < 0:
                continue  # padding row
            phys = self._phys[i]
            anchor = int(toks[i])
            row = pts[i].reshape(-1) if pts is not None else None
            if self._pending[phys] is not None:
                T, pt_row = self._pending[phys]
                self._pending[phys] = None
                if int(poss[i]) != T:
                    logger.warning(f"Qwen36DFlash: slot {phys} first decode start_pos {int(poss[i])} != prompt_len {T}")
                dec.begin(phys, anchor, T, row if row is not None else pt_row)
                self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None
                self._anchor_warned = False
            elif dec.active[phys]:
                if row is not None:
                    dec.set_table(phys, row)  # vLLM allocates a new block every block_size tokens
                if self._prev_tail[phys] is not None and anchor != self._prev_tail[phys] and not self._anchor_warned:
                    self._anchor_warned = True
                    logger.warning(
                        f"Qwen36DFlash: runner anchor {anchor} != last emitted token {self._prev_tail[phys]} (slot {phys}); "
                        "the session owns the trajectory (expected only after an EOS-filled block under ignore_eos)"
                    )
            else:
                if not self._nosession_warned:
                    self._nosession_warned = True
                    logger.warning(f"Qwen36DFlash: live row {i} (slot {phys}) has no speculative session; EOS-filling")
                continue
            live_rows.append((i, phys))
        # Step until every live row can fill its block (a stopped row is EOS-filled and needs nothing).
        t0 = time.perf_counter() if _DEBUG else 0.0
        iters = 0

        def _needs(phys):
            return dec.active[phys] and not self._stopped[phys] and len(self._carry[phys]) < _W

        while True:
            need = [phys for _, phys in live_rows if _needs(phys)]
            if not need:
                break
            # Only the slots still short of a block step; the others hold, so a fast slot never runs
            # more than one block + one iteration ahead of what vLLM has consumed (its KV write reach
            # stays inside the declared lookahead).
            com = dec.step(only=need)
            iters += 1
            if not com:
                break
            for phys, ids in com.items():
                self._carry[phys].extend(int(t) for t in ids)
                if any(t in self._eos for t in ids):
                    self._stopped[phys] = True
        out = torch.full((Bp, _W), self._eos_fill, dtype=torch.int32)
        for i, phys in live_rows:
            carry = self._carry[phys]
            stop_i = next((k for k, t in enumerate(carry) if t in self._eos), None) if self._stopped[phys] else None
            if stop_i is not None and stop_i < _W:
                # Emit through the first stop token and EOS-fill the rest; tokens speculated past the stop
                # are dropped (the scheduler normally releases the request next; an ignore_eos client
                # continues from a fresh block).
                block, self._carry[phys], self._stopped[phys] = carry[: stop_i + 1], [], False
            else:
                block, self._carry[phys] = carry[:_W], carry[_W:]
                # A stop still sitting in the carry keeps the row marked; it is emitted next step.
                self._stopped[phys] = stop_i is not None
            out[i] = self._fill_block(block)
            self._prev_tail[phys] = int(out[i, -1])
        if _DEBUG:
            logger.info(
                f"[dflash2-serve] step: rows={[i for i, _ in live_rows]} iters={iters} "
                f"{(time.perf_counter() - t0) * 1e3:.1f} ms carry={[len(self._carry[p]) for _, p in live_rows]}"
            )
        return out

    def read_decode_output(self, tt_out, async_read=False, *args, **kwargs):
        # A spec block step returns committed HOST tokens: nothing to read.
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *args, **kwargs)

    # ------------------------------------------------------------------ plugin lifecycle hooks
    def release_request(self, row: int) -> None:
        if self._spec is None:
            return
        phys = self._phys[int(row)] if 0 <= int(row) < self._B else None
        if phys is None:
            return
        self._spec.end(phys)
        self._pending[phys] = None
        self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None

    def release_persistent_capture(self) -> None:
        if self._spec is None:
            return
        self._spec.release()
        self._spec = None
