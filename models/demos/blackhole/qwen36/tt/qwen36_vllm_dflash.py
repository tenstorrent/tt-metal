# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.x on Blackhole served by vLLM WITH the DFlash2 speculative drafter (single-sequence profile).

vLLM's own speculative_config stays unset (the TT plugin rejects it); speculation is model-internal
through the plugin's ADAPTIVE BLOCK-OUTPUT contract (vllm-tt-plugin arg/gemma4_spec_serving):

  * prefill emits one host-sampled anchor token (width 1), as before;
  * a SOLO decode step of a prompt <= QWEN36_DFLASH_MAX_PROMPT tokens runs the DFlash2 draft/verify
    loop INSIDE decode_forward until QWEN36_DFLASH_SERVE_BLOCK tokens are committed and returns them as
    one host [1, W] row (EOS-filled at a genuine stop; vLLM trims at the first stop token);
  * longer prompts, multimodal prompts and batched steps decode as the plain baseline (width 1);
  * the scheduler reserves W placeholders + KV lookahead exactly for the steps the model blocks on.

Greedy: a solo block is the target's greedy trajectory from the sampled anchor (the same tokens plain
greedy decode would produce). max_num_seqs must be 1: the spec substrate owns the single GDN state
slot. QWEN36_DRAFTER=mtp (or QWEN36_DFLASH_SERVE_BLOCK=1) turns speculation off and this class serves
exactly like Qwen36ForCausalLM (including batched max_num_seqs>1), so one bundle covers both profiles.

Warmup (server start): after the plain decode trace is captured, every spec buffer is allocated and
every spec program compiled (eager tap-capturing prefill for each mask bucket, seed, draft, extend),
then the verify/commit/draft/extend traces are captured ONCE and reused by every request
(DFlash2ServingDecoder). Per request: prompt taps -> drafter context, eager seed, then W-token blocks.
"""
import json
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
    _gate = os.environ.get("QWEN36_PREFILL_BUCKET_TRACE")
    if _gate not in (None, "", "0", "off", "false"):
        logger.warning(
            f"QWEN36_PREFILL_BUCKET_TRACE={_gate!r} is incompatible with DFlash2 serving (the traced masked "
            "bucket captures no drafter taps); forcing the eager masked-bucket prefill"
        )
    os.environ["QWEN36_PREFILL_BUCKET_TRACE"] = "0"

from vllm.model_executor.models.qwen3_5 import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor  # noqa: E402
from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.tt.dflash2_serving import (  # noqa: E402
    DFlash2ServingDecoder,
    dummy_prompt,
    serve_block_size,
)
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, TT_Qwen3_5ProcessingInfo  # noqa: E402

_W = serve_block_size() if _SPEC_ON else 1
# Spec-eligible prompt length. Capped at one prefill chunk MINUS ONE: a prompt that is an exact multiple of
# the 2048-token chunk replays the chunk-prefill TRACE (no python-side tap clones), so it cannot seed the
# drafter; every shorter prompt takes the eager masked bucket, which does. The scheduler gates on the same
# number, so a 2048-token prompt is served as plain width-1 decode by both sides.
_PREFILL_CHUNK = 2048
_MAX_PROMPT = min(int(os.environ.get("QWEN36_DFLASH_MAX_PROMPT", "2048")), _PREFILL_CHUNK - 1)
# Every DFlash checkpoint drafts at most block-1 <= 15 tokens; the KV lookahead only has to bound the
# verify's reach past the block (W committed positions + K+1 candidate rows).
_MAX_DRAFT = 15
_DEBUG = os.environ.get("QWEN36_DFLASH_DEBUG", "0") == "1"


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36DFlashForCausalLM(Qwen36ForCausalLM):
    """Qwen36ForCausalLM + model-internal DFlash2 speculation on solo decode steps (see module doc)."""

    model_capabilities = {
        **Qwen36ForCausalLM.model_capabilities,
        # A solo decode step of a spec-eligible request commits exactly _W tokens (EOS-filled at a stop).
        "output_tokens_per_step": _W,
        # Block only when decoding alone; prefill anchors and batched steps are plain width-1 rows.
        "tt_adaptive_block_output": _W > 1,
        # Prompts longer than this are served as plain baseline for their whole lifetime (the drafter
        # only sees a 2048-token context window, where native decode is faster anyway).
        "tt_adaptive_block_max_prompt_tokens": _MAX_PROMPT if _W > 1 else 0,
        # The block step writes the W committed positions AND the last verify's K+1 candidate rows into
        # the paged KV inside one step: have the scheduler allocate that reach up front.
        "tt_block_output_kv_lookahead_tokens": (_W + _MAX_DRAFT + 1) if _W > 1 else 0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec = None  # DFlash2ServingDecoder, armed at warmup
        self._spec_pending = None  # (taps, prompt_len, page_table_row) from the last spec-eligible prefill
        self._spec_req_prompt_len = None  # prompt length of the live single request (block gate mirror)
        self._spec_last_pt = None
        self._spec_carry = []  # committed-but-unemitted tokens (block boundary overflow)
        self._spec_prev_tail = None
        self._spec_anchor_warned = False
        self._spec_oov_warned = False
        self._spec_plain_steps = 0
        model = self.model[0]
        self._eos = self._load_eos_ids(model)
        self._eos_fill = min(self._eos)
        if _W > 1:
            mb = int(model.args.max_batch_size)
            if mb != 1:
                raise RuntimeError(
                    "Qwen36DFlash speculative serving is single-sequence: launch with --max-num-seqs 1 "
                    f"(or QWEN36_DRAFTER=mtp for the plain batched profile); got max_batch_size={mb}"
                )
            if model.num_devices <= 1:
                raise RuntimeError("Qwen36DFlash speculative serving needs the TP mesh (MESH_DEVICE=P150x4)")
            logger.info(
                f"Qwen36DFlash serving: block W={_W} tokens/step, spec prompts <= {_MAX_PROMPT}, "
                f"eos={sorted(self._eos)}, drafter={os.environ.get('DFLASH_WEIGHTS')}"
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
        assert ids, "no eos_token_id found for the served checkpoint"
        return ids

    def _spec_drop_pending(self):
        if self._spec_pending is not None:
            taps, _, _ = self._spec_pending
            for t in taps:
                ttnn.deallocate(t)
            self._spec_pending = None

    def _spec_prefill_taps(self, model, tokens, page_table, T, **kwargs):
        """The serving TP prefill with the drafter taps armed; returns (prefill output, taps)."""
        model._dflash_tap = True
        try:
            out = super().prefill_forward(tokens, page_table, kwargs.pop("kv_cache", None), torch.tensor([T]), **kwargs)
        finally:
            model._dflash_tap = False
        return out, model.take_dflash_eager_taps()

    def _fill_block(self, block):
        """Exactly _W ids: a short block happens only at a genuine stop (EOS / capacity) -> EOS-fill; the
        scheduler trims at the first stop token."""
        block = list(block)[:_W]
        vocab = self.model[0].vocab_size
        oov = [t for t in block if not 0 <= t < vocab]
        if oov:
            if not self._spec_oov_warned:
                self._spec_oov_warned = True
                logger.warning(
                    f"Qwen36DFlash: {len(oov)} out-of-vocab committed id(s) (first={oov[0]}); substituting EOS"
                )
            block = [t if 0 <= t < vocab else self._eos_fill for t in block]
        out = torch.full((1, _W), self._eos_fill, dtype=torch.int32)
        if block:
            out[0, : len(block)] = torch.tensor(block, dtype=torch.int32)
        return out

    # ------------------------------------------------------------------ warmup: arm the session once
    def warmup_model_decode(self, *args, **kwargs):
        out = super().warmup_model_decode(*args, **kwargs)
        if _W > 1 and kwargs.get("enable_trace") and self._spec is None:
            num_blocks = kwargs.get("num_blocks")
            if not num_blocks:
                kv = kwargs.get("kv_cache")
                num_blocks = int(kv[0][0].shape[0]) if kv else 4096
            self._spec_warmup(int(num_blocks))
        return out

    def _spec_warmup(self, num_blocks):
        model = self.model[0]
        t0 = time.perf_counter()
        ctx_blocks = int(os.environ.get("QWEN36_DFLASH_SERVE_CTX_BLOCKS", "0")) or num_blocks
        dec = DFlash2ServingDecoder(model, num_blocks, ctx_blocks=ctx_blocks, stop_tokens=self._eos)
        dec.alloc()
        pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        buckets = [b for b in model._PREFILL_MASK_BUCKETS if b <= max(_MAX_PROMPT, model._PREFILL_MASK_BUCKETS[0])]
        # 1) Compile pass: the eager tap-capturing prefill + drafter context fill for EVERY mask bucket a
        #    spec-eligible prompt can take (each bucket is its own program set), before any spec capture.
        for S in buckets:
            # The largest bucket is exercised with a (chunk-1)-token prompt: an exact chunk multiple would
            # replay the chunk trace instead of the eager masked bucket (see _MAX_PROMPT).
            T = min(S, _MAX_PROMPT)
            _, taps = self._spec_prefill_taps(model, dummy_prompt(T), pt, T)
            assert taps is not None, "eager masked prefill captured no drafter taps (bucket trace gate on?)"
            dec.ingest_prompt(taps, T)
        S = min(buckets[-1], _MAX_PROMPT)
        # 2) Seed + draft warmup, then the one-time verify/commit captures; the first two steps capture
        #    the drafter's draft/extend traces and replay them once.
        dec.begin(first=int(dummy_prompt(1, seed=99)[0, 0]), T=S, page_table_row=pt)
        dec.capture()
        for _ in range(2):
            dec.step()
        dec.end()
        # 3) A second, replay-only session on the smallest bucket: proves every program is cached (no
        #    compile with the traces parked) and reports the steady-state iteration time.
        S2 = min(buckets[0], _MAX_PROMPT)
        _, taps = self._spec_prefill_taps(model, dummy_prompt(S2, seed=7), pt, S2)
        dec.ingest_prompt(taps, S2)
        dec.begin(first=int(dummy_prompt(1, seed=5)[0, 0]), T=S2, page_table_row=pt)
        ttnn.synchronize_device(model.mesh_device)
        t1 = time.perf_counter()
        n = 0
        for _ in range(3):
            n += len(dec.step() or [])
        ttnn.synchronize_device(model.mesh_device)
        t2 = time.perf_counter()
        dec.end()
        self._spec = dec
        logger.info(
            f"Qwen36DFlash warmup done in {time.perf_counter() - t0:.1f}s: {(t2 - t1) / 3 * 1e3:.1f} ms/iter, "
            f"{n / 3:.2f} tok/iter on the dummy session (K={dec.K}, W={_W})"
        )

    # ------------------------------------------------------------------ prefill: capture the prompt taps
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        if _W <= 1 or self._spec is None:
            return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        model = self.model[0]
        # B=1 serving: a new prompt means the previous request is gone (finished or aborted).
        self._spec_drop_pending()
        if self._spec.active:
            self._spec.end()
        self._spec_carry = []
        self._spec_prev_tail = None
        T = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        self._spec_req_prompt_len = T
        eligible = (
            int(tokens.shape[0]) == 1
            and T <= _MAX_PROMPT
            and not self._has_visual(kwargs, "pixel_values")
            and not self._has_visual(kwargs, "pixel_values_videos")
        )
        if not eligible:
            model._dflash_tap = False
            return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        model._dflash_tap = True
        try:
            out = super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        finally:
            model._dflash_tap = False
        taps = model.take_dflash_eager_taps()
        if taps is None:
            logger.warning(
                f"Qwen36DFlash: prefill T={T} captured no drafter taps; this request decodes in plain blocks"
            )
            return out
        row = torch.as_tensor(page_table)[:1].clone()
        self._spec_pending = (taps, T, row)
        return out

    # ------------------------------------------------------------------ decode: one block per solo step
    def decode_forward(self, *args, **kwargs):
        if _W <= 1 or self._spec is None:
            return super().decode_forward(*args, **kwargs)
        tokens = kwargs.get("tokens", args[0] if args else None)
        start_pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        page_table = kwargs.get("page_table", args[2] if len(args) > 2 else None)
        assert tokens is not None, "Qwen36DFlash decode expects token input"
        if int(tokens.shape[0]) != 1:
            # Batched step (adaptive contract): plain baseline, width 1. Unreachable at max_num_seqs=1.
            self._spec_drop_pending()
            if self._spec.active:
                self._spec.end()
            return super().decode_forward(*args, **kwargs)
        anchor = int(tokens.reshape(-1)[0])
        pos = int(start_pos.reshape(-1)[0]) if start_pos is not None else None
        row = torch.as_tensor(page_table)[:1] if page_table is not None else None
        dec = self._spec
        if self._spec_pending is not None:
            taps, T, pt_row = self._spec_pending
            self._spec_pending = None
            if pos is not None and pos != T:
                logger.warning(f"Qwen36DFlash: first decode start_pos {pos} != prompt_len {T}")
            dec.ingest_prompt(taps, T)
            dec.begin(anchor, T, row if row is not None else pt_row)
            self._spec_last_pt = dec.page_table.clone()
            self._spec_carry = []
            self._spec_prev_tail = None
            self._spec_anchor_warned = False
        block_step = self._spec_req_prompt_len is not None and self._spec_req_prompt_len <= _MAX_PROMPT
        if not dec.active:
            if not block_step:
                # Long prompt: the scheduler reserved width 1 for every step of this request.
                return super().decode_forward(*args, **kwargs)
            # Spec-eligible request without a session (no taps / multimodal): the scheduler still
            # reserved a W-token block, so honour the contract with a plain greedy block.
            return self._plain_block(anchor, pos, row)
        # vLLM allocates a new block every block_size tokens: re-point the verify trace when its row changes.
        if row is not None:
            fitted = dec.fit_page_table(row)
            if self._spec_last_pt is None or not torch.equal(fitted, self._spec_last_pt):
                dec.set_page_table(fitted)
                self._spec_last_pt = fitted
        if self._spec_prev_tail is not None and anchor != self._spec_prev_tail and not self._spec_anchor_warned:
            self._spec_anchor_warned = True
            logger.warning(
                f"Qwen36DFlash: runner anchor {anchor} != last emitted token {self._spec_prev_tail}; the session "
                "owns the trajectory (expected only after an EOS-filled block under ignore_eos)"
            )
        block = []
        carry = self._spec_carry
        while carry and len(block) < _W:
            block.append(carry.pop(0))
        stopped = any(t in self._eos for t in block)
        t0 = time.perf_counter() if _DEBUG else 0.0
        iters = 0
        while not stopped and len(block) < _W:
            committed = dec.step()
            if committed is None:  # capacity: end the request cleanly
                stopped = True
                break
            iters += 1
            for t in committed:
                (block if len(block) < _W else carry).append(t)
            stopped = any(t in self._eos for t in committed)
        if _DEBUG:
            logger.info(
                f"[dflash2-serve] block: {len(block)} tokens in {iters} iters, {(time.perf_counter() - t0) * 1e3:.1f} ms, "
                f"carry={len(carry)}, stopped={stopped}"
            )
        self._spec_prev_tail = block[-1] if block else None
        return self._fill_block(block)

    def _plain_block(self, anchor, pos, row):
        """W plain greedy decode steps in one call (fallback that keeps the block contract)."""
        model = self.model[0]
        pt = self._spec.fit_page_table(row) if row is not None else self._spec.page_table
        self._spec_plain_steps += 1
        if self._spec_plain_steps == 1:
            logger.warning("Qwen36DFlash: serving a spec-eligible request without a drafter session (plain blocks)")
        out = []
        tok, p = anchor, int(pos)
        for _ in range(_W):
            logits, hidden = model.decode_step_paged(tok, p, pt)
            ttnn.deallocate(hidden)
            tok = int(logits.argmax())
            out.append(tok)
            p += 1
            if tok in self._eos:
                break
        return self._fill_block(out)

    def read_decode_output(self, tt_out, async_read=False, *args, **kwargs):
        # A solo block step returns committed HOST tokens: nothing to read.
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        return super().read_decode_output(tt_out, async_read, *args, **kwargs)

    # ------------------------------------------------------------------ plugin lifecycle hooks
    def release_request(self, row: int) -> None:
        if self._spec is None:
            return
        self._spec_drop_pending()
        if self._spec.active:
            self._spec.end()
        self._spec_carry = []
        self._spec_prev_tail = None
        self._spec_req_prompt_len = None

    def release_persistent_capture(self) -> None:
        if self._spec is None:
            return
        self._spec_drop_pending()
        self._spec.release()
        self._spec = None
