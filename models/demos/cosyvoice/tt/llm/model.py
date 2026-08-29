# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TransformerLM: text -> semantic speech tokens, autoregressively, on device.

    text tokens ------> text_embedding (51866 x 512)
                    --> ConformerEncoder, 6 blocks, 16 heads, d=1024, **causal**
                    --> text_encoder_affine (1024 -> 1024)
    speaker x-vector -> L2 normalise, affine (192 -> 1024)
    prompt tokens ----> speech_embedding (4096 x 1024)

    prefix = [sos | speaker | text | task_id | prompt speech tokens]
    loop: AR decoder (14 blocks, KV cache) -> llm_decoder (1024 -> 4097)
          -> RAS sampling -> next token -> speech_embedding -> loop

The two encoders in this checkpoint are the same class family with **different
attention patterns and different feed-forward activations**, which the config only
implies:

* the text encoder sets `static_chunk_size: 1`, and `subsequent_chunk_mask` turns
  that into a plain causal mask -- unlike the flow encoder, whose `static_chunk_size`
  is 0 and which therefore attends fully;
* `ConformerEncoder` defaults `activation_type` to `"swish"` while
  `TransformerEncoder` defaults it to `"relu"`, and cosyvoice.yaml overrides
  neither -- so the text encoder's FFN is SiLU and the AR decoder's is ReLU.

Both facts are carried in the exported metadata rather than hardcoded here.

Generation length is bounded by the *text* length: `min_len = 2 * text_len` and
`max_len = 20 * text_len`, with the EOS token masked out until `min_len`. Those
bounds are what stop the sampler ending an utterance after one token.
"""
from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn

from ..flow.encoder import TtConformerEncoder, _linear, espnet_rel_positional_encoding
from ..hifigan.conv import accurate_compute_config
from .decoder import TtARDecoder, causal_bias
from .sampling import greedy, ras_sampling


class TtTransformerLM:
    """The LLM stage. Activations are `[1, T, C]`."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16, weights_dtype=None):
        """`weights_dtype=ttnn.bfloat8_b` stores the AR decoder's matrices at half
        the width of the activations.

        It applies to the **AR decoder only**, which is the stage that reads every
        weight from DRAM to produce one token -- at batch 1 that is a bandwidth
        problem, not an arithmetic one. The text encoder runs once per utterance
        and the output head is a single matmul, so neither is worth the accuracy.
        """
        self.device, self.dtype, self.meta = device, dtype, meta
        self.weights_dtype = weights_dtype or dtype
        self.cc = accurate_compute_config(device)
        self.text_meta = meta["text_encoder"]
        self.ar_meta = meta["ar_decoder"]
        self.speech_token_size = meta["speech_token_size"]
        self.eos_token = meta["eos_token"]
        self.sos, self.task_id = meta["sos"], meta["task_id"]

        self.text_embedding = ttnn.from_torch(
            bag.tensor("text_embedding.weight"), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.text_encoder = TtConformerEncoder(device, bag.sub("text_encoder"), self.text_meta, dtype)
        self.affine_w, self.affine_b = _linear(device, bag, "text_encoder_affine_layer", dtype)
        self.spk_w, self.spk_b = _linear(device, bag, "spk_embed_affine_layer", dtype)

        # llm_embedding holds exactly two rows: sos and task_id.
        self.llm_embedding = bag.tensor("llm_embedding.weight")
        self.speech_embedding_host = bag.tensor("speech_embedding.weight")
        self.speech_embedding = ttnn.from_torch(
            self.speech_embedding_host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.decoder = TtARDecoder(device, bag.sub("llm"), self.ar_meta, dtype, weights_dtype)
        self.head_w, self.head_b = _linear(device, bag, "llm_decoder", dtype)
        self._causal: dict[int, object] = {}

    # ----------------------------------------------------------------------
    def causal_mask(self, size: int):
        """Additive causal bias, cached per size. Prefill only."""
        if size not in self._causal:
            self._causal[size] = ttnn.from_torch(
                causal_bias(size), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._causal[size]

    def encode_text(self, text_tokens):
        """Token IDs `[1, T]` -> `[1, T, 1024]`, causally masked."""
        emb = ttnn.embedding(text_tokens, self.text_embedding, layout=ttnn.TILE_LAYOUT)
        t = emb.shape[1]
        pos = ttnn.from_torch(
            espnet_rel_positional_encoding(t, self.text_meta["d_model"]),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        h = self.text_encoder(emb, pos, mask=self.causal_mask(t))
        ttnn.deallocate(emb)
        ttnn.deallocate(pos)
        out = ttnn.linear(h, self.affine_w, bias=self.affine_b, compute_kernel_config=self.cc)
        ttnn.deallocate(h)
        return out

    def speaker_embedding(self, embedding):
        """`[1, 1, 192]` -> `[1, 1, 1024]`: L2 normalise then affine."""
        sq = ttnn.multiply(embedding, embedding)
        s = ttnn.sum(sq, dim=-1, keepdim=True)
        ttnn.deallocate(sq)
        inv = ttnn.rsqrt(s)
        ttnn.deallocate(s)
        unit = ttnn.multiply(embedding, inv)
        ttnn.deallocate(inv)
        out = ttnn.linear(unit, self.spk_w, bias=self.spk_b, compute_kernel_config=self.cc)
        ttnn.deallocate(unit)
        return out

    def build_prefix(self, text_enc, spk_emb=None, prompt_speech_tokens=None):
        """`[sos | speaker | encoded text | task_id | prompt speech tokens]`.

        The speaker row is omitted entirely when there is no x-vector -- the
        reference builds a zero-width tensor, which is the same thing.
        """
        d = self.ar_meta["input_size"]
        sos = ttnn.from_torch(
            self.llm_embedding[self.sos].reshape(1, 1, d), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        task = ttnn.from_torch(
            self.llm_embedding[self.task_id].reshape(1, 1, d),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        parts = [sos]
        if spk_emb is not None:
            parts.append(spk_emb)
        parts.append(text_enc)
        parts.append(task)
        if prompt_speech_tokens is not None and prompt_speech_tokens.shape[1] > 0:
            parts.append(ttnn.embedding(prompt_speech_tokens, self.speech_embedding, layout=ttnn.TILE_LAYOUT))
        out = ttnn.concat(parts, dim=1)
        ttnn.deallocate(sos)
        ttnn.deallocate(task)
        if len(parts) > 4 and parts[-1] is not text_enc:
            ttnn.deallocate(parts[-1])
        return out

    # ----------------------------------------------------------------------
    def logits_for_last(self, ys):
        """`llm_decoder(y[:, -1])` -> `[B, 4097]` on the host, ready for sampling.

        Batch 1 -- every published figure -- returns a flat `[4097]`, unchanged; a
        batched decode returns one row per sequence.

        The head is the one place a host round trip is unavoidable: RAS needs the
        full 4097-way distribution *and* the history of emitted tokens, and its
        repetition branch rewrites a score before resampling.

        Moving it to `ttnn.sampling` was measured and rejected -- the whole
        per-token tail is 0.352 ms, 2.7 % of a token, and on-device sampling could
        remove at most 0.217 ms of it while giving up exact agreement with the
        reference. See the note in `sampling.py` and `scripts/profile_token_tail.py`.
        """
        t = ys.shape[1]
        # A single-row `ys` needs no slice, and taking one anyway is actively
        # harmful: `ttnn.slice` over the full extent returns an **alias of the
        # input**, so the `deallocate` below would free the caller's tensor. That
        # is invisible while `ys` is a fresh per-step allocation the caller frees
        # regardless -- and fatal once it is the trace's persistent output buffer,
        # where it surfaces as "Input Tensor is not allocated" on the *next* step.
        b = ys.shape[0]
        owned = t > 1
        last = ttnn.slice(ys, [0, t - 1, 0], [b, t, ys.shape[2]]) if owned else ys
        logits = ttnn.linear(last, self.head_w, bias=self.head_b, compute_kernel_config=self.cc)
        if owned:
            ttnn.deallocate(last)
        out = ttnn.to_torch(logits).float()
        ttnn.deallocate(logits)
        return out.reshape(-1) if b == 1 else out.reshape(b, -1)

    @staticmethod
    def cache_width(prefix_len: int, max_tokens: int, bucket: int = 128) -> int:
        """How wide the KV cache buffer must be, rounded up to a bucket.

        The rounding is not cosmetic. Every distinct width is a separate kernel
        compile, so sizing the buffer exactly to each utterance would put a compile
        at the start of every request; bucketing to 128 means a handful of widths
        cover every utterance the model will ever see, and they are warm after the
        first few. Attention then runs over at most 127 slots more than it needs.
        """
        need = prefix_len + max_tokens + 1
        return ((need + bucket - 1) // bucket) * bucket

    def prefill(self, prefix, max_len: int):
        """The whole prompt in one chunk, right-aligned in a `max_len` buffer."""
        from .decoder import right_aligned_bias

        length = prefix.shape[1]
        caches = self.decoder.empty_cache(max_len, length)
        mask = self._dev_mask(right_aligned_bias(max_len, length, length, causal=True))
        out = self.decoder.forward_chunk_fixed(prefix, caches, max_len, valid=length, mask=mask)
        ttnn.deallocate(mask)
        return out

    def decode_step(self, token_id: int, caches, max_len: int, valid: int):
        """One token in, one `[1, 1, 1024]` hidden state out.

        The mask suppresses the padding slots at the front of the buffer; its
        *values* change every step but its *shape* does not, which is the entire
        point -- see `forward_chunk_fixed`.
        """
        from .decoder import right_aligned_bias

        row = self.speech_embedding_host[token_id].reshape(1, 1, -1)
        x = ttnn.from_torch(row, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask = self._dev_mask(right_aligned_bias(max_len, min(valid, max_len), 1))
        ys, caches = self.decoder.forward_chunk_fixed(x, caches, max_len, valid=valid, mask=mask)
        ttnn.deallocate(x)
        # Freed every step, not cached: the *values* change with `valid` even though
        # the shape does not, and a few hundred decode steps of un-freed masks is
        # enough to exhaust the allocator part-way through a sweep.
        ttnn.deallocate(mask)
        return ys, caches

    def _dev_mask(self, m):
        return ttnn.from_torch(m, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def release_caches(self):
        """Drop the per-length constants held between utterances.

        `positional()` and `causal_mask()` memoise by size, which is right within
        one utterance and wrong across a sweep: every utterance has its own prefix
        length and its own bucket, so the tables grow without bound and the
        allocator runs out somewhere in the middle. The weights are untouched.
        """
        for cache in (self._causal, self.decoder._pos_cache):
            for t in cache.values():
                ttnn.deallocate(t)
            cache.clear()

    # ----------------------------------------------------------------------
    def generate(
        self,
        text_tokens,
        spk_emb=None,
        prompt_speech_tokens=None,
        *,
        text_len: int | None = None,
        sampler: str = "ras",
        max_tokens: int | None = None,
        seed: int | None = None,
        use_trace: bool = True,
    ) -> list[int]:
        """Text token IDs -> semantic speech token IDs.

        `sampler='greedy'` makes the stream deterministic, which is what the device
        tests use; `'ras'` reproduces the reference's stochastic policy.
        """
        if seed is not None:
            torch.manual_seed(seed)
        sample = greedy if sampler == "greedy" else None
        cfg = self.meta.get("sampling", {})

        text_enc = self.encode_text(text_tokens)
        prefix = self.build_prefix(text_enc, spk_emb, prompt_speech_tokens)
        ttnn.deallocate(text_enc)

        n_text = text_len if text_len is not None else text_tokens.shape[1]
        min_len = int(n_text * self.meta.get("min_token_text_ratio", 2))
        cap = max_tokens or int(n_text * self.meta.get("max_token_text_ratio", 20))

        prefix_len = prefix.shape[1]
        max_len = self.cache_width(prefix_len, cap)
        ys, caches = self.prefill(prefix, max_len)
        ttnn.deallocate(prefix)
        # The prefill's logits are read before capture. Not strictly required --
        # the prefill's `ys` does survive `begin_trace_capture` -- but it keeps the
        # loop below to a single source of logits.
        pending_logits = self.logits_for_last(ys)
        ttnn.deallocate(ys)
        ys = None

        # Trace capture is worth ~2.2x on the decode step, verified bit-exact
        # against the untraced path; PERF.md carries the current figures and is
        # the one place they are maintained. Only the decode step is
        # traced -- prefill runs once per utterance with a different shape, so
        # tracing it would buy one dispatch saving for a second capture. Capture
        # costs two warm-up passes, so it is skipped for very short generations.
        traced = None
        if use_trace and cap > 8:
            from .decoder import TracedDecodeStep, TracedDecodeStepInPlace, kv_inplace_default

            # `COSYVOICE_KV_INPLACE` writes the KV cache in place instead of rebuilding
            # it; unset, `kv_inplace_default` follows the architecture (on for
            # Wormhole, off for Blackhole -- see its docstring for the numbers). It
            # costs two things the moving cache does not: 65 captured traces instead
            # of one, so it needs a much larger trace region (384 MB observed to work;
            # 64 MB fails); and it is not bit-exact against the moving cache (worst
            # PCC 0.9986 over 72 steps, non-accumulating). The fallback below turns a
            # too-small trace region into a warning and an untraced decode (2.2x
            # slower than either traced path) rather than a crash, which is what makes
            # defaulting it on for Wormhole safe for a caller that has not resized its
            # device: it degrades audibly, it does not fail.
            _kv_env = os.environ.get("COSYVOICE_KV_INPLACE")
            use_inplace = (_kv_env == "1") if _kv_env is not None else kv_inplace_default(self.decoder.device)
            kls = TracedDecodeStepInPlace if use_inplace else TracedDecodeStep
            try:
                traced = kls(self.decoder, max_len).capture()
                traced.seed(caches)
                TtARDecoder.free_caches(caches)
                caches = None
            except Exception as e:  # noqa: BLE001
                # Capture needs the device opened with a `trace_region_size`, which
                # not every caller does. Tracing is an optimisation, so a failure
                # here degrades to the untraced path rather than failing the
                # generation -- but it says so, because silently running 2.2x slower
                # is exactly the kind of regression that hides for months.
                logger.warning(f"trace capture unavailable, falling back to untraced decode: {e}")
                if traced is not None:
                    traced.release()
                traced = None

        out: list[int] = []
        logits = pending_logits
        cfg_kw = dict(
            top_p=cfg.get("top_p", 0.8),
            top_k=cfg.get("top_k", 25),
            win_size=cfg.get("win_size", 10),
            tau_r=cfg.get("tau_r", 0.1),
        )
        for i in range(cap):
            logp = torch.log_softmax(logits, dim=-1)
            if i < min_len:
                # EOS is suppressed until the utterance is long enough for the text
                logp[self.eos_token] = -float("inf")
            token = sample(logp) if sample is not None else ras_sampling(logp, out, **cfg_kw)
            if token == self.eos_token:
                break
            out.append(token)

            # Logits are read in the SAME iteration that produces them, rather than
            # carried to the next. The traced path writes into one persistent output
            # buffer, so holding a reference across a step is a lifetime question
            # that does not need to exist.
            row = self.speech_embedding_host[token].reshape(1, 1, -1)
            if traced is not None:
                ys = traced.step(row, prefix_len + len(out))
                ttnn.synchronize_device(self.device)
                logits = self.logits_for_last(ys)
            else:
                ys, caches = self.decode_step(token, caches, max_len, prefix_len + len(out))
                logits = self.logits_for_last(ys)
                ttnn.deallocate(ys)

        if traced is not None:
            # `ys` is the trace's persistent output buffer -- released with the
            # trace, not separately.
            traced.release()
        else:
            TtARDecoder.free_caches(caches)
        return out

    # ----------------------------------------------------------------------
    def generate_batch(
        self,
        requests: list[dict],
        *,
        sampler: str = "ras",
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> list[list[int]]:
        """Generate semantic tokens for several utterances in one decode loop.

        `requests` is a list of the keyword sets `generate` takes -- `text_tokens`,
        and optionally `spk_emb`, `prompt_speech_tokens`, `text_len`. One list of
        token IDs comes back per request, in order.

        **Why this exists.** A decode step at one row is bound by reading the AR
        decoder's 14 blocks of weights out of DRAM: every matmul is a matrix against
        a single row, so there is no reuse to amortise the read against. That is what
        `test_device_decode_bfloat8_weights` measures from the other side -- halving
        the weight *width* moves the step, because the step is the read. Batching
        attacks the same bottleneck from the numerator: the same weight read serves
        `B` rows, so the per-utterance cost falls without any kernel changing.

        **Why the batch can be ragged.** Utterances have different prompt lengths and
        stop at different tokens. Both are absorbed by the cache being *right*
        aligned: each row's live history ends at the last slot and differs only in
        where it begins, which is exactly one number per row in the mask
        (`right_aligned_bias` takes a list). Nothing gathers, nothing re-packs.

        **What it costs.** Rows that hit EOS early keep stepping until the longest
        row finishes -- their outputs are discarded. So the wall clock is set by the
        longest utterance in the batch and the useful fraction is
        `mean(length) / max(length)`; a batch of similar-length utterances is worth
        much more than a mixed one. The alternative -- compacting the batch when a
        row retires -- would rebuild the KV cache at a new batch size, and so pay a
        fresh trace capture, several times per batch. Reported rather than hidden:
        the perf test prints the padding waste alongside the throughput.

        Prefill stays per-utterance and untraced. Each prompt is a different length,
        so batching prefill would mean padding every prompt to the longest and
        running the encoder over the padding; it runs once per utterance against
        `n` decode steps, so it is not where the time is.

        Sampling is unchanged and still per row on the host -- RAS needs each row's
        own emitted history, which is not a batched operation in any useful sense.
        """
        from .decoder import TracedDecodeStep

        if seed is not None:
            torch.manual_seed(seed)
        if not requests:
            return []
        b = len(requests)
        sample = greedy if sampler == "greedy" else None
        cfg = self.meta.get("sampling", {})
        cfg_kw = dict(
            top_p=cfg.get("top_p", 0.8),
            top_k=cfg.get("top_k", 25),
            win_size=cfg.get("win_size", 10),
            tau_r=cfg.get("tau_r", 0.1),
        )

        # ---- per-request prefixes, and the one cache width they will share
        prefixes, prefix_lens, min_lens, caps = [], [], [], []
        for req in requests:
            text_tokens = req["text_tokens"]
            text_enc = self.encode_text(text_tokens)
            prefix = self.build_prefix(text_enc, req.get("spk_emb"), req.get("prompt_speech_tokens"))
            ttnn.deallocate(text_enc)
            n_text = req.get("text_len") or text_tokens.shape[1]
            prefixes.append(prefix)
            prefix_lens.append(prefix.shape[1])
            min_lens.append(int(n_text * self.meta.get("min_token_text_ratio", 2)))
            caps.append(max_tokens or int(n_text * self.meta.get("max_token_text_ratio", 20)))
        cap = max(caps)
        # One width for the batch: the rows share a buffer, so the widest requirement
        # sets it and shorter rows simply carry more suppressed padding.
        max_len = max(self.cache_width(pl, cap) for pl in prefix_lens)

        # ---- prefill each row, then stack the caches on the batch axis
        per_row_caches = []
        pending = []
        for prefix in prefixes:
            ys, caches = self.prefill(prefix, max_len)
            ttnn.deallocate(prefix)
            pending.append(self.logits_for_last(ys))
            ttnn.deallocate(ys)
            per_row_caches.append(caches)

        if b == 1:
            # `ttnn.concat` of a single tensor returns an **alias** of it, so the
            # stack-then-free below would free the very buffers it just produced --
            # the same aliasing trap `logits_for_last` documents for a full-extent
            # `ttnn.slice`. One row needs no stacking.
            stacked = per_row_caches[0]
        else:
            stacked = []
            for layer in range(len(per_row_caches[0])):
                k = ttnn.concat([c[layer][0] for c in per_row_caches], dim=0)
                v = ttnn.concat([c[layer][1] for c in per_row_caches], dim=0)
                stacked.append((k, v))
            for caches in per_row_caches:
                TtARDecoder.free_caches(caches)

        traced = TracedDecodeStep(self.decoder, max_len, batch=b).capture()
        traced.seed(stacked)
        TtARDecoder.free_caches(stacked)

        # ---- the shared decode loop
        out: list[list[int]] = [[] for _ in range(b)]
        done = [False] * b
        logits = torch.stack(pending)  # [B, vocab]
        d_in = self.decoder.meta["input_size"]
        try:
            for _ in range(cap):
                rows = torch.zeros(b, 1, d_in)
                for i in range(b):
                    if done[i]:
                        continue
                    logp = torch.log_softmax(logits[i], dim=-1)
                    if len(out[i]) < min_lens[i]:
                        logp[self.eos_token] = -float("inf")
                    token = sample(logp) if sample is not None else ras_sampling(logp, out[i], **cfg_kw)
                    if token == self.eos_token or len(out[i]) >= caps[i]:
                        done[i] = True
                        continue
                    out[i].append(token)
                    rows[i] = self.speech_embedding_host[token].reshape(1, -1)
                if all(done):
                    break
                # A retired row is still stepped -- see the docstring. Its embedding
                # row stays zero and its output is thrown away; what it must not do is
                # change the batch size, which would cost a fresh trace capture.
                valids = [prefix_lens[i] + len(out[i]) for i in range(b)]
                ys = traced.step(rows, valids)
                ttnn.synchronize_device(self.device)
                logits = self.logits_for_last(ys)
        finally:
            traced.release()
        return out
