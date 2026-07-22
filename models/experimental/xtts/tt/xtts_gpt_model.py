# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 GPT decoder *with embeddings and heads*.

Mirrors ``reference/xtts_gpt_model.py``:

    text_emb = text_embedding(text_ids) + text_pos_embedding(0..text_len)
    mel_emb  = mel_embedding(mel_ids)   + mel_pos_embedding(0..mel_len)
    emb = concat([text_emb, mel_emb], dim=1)
    enc = final_norm(stack(emb))          # stack = 30 blocks + ln_f
    text_logits = text_head(enc[:, :text_len])
    mel_logits  = mel_head(enc[:, text_len:])

Token/position lookups use ``ttnn.embedding``; the two heads are ``ttnn.linear``.
GPT-2 ``nn.Linear`` head weights are stored ``[out, in]`` so they are transposed
to the ``[in, out]`` layout ``ttnn.linear`` expects (y = x @ W + b).

NOTE: the ``[text] + [mel]`` concat and the text/mel split slice along the
sequence dim. In TILE_LAYOUT these are cleanest when ``text_len`` and ``mel_len``
are multiples of the 32-row tile; the PCC test picks tile-aligned lengths.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    NUM_HEADS,
    NUM_LAYERS,
)
from models.experimental.xtts.tt.xtts_gpt_block import _to_device
from models.experimental.xtts.tt.xtts_gpt_stack import TtXttsGptStack


def _to_device_rm(torch_tensor, device):
    """torch -> ttnn bf16 ROW_MAJOR tensor on device (weight table for ttnn.embedding)."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


class TtXttsGptModel(LightweightModule):
    def __init__(self, state_dict, device, num_layers=NUM_LAYERS):
        super().__init__()
        self.device = device

        # Token + learned-position embedding tables (row-major for ttnn.embedding).
        self.text_emb_weight = _to_device_rm(state_dict["gpt.text_embedding.weight"], device)
        self.mel_emb_weight = _to_device_rm(state_dict["gpt.mel_embedding.weight"], device)
        self.text_pos_weight = _to_device_rm(state_dict["gpt.text_pos_embedding.emb.weight"], device)
        self.mel_pos_weight = _to_device_rm(state_dict["gpt.mel_pos_embedding.emb.weight"], device)

        # The 30 decoder blocks + ln_f.
        self.stack = TtXttsGptStack(state_dict, device, num_layers=num_layers)

        # Second final LayerNorm.
        self.final_norm_weight = _to_device(state_dict["gpt.final_norm.weight"], device)
        self.final_norm_bias = _to_device(state_dict["gpt.final_norm.bias"], device)

        # Heads. nn.Linear weight is [out, in]; ttnn.linear wants [in, out] -> transpose.
        self.text_head_weight = _to_device(state_dict["gpt.text_head.weight"].t().contiguous(), device)
        self.text_head_bias = _to_device(state_dict["gpt.text_head.bias"], device)
        self.mel_head_weight = _to_device(state_dict["gpt.mel_head.weight"].t().contiguous(), device)
        self.mel_head_bias = _to_device(state_dict["gpt.mel_head.bias"], device)

    def _embed(self, ids, tok_weight, pos_weight):
        """ids: torch int tensor [batch, seq] -> ttnn [batch, seq, hidden] (token + position).

        Token ids arrive from the host; position ids 0..seq-1 are generated on
        device with ``ttnn.arange`` (no torch fallback).
        """
        seq = ids.shape[1]
        ids_tt = ttnn.from_torch(
            ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )
        pos_tt = ttnn.arange(0, seq, 1, dtype=ttnn.uint32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_tt = ttnn.reshape(pos_tt, (1, seq))  # [seq] -> [1, seq]; broadcasts over batch on add

        tok = ttnn.to_layout(ttnn.embedding(ids_tt, tok_weight), ttnn.TILE_LAYOUT)
        ttnn.deallocate(ids_tt)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, pos_weight), ttnn.TILE_LAYOUT)
        ttnn.deallocate(pos_tt)
        emb = ttnn.add(tok, pos)
        ttnn.deallocate(tok)
        ttnn.deallocate(pos)
        return emb

    def forward(self, text_ids, mel_ids, cond_latents=None):
        """text_ids / mel_ids are torch int tensors ``[batch, seq]``.

        ``cond_latents`` (optional) is a ttnn tensor ``[b, n_cond, hidden]`` of
        audio conditioning latents prepended as the GPT prompt; it is stripped
        (via ``offset``) after the stack, before the heads. ``n_cond`` should be
        tile-aligned (32 for XTTS). Returns ``(text_logits, mel_logits)`` on device.
        """
        text_len, mel_len = text_ids.shape[1], mel_ids.shape[1]

        text_emb = self._embed(text_ids, self.text_emb_weight, self.text_pos_weight)
        mel_emb = self._embed(mel_ids, self.mel_emb_weight, self.mel_pos_weight)

        parts, offset = [text_emb, mel_emb], 0
        if cond_latents is not None:
            parts = [cond_latents] + parts
            offset = cond_latents.shape[1]

        emb = ttnn.concat(parts, dim=1)  # [b, (n_cond +) text_len + mel_len, hidden]
        ttnn.deallocate(text_emb)  # copied into emb; cond_latents is the caller's, left alone
        ttnn.deallocate(mel_emb)
        enc = self.stack(emb)  # frees emb internally
        if offset:
            enc_stripped = ttnn.slice(enc, [0, offset, 0], [enc.shape[0], enc.shape[1], HIDDEN_SIZE])  # strip prompt
            ttnn.deallocate(enc)
            enc = enc_stripped
        enc_n = ttnn.layer_norm(enc, weight=self.final_norm_weight, bias=self.final_norm_bias, epsilon=LAYER_NORM_EPS)
        ttnn.deallocate(enc)

        b = enc_n.shape[0]
        text_part = ttnn.slice(enc_n, [0, 0, 0], [b, text_len, HIDDEN_SIZE])
        mel_part = ttnn.slice(enc_n, [0, text_len, 0], [b, text_len + mel_len, HIDDEN_SIZE])
        ttnn.deallocate(enc_n)

        text_logits = ttnn.linear(text_part, self.text_head_weight, bias=self.text_head_bias)
        ttnn.deallocate(text_part)
        mel_logits = ttnn.linear(mel_part, self.mel_head_weight, bias=self.mel_head_bias)
        ttnn.deallocate(mel_part)
        return text_logits, mel_logits

    # ------------------------------------------------------------------ #
    # Autoregressive KV-cache decode (see tt/xtts_generator.py).
    # Prefill covers only [cond | text]; the start_audio token is fed as the first
    # decode step, so mel positions match the reference (start=0, code_i=i+1).
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Autoregressive decode over a FIXED-size KV cache (the model's PREFILL + DECODE — mirrors
    # the block's two forwards). The cache is a [1, heads, max_seq, head_dim] buffer written in
    # place at a device-driven position, so every decode step is the SAME static-shape op sequence
    # (no concat growth, and capturable as one trace). Prefill covers only [cond | text]; the
    # start_audio token is fed as the first decode step (mel positions: start=0, code_i=i+1).
    # ------------------------------------------------------------------ #
    def _prefill_kv(self, text_ids, cond_latents):
        """Compute per-layer K/V from the prompt ``[cond_latents | text]`` (internal helper used to
        seed the fixed cache). ``text_ids`` torch ``[b, text_len]``; ``cond_latents`` ttnn
        ``[b, n_cond, hidden]`` (TILE). Returns the per-layer ``(k, v)`` list; no logits."""
        text_emb = self._embed(text_ids, self.text_emb_weight, self.text_pos_weight)
        prefix = ttnn.concat([cond_latents, text_emb], dim=1)  # [b, n_cond + text_len, hidden]
        ttnn.deallocate(text_emb)  # copied into prefix; cond_latents is the caller's
        prompt_ln, kv = self.stack.forward_prefill(prefix)  # frees prefix internally
        ttnn.deallocate(prompt_ln)  # prompt-position outputs are unused (only the cache is kept)
        return kv

    def prefill(self, text_ids, cond_latents, max_seq):
        """PREFILL: enable the fixed-size KV cache (``max_seq``), seed it from the ``[cond | text]``
        prompt, and remember ``prompt_len`` (mel token ``i`` occupies cache position
        ``prompt_len + i``). Returns the per-layer fixed KV cache the decode loop updates in place."""
        self.init_static_decode(max_seq)
        kv, prompt_len = self.prefill_static(text_ids, cond_latents)
        self.prompt_len = prompt_len
        return kv

    def decode(self, token_id, mel_pos, kv):
        """One DECODE step (single mel token) over the FIXED cache — no concat growth. ``token_id`` /
        ``mel_pos`` are Python ints. Builds the per-step device index/position tensors and runs the
        static in-place decode. Returns ``(logits [1, 1, NUM_AUDIO_TOKENS], latent [1, 1, hidden],
        kv)`` — ``kv`` is the same fixed cache, updated in place."""
        logits, latent = self.decode_static(
            self._pos_ids(token_id), self._pos_ids(mel_pos), self.cache_pos(self.prompt_len + mel_pos), kv
        )
        return logits, latent, kv

    # ------------------------------------------------------------------ #
    # Static-KV helpers shared by the eager decode above and the traced pipeline.
    # ------------------------------------------------------------------ #
    def init_static_decode(self, max_seq):
        """Enable the static-KV decode path for a fixed context length ``max_seq``."""
        self.max_seq = max_seq
        self.stack.init_static(max_seq)

    def _pos_ids(self, value):
        """``[1, 1]`` uint32 index tensor (token id or embedding position) on device."""
        return ttnn.from_torch(
            torch.tensor([[value]], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
        )

    def cache_pos(self, value):
        """``[1, 1, 1, max_seq]`` tensor filled with the absolute cache position ``value``."""
        return ttnn.from_torch(
            torch.full((1, 1, 1, self.max_seq), float(value), dtype=torch.float32),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.float32,
        )

    def prefill_static(self, text_ids, cond_latents):
        """Seed fixed-size caches from the ``[cond | text]`` prompt. Reuses the (eager, one-shot)
        concat prefill, then pads each layer's K/V out to ``max_seq``. Returns ``(kv, prompt_len)``
        where the mel token for step ``i`` occupies absolute cache position ``prompt_len + i``."""
        kv = self._prefill_kv(text_ids, cond_latents)  # list of (k, v) [1, heads, prompt_len, head_dim]
        prompt_len = kv[0][0].shape[2]
        pad = self.max_seq - prompt_len
        kv_static = []
        for k, v in kv:
            zeros = ttnn.from_torch(
                torch.zeros(1, NUM_HEADS, pad, HEAD_DIM),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            k2 = ttnn.concat([k, zeros], dim=2)
            zeros2 = ttnn.from_torch(
                torch.zeros(1, NUM_HEADS, pad, HEAD_DIM),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            v2 = ttnn.concat([v, zeros2], dim=2)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            kv_static.append((k2, v2))
        return kv_static, prompt_len

    def decode_static(self, token_ids, mel_pos_ids, cache_pos, kv):
        """One trace-compatible decode step. All per-step inputs are DEVICE tensors so a single
        capture replays at any position: ``token_ids``/``mel_pos_ids`` are ``[1, 1]`` uint32
        (audio code id + mel position for the embeddings); ``cache_pos`` is ``[1, 1, 1, max_seq]``
        (absolute cache write/attention position). Returns ``(logits, latent, new_kv)``."""
        # Gather both embeddings in ROW_MAJOR and add there, then a SINGLE tilize to TILE. A
        # per-embedding to_layout(TILE) is two TilizeWithValPadding ops (each padding the seq-1 row
        # to a full tile) on the decode hot path; adding first folds them into one conversion.
        tok = ttnn.embedding(token_ids, self.mel_emb_weight)  # ROW_MAJOR
        posn = ttnn.embedding(mel_pos_ids, self.mel_pos_weight)  # ROW_MAJOR
        x = ttnn.to_layout(ttnn.add(tok, posn), ttnn.TILE_LAYOUT)  # [1, 1, hidden], one tilize
        ttnn.deallocate(tok)
        ttnn.deallocate(posn)
        hidden = self.stack.forward_decode(x, kv, cache_pos)  # kv caches updated in place
        latent = ttnn.layer_norm(
            hidden, weight=self.final_norm_weight, bias=self.final_norm_bias, epsilon=LAYER_NORM_EPS
        )
        ttnn.deallocate(hidden)
        logits = ttnn.linear(latent, self.mel_head_weight, bias=self.mel_head_bias)
        return logits, latent

    # ------------------------------------------------------------------ #
    # Fully-traced setup: device-input variants (no host->device write inside — writes are FATAL
    # during trace capture) + a persistent pre-allocated KV cache seeded via ``fill_cache``, so
    # the whole prompt path (conditioning->prefill) can be captured as one "setup" trace whose
    # output caches the decode trace then reads/updates in place.
    # ------------------------------------------------------------------ #
    def alloc_static_kv(self, max_seq):
        """Enable static decode and pre-allocate the persistent per-layer KV cache (zeros). These
        buffers are seeded by ``prefill_dev`` and updated by ``decode_static`` — both in place."""
        self.init_static_decode(max_seq)
        # Precompute text position ids [1, max_seq] uint32 ONCE (host->device write here, outside any
        # capture) so prefill_dev's _embed_dev can SLICE it instead of calling ttnn.arange — arange
        # is a host write and is fatal inside a trace capture.
        self._text_pos_full = ttnn.from_torch(
            torch.arange(max_seq, dtype=torch.int32).reshape(1, max_seq),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
        )
        self._static_kv = []
        for _ in range(self.stack.num_layers):
            k = ttnn.from_torch(
                torch.zeros(1, NUM_HEADS, max_seq, HEAD_DIM),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            v = ttnn.from_torch(
                torch.zeros(1, NUM_HEADS, max_seq, HEAD_DIM),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.bfloat16,
            )
            self._static_kv.append((k, v))
        return self._static_kv

    def text_ids_to_device(self, text_ids):
        """Host text ids ``[1, seq]`` -> device uint32 ROW_MAJOR (the ``from_torch`` host->device
        write, kept OUTSIDE any trace capture)."""
        return ttnn.from_torch(
            text_ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )

    def _embed_dev(self, ids_tt, tok_weight, pos_weight):
        """Like ``_embed`` but ``ids_tt`` is an already-on-device uint32 ``[1, seq]`` tensor and the
        positions are SLICED from the precomputed ``self._text_pos_full`` (a device slice, no host
        write), so the whole thing is trace-capturable (``ttnn.arange`` would be a fatal in-capture write)."""
        seq = ids_tt.shape[1]
        pos_tt = ttnn.slice(self._text_pos_full, [0, 0], [1, seq])  # [1, seq] uint32, device slice
        tok = ttnn.to_layout(ttnn.embedding(ids_tt, tok_weight), ttnn.TILE_LAYOUT)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, pos_weight), ttnn.TILE_LAYOUT)
        ttnn.deallocate(pos_tt)
        emb = ttnn.add(tok, pos)
        ttnn.deallocate(tok)
        ttnn.deallocate(pos)
        return emb

    def prefill_dev(self, text_ids_tt, cond_latents):
        """Trace-capturable prefill: device text ids + cond_latents -> full causal prefill over
        ``[cond | text]`` -> seed the persistent ``self._static_kv`` in place with ``fill_cache``.
        No host->device write inside. Returns ``prompt_len`` (mel token i -> cache pos prompt_len+i)."""
        text_emb = self._embed_dev(text_ids_tt, self.text_emb_weight, self.text_pos_weight)
        prefix = ttnn.concat([cond_latents, text_emb], dim=1)  # [1, n_cond + text_len, hidden]
        ttnn.deallocate(text_emb)
        prompt_ln, kv = self.stack.forward_prefill(prefix)  # per-layer (k, v) [1, heads, prompt_len, head_dim]
        ttnn.deallocate(prompt_ln)
        prompt_len = kv[0][0].shape[2]
        for i, (k, v) in enumerate(kv):
            ttnn.fill_cache(self._static_kv[i][0], k, 0)  # write prompt K into positions 0..prompt_len-1
            ttnn.fill_cache(self._static_kv[i][1], v, 0)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        return prompt_len
