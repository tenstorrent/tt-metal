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
from models.experimental.xtts.tt.xtts_gpt_block import _mm_1d_config, _to_device, _to_device_w8, sharded_decode_ln
from models.experimental.xtts.tt.xtts_gpt_stack import TtXttsGptStack

###############################debugging###############################


def _tensor_mem_tag(t: ttnn.Tensor) -> str:
    """Compact memory-config label: buffer (L1/DRAM) + interleaved vs sharded layout."""
    mem = t.memory_config()
    buf = "L1" if mem.buffer_type == ttnn.BufferType.L1 else "DRAM"
    layout = mem.memory_layout
    if layout == ttnn.TensorMemoryLayout.INTERLEAVED or not mem.is_sharded():
        placement = "INTERLEAVED"
    elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        placement = "BLOCK_SHARDED"
    elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        placement = "WIDTH_SHARDED"
    elif layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        placement = "HEIGHT_SHARDED"
    else:
        placement = str(layout).split(".")[-1] if layout is not None else "UNKNOWN"
    tag = f"{buf}/{placement}"
    shard_spec = mem.shard_spec
    if shard_spec is not None:
        tag += f" grid={shard_spec.grid} shard={tuple(shard_spec.shape)}"
    return tag


def _debug_tensor_mem(label: str, t: ttnn.Tensor) -> None:
    """Print tensor shape, dtype, and memory placement between vision-block ops."""
    print(f"[block] {label}: shape={list(t.shape)} dtype={t.dtype} mem={_tensor_mem_tag(t)}")


###############################debugging###############################


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
        # mel_head weight in bfloat8_b (decode matmul weight, like the block's mlp weights) — half the
        # DRAM bytes of bf16; validated output-neutral (exact 16/16) by the generate test.
        self.mel_head_weight = _to_device_w8(state_dict["gpt.mel_head.weight"].t().contiguous(), device)
        # [1, N] (rank>=2) so the bias fuses into the tuned mel_head matmul epilogue (see decode_on_device).
        self.mel_head_bias = _to_device(state_dict["gpt.mel_head.bias"].reshape(1, -1), device)

    def _embed(self, ids, tok_weight, pos_weight):
        """ids: torch int tensor [batch, seq] -> ttnn [batch, seq, hidden] (token + position).

        Token ids arrive from the host; position ids 0..seq-1 are generated on
        device with ``ttnn.arange`` (no torch fallback).
        """
        print(f"[TtXttsGptModel._embed] ids={list(ids.shape)}")
        seq = ids.shape[1]
        ids_tt = ttnn.from_torch(
            ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )
        pos_tt = ttnn.arange(0, seq, 1, dtype=ttnn.uint32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_tt = ttnn.reshape(pos_tt, (1, seq))  # [seq] -> [1, seq]; broadcasts over batch on add

        tok = ttnn.to_layout(ttnn.embedding(ids_tt, tok_weight), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ids_tt)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, pos_weight), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_tt)
        emb = ttnn.add(tok, pos, memory_config=ttnn.L1_MEMORY_CONFIG)
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
        print(
            f"[TtXttsGptModel.forward] text_ids={list(text_ids.shape)} mel_ids={list(mel_ids.shape)} cond_latents={list(cond_latents.shape) if cond_latents is not None else None}"
        )
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
        enc_n = ttnn.layer_norm(
            enc,
            weight=self.final_norm_weight,
            bias=self.final_norm_bias,
            epsilon=LAYER_NORM_EPS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
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

    # ================================================================== #
    # The model's TWO main ops: PREFILL (seed the KV cache from the prompt) and DECODE (one mel
    # token). Both use a FIXED-size [1, heads, max_seq, head_dim] cache written in place at a
    # device-driven position — no concat growth, same static-shape op sequence every step (so the
    # decode step is capturable as one trace). Prefill covers [cond | text]; the start_audio token
    # is fed as the first decode step (mel positions: start=0, code_i=i+1). Each main is a thin
    # EAGER wrapper (builds its device inputs with host writes) over a device-input CORE
    # (``prefill_on_device`` / ``decode_on_device``) that the traced pipeline calls directly inside
    # a capture, where host->device writes are illegal.
    # ================================================================== #
    def prefill(self, text_ids, cond_latents, max_seq):
        """PREFILL (main). Allocate the fixed KV cache (``max_seq``) and seed it from the
        ``[cond | text]`` prompt; remember ``prompt_len`` (mel token ``i`` -> cache pos
        ``prompt_len + i``). ``text_ids`` torch ``[1, text_len]``; ``cond_latents`` ttnn
        ``[1, n_cond, hidden]``. Returns the per-layer fixed cache the decode loop updates in place."""
        print(
            f"[TtXttsGptModel.prefill] text_ids={list(text_ids.shape)} cond_latents={list(cond_latents.shape) if cond_latents is not None else None} max_seq={max_seq}"
        )
        self.alloc_static_kv(max_seq)
        self.prompt_len = self.prefill_on_device(self.text_ids_to_device(text_ids), cond_latents)
        return self._static_kv

    def decode(self, token_id, mel_pos, kv):
        """DECODE (main). One mel token over the fixed cache. ``token_id`` / ``mel_pos`` are Python
        ints; builds the per-step device index/position tensors and runs the core. Returns
        ``(logits [1, 1, NUM_AUDIO_TOKENS], latent [1, 1, hidden], kv)`` — ``kv`` updated in place."""
        pos = self.prompt_len + mel_pos
        logits, latent = self.decode_on_device(
            self._pos_ids(token_id), self._pos_ids(mel_pos), self.cache_pos(pos), kv, write_idx=pos
        )
        return logits, latent, kv

    # ------------------------------------------------------------------ #
    # Device-input cores (trace-safe: NO host->device writes) + the fixed-cache helpers, shared by
    # the eager mains above and the traced pipeline (tt/xtts_generator.py, tt/xtts_inference.py).
    # ------------------------------------------------------------------ #
    def init_static_decode(self, max_seq):
        """Enable the static-KV decode path for a fixed context length ``max_seq``."""
        self.max_seq = max_seq
        self.stack.init_static(max_seq)

    def _pos_ids(self, value):
        """``[1, 1]`` uint32 index tensor (token id or embedding position) on device."""
        # print(f"[TtXttsGptModel._pos_ids] value={value}")  # per-decode-step: kept quiet on the hot path
        return ttnn.from_torch(
            torch.tensor([[value]], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
        )

    def cache_pos(self, value):
        """``[1, 1, 1, max_seq]`` tensor filled with the absolute cache position ``value``."""
        # print(f"[TtXttsGptModel.cache_pos] value={value} max_seq={self.max_seq}")  # per-decode-step: quiet
        return ttnn.from_torch(
            torch.full((1, 1, 1, self.max_seq), float(value), dtype=torch.float32),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.float32,
        )

    def decode_on_device(self, token_ids, mel_pos_ids, cache_pos, kv, write_idx=None):
        """DECODE core (device inputs, trace-safe — no host->device write). All per-step inputs are
        DEVICE tensors so one capture replays at any position: ``token_ids``/``mel_pos_ids`` are
        ``[1, 1]`` uint32 (code id + mel position for the embeddings); ``cache_pos`` is
        ``[1, 1, 1, max_seq]`` (absolute write/attention position). ``write_idx`` (eager Python int)
        routes the cache write through the O(1) ``ttnn.update_cache``; None (traced) uses the
        data-driven one-hot write. Returns ``(logits, latent)``; the fixed cache is updated in place."""
        # Gather both embeddings in ROW_MAJOR and add there, then a SINGLE tilize to TILE. A
        # per-embedding to_layout(TILE) is two TilizeWithValPadding ops (each padding the seq-1 row
        # to a full tile) on the decode hot path; adding first folds them into one conversion.
        # Activations stay in L1 through the decode (weights remain DRAM) so the matmuls read
        # input-0 from L1 instead of a per-op DRAM round-trip.
        tok = ttnn.embedding(token_ids, self.mel_emb_weight, memory_config=ttnn.L1_MEMORY_CONFIG)  # ROW_MAJOR
        posn = ttnn.embedding(mel_pos_ids, self.mel_pos_weight, memory_config=ttnn.L1_MEMORY_CONFIG)  # ROW_MAJOR
        x = ttnn.to_layout(
            ttnn.add(tok, posn, memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # [1,1,hidden] — L1->L1 tilize (embeddings gathered in L1, no DRAM round-trip)
        ttnn.deallocate(tok)
        ttnn.deallocate(posn)
        hidden = self.stack.forward_decode(x, kv, cache_pos, write_idx=write_idx)  # kv updated in place
        latent = sharded_decode_ln(
            hidden, self.final_norm_weight, self.final_norm_bias, self.device
        )  # sharded final_norm
        # mel_head: tuned 1D-multicast config (same decode-optimal layout as the block linears) instead
        # of the auto config, so this per-token head matmul streams the DRAM weight over fewer cores.
        logits = ttnn.linear(
            latent,
            self.mel_head_weight,
            bias=self.mel_head_bias,
            program_config=_mm_1d_config(
                self.device, latent.shape[-2], latent.shape[-1], self.mel_head_weight.shape[-1]
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
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
        print(f"[TtXttsGptModel.alloc_static_kv] max_seq={max_seq}")
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
        # KV cache stays in DRAM: it's a large persistent buffer (~31 MB at max_seq=256 across 30
        # layers) AND — decisively — the traced decode does an in-place cache write
        # (where(onehot, k, k_cache, out=k_cache)); with the cache in L1 that write fails trace
        # capture (program.cpp:1612). L1 was validated to fit + stay exact in EAGER but broke the
        # traced path (the fast 176 tok/s path), and gave no measurable speedup — so DRAM it is.
        # KV cache stays bf16: ttnn.update_cache requires input.dtype == cache.dtype, so a bfp8 cache
        # would force casting every new k/v to bfp8 first (+2 ops/layer = +60 ops/token) on an
        # op-count-bound decode, plus bfp8 K/V degrades attention precision — a memory-only change
        # (31->15 MB) that isn't worth the op overhead + output risk.
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
        print(f"[TtXttsGptModel.text_ids_to_device] text_ids={list(text_ids.shape)}")
        return ttnn.from_torch(
            text_ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )

    def _embed_dev(self, ids_tt, tok_weight, pos_weight):
        """Like ``_embed`` but ``ids_tt`` is an already-on-device uint32 ``[1, seq]`` tensor and the
        positions are SLICED from the precomputed ``self._text_pos_full`` (a device slice, no host
        write), so the whole thing is trace-capturable (``ttnn.arange`` would be a fatal in-capture write)."""
        print(
            f"[TtXttsGptModel._embed_dev] ids_tt={list(ids_tt.shape)} tok_weight={list(tok_weight.shape)} pos_weight={list(pos_weight.shape)}"
        )
        seq = ids_tt.shape[1]
        pos_tt = ttnn.slice(self._text_pos_full, [0, 0], [1, seq])  # [1, seq] uint32, device slice
        tok = ttnn.to_layout(ttnn.embedding(ids_tt, tok_weight), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, pos_weight), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_tt)
        emb = ttnn.add(tok, pos, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        ttnn.deallocate(pos)
        return emb

    def prefill_on_device(self, text_ids_tt, cond_latents):
        """PREFILL core (device inputs, trace-safe — no host->device write). Device text ids +
        cond_latents -> full causal prefill over ``[cond | text]`` -> seed the pre-allocated
        ``self._static_kv`` in place with ``fill_cache``. Returns ``prompt_len`` (mel token i ->
        cache pos prompt_len + i). Requires ``alloc_static_kv`` first (zero cache + text-pos table)."""
        print(
            f"[TtXttsGptModel.prefill_on_device] text_ids_tt={list(text_ids_tt.shape)} cond_latents={list(cond_latents.shape) if cond_latents is not None else None}"
        )
        text_emb = self._embed_dev(text_ids_tt, self.text_emb_weight, self.text_pos_weight)
        prefix = ttnn.concat(
            [cond_latents, text_emb], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG
        )  # [1, n_cond+text, hidden]
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
