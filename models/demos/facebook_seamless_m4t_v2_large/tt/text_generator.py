# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T-v2 autoregressive text-decoder generation loop in TTNN.

This module composes:

    - :class:`TextDecoder`           (24-layer NLLB-style decoder)
    - :class:`TextDecoderKVCache`    (self + cross attention caches)
    - inline ``ttnn.linear`` LM head (tied to ``shared.weight``)

and drives the autoregressive ``generate(...)`` loop the way HF's
``SeamlessM4Tv2ForTextToText.generate(..., do_sample=False)`` does:

    1.  decoder_input_ids = [decoder_start_token_id, tgt_lang_id]
        (decoder_start = eos = 3 for SeamlessM4T-v2; tgt_lang_id from
         ``generation_config.text_decoder_lang_to_code_id[lang]``).
    2.  Populate the per-layer cross-attention K/V cache once from the
        encoder hidden states.
    3.  Warm up the self-attention KV cache with positions 0 and 1
        (the two prefix tokens). The logits produced at position 1
        are the first ones we sample from.
    4.  Loop: greedy ``argmax`` -> append -> next decode_step.
    5.  Stop on ``eos_token_id`` or when ``max_new_tokens`` reached.

This file deliberately does NOT touch any existing TTNN block files —
Phase-3 already added all the hooks we need (cross-attn prefill via
``TextDecoder.populate_cross_attention_cache`` and per-step decode via
``TextDecoder.decode_step``).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.kv_cache import TextDecoderKVCache
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder import TextDecoder

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


class TextGenerator(LightweightModule):
    """SeamlessM4T-v2 text-decoder generation driver.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: attention head count (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large).
        num_layers: number of decoder layers (24 for v2-Large).
        text_decoder_state_dict: nested dict consumed by
            :class:`TextDecoder` with keys
            ``{"embed_tokens": {"weight": ...},
               "embed_positions_weights": ...,
               "layers": [...],
               "layer_norm": {"weight", "bias"}}`` — same shape as the
            output of :func:`weight_loader.text_decoder_weights`.
        lm_head_state_dict: ``{"weight": (vocab_size, hidden_size)}``
            (HF ties this to ``shared.weight``).
        max_decode_seq_len: KV cache capacity in decoder tokens.
            Must be a multiple of 32. Limits the total length of the
            generated sequence (including the 2 prefix tokens).
        encoder_seq_len: tile-padded encoder sequence length to allocate
            the cross-attention cache for. Caller passes the SAME number
            they used to tile-pad the encoder output (which is what gets
            fed into populate_encoder_cache).
        eps: LayerNorm epsilon (default 1e-5).
        padding_idx: decoder padding token id (default 0).
        embed_scale: scalar applied to token embeddings; defaults to
            ``sqrt(embed_dim)``.
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM).

    Typical usage::

        gen = TextGenerator(device, hf_config, ...)
        gen.load_weights(text_decoder_state_dict, lm_head_state_dict)
        tokens = gen.generate(
            encoder_hidden_states=enc_hidden,     # [1, S, H] host torch
            encoder_attention_mask=mask_2d,       # [1, S] host torch (1=keep)
            decoder_start_token_id=3,
            tgt_lang_id=256026,                   # fra
            eos_token_id=3,
            max_new_tokens=24,
        )
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        text_decoder_state_dict,
        lm_head_state_dict,
        max_decode_seq_len: int,
        encoder_seq_len: int,
        eps: float = 1e-5,
        padding_idx: int = 0,
        embed_scale: Optional[float] = None,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        if max_decode_seq_len % _TILE != 0:
            raise ValueError(
                f"max_decode_seq_len({max_decode_seq_len}) must be a multiple of {_TILE} (KV cache tile alignment)."
            )
        if encoder_seq_len % _TILE != 0:
            raise ValueError(
                f"encoder_seq_len({encoder_seq_len}) must be a multiple of {_TILE} (cross-attn cache tile alignment)."
            )

        self.device = device
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_layers = int(num_layers)
        self.max_decode_seq_len = int(max_decode_seq_len)
        self.encoder_seq_len = int(encoder_seq_len)
        self.padding_idx = int(padding_idx)
        self.eps = float(eps)
        if embed_scale is None:
            embed_scale = math.sqrt(embed_dim)
        self.embed_scale = float(embed_scale)

        # 1. Build the (heavy) 24-layer text decoder.
        if len(text_decoder_state_dict["layers"]) != self.num_layers:
            raise ValueError(
                f"text_decoder_state_dict has {len(text_decoder_state_dict['layers'])} layers, "
                f"expected {self.num_layers}."
            )
        self.text_decoder = TextDecoder(
            device=device,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            embed_tokens_weight=text_decoder_state_dict["embed_tokens"]["weight"],
            embed_positions_weights=text_decoder_state_dict["embed_positions_weights"],
            layers_state_dict=text_decoder_state_dict["layers"],
            final_layer_norm_state_dict=text_decoder_state_dict["layer_norm"],
            eps=self.eps,
            padding_idx=self.padding_idx,
            embed_scale=self.embed_scale,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. LM head — bias=False, weight tied to shared.weight (HF). The
        # raw HF weight is (vocab, hidden); ttnn.linear expects
        # (in_features, out_features), so transpose on host.
        lm_w = lm_head_state_dict["weight"]
        self.lm_head_weight = ttnn.from_torch(
            lm_w.t().contiguous(),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # 3. Allocate the KV cache once. Reset between generate() calls.
        self.past_key_values = TextDecoderKVCache(
            device=device,
            num_layers=self.num_layers,
            batch=1,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_decode_seq_len=self.max_decode_seq_len,
            encoder_seq_len=self.encoder_seq_len,
            dtype=weight_dtype,
        )

        # 4. Persistent buffers for the metal-trace AR loop. Lazily filled
        # in by :meth:`_ensure_persistent_buffers` because the vocab size
        # is implied only by ``lm_head_state_dict["weight"].shape[0]``.
        self._persistent_input_ids_tt: Optional[ttnn.Tensor] = None
        self._persistent_position_ids_tt: Optional[ttnn.Tensor] = None
        self._persistent_self_masks_tt: dict = {}  # pos -> ttnn.Tensor
        self._persistent_encoder_mask_tt: Optional[ttnn.Tensor] = None
        self._vocab_size = int(lm_head_state_dict["weight"].shape[0])
        # Trace plumbing: one trace per position because ttnn.update_cache
        # only accepts an integer update_idx -- the constant is baked into
        # the captured trace, so each position needs its own. Map from
        # position -> trace_id and from position -> output logits buffer.
        self._decode_trace_ids: dict = {}
        self._decode_trace_outputs: dict = {}
        # Track which positions have a captured trace, so we can lazily
        # capture only as needed (the e2e tests rarely run to max_seq).
        self._captured_positions: set = set()

    # ----------------------------------------------------------------- helpers

    def _pack_token(self, token_id: int) -> torch.Tensor:
        """Pack one scalar token id into a host ``[1, 1]`` int tensor."""
        return torch.tensor([[int(token_id)]], dtype=torch.long)

    # ------------------------------------------------------------- trace plumbing
    def _ensure_persistent_buffers(self) -> None:
        """Lazily allocate the persistent device tensors used by the AR trace.

        These are written into via :func:`ttnn.copy_host_to_device_tensor`
        per decode step (no per-step allocation), then read by the
        captured trace. The output ``logits_out`` slot is also persistent
        so we can ``ttnn.to_torch`` it after each ``execute_trace``.
        """
        if self._persistent_input_ids_tt is not None:
            return
        # Persistent uint32 ROW_MAJOR DRAM buffer for the single new token id.
        self._persistent_input_ids_tt = ttnn.from_torch(
            torch.zeros((1, 1), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Persistent uint32 ROW_MAJOR DRAM buffer for the position id we
        # gather into the sinusoidal embedding table. Shape [1, 1] matches
        # what ttnn.embedding wants.
        self._persistent_position_ids_tt = ttnn.from_torch(
            torch.zeros((1, 1), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Persistent bf16 TILE DRAM buffer for the cross-attention mask.
        # Shape [1, 1, 1, encoder_seq_len_total]. Content is overwritten
        # per generate() via copy_host_to_device_tensor so the buffer
        # ADDRESS captured in the trace stays valid across generate
        # calls (and across captured traces). Initialized to the
        # all-keep-followed-by-tile-pad-fill pattern.
        enc_seq_total = self.encoder_seq_len
        fill = float(torch.finfo(torch.float32).min)
        # Default content: all-keep within src bounds (we'll overwrite
        # for real per generate()). bf16 storage to match the dtype the
        # captured trace reads.
        default_enc_mask = torch.full((1, 1, 1, enc_seq_total), fill, dtype=torch.float32)
        # No 2D mask info at __init__ time -- treat the whole seq as keep.
        default_enc_mask[:, :, :, :] = 0.0
        self._persistent_encoder_mask_tt = ttnn.from_torch(
            default_enc_mask,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _ensure_persistent_self_mask(self, position: int) -> ttnn.Tensor:
        """Build (and cache) the ``[1,1,1,max_seq]`` self-attn mask for ``position``.

        Positions ``[0..position]`` get additive zero; positions
        ``[position+1..max_seq-1]`` get the additive ``-inf`` fill so SDPA
        ignores empty cache slots beyond what we have written so far.

        These are uploaded ONCE during trace capture and reused on every
        replay of that particular position. We keep the masks in a dict
        keyed by position because each captured trace points at the mask
        buffer it was captured against — sharing one mask across all
        positions would force a copy per step (which we want to avoid).
        """
        tt = self._persistent_self_masks_tt.get(position)
        if tt is not None:
            return tt
        max_seq = self.past_key_values.self_attn.max_seq_len
        mask = self.text_decoder._build_decode_self_attention_mask(
            batch=1,
            position=position,
            max_seq_len=max_seq,
            dtype=torch.float32,
        )
        tt = self.text_decoder._to_tt(mask)
        self._persistent_self_masks_tt[position] = tt
        return tt

    def _run_decode_body(
        self,
        position: int,
        precomputed_encoder_mask_tt: Optional[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Run the decode_step + lm_head and return the logits tensor.

        Inputs are taken from the persistent buffers. Returns the device
        tensor produced by the LM head linear (no D2H -- the caller is
        responsible for ``ttnn.to_torch`` after the trace replay).
        """
        self_mask_tt = self._ensure_persistent_self_mask(position)
        hidden_tt = self.text_decoder.decode_step(
            input_ids=torch.zeros((1, 1), dtype=torch.long),  # unused (persistent buf path)
            position=int(position),
            past_key_values=self.past_key_values,
            precomputed_self_mask_tt=self_mask_tt,
            precomputed_encoder_mask_tt=precomputed_encoder_mask_tt,
            persistent_input_ids_tt=self._persistent_input_ids_tt,
            persistent_position_ids_tt=self._persistent_position_ids_tt,
        )
        logits_tt = ttnn.linear(
            hidden_tt,
            self.lm_head_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden_tt)
        return logits_tt

    def _capture_decode_trace(
        self,
        position: int,
        precomputed_encoder_mask_tt: Optional[ttnn.Tensor],
    ) -> int:
        """Capture (or reuse) a trace for decode at ``position`` and return its id.

        Captures exactly one trace per position. Subsequent calls with
        the same position return the cached id. The output device
        tensor (the LM-head linear's result) is retained in
        :attr:`_decode_trace_outputs[position]` so the caller can D2H
        from the SAME buffer the captured trace writes to.

        Pre-condition: the program cache must already be warm from a
        prior un-traced run of the same body at this position (otherwise
        capture will try to compile kernels which is illegal during
        capture). The caller arranges this by running the body once via
        :meth:`_run_decode_body` immediately before calling us.
        """
        if position in self._decode_trace_ids:
            return self._decode_trace_ids[position]
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            logits_tt = self._run_decode_body(
                position=position,
                precomputed_encoder_mask_tt=precomputed_encoder_mask_tt,
            )
        finally:
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._decode_trace_ids[position] = trace_id
        # Stash the output handle so `execute_trace`-then-D2H knows
        # which device tensor holds the freshly computed logits.
        self._decode_trace_outputs[position] = logits_tt
        self._captured_positions.add(position)
        return trace_id

    def _release_decode_traces(self) -> None:
        """Release every captured decode trace + persistent buffers.

        Optional; safe to skip when reusing the generator across multiple
        ``generate()`` calls.
        """
        for trace_id in self._decode_trace_ids.values():
            try:
                ttnn.release_trace(self.device, trace_id)
            except Exception:
                pass
        self._decode_trace_ids.clear()
        self._decode_trace_outputs.clear()
        self._captured_positions.clear()

    def _logits_from_hidden(self, hidden_tt: ttnn.Tensor) -> torch.Tensor:
        """Run hidden state through the LM head and return host ``[vocab]`` logits.

        ``hidden_tt`` is the decoder ``last_hidden_state`` for one decode
        step — shape ``[1, 1, embed_dim]`` in TILE_LAYOUT DRAM.
        """
        logits_tt = ttnn.linear(
            hidden_tt,
            self.lm_head_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden_tt)
        logits_torch = ttnn.to_torch(logits_tt).to(torch.float32)
        ttnn.deallocate(logits_tt)
        # Strip any leading singleton dims tt sometimes adds.
        while logits_torch.dim() > 1:
            if logits_torch.shape[0] == 1:
                logits_torch = logits_torch.squeeze(0)
            else:
                # [batch, seq, vocab] -> use last seq, batch 0.
                logits_torch = logits_torch[0, -1, :]
                break
        return logits_torch

    # ----------------------------------------------------------------- prefill
    def populate_encoder_cache(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """Populate the cross-attention K/V cache from encoder output.

        Returns the *logical* (un-padded) encoder sequence length so the
        caller can pass it through to ``decode_step`` for mask building.

        Args:
            encoder_hidden_states: host ``[1, S, H]`` float tensor — the
                encoder output. S need not be a multiple of 32; we
                tile-pad along S internally.
            encoder_attention_mask: optional host ``[1, S]`` mask
                (1=keep, 0=pad). Not used here, just returned for the
                caller's convenience.
        """
        if encoder_hidden_states.dim() != 3 or int(encoder_hidden_states.shape[0]) != 1:
            raise ValueError(f"encoder_hidden_states must be [1, S, H]; got {tuple(encoder_hidden_states.shape)}")
        src_logical = int(encoder_hidden_states.shape[1])
        src_padded = src_logical + _pad_to_tile(src_logical)
        if src_padded != self.encoder_seq_len:
            raise ValueError(
                f"encoder_hidden_states pads to {src_padded} along S; "
                f"TextGenerator was constructed with encoder_seq_len={self.encoder_seq_len}. "
                "Re-construct with the matching length."
            )
        # Always reset before re-populating to avoid double-allocation leaks.
        self.past_key_values.cross_attn.reset()
        self.text_decoder.populate_cross_attention_cache(
            past_key_values=self.past_key_values,
            encoder_hidden_states_torch=encoder_hidden_states,
        )
        return src_logical

    # ------------------------------------------------------------------ step
    def decode_step(
        self,
        prev_token: int,
        position: int,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_seq_len_logical: Optional[int] = None,
        precomputed_self_mask_tt: Optional[ttnn.Tensor] = None,
        precomputed_encoder_mask_tt: Optional[ttnn.Tensor] = None,
    ) -> torch.Tensor:
        """Run one autoregressive step and return ``[vocab_size]`` host logits.

        Pre-condition: ``populate_encoder_cache`` has been called.

        Args:
            prev_token: the token id to insert at ``position``. For
                position 0 this is ``decoder_start_token_id``; for
                position 1 it is ``tgt_lang_id``; for ``>=2`` it is the
                previously sampled token.
            position: 0-based KV-cache slot to write the new K/V into.
            encoder_attention_mask: optional host ``[1, S]`` mask
                (1=keep, 0=pad).
            encoder_seq_len_logical: pre-padding length of the encoder
                output. When None we let the decoder use the full
                tile-padded length (correct only when there was no
                tile padding).
            precomputed_self_mask_tt: optional pre-built ``[B, 1, 1, max_seq]``
                self-attn mask for THIS position. Skips the host-side
                build + upload per step.
            precomputed_encoder_mask_tt: optional pre-built
                ``[B, 1, 1, enc_seq_total]`` cross-attn mask. Invariant
                across all decode steps of one generate() call.
        """
        if not (0 <= position < self.max_decode_seq_len):
            raise ValueError(f"position({position}) outside [0, max_decode_seq_len={self.max_decode_seq_len})")
        input_ids = self._pack_token(prev_token)
        hidden_tt = self.text_decoder.decode_step(
            input_ids=input_ids,
            position=int(position),
            past_key_values=self.past_key_values,
            encoder_attention_mask=encoder_attention_mask,
            encoder_seq_len_logical=encoder_seq_len_logical,
            precomputed_self_mask_tt=precomputed_self_mask_tt,
            precomputed_encoder_mask_tt=precomputed_encoder_mask_tt,
        )
        return self._logits_from_hidden(hidden_tt)

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        decoder_start_token_id: int,
        tgt_lang_id: int,
        eos_token_id: int,
        max_new_tokens: int = 128,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        do_sample: bool = False,
        use_trace: bool = False,
    ) -> List[int]:
        """Run the autoregressive loop and return the generated token list.

        The returned list MATCHES the convention that HF's
        ``model.generate()`` returns for SeamlessM4Tv2ForTextToText:
        the leading prefix tokens ``[decoder_start, tgt_lang]`` are
        included, followed by the generated tokens (up to and including
        ``eos_token_id`` if EOS was hit before ``max_new_tokens``).
        """
        if do_sample:
            raise NotImplementedError(
                "TextGenerator.generate(do_sample=True) is not implemented yet — "
                "greedy decoding only (do_sample=False). Top-k / top-p can land later."
            )

        # ----- 1. Encoder cross-attn cache prefill -----
        # Reset self-attn cache too — generator instance may be re-used
        # across multiple generate() calls.
        self.past_key_values.self_attn.reset()
        src_logical = self.populate_encoder_cache(
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # In trace mode, ensure the persistent buffers exist before the
        # encoder-mask precompute so the mask is uploaded INTO the
        # persistent buffer (whose address is later captured into every
        # decode trace and must remain stable across generate() calls).
        if use_trace:
            self._ensure_persistent_buffers()
            # NOTE on trace lifetime: the per-position trace approach
            # used here bakes a constant ``update_idx=pos`` into every
            # captured trace (because the underlying ``ttnn.update_cache``
            # primitive only accepts a Python int, not a device tensor).
            # We've empirically observed that replaying these traces in a
            # second ``generate()`` call (with the captured K/V cache
            # buffers zero-reset between calls) produces incorrect logits
            # at the second decode position onward, despite all input
            # tensor addresses being preserved. This points at additional
            # implicit state captured by the trace (likely program-cache
            # IDs or pre-allocated workspace) that doesn't survive a
            # cross-call replay. Until we migrate the KV cache update to
            # ``paged_update_cache`` (which takes a ``cur_pos_tensor``
            # and so allows a SINGLE re-usable trace), we release the
            # captured traces here at the START of every generate() and
            # re-capture lazily in the AR loop. This keeps the
            # correctness guarantee while preserving the (smaller)
            # within-generation win of trace replay over un-traced.
            self._release_decode_traces()

        tokens: List[int] = [int(decoder_start_token_id), int(tgt_lang_id)]

        max_total = min(int(max_new_tokens), self.max_decode_seq_len)
        if max_total < 2:
            return tokens[:max_total]

        # ----- 1b. Pre-compute the per-call invariant cross-attn mask -----
        # Optimization (Phase-9): the cross-attention mask is invariant across
        # every decode step in this generate() call (encoder output is fixed
        # and the 2D padding mask is fixed). Uploading it ONCE here saves
        # max_total - 1 redundant host->device transfers (and the matching
        # max_total tilize passes) versus rebuilding inside decode_step.
        # The self-attention mask still depends on `position`, so we leave
        # that per-step for now -- it could in principle also be batched
        # but the encoder mask is the bigger win on Blackhole here (it's the
        # wider tensor: max_seq=128 vs. enc_seq up to 64-128 depending on
        # prompt) and the simpler change.
        precomp_encoder_mask_tt = self._precompute_encoder_mask(
            encoder_attention_mask=encoder_attention_mask,
            encoder_seq_len_logical=src_logical,
            batch=1,
        )

        try:
            if use_trace:
                tokens = self._generate_traced(
                    tokens=tokens,
                    max_total=max_total,
                    eos_token_id=int(eos_token_id),
                    encoder_attention_mask=encoder_attention_mask,
                    src_logical=src_logical,
                    precomp_encoder_mask_tt=precomp_encoder_mask_tt,
                )
            else:
                # ----- 2. Warm-up: feed positions 0 and 1 to populate self-attn cache -----
                # decoder_start_token_id (position 0) — its output logits are
                # discarded (HF's generation loop never reads them).
                _ = self.decode_step(
                    prev_token=tokens[0],
                    position=0,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_seq_len_logical=src_logical,
                    precomputed_encoder_mask_tt=precomp_encoder_mask_tt,
                )

                # tgt_lang_id (position 1) — its output IS the first sampled position.
                logits = self.decode_step(
                    prev_token=tokens[1],
                    position=1,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_seq_len_logical=src_logical,
                    precomputed_encoder_mask_tt=precomp_encoder_mask_tt,
                )

                # ----- 3. AR loop — positions 2, 3, ... -----
                # We sample after each forward; total length is bounded by
                # max_new_tokens AND max_decode_seq_len (cache capacity).
                # ``max_new_tokens`` counts the prefix tokens too (matches HF
                # default semantics for short-form generation).
                for pos in range(2, max_total):
                    # Greedy: argmax over vocab. The logits we hold were produced
                    # by the PREVIOUS decode_step (whose KV was written at
                    # position pos-1) -- this is the token prediction for slot pos.
                    next_token = int(torch.argmax(logits).item())
                    tokens.append(next_token)
                    if next_token == int(eos_token_id):
                        break
                    if pos + 1 >= max_total:
                        # Reached the budget; no point in running one more step.
                        break
                    # Run one more step to get logits for slot pos+1.
                    logits = self.decode_step(
                        prev_token=next_token,
                        position=pos,
                        encoder_attention_mask=encoder_attention_mask,
                        encoder_seq_len_logical=src_logical,
                        precomputed_encoder_mask_tt=precomp_encoder_mask_tt,
                    )
        finally:
            # In trace mode, ``precomp_encoder_mask_tt`` is the persistent
            # buffer -- do NOT deallocate it (the captured traces hold a
            # pointer to it). In untraced mode it's a per-call upload, so
            # release it here.
            if precomp_encoder_mask_tt is not None and precomp_encoder_mask_tt is not self._persistent_encoder_mask_tt:
                ttnn.deallocate(precomp_encoder_mask_tt)

        return tokens

    # --------------------------------------------------- trace AR loop
    def _write_input_ids(self, token_id: int) -> None:
        """Write a single token id into :attr:`_persistent_input_ids_tt` via H2D."""
        host = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host, self._persistent_input_ids_tt)

    def _write_position_id(self, position: int) -> None:
        """Write the sinusoidal-table position id (== position + 1 for non-pad)."""
        host = ttnn.from_torch(
            torch.tensor([[int(position) + 1]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host, self._persistent_position_ids_tt)

    # Optional callback invoked once per AR step in _generate_traced after
    # ``ttnn.synchronize_device`` so profilers can measure host-perceived
    # per-step latency. Signature: ``fn(position: int, ms: float, kind: str)``
    # where kind is "warmup" (untraced warmup pass), "capture" (trace
    # capture), or "replay" (steady-state execute_trace).
    step_callback = None

    def _generate_traced(
        self,
        tokens: List[int],
        max_total: int,
        eos_token_id: int,
        encoder_attention_mask: Optional[torch.Tensor],
        src_logical: int,
        precomp_encoder_mask_tt: Optional[ttnn.Tensor],
    ) -> List[int]:
        """Metal-trace AR loop.

        Strategy: one trace per absolute position (positions 1..max-1),
        because ``ttnn.update_cache(cache, k_new, update_idx=int(pos))``
        captures ``pos`` as a static constant — a single re-usable trace
        would always write into slot 0 on replay.

        We capture lazily on the first hit of a given position. The first
        decode call at each new position pays warmup+capture latency; all
        subsequent calls at that position replay the trace and skip
        host-side op dispatch entirely.
        """
        import time as _time

        # Persistent input buffers were allocated in generate() before
        # the encoder-mask precompute (so the mask lands in the persistent
        # buffer). Call again here for safety -- idempotent.
        self._ensure_persistent_buffers()

        # ---- positions 0 and 1: warmup, no trace ----
        # Position 0 (decoder_start_token_id): its logits are discarded.
        self._write_input_ids(tokens[0])
        self._write_position_id(0)
        _t0 = _time.perf_counter()
        _wu0 = self._run_decode_body(position=0, precomputed_encoder_mask_tt=precomp_encoder_mask_tt)
        ttnn.synchronize_device(self.device)
        if self.step_callback is not None:
            self.step_callback(0, (_time.perf_counter() - _t0) * 1000.0, "warmup")
        ttnn.deallocate(_wu0)

        # Position 1 (tgt_lang_id): first logits we actually sample from.
        self._write_input_ids(tokens[1])
        self._write_position_id(1)
        # Capture trace for pos=1 so we get this step ALSO under trace if
        # we ever return to it; but actually pos 1 only happens once per
        # generate(), so we just run + read.
        _t0 = _time.perf_counter()
        logits_tt = self._run_decode_body(position=1, precomputed_encoder_mask_tt=precomp_encoder_mask_tt)
        logits = ttnn.to_torch(logits_tt).to(torch.float32)
        if self.step_callback is not None:
            self.step_callback(1, (_time.perf_counter() - _t0) * 1000.0, "warmup")
        ttnn.deallocate(logits_tt)
        # Strip leading singleton dims.
        while logits.dim() > 1:
            if logits.shape[0] == 1:
                logits = logits.squeeze(0)
            else:
                logits = logits[0, -1, :]
                break

        # ---- AR loop: positions 2..max_total-1 ----
        for pos in range(2, max_total):
            next_token = int(torch.argmax(logits).item())
            tokens.append(next_token)
            if next_token == int(eos_token_id):
                break
            if pos + 1 >= max_total:
                break

            # Write inputs for THIS step (decode at slot pos).
            self._write_input_ids(next_token)
            self._write_position_id(pos)

            if pos not in self._decode_trace_ids:
                # First time hitting this position: untraced warmup so
                # the program cache holds every kernel, THEN capture.
                _t_wu = _time.perf_counter()
                _wu = self._run_decode_body(
                    position=pos,
                    precomputed_encoder_mask_tt=precomp_encoder_mask_tt,
                )
                ttnn.synchronize_device(self.device)
                if self.step_callback is not None:
                    self.step_callback(pos, (_time.perf_counter() - _t_wu) * 1000.0, "warmup")
                ttnn.deallocate(_wu)

                _t_cap = _time.perf_counter()
                self._capture_decode_trace(pos, precomp_encoder_mask_tt)
                if self.step_callback is not None:
                    self.step_callback(pos, (_time.perf_counter() - _t_cap) * 1000.0, "capture")

            # Replay the captured trace.
            _t_rep = _time.perf_counter()
            ttnn.execute_trace(self.device, self._decode_trace_ids[pos], cq_id=0, blocking=True)
            logits_tt = self._decode_trace_outputs[pos]
            logits = ttnn.to_torch(logits_tt).to(torch.float32)
            if self.step_callback is not None:
                self.step_callback(pos, (_time.perf_counter() - _t_rep) * 1000.0, "replay")
            # NB: do NOT deallocate logits_tt — it's the persistent
            # output of the trace and we'll read it again next time the
            # same position runs (across multiple generate() calls).
            while logits.dim() > 1:
                if logits.shape[0] == 1:
                    logits = logits.squeeze(0)
                else:
                    logits = logits[0, -1, :]
                    break

        return tokens

    # ------------------------------------------------------------------ mask precompute helpers
    def _precompute_encoder_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        encoder_seq_len_logical: int,
        batch: int,
    ) -> Optional["ttnn.Tensor"]:
        """Build and upload the (per-call invariant) cross-attention mask.

        When trace mode is active and the persistent encoder-mask buffer
        has been allocated, we write the host-built mask INTO it via
        ``ttnn.copy_host_to_device_tensor`` and return that buffer (so
        the SAME device address is used across generate() calls and
        consequently across all captured decode traces).

        Returns None if no mask is needed (no key padding + no tile pad)
        AND we're NOT in trace mode (the persistent buffer always has
        valid content, so the trace path always passes a real mask).
        """
        td = self.text_decoder
        enc_seq_total = self.past_key_values.cross_attn.encoder_seq_len
        if encoder_attention_mask is not None:
            src_len = int(encoder_attention_mask.shape[1])
        else:
            src_len = int(encoder_seq_len_logical)

        if self._persistent_encoder_mask_tt is not None:
            # Trace mode: write into the persistent buffer.
            enc_mask_torch = td._build_decode_encoder_attention_mask(
                encoder_attention_mask_2d=encoder_attention_mask,
                batch=batch,
                encoder_seq_len_total=enc_seq_total,
                src_len=src_len,
                dtype=torch.float32,
            )
            host_tt = ttnn.from_torch(
                enc_mask_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_tt, self._persistent_encoder_mask_tt)
            return self._persistent_encoder_mask_tt

        # Non-trace mode: original behaviour.
        if encoder_attention_mask is None and enc_seq_total == src_len:
            return None
        enc_mask_torch = td._build_decode_encoder_attention_mask(
            encoder_attention_mask_2d=encoder_attention_mask,
            batch=batch,
            encoder_seq_len_total=enc_seq_total,
            src_len=src_len,
            dtype=torch.float32,
        )
        return td._to_tt(enc_mask_torch)
