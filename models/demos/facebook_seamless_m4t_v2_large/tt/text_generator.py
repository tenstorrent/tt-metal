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
        # Self-attn mask is now a SINGLE persistent buffer (one slot for
        # all positions). We rewrite its content per step via H2D copy --
        # because the captured trace reads from the same device address,
        # this lets a single trace handle all positions.
        self._persistent_self_mask_tt: Optional[ttnn.Tensor] = None
        self._persistent_encoder_mask_tt: Optional[ttnn.Tensor] = None
        self._vocab_size = int(lm_head_state_dict["weight"].shape[0])
        # Single decode trace (re-usable across all positions thanks to
        # paged_update_cache + tensor-valued cur_pos + persistent self mask
        # buffer). We retain the captured trace id and its output logits
        # buffer; re-using both across positions AND across generate()
        # calls is correct because all device addresses are stable.
        self._decode_trace_id: Optional[int] = None
        self._decode_trace_output_tt: Optional[ttnn.Tensor] = None
        # Track whether the program cache for the decode body has been
        # warmed up at least once (gated by an untraced warmup call).
        self._decode_kernels_compiled: bool = False

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
        # Persistent bf16 TILE DRAM buffer for the self-attention mask.
        # Shape [1, 1, 1, max_decode_seq_len]. We OVERWRITE its content
        # per step via copy_host_to_device_tensor — because the buffer
        # ADDRESS is stable, a single captured trace can re-read the
        # mask each replay and effectively decode any position.
        max_seq = self.past_key_values.self_attn.max_seq_len
        default_self_mask = torch.full((1, 1, 1, max_seq), fill, dtype=torch.float32)
        default_self_mask[:, :, :, :1] = 0.0  # position 0 only at startup
        self._persistent_self_mask_tt = ttnn.from_torch(
            default_self_mask,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Pre-build the per-position host-resident tile tensors once.
        # Per-step we only need a copy_host_to_device_tensor (no fresh
        # from_torch/tilize). This trades a small constant memory cost
        # for a fast per-step path.
        self._self_mask_hosts: dict = {}
        for p in range(max_seq):
            m = self.text_decoder._build_decode_self_attention_mask(
                batch=1,
                position=p,
                max_seq_len=max_seq,
                dtype=torch.float32,
            )
            self._self_mask_hosts[p] = ttnn.from_torch(
                m,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        # Pre-build per-position int32 host tensors for the KV-cache
        # update_idxs_tensor write. Avoids per-step from_torch.
        self._cache_pos_hosts: dict = {}
        for p in range(max_seq):
            self._cache_pos_hosts[p] = ttnn.from_torch(
                torch.tensor([int(p)], dtype=torch.int32),
                dtype=ttnn.int32,
            )
        # Pre-build per-position uint32 host tensors for the sinusoidal
        # position-id write (position + 1 because pad_idx=0 occupies row
        # 0 of the sinusoidal table).
        self._sin_pos_hosts: dict = {}
        for p in range(max_seq):
            self._sin_pos_hosts[p] = ttnn.from_torch(
                torch.tensor([[int(p) + 1]], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

    def _write_self_mask(self, position: int) -> ttnn.Tensor:
        """Stream the pre-built ``[1,1,1,max_seq]`` self-attn mask for
        ``position`` into the persistent mask buffer and return that
        buffer.

        Positions ``[0..position]`` get additive zero; positions
        ``[position+1..max_seq-1]`` get the additive ``-inf`` fill so
        SDPA ignores empty cache slots beyond what we have written so
        far. The buffer ADDRESS is unchanged across calls so any trace
        captured against it stays valid.
        """
        host = self._self_mask_hosts[int(position)]
        ttnn.copy_host_to_device_tensor(host, self._persistent_self_mask_tt)
        return self._persistent_self_mask_tt

    def _run_decode_body(
        self,
        position: int,
        precomputed_encoder_mask_tt: Optional[ttnn.Tensor],
        use_pos_tensor: bool = False,
    ) -> ttnn.Tensor:
        """Run the decode_step + lm_head and return the logits tensor.

        Inputs are taken from the persistent buffers. Returns the device
        tensor produced by the LM head linear (no D2H -- the caller is
        responsible for ``ttnn.to_torch`` after the trace replay).

        Args:
            position: integer position (still required because the decode
                body builds per-position artifacts like the self-mask
                outside this call; the in-trace cache update uses the
                persistent device tensor when ``use_pos_tensor`` is True).
            precomputed_encoder_mask_tt: cross-attn mask buffer.
            use_pos_tensor: when True, pass the persistent position
                buffer (a ttnn.Tensor) to the cache update instead of
                the Python int. Required for trace capture so the
                update_idxs read comes from device memory.
        """
        self_mask_tt = self._persistent_self_mask_tt
        if use_pos_tensor:
            pos_arg = self.past_key_values.self_attn.get_persistent_pos_buffer()
        else:
            pos_arg = int(position)
        hidden_tt = self.text_decoder.decode_step(
            input_ids=torch.zeros((1, 1), dtype=torch.long),  # unused (persistent buf path)
            position=pos_arg,
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
        precomputed_encoder_mask_tt: Optional[ttnn.Tensor],
    ) -> int:
        """Capture (or reuse) the single decode trace.

        The captured trace handles ALL positions because:
          * the new token id is read from ``_persistent_input_ids_tt``
          * the position id (for sinusoidal lookup) is read from
            ``_persistent_position_ids_tt``
          * the KV-cache write position is read from the
            ``self_attn.get_persistent_pos_buffer()`` int32 tensor
            (consumed by ``paged_update_cache``'s ``update_idxs_tensor``)
          * the self-attn mask is read from
            ``_persistent_self_mask_tt``
          * the cross-attn mask is read from
            ``_persistent_encoder_mask_tt``

        All five buffers have stable device addresses, so the trace
        replays correctly for any position once the caller writes the
        per-step inputs into them via ``copy_host_to_device_tensor``.
        """
        if self._decode_trace_id is not None:
            return self._decode_trace_id
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            logits_tt = self._run_decode_body(
                position=0,  # not used directly inside the trace body
                precomputed_encoder_mask_tt=precomputed_encoder_mask_tt,
                use_pos_tensor=True,
            )
        finally:
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._decode_trace_id = trace_id
        self._decode_trace_output_tt = logits_tt
        return trace_id

    def _release_decode_traces(self) -> None:
        """Release the captured decode trace.

        Optional; safe to skip when reusing the generator across multiple
        ``generate()`` calls (the captured trace is CROSS-CALL safe by
        design now that paged_update_cache + persistent position buffers
        replace the int-baked ``ttnn.update_cache``).
        """
        if self._decode_trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._decode_trace_id)
            except Exception:
                pass
        self._decode_trace_id = None
        self._decode_trace_output_tt = None
        self._decode_kernels_compiled = False

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
        # persistent buffer (whose address is captured into the decode
        # trace and must remain stable across generate() calls).
        if use_trace:
            self._ensure_persistent_buffers()
            # The decode trace is CROSS-CALL safe: it reads the position
            # from a device tensor (via paged_update_cache's
            # update_idxs_tensor), and all persistent buffers retain
            # their device addresses across generate() calls. So we do
            # NOT release the trace here -- subsequent generate() calls
            # just rewrite the persistent buffers and replay.

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
        host = self._sin_pos_hosts[int(position)]
        ttnn.copy_host_to_device_tensor(host, self._persistent_position_ids_tt)

    def _write_cache_pos(self, position: int) -> None:
        """Write the KV-cache target position into the persistent int32 buffer
        that ``paged_update_cache`` reads via its ``update_idxs_tensor`` arg.
        """
        host = self._cache_pos_hosts[int(position)]
        ttnn.copy_host_to_device_tensor(host, self.past_key_values.self_attn.get_persistent_pos_buffer())

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
        """Metal-trace AR loop using a SINGLE re-usable trace.

        Strategy: thanks to ``paged_update_cache(update_idxs_tensor=...)``
        + persistent self/encoder masks + persistent input/position id
        buffers, ONE captured trace handles all decode positions. We
        capture once during the first ``generate()`` (after a single
        untraced warmup so the program cache is warm) and replay for
        every step thereafter -- including across subsequent
        ``generate()`` calls.

        Position 0 (decoder_start_token_id) is always run untraced
        because its logits are discarded -- folding it into the trace
        would just add complexity.
        """
        import time as _time

        # Persistent input buffers were allocated in generate() before
        # the encoder-mask precompute (so the mask lands in the persistent
        # buffer). Call again here for safety -- idempotent.
        self._ensure_persistent_buffers()

        # ---- position 0: warmup, no trace. Logits discarded. ----
        self._write_input_ids(tokens[0])
        self._write_position_id(0)
        self._write_cache_pos(0)
        self._write_self_mask(0)
        _t0 = _time.perf_counter()
        if not self._decode_kernels_compiled:
            _wu0 = self._run_decode_body(
                position=0,
                precomputed_encoder_mask_tt=precomp_encoder_mask_tt,
                use_pos_tensor=True,
            )
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(_wu0)
            self._decode_kernels_compiled = True
            if self.step_callback is not None:
                self.step_callback(0, (_time.perf_counter() - _t0) * 1000.0, "warmup")
        else:
            # Subsequent generate() calls: program cache already warm.
            # Just replay the trace at pos=0 (logits discarded).
            if self._decode_trace_id is None:
                # First-call warmup happened before but trace not yet
                # captured (shouldn't happen given the ordering below).
                pass
            else:
                ttnn.execute_trace(self.device, self._decode_trace_id, cq_id=0, blocking=True)
                if self.step_callback is not None:
                    self.step_callback(0, (_time.perf_counter() - _t0) * 1000.0, "replay")

        # ---- capture the single decode trace if not already captured ----
        if self._decode_trace_id is None:
            _t_cap = _time.perf_counter()
            self._capture_decode_trace(precomp_encoder_mask_tt)
            if self.step_callback is not None:
                self.step_callback(0, (_time.perf_counter() - _t_cap) * 1000.0, "capture")

        # ---- position 1 (tgt_lang_id): first logits we sample from. ----
        self._write_input_ids(tokens[1])
        self._write_position_id(1)
        self._write_cache_pos(1)
        self._write_self_mask(1)
        _t0 = _time.perf_counter()
        ttnn.execute_trace(self.device, self._decode_trace_id, cq_id=0, blocking=True)
        logits_tt = self._decode_trace_output_tt
        logits = ttnn.to_torch(logits_tt).to(torch.float32)
        if self.step_callback is not None:
            self.step_callback(1, (_time.perf_counter() - _t0) * 1000.0, "replay")
        while logits.dim() > 1:
            if logits.shape[0] == 1:
                logits = logits.squeeze(0)
            else:
                logits = logits[0, -1, :]
                break

        # ---- AR loop: positions 2..max_total-1, ALL via the single trace ----
        for pos in range(2, max_total):
            next_token = int(torch.argmax(logits).item())
            tokens.append(next_token)
            if next_token == int(eos_token_id):
                break
            if pos + 1 >= max_total:
                break

            # Write per-step inputs into the persistent buffers.
            self._write_input_ids(next_token)
            self._write_position_id(pos)
            self._write_cache_pos(pos)
            self._write_self_mask(pos)

            _t_rep = _time.perf_counter()
            ttnn.execute_trace(self.device, self._decode_trace_id, cq_id=0, blocking=True)
            logits_tt = self._decode_trace_output_tt
            logits = ttnn.to_torch(logits_tt).to(torch.float32)
            if self.step_callback is not None:
                self.step_callback(pos, (_time.perf_counter() - _t_rep) * 1000.0, "replay")
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
