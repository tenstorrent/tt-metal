# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 NLLB-style text decoder.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::text_decoder_forward``,
which reproduces the full ``SeamlessM4Tv2Decoder`` (NLLB text decoder branch).
Op sequence::

    inputs_embeds = scaled_word_embedding(input_ids)
    embed_pos     = sinusoidal_positional_embedding(input_ids, ...)
    hidden        = inputs_embeds + embed_pos
    # build 4D causal additive log-mask for self-attn (HF helper)
    # expand encoder 2D padding mask to 4D additive log-mask (HF helper)
    for layer in self.layers:
        hidden = text_decoder_layer(hidden, encoder_hidden_states,
                                    self_attention_mask, encoder_attention_mask)
    hidden = final_layer_norm(hidden)

Composed over already-verified TTNN leaf modules:

    - :class:`ScaledWordEmbedding`             (1 instance)
    - :class:`SinusoidalPositionalEmbedding`   (1 instance)
    - :class:`TextDecoderLayer`                (``num_layers`` instances)
    - :class:`LayerNorm`                       (1 instance, final)

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - decoder_layers = 24 (golden uses 2 for size)
    - decoder_attention_heads = 16  (head_dim = 64)
    - decoder_ffn_dim = 8192
    - activation_function = "relu"
    - layer_norm_eps = 1e-5
    - scale_embedding = True  -> embed_scale = sqrt(1024) = 32.0
    - pad_token_id = 0

**Tile padding for SDPA**

SDPA reads off-grid tile pads as garbage when seq_len % 32 != 0. The decoder
golden uses tgt_len=8 and src_len=16 — both off-tile. We right-pad:
    - decoder_input_ids to a tile multiple with ``padding_idx`` (embed and
      pos lookup of the padding row are zero, so padded queries are zero
      pre-attention),
    - encoder_hidden_states to a tile multiple with zeros,
    - self_attention_mask along its last two dims with the additive
      "minus" fill so the padded Q/K rows produce no contribution,
    - encoder_attention_mask along both Q (tgt) and K (src) dims with the
      same "minus" fill.

After the final LayerNorm we slice the tail off to return the logical
``[B, T, hidden]`` answer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm
from models.demos.facebook_seamless_m4t_v2_large.tt.scaled_word_embedding import ScaledWordEmbedding
from models.demos.facebook_seamless_m4t_v2_large.tt.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder_layer import TextDecoderLayer

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


class TextDecoder(LightweightModule):
    """Full NLLB-style text decoder in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of self/cross-attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        embed_tokens_weight: torch.Tensor of shape ``(vocab_size, embed_dim)``
            for the scaled word embedding table.
        embed_positions_weights: torch.Tensor of shape
            ``(num_embeddings, embed_dim)`` — the pre-computed sinusoidal
            positional embedding table (with row ``padding_idx`` zeroed).
        layers_state_dict: list of per-layer state dicts in the format
            consumed by :class:`TextDecoderLayer`.
        final_layer_norm_state_dict: ``{"weight", "bias"}`` for the
            terminal LayerNorm.
        eps: LayerNorm epsilon (default 1e-5).
        padding_idx: padding token id (default 0, matches v2 default config).
        embed_scale: scalar applied to token embeddings; defaults to
            ``sqrt(embed_dim)`` when ``None``.
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM).
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        embed_tokens_weight: torch.Tensor,
        embed_positions_weights: torch.Tensor,
        layers_state_dict,
        final_layer_norm_state_dict,
        eps: float = 1e-5,
        padding_idx: int = 0,
        embed_scale: Optional[float] = None,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.padding_idx = int(padding_idx)
        self.eps = float(eps)
        if embed_scale is None:
            embed_scale = math.sqrt(embed_dim)
        self.embed_scale = float(embed_scale)

        # 1. Scaled word embedding (lookup * sqrt(hidden_size)).
        self.embed_tokens = ScaledWordEmbedding(
            device=device,
            weight=embed_tokens_weight,
            scale=self.embed_scale,
            padding_idx=self.padding_idx,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. Padding-aware sinusoidal positional embedding.
        self.embed_positions = SinusoidalPositionalEmbedding(
            device=device,
            weights=embed_positions_weights,
            padding_idx=self.padding_idx,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 3. Stack of decoder layers.
        self.layers = [
            TextDecoderLayer(
                device=device,
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                state_dict=layer_sd,
                eps=self.eps,
                weight_dtype=weight_dtype,
                weight_memory_config=weight_memory_config,
            )
            for layer_sd in layers_state_dict
        ]

        # 4. Final LayerNorm.
        self.final_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=final_layer_norm_state_dict["weight"],
            bias=final_layer_norm_state_dict["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

    # ------------------------------------------------------------------ helpers

    def _pad_input_ids(self, input_ids: torch.Tensor, pad_to: int) -> torch.Tensor:
        """Right-pad ids with ``padding_idx`` (zero embed + zero pos rows)."""
        batch = int(input_ids.shape[0])
        pad = torch.full(
            (batch, pad_to),
            self.padding_idx,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, pad], dim=1)

    def _pad_encoder_hidden(self, enc_torch: torch.Tensor, pad_to: int) -> torch.Tensor:
        """Right-pad encoder hidden states (along the src seq dim) with zeros.

        The corresponding mask rows get the "minus" log fill so the padded
        K/V positions contribute nothing post-softmax regardless of K/V
        magnitude — but zero-init is the cleanest invariant.
        """
        batch, src_len, hidden = enc_torch.shape
        pad = torch.zeros((batch, pad_to, hidden), dtype=enc_torch.dtype, device=enc_torch.device)
        return torch.cat([enc_torch, pad], dim=1)

    def _build_self_attention_mask(
        self,
        attention_mask_2d: Optional[torch.Tensor],
        batch: int,
        tgt_len: int,
        tgt_pad: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a 4D additive causal log-mask shaped ``[B, 1, T+pad, T+pad]``.

        Mirrors HF ``_prepare_4d_causal_attention_mask``: lower-triangular
        causal mask, with any 2D padding column added in. The tail
        ``[tgt_len:]`` is filled with ``finfo(dtype).min`` along the *last*
        axis (both Q and K side) so SDPA's tile-aligned read of the mask
        respects the logical sequence length.
        """
        full = tgt_len + tgt_pad
        fill = float(torch.finfo(dtype).min)
        # Build the [tgt_len, tgt_len] causal core: 0 on/below diagonal, fill above.
        core = torch.zeros((tgt_len, tgt_len), dtype=dtype)
        upper_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.bool), diagonal=1)
        core = core.masked_fill(upper_mask, fill)

        # Embed the core into a fill-padded full tile-multiple square.
        mask = torch.full((batch, 1, full, full), fill, dtype=dtype)
        # Broadcast core across batch.
        mask[:, 0, :tgt_len, :tgt_len] = core

        # Apply 2D key-padding (per-batch) by OR-ing with column-wise fill on K-side.
        if attention_mask_2d is not None:
            # attention_mask_2d: [B, T] (1=keep, 0=pad). Padded K columns -> fill.
            key_pad = attention_mask_2d == 0  # [B, T], True for padded
            for b in range(batch):
                cols = key_pad[b]
                # Mask the padded K columns within the logical region (broadcast over Q rows).
                if cols.any():
                    idx = cols.nonzero(as_tuple=False).flatten().tolist()
                    for j in idx:
                        mask[b, 0, :tgt_len, j] = fill
        return mask

    def _build_encoder_attention_mask(
        self,
        encoder_attention_mask_2d: Optional[torch.Tensor],
        batch: int,
        tgt_len: int,
        tgt_pad: int,
        src_len: int,
        src_pad: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build 4D additive log-mask shaped ``[B, 1, T+tpad, S+spad]``.

        Mirrors HF ``_prepare_4d_attention_mask`` over the unpadded core,
        with the surrounding tile-pad region set to the "minus" fill.
        """
        full_t = tgt_len + tgt_pad
        full_s = src_len + src_pad
        fill = float(torch.finfo(dtype).min)

        # Start fully-padded then carve out the logical box.
        mask = torch.full((batch, 1, full_t, full_s), fill, dtype=dtype)

        # The logical core: rows 0..tgt_len, cols 0..src_len.
        # HF's _prepare_4d_attention_mask broadcasts the [B,S] mask across T queries.
        if encoder_attention_mask_2d is None:
            # All-keep core.
            mask[:, 0, :tgt_len, :src_len] = 0.0
        else:
            # Expand 2D [B,S] -> per-batch row of size S, broadcast over T.
            keep = encoder_attention_mask_2d.to(torch.bool)  # [B, S]
            for b in range(batch):
                row = torch.where(
                    keep[b],
                    torch.zeros(src_len, dtype=dtype),
                    torch.full((src_len,), fill, dtype=dtype),
                )
                mask[b, 0, :tgt_len, :src_len] = row.unsqueeze(0).expand(tgt_len, src_len)
        return mask

    def _to_tt(self, t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ----------------------------------------------------------------- forward

    # ------------------------------------------------------------------ cross-attn prefill
    def populate_cross_attention_cache(
        self,
        past_key_values,
        encoder_hidden_states_torch: torch.Tensor,
    ) -> None:
        """Project encoder hidden states through each layer's cross-attn K/V
        weights and store the result in the cross-attention KV cache.

        Run this ONCE per generation, immediately after the encoder forward,
        BEFORE the first decode step.

        Args:
            past_key_values: a
                :class:`models.demos.facebook_seamless_m4t_v2_large.tt.kv_cache.TextDecoderKVCache`
                whose ``cross_attn.encoder_seq_len`` matches the
                tile-padded encoder length we feed in here.
            encoder_hidden_states_torch: host-side ``[B, S, H]`` float
                tensor. We tile-pad along ``S`` with zeros to match the
                cross-attn cache slot before projecting.
        """
        src_len = int(encoder_hidden_states_torch.shape[1])
        src_pad = _pad_to_tile(src_len)
        if src_pad > 0:
            enc_padded = self._pad_encoder_hidden(encoder_hidden_states_torch, src_pad)
        else:
            enc_padded = encoder_hidden_states_torch
        encoder_hidden_tt = self._to_tt(enc_padded.to(torch.float32))

        for i, layer in enumerate(self.layers):
            # NB: the pre-norm in HF BART-style cross-attention applies to
            # the *decoder* (Q) side, NOT to the encoder K/V side. So we
            # feed encoder_hidden_states straight into K/V projections.
            k_cross, v_cross = layer.cross_attention.project_kv(encoder_hidden_tt)
            past_key_values.cross_attn.populate(i, k_cross, v_cross)

        ttnn.deallocate(encoder_hidden_tt)

    # ------------------------------------------------------------------ decode-step mask
    def _build_decode_self_attention_mask(
        self,
        batch: int,
        position: int,
        max_seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a ``[B, 1, 1, max_seq_len]`` additive mask for one decode step.

        Positions ``[0 .. position]`` are unmasked (additive 0); positions
        ``[position+1 .. max_seq_len-1]`` get ``finfo(dtype).min`` so SDPA
        ignores empty cache slots.
        """
        fill = float(torch.finfo(dtype).min)
        mask = torch.zeros((batch, 1, 1, max_seq_len), dtype=dtype)
        if position + 1 < max_seq_len:
            mask[:, :, :, position + 1 :] = fill
        return mask

    def _build_decode_encoder_attention_mask(
        self,
        encoder_attention_mask_2d: Optional[torch.Tensor],
        batch: int,
        encoder_seq_len_total: int,
        src_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a ``[B, 1, 1, encoder_seq_len_total]`` cross-attn mask.

        Positions ``[0 .. src_len)`` carry the (optional) caller-supplied
        2D padding mask; positions ``[src_len .. encoder_seq_len_total)``
        are tile-padding and get the "minus" fill so they contribute
        nothing post-softmax.
        """
        fill = float(torch.finfo(dtype).min)
        mask = torch.full((batch, 1, 1, encoder_seq_len_total), fill, dtype=dtype)
        if encoder_attention_mask_2d is None:
            mask[:, :, :, :src_len] = 0.0
        else:
            keep = encoder_attention_mask_2d.to(torch.bool)  # [B, S]
            for b in range(batch):
                row = torch.where(
                    keep[b],
                    torch.zeros(src_len, dtype=dtype),
                    torch.full((src_len,), fill, dtype=dtype),
                )
                mask[b, 0, 0, :src_len] = row
        return mask

    # ------------------------------------------------------------------ decode step
    def decode_step(
        self,
        input_ids: torch.Tensor,
        position: int,
        past_key_values,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_seq_len_logical: Optional[int] = None,
        precomputed_self_mask_tt: Optional[ttnn.Tensor] = None,
        precomputed_encoder_mask_tt: Optional[ttnn.Tensor] = None,
        persistent_input_ids_tt: Optional[ttnn.Tensor] = None,
        persistent_position_ids_tt: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run one autoregressive decode step.

        Args:
            input_ids: host int tensor of shape ``[B, 1]`` — the single
                token to append to the cache and consume.
            position: 0-based index of this new token within the cache.
                Must satisfy ``position < past_key_values.self_attn.max_seq_len``.
            past_key_values: a populated
                :class:`models.demos.facebook_seamless_m4t_v2_large.tt.kv_cache.TextDecoderKVCache`
                (cross-attn already populated via
                :meth:`populate_cross_attention_cache`).
            encoder_attention_mask: optional 2D host ``[B, S]`` padding
                mask. Combined with the tile-pad mask internally.
            precomputed_self_mask_tt: optional pre-built ttnn tensor of
                shape ``[B, 1, 1, max_seq_len]`` for the self-attention
                mask AT THIS POSITION. When supplied, we skip the host-
                side mask build + tile upload (a meaningful per-step win
                in the AR loop). Caller is responsible for keeping the
                tensor valid until the step returns, and for deallocating
                it.
            precomputed_encoder_mask_tt: optional pre-built ttnn tensor of
                shape ``[B, 1, 1, encoder_seq_len_total]`` for the cross-
                attention mask. Invariant across all decode steps of one
                generate() call, so it can be uploaded ONCE and reused.
                When supplied, ``encoder_attention_mask`` and
                ``encoder_seq_len_logical`` are ignored.
            persistent_input_ids_tt: optional pre-allocated device uint32
                ROW_MAJOR ``[B, 1]`` tensor holding the new token id.
                When supplied we DO NOT re-upload from host -- the caller
                is expected to have just written the right value via
                ``ttnn.copy_host_to_device_tensor``. Required for trace
                capture; the host upload path is otherwise unchanged.
            persistent_position_ids_tt: optional pre-allocated device
                uint32 ROW_MAJOR ``[B, 1]`` tensor holding the absolute
                positional index to gather into the sinusoidal table
                (i.e. for our padding-aware case, ``position + 1`` when
                the new token is not padding). When supplied we route
                the sinusoidal embed through
                ``SinusoidalPositionalEmbedding`` 's tensor-input path
                which skips all host-side cumsum + H2D.

        Returns:
            ttnn tensor of shape ``[B, 1, embed_dim]`` — the decoder's
            ``last_hidden_state`` for this step.
        """
        if input_ids.shape[1] != 1:
            raise ValueError(f"decode_step expects T=1, got input_ids.shape={tuple(input_ids.shape)}")
        batch = int(input_ids.shape[0])

        # 1. Token embedding (scaled) for the single new token. ttnn.embedding
        #    requires uint32 ROW_MAJOR ids.
        owns_input_ids = False
        if persistent_input_ids_tt is not None:
            tt_input_ids = persistent_input_ids_tt
        else:
            tt_input_ids = ttnn.from_torch(
                input_ids.to(torch.int32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            owns_input_ids = True
        inputs_embeds = self.embed_tokens(tt_input_ids)
        if owns_input_ids:
            ttnn.deallocate(tt_input_ids)

        # 2. Positional embedding for absolute position `position`.
        if persistent_position_ids_tt is not None:
            embed_pos = self.embed_positions(
                precomputed_position_ids_tt=persistent_position_ids_tt,
            )
        else:
            # Host-side padding-aware indexing (cumsum of non-pad mask +
            # past_key_values_length) gives the correct row when we hand
            # the layer the single new id with
            # ``past_key_values_length=position``.
            embed_pos = self.embed_positions(
                input_ids=input_ids,
                past_key_values_length=position,
            )
        hidden_states = ttnn.add(inputs_embeds, embed_pos)
        ttnn.deallocate(inputs_embeds)
        ttnn.deallocate(embed_pos)

        # 3. Self-attention mask: prefer the caller-supplied precomputed
        #    tensor. Fallback: build it host-side + upload per step (the
        #    original / single-call path).
        owns_self_mask = False
        if precomputed_self_mask_tt is not None:
            self_attention_mask_tt = precomputed_self_mask_tt
        else:
            max_seq_len = past_key_values.self_attn.max_seq_len
            self_mask_torch = self._build_decode_self_attention_mask(
                batch=batch,
                position=position,
                max_seq_len=max_seq_len,
                dtype=torch.float32,
            )
            self_attention_mask_tt = self._to_tt(self_mask_torch)
            owns_self_mask = True

        # 4. Cross-attention mask: prefer the caller-supplied precomputed
        #    tensor (uploaded once per generate() — invariant across
        #    decode steps). Fallback: rebuild every step.
        owns_encoder_mask = False
        if precomputed_encoder_mask_tt is not None:
            encoder_mask_tt = precomputed_encoder_mask_tt
        else:
            enc_seq_total = past_key_values.cross_attn.encoder_seq_len
            if encoder_attention_mask is not None:
                src_len = int(encoder_attention_mask.shape[1])
            elif encoder_seq_len_logical is not None:
                src_len = int(encoder_seq_len_logical)
            else:
                src_len = enc_seq_total
            if encoder_attention_mask is not None or enc_seq_total != src_len:
                enc_mask_torch = self._build_decode_encoder_attention_mask(
                    encoder_attention_mask_2d=encoder_attention_mask,
                    batch=batch,
                    encoder_seq_len_total=enc_seq_total,
                    src_len=src_len,
                    dtype=torch.float32,
                )
                encoder_mask_tt = self._to_tt(enc_mask_torch)
                owns_encoder_mask = True
            else:
                encoder_mask_tt = None

        # 5. Stack of decoder layers with KV cache.
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                self_attention_mask=self_attention_mask_tt,
                encoder_attention_mask=encoder_mask_tt,
                past_key_values=past_key_values,
                position=position,
                layer_idx=i,
            )

        # 6. Final LayerNorm.
        hidden_states = self.final_layer_norm(hidden_states)

        # Clean up scratch device tensors we ALLOCATED in this call. Caller-
        # supplied precomputed masks are NOT deallocated here.
        if owns_self_mask:
            ttnn.deallocate(self_attention_mask_tt)
        if owns_encoder_mask and encoder_mask_tt is not None:
            ttnn.deallocate(encoder_mask_tt)

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states_torch: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> ttnn.Tensor:
        """Run the full text decoder.

        Args:
            input_ids: host-side torch int tensor of shape ``[B, T]`` holding
                decoder token ids.
            encoder_hidden_states_torch: optional host-side ``[B, S, H]``
                float tensor — encoder output that K/V are projected from in
                the cross-attention block. When ``None`` every layer's
                cross-attention block is skipped.
            attention_mask: optional 2D host ``[B, T]`` padding mask
                (1=keep, 0=pad). Combined with a triangular causal mask
                internally.
            encoder_attention_mask: optional 2D host ``[B, S]`` padding mask
                (1=keep, 0=pad). Expanded to a 4D additive log-mask
                internally. Ignored when ``encoder_hidden_states_torch`` is
                ``None``.
            past_key_values_length: number of cached tokens to offset
                sinusoidal positions by (default 0).

        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]`` in TILE_LAYOUT (DRAM)
            — the decoder's ``last_hidden_state``.
        """
        batch, tgt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        tgt_pad = _pad_to_tile(tgt_len)

        # Right-pad decoder ids so embeds and pos rows are zero in the tail.
        if tgt_pad > 0:
            input_ids_eff = self._pad_input_ids(input_ids, tgt_pad)
        else:
            input_ids_eff = input_ids

        # 1. Token embeddings (scaled). ttnn.embedding wants uint32 ROW_MAJOR ids.
        tt_input_ids = ttnn.from_torch(
            input_ids_eff.to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inputs_embeds = self.embed_tokens(tt_input_ids)

        # 2. Sinusoidal positional embeddings — padding-aware index path
        #    derived from the host-side input_ids.
        embed_pos = self.embed_positions(
            input_ids=input_ids_eff,
            past_key_values_length=past_key_values_length,
        )

        # 3. Sum embeddings -> initial hidden state.
        hidden_states = ttnn.add(inputs_embeds, embed_pos)
        ttnn.deallocate(inputs_embeds)
        ttnn.deallocate(embed_pos)

        # 4. Build the 4D causal self-attention mask, tile-padded.
        self_mask_torch = self._build_self_attention_mask(
            attention_mask,
            batch=batch,
            tgt_len=tgt_len,
            tgt_pad=tgt_pad,
            dtype=torch.float32,
        )
        self_attention_mask_tt = self._to_tt(self_mask_torch)

        # 5. Cross-attention prep: pad encoder hidden states and build the
        #    4D encoder mask if encoder context is provided.
        encoder_hidden_tt = None
        encoder_mask_tt = None
        if encoder_hidden_states_torch is not None:
            src_len = int(encoder_hidden_states_torch.shape[1])
            src_pad = _pad_to_tile(src_len)
            if src_pad > 0:
                enc_padded = self._pad_encoder_hidden(encoder_hidden_states_torch, src_pad)
            else:
                enc_padded = encoder_hidden_states_torch
            encoder_hidden_tt = self._to_tt(enc_padded.to(torch.float32))

            enc_mask_torch = self._build_encoder_attention_mask(
                encoder_attention_mask,
                batch=batch,
                tgt_len=tgt_len,
                tgt_pad=tgt_pad,
                src_len=src_len,
                src_pad=src_pad,
                dtype=torch.float32,
            )
            encoder_mask_tt = self._to_tt(enc_mask_torch)

        # 6. Stack of decoder layers.
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_tt,
                self_attention_mask=self_attention_mask_tt,
                encoder_attention_mask=encoder_mask_tt,
            )

        # 7. Final LayerNorm.
        hidden_states = self.final_layer_norm(hidden_states)

        # 8. Slice back to the original (unpadded) decoder sequence length.
        if tgt_pad > 0:
            hidden_states = hidden_states[:, :tgt_len, :]

        # Clean up scratch device tensors.
        ttnn.deallocate(self_attention_mask_tt)
        if encoder_hidden_tt is not None:
            ttnn.deallocate(encoder_hidden_tt)
        if encoder_mask_tt is not None:
            ttnn.deallocate(encoder_mask_tt)

        return hidden_states
