# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""High-level T2U (text-to-unit) NAR generator for SeamlessM4T-v2.

Composes the verified TTNN sub-models:

    - :class:`T2uEncoder`   (6-layer NLLB-style T2U encoder, no embeddings)
    - :class:`T2uDecoder`   (6-layer NAR decoder with duration upsample)
    - Tied LM head: ``t2u_model.lm_head.weight`` is tied to
      ``t2u_model.model.decoder.embed_tokens.weight`` per HF config.

The result is unit token ids ready for the code HiFi-GAN vocoder.

This block is NAR (no AR loop), so synthesis is a single forward pass
through encoder + decoder + linear + argmax.

Inputs (host-side torch):
    - ``text_decoder_hidden``: ``[B, T, H]`` last_hidden_state from the
      preceding text decoder (HF ``self.text_decoder`` call with
      ``sequences[:, :-1]`` -- i.e. tokens with trailing EOS stripped).
    - ``char_input_ids``: ``[B, char_seq_len]`` int. From HF
      ``self._get_char_input_ids(...)`` over the same sequences.
    - ``char_count_per_id``: ``[B, T]`` int (with leading + trailing zero
      pads for lang/EOS, per HF).
    - ``t2u_attention_mask``: ``[B, T]`` 0/1 float built by HF's
      ``_compute_new_attention_mask`` over ``text_decoder_hidden`` with
      ``seq_lens = (sequences[:, :-1] != pad_token_id).sum(1)``.

Output:
    - ``unit_token_ids``: ``[B, unit_seq_len]`` int64 in the *t2u* vocab,
      with EOS replaced by pad and ``vocoder_offset`` subtracted (so the
      result is the channel index in ``vocoder.unit_embedding``).
    - ``unit_padding_mask``: ``[B, unit_seq_len]`` int64 (1=keep, 0=pad).

Both follow HF's bookkeeping in
``SeamlessM4Tv2ForTextToSpeech.generate`` after the t2u_model call
(lines around ``t2u_logits.argmax`` ... ``unit_ids - vocoder_offset``).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder import T2uDecoder
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_encoder import T2uEncoder

# ---------------------------------------------------------------------------
# Model config (matches SeamlessM4T-v2-Large defaults)
# ---------------------------------------------------------------------------
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
EPS = 1e-5
T2U_PAD_TOKEN_ID = 1
T2U_EOS_TOKEN_ID = 2
T2U_VOCAB_SIZE = 10082
VOCODER_OFFSET = 4
T2U_MAX_POSITION_EMBEDDINGS = 4096


class T2UGenerator:
    """SeamlessM4T-v2 NAR text-to-unit generator (TTNN).

    Args:
        device: opened ttnn device.
        hf_state_dict: result of :func:`weight_loader.load_hf_state_dict`.
        weight_dtype: TTNN storage dtype for all sub-module weights.
        num_encoder_layers: number of T2U encoder layers (default 6).
        num_decoder_layers: number of T2U decoder layers (default 6).
        t2u_vocab_size: t2u vocabulary size (default 10082).
        t2u_pad_token_id: t2u padding id (default 1).
        t2u_eos_token_id: t2u EOS id (default 2).
        vocoder_offset: control-symbol offset subtracted before vocoder
            (default 4).
    """

    def __init__(
        self,
        device,
        hf_state_dict: Dict[str, torch.Tensor],
        weight_dtype=ttnn.bfloat16,
        num_encoder_layers: int = NUM_ENCODER_LAYERS,
        num_decoder_layers: int = NUM_DECODER_LAYERS,
        t2u_vocab_size: int = T2U_VOCAB_SIZE,
        t2u_pad_token_id: int = T2U_PAD_TOKEN_ID,
        t2u_eos_token_id: int = T2U_EOS_TOKEN_ID,
        vocoder_offset: int = VOCODER_OFFSET,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.t2u_pad_token_id = int(t2u_pad_token_id)
        self.t2u_eos_token_id = int(t2u_eos_token_id)
        self.vocoder_offset = int(vocoder_offset)

        # --- T2U encoder (6 layers, no token / position embeddings) ---------
        t2u_enc_sd = wl.t2u_encoder_weights(hf_state_dict, num_layers=num_encoder_layers)
        self.t2u_encoder = T2uEncoder(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            layers_state_dict=t2u_enc_sd["layers"],
            final_layer_norm_state_dict=t2u_enc_sd["final_layer_norm"],
            eps=EPS,
            weight_dtype=weight_dtype,
        )

        # --- T2U decoder (6 layers, NAR, duration upsample) -----------------
        t2u_dec = wl.t2u_decoder_weights(
            hf_state_dict,
            num_layers=num_decoder_layers,
            hidden_size=EMBED_DIM,
            max_position_embeddings=T2U_MAX_POSITION_EMBEDDINGS,
            t2u_padding_idx=self.t2u_pad_token_id,
        )
        self.t2u_decoder = T2uDecoder(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            state_dict=t2u_dec["state_dict"],
            char_positional_weights=t2u_dec["char_positional_weights"],
            positional_weights=t2u_dec["positional_weights"],
            embed_scale=math.sqrt(EMBED_DIM),
            padding_idx=self.t2u_pad_token_id,
            eps=EPS,
            weight_dtype=weight_dtype,
        )

        # --- LM head -- tied to t2u_model.model.decoder.embed_tokens.weight -
        # HF instantiates this as nn.Linear(hidden_size, t2u_vocab_size, bias=False)
        # and the post_init tie_weights step copies the embedding weight in.
        # Shape: [t2u_vocab_size, hidden_size] (PyTorch's Linear stores weight
        # as [out, in]).
        lm_head_w = hf_state_dict["t2u_model.model.decoder.embed_tokens.weight"]
        if int(lm_head_w.shape[0]) != t2u_vocab_size:
            raise RuntimeError(
                f"t2u_model.model.decoder.embed_tokens.weight shape {tuple(lm_head_w.shape)} "
                f"does not match t2u_vocab_size={t2u_vocab_size}; check HF checkpoint."
            )
        # Linear y = x @ W^T so we ship the transposed [H, V] matrix to DRAM
        # for ttnn.linear-with-shared-weights.
        self._lm_head_weight_tt = ttnn.from_torch(
            lm_head_w.t().contiguous().to(torch.bfloat16),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _compute_new_attention_mask(seq_lens: torch.Tensor, batch: int, mask_seq_len: int) -> torch.Tensor:
        """Port of HF ``_compute_new_attention_mask`` (1 valid, 0 pad)."""
        indices = torch.arange(mask_seq_len).expand(batch, -1)
        bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
        mask = torch.ones((batch, mask_seq_len), dtype=torch.float32)
        return mask.masked_fill(bool_mask, 0.0)

    @staticmethod
    def _padding_mask_to_4d_additive(
        attention_mask_2d: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert ``[B, S]`` 0/1 mask to ``[B, 1, S, S]`` additive log-mask.

        Mirrors HF ``_prepare_4d_attention_mask`` but explicitly broadcasts to
        a full ``[S, S]`` so SDPA tile reads see well-defined data on every
        tile boundary.
        """
        b, s = attention_mask_2d.shape
        fill = float(torch.finfo(dtype).min)
        # invert + broadcast: rows fully filled if pad, columns fully filled if pad.
        # Real path: zero where keep, fill where pad. Use column-only since HF
        # uses [B,1,1,S] broadcast. We materialise to [B,1,S,S] though to
        # match TextEncoder's pattern of pre-broadcasting.
        keep_k = attention_mask_2d.to(torch.bool)  # [B, S]
        mask = torch.full((b, 1, s, s), fill, dtype=dtype)
        for i in range(b):
            cols = keep_k[i]
            if cols.any():
                mask[i, 0, :, cols] = 0.0
        return mask

    # ------------------------------------------------------------------ public API

    def synthesize_units(
        self,
        text_decoder_hidden: torch.Tensor,
        char_input_ids: torch.Tensor,
        char_count_per_id: torch.Tensor,
        t2u_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """One forward pass through T2U encoder + decoder.

        Args:
            text_decoder_hidden: ``[B, T, H]`` float -- HF text_decoder
                ``last_hidden_state`` over ``sequences[:, :-1]``.
            char_input_ids: ``[B, char_seq_len]`` int. From
                ``self._get_char_input_ids(...)``.
            char_count_per_id: ``[B, T]`` int with leading + trailing zero pad
                for lang_id / EOS, per HF.
            t2u_attention_mask: optional ``[B, T]`` 0/1 mask. If ``None`` we
                derive it as ``(text_decoder_hidden != 0).any(-1)``, which
                works for the common case where the host runner stripped
                pads. Recommended: pass the mask explicitly.

        Returns:
            dict with:
                ``unit_token_ids``  -- ``[B, unit_seq_len]`` int64 (offset-stripped, EOS->pad).
                ``raw_unit_ids``    -- ``[B, unit_seq_len]`` int64 (raw argmax before remap).
                ``unit_padding_mask`` -- ``[B, unit_seq_len]`` int64 (1=keep, 0=pad).
                ``dur_out``         -- ``[B, char_seq_len]`` int64 (durations used).
        """
        if text_decoder_hidden.dim() != 3:
            raise ValueError(f"text_decoder_hidden must be [B, T, H], got {tuple(text_decoder_hidden.shape)}")
        if char_input_ids.dim() != 2 or char_count_per_id.dim() != 2:
            raise ValueError("char_input_ids and char_count_per_id must be [B, _]")

        batch = int(text_decoder_hidden.shape[0])
        text_seq_len = int(text_decoder_hidden.shape[1])

        text_decoder_hidden = text_decoder_hidden.to(torch.float32).contiguous()

        # 1. Build / accept the [B, T] attention mask for the T2U encoder.
        if t2u_attention_mask is None:
            # Best-effort: keep all rows (no padding info available).
            t2u_attention_mask = torch.ones((batch, text_seq_len), dtype=torch.float32)
        else:
            t2u_attention_mask = t2u_attention_mask.to(torch.float32)

        # 2. T2U encoder forward (consumes pre-embedded hidden states).
        attn_mask_4d = self._padding_mask_to_4d_additive(t2u_attention_mask)
        enc_out_tt = self.t2u_encoder(
            text_decoder_hidden,
            attention_mask_torch=attn_mask_4d,
        )
        enc_out_torch = ttnn.to_torch(enc_out_tt).to(torch.float32)
        ttnn.deallocate(enc_out_tt)
        # Strip extra batch dim if the encoder returned [1, B, T, H].
        while enc_out_torch.dim() > 3:
            enc_out_torch = enc_out_torch.squeeze(0)
        enc_out_torch = enc_out_torch.reshape(batch, text_seq_len, EMBED_DIM)

        # 3. T2U decoder forward (NAR, with duration upsample).
        dec_out = self.t2u_decoder(
            char_input_ids=char_input_ids.to(torch.int64),
            char_count_per_id=char_count_per_id.to(torch.int64),
            encoder_hidden_states=enc_out_torch,
        )
        dec_hidden_tt = dec_out["last_hidden_state"]
        unit_padding_mask = dec_out["padding_mask"]  # [B, unit_seq_len] float (host)
        dur_out = dec_out["dur_out"]  # [B, char_seq_len] int (host)

        # 4. LM head -> argmax -> unit token ids.
        logits_tt = ttnn.linear(
            dec_hidden_tt,
            self._lm_head_weight_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(dec_hidden_tt)
        logits_torch = ttnn.to_torch(logits_tt).to(torch.float32)
        ttnn.deallocate(logits_tt)
        # logits may come back as [1, B, T, V] or [B, T, V] -- normalise.
        while logits_torch.dim() > 3:
            logits_torch = logits_torch.squeeze(0)
        unit_seq_len = int(unit_padding_mask.shape[1])
        logits_torch = logits_torch.reshape(batch, unit_seq_len, -1)

        raw_unit_ids = logits_torch.argmax(dim=-1).to(torch.int64)  # [B, T_u]

        # 5. HF post-processing (see SeamlessM4Tv2ForTextToSpeech.generate):
        #    a. replace EOS or pad-mask positions with t2u_pad_token_id.
        #    b. subtract vocoder_offset for non-pad positions.
        padding_bool = unit_padding_mask.to(torch.bool)
        replace_mask = (raw_unit_ids == self.t2u_eos_token_id) | (~padding_bool)
        unit_ids = raw_unit_ids.masked_fill(replace_mask, self.t2u_pad_token_id)
        unit_ids = torch.where(
            unit_ids == self.t2u_pad_token_id,
            unit_ids,
            unit_ids - self.vocoder_offset,
        )

        return {
            "unit_token_ids": unit_ids,
            "raw_unit_ids": raw_unit_ids,
            "unit_padding_mask": unit_padding_mask.to(torch.int64),
            "dur_out": dur_out,
        }
