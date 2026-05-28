# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Text-to-Unit (T2U) decoder.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::t2u_decoder_forward``,
which reproduces ``SeamlessM4Tv2TextToUnitDecoder`` bit-for-bit (NAR T2U
decoder).

Op sequence (matches HF / reference exactly)::

    char_padding_mask = compute_new_attention_mask(char_ids, char_count.sum(1))
    char_hidden = hard_upsample(encoder_hidden, char_count_per_id)
    char_pos    = pos_emb_alpha_char * embed_char_positions(inputs_embeds=char_hidden)
    char_embed  = embed_char(char_ids) * embed_scale
    char_hidden = char_embed + char_pos + char_hidden

    log_dur = duration_predictor(char_hidden, padding_mask=char_padding_mask)
    dur_out = clamp(round(expm1(log_dur)).long(), min=1)
    dur_out = dur_out.masked_fill(~char_padding_mask.bool(), 0)

    char_hidden = hard_upsample(char_hidden, dur_out)
    positions   = pos_emb_alpha * embed_positions(inputs_embeds=char_hidden)
    hidden      = char_hidden + positions

    padding_mask = compute_new_attention_mask(hidden, dur_out.sum(1))
    attention_mask = _prepare_4d_attention_mask(padding_mask, dtype)  # [B,1,1,T]

    for layer in self.layers:
        hidden = t2u_decoder_layer(hidden, attention_mask, padding_mask)
    hidden = final_layer_norm(hidden)

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - t2u_decoder_layers = 6 (golden uses 2 to keep file size small)
    - t2u_decoder_attention_heads = 16  (head_dim = 64)
    - activation_function = "relu"
    - layer_norm_eps = 1e-5
    - scale_embedding = True  -> embed_scale = sqrt(1024) = 32.0
    - char_vocab_size = 10943
    - pad_token_id = 1 (after t2u_ strip)
    - variance_predictor_kernel_size = 3
    - conv_kernel_size = 7

This block is composed over the already-verified TTNN leaf modules:

    - :class:`ScaledWordEmbedding`           (1 instance, char-level)
    - :class:`SinusoidalPositionalEmbedding` (2 instances; char-pos + unit-pos)
    - :class:`VariancePredictor`             (1 instance, duration predictor)
    - :class:`T2UDecoderLayer`               (``num_layers`` instances)
    - :class:`LayerNorm`                     (1 instance, final)

**Host-resident exception: hard-upsampling.** The duration-driven repeat
(``_hard_upsample``) is fundamentally a discrete gather: each char index is
copied ``dur_out[char_idx]`` times in a row, where ``dur_out`` is an integer
tensor produced *during* the forward by the variance predictor. The output
sequence length is data-dependent and not known statically. We therefore
bring the relevant tensors back to host, run ``torch.repeat_interleave``
there, then reupload to device for the rest of the forward pass. This is a
documented intentional exception, not a shortcut.

**Tile-padding for SDPA.** After the second hard-upsample, the new sequence
length may not be a multiple of 32. Mirroring ``text_encoder``/``text_decoder``
we right-pad the upsampled embeddings with zeros, build the 4D additive
log-mask at the padded length (with the tile-pad tail filled with
``finfo.min``), and slice the un-padded tail off the final-LN output.
"""

from __future__ import annotations

from typing import Dict

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm
from models.demos.facebook_seamless_m4t_v2_large.tt.scaled_word_embedding import ScaledWordEmbedding
from models.demos.facebook_seamless_m4t_v2_large.tt.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder_layer import T2UDecoderLayer
from models.demos.facebook_seamless_m4t_v2_large.tt.variance_predictor import VariancePredictor

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


class T2uDecoder(LightweightModule):
    """Full SeamlessM4T-v2 Text-to-Unit (NAR) decoder in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of self-attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        state_dict: nested mapping mirroring the reference's
            ``t2u_decoder_forward`` ``state_dict`` arg with keys
            ``{"embed_char", "pos_emb_alpha_char", "pos_emb_alpha",
              "duration_predictor", "layers", "layer_norm"}``.
        char_positional_weights: torch.Tensor of shape
            ``(num_embeddings, embed_dim)`` — the sinusoidal table used for
            ``embed_char_positions`` (char-level positions).
        positional_weights: torch.Tensor of shape
            ``(num_embeddings, embed_dim)`` — the sinusoidal table used for
            ``embed_positions`` (unit-level positions after the duration
            upsample).
        embed_scale: scalar applied to char-token embeddings (``sqrt(hidden)``).
        padding_idx: padding token id used by BOTH positional embeddings
            (default 1, matches the T2U-prefix-stripped pad_token_id).
        eps: LayerNorm epsilon (default 1e-5).
        variance_predictor_kernel_size: Conv1d kernel size for the duration
            predictor (default 3).
        conv_kernel_size: Conv1d kernel size inside each decoder layer's
            conv branch (default 7, hardcoded in HF).
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM).
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        state_dict: Dict,
        char_positional_weights: torch.Tensor,
        positional_weights: torch.Tensor,
        embed_scale: float,
        padding_idx: int = 1,
        eps: float = 1e-5,
        variance_predictor_kernel_size: int = 3,
        conv_kernel_size: int = 7,
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
        self.embed_scale = float(embed_scale)

        # ------------------------------------------------------------------
        # 1. Char-level scaled word embedding.
        # ------------------------------------------------------------------
        self.embed_char = ScaledWordEmbedding(
            device=device,
            weight=state_dict["embed_char"]["weight"],
            scale=self.embed_scale,
            padding_idx=None,  # HF embed_char is a plain nn.Embedding (no padding_idx).
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # ------------------------------------------------------------------
        # 2. Char-level and unit-level sinusoidal positional embeddings.
        #    Both use the same padding_idx.
        # ------------------------------------------------------------------
        self.embed_char_positions = SinusoidalPositionalEmbedding(
            device=device,
            weights=char_positional_weights,
            padding_idx=self.padding_idx,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )
        self.embed_positions = SinusoidalPositionalEmbedding(
            device=device,
            weights=positional_weights,
            padding_idx=self.padding_idx,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # Scalar HF nn.Parameter alphas (shape (1,)). Keep as host scalars to
        # apply via ttnn.multiply(scalar, ...).
        self.pos_emb_alpha_char = float(state_dict["pos_emb_alpha_char"].item())
        self.pos_emb_alpha = float(state_dict["pos_emb_alpha"].item())

        # ------------------------------------------------------------------
        # 3. Duration predictor (variance predictor with kernel_size=3).
        # ------------------------------------------------------------------
        dp_sd = state_dict["duration_predictor"]
        self.duration_predictor = VariancePredictor(
            device=device,
            conv1_weight=dp_sd["conv1"]["weight"],
            conv1_bias=dp_sd["conv1"]["bias"],
            ln1_weight=dp_sd["ln1"]["weight"],
            ln1_bias=dp_sd["ln1"]["bias"],
            conv2_weight=dp_sd["conv2"]["weight"],
            conv2_bias=dp_sd["conv2"]["bias"],
            ln2_weight=dp_sd["ln2"]["weight"],
            ln2_bias=dp_sd["ln2"]["bias"],
            proj_weight=dp_sd["proj"]["weight"],
            proj_bias=dp_sd["proj"]["bias"],
            kernel_size=int(variance_predictor_kernel_size),
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # ------------------------------------------------------------------
        # 4. Stack of T2U decoder layers (POST-norm, conv FFN, k=7).
        # ------------------------------------------------------------------
        self.layers = [
            T2UDecoderLayer(
                device=device,
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                state_dict=layer_sd,
                conv_kernel_size=int(conv_kernel_size),
                eps=self.eps,
                weight_dtype=weight_dtype,
                weight_memory_config=weight_memory_config,
            )
            for layer_sd in state_dict["layers"]
        ]

        # ------------------------------------------------------------------
        # 5. Final LayerNorm.
        # ------------------------------------------------------------------
        self.final_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["layer_norm"]["weight"],
            bias=state_dict["layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _compute_new_attention_mask(seq_lens: torch.Tensor, batch: int, mask_seq_len: int) -> torch.Tensor:
        """Port of HF ``_compute_new_attention_mask``: ``[B, T]`` float 1/0
        with 0.0 marking positions at index >= seq_lens[b].
        """
        indices = torch.arange(mask_seq_len).expand(batch, -1)
        bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
        mask = torch.ones((batch, mask_seq_len), dtype=torch.float32)
        mask = mask.masked_fill(bool_mask, 0.0)
        return mask

    @staticmethod
    def _hard_upsample(hidden_states: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Host-side port of HF ``_hard_upsample`` (repeat-by-duration).

        For ``batch == 1`` HF uses a single ``torch.repeat_interleave``; for
        ``batch > 1`` HF interleaves per-sample and right-pads with
        ``pad_sequence``. We mirror that exactly.
        """
        if hidden_states.size(0) == 1:
            return torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
        pieces = [
            torch.repeat_interleave(hidden_state, duration, dim=0)
            for (hidden_state, duration) in zip(hidden_states, durations)
        ]
        return torch.nn.utils.rnn.pad_sequence(pieces, batch_first=True)

    def _to_tt_tile(self, t: torch.Tensor, *, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t,
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _padding_mask_to_btN(self, padding_mask: torch.Tensor) -> ttnn.Tensor:
        """Convert ``(B, T)`` float padding mask -> TTNN ``[B, T, 1]`` multiplier."""
        pm = padding_mask.to(torch.float32).unsqueeze(-1).contiguous()
        return self._to_tt_tile(pm)

    def _build_4d_additive_mask(
        self,
        padding_mask: torch.Tensor,
        pad_to: int,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Build a ``[B, 1, T+pad_to, T+pad_to]`` additive log-mask for SDPA.

        Mirrors HF ``_prepare_4d_attention_mask`` (which builds ``[B,1,1,T]``
        and relies on broadcast inside the attention op). TTNN ``SDPA``
        instead requires ``mask_shape[2] == q_shape[2]``, so we explicitly
        materialise the broadcasted ``[B, 1, T, T]`` (and additively mask
        the tile-pad tail along the K axis with ``finfo.min``). The Q axis
        is treated identically along the padded tail so padded rows are
        isolated from real rows.
        """
        fill = float(torch.finfo(dtype).min)
        batch, seq_len = padding_mask.shape
        full = seq_len + pad_to
        # Start fully-masked then carve the real [seq_len, seq_len] core.
        mask = torch.full((batch, 1, full, full), fill, dtype=dtype)
        # K-side: 0.0 where padding_mask is 1.0 (keep), fill where it's 0.0.
        keep_k = padding_mask.to(torch.bool)  # [B, T]
        for b in range(batch):
            cols = keep_k[b].nonzero(as_tuple=False).flatten().tolist()
            if not cols:
                continue
            for j in cols:
                mask[b, 0, :seq_len, j] = 0.0
        return mask

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        char_input_ids: torch.Tensor,
        char_count_per_id: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Dict[str, object]:
        """Run the full T2U decoder.

        Args:
            char_input_ids: host-side ``[B, char_seq_len]`` long tensor of
                character ids.
            char_count_per_id: host-side ``[B, encoder_seq_len]`` long tensor
                of integer per-text-token character counts (used to upsample
                ``encoder_hidden_states`` to character resolution).
            encoder_hidden_states: host-side ``[B, encoder_seq_len, hidden]``
                float tensor (the T2U encoder's output).

        Returns:
            ``dict`` with three entries:

            - ``"last_hidden_state"``: ttnn TILE_LAYOUT tensor of shape
              ``[B, unit_seq_len, hidden]`` (the decoder's final hidden
              states, sliced back to the logical unit-seq length).
            - ``"padding_mask"``: host-side ``[B, unit_seq_len]`` float mask
              (1.0 valid, 0.0 pad). Returned on host so callers can pipe it
              straight to the vocoder without an extra device->host trip.
            - ``"dur_out"``: host-side ``[B, char_seq_len]`` long tensor of
              the per-character integer durations actually used (useful for
              upstream tests).
        """
        if char_input_ids.dim() != 2:
            raise ValueError(f"char_input_ids must be [B, T], got shape {tuple(char_input_ids.shape)}")
        if char_count_per_id.dim() != 2:
            raise ValueError(f"char_count_per_id must be [B, E], got shape {tuple(char_count_per_id.shape)}")
        if encoder_hidden_states.dim() != 3:
            raise ValueError(f"encoder_hidden_states must be [B, E, H], got shape {tuple(encoder_hidden_states.shape)}")

        batch = int(char_input_ids.shape[0])
        char_seq_len = int(char_input_ids.shape[1])

        # ----------------------------------------------------------------
        # 1. Char-level padding mask.
        # ----------------------------------------------------------------
        char_padding_mask = self._compute_new_attention_mask(
            char_count_per_id.sum(1),
            batch=batch,
            mask_seq_len=char_seq_len,
        )

        # ----------------------------------------------------------------
        # 2. Upsample encoder hidden states to char resolution (host gather).
        #    Host-resident step #1: encoder->char repeat_interleave.
        # ----------------------------------------------------------------
        char_hidden_torch = self._hard_upsample(encoder_hidden_states, char_count_per_id)
        # Sanity: char-level seq_len after expand must equal char_input_ids' len.
        assert int(char_hidden_torch.shape[1]) == char_seq_len, (
            f"char-level upsample mismatch: got {tuple(char_hidden_torch.shape)} vs " f"char_seq_len={char_seq_len}"
        )

        # ----------------------------------------------------------------
        # 3. Char positional embedding + char token embedding (scaled).
        #    All happens on device; positional ids derived on host.
        # ----------------------------------------------------------------
        # Char positions: HF passes inputs_embeds -> positions are simply
        # padding_idx+1 .. padding_idx+seq_len.
        char_positions_tt = self.embed_char_positions(
            input_ids=None,
            inputs_embeds_shape=(batch, char_seq_len),
            past_key_values_length=0,
        )
        # Apply pos_emb_alpha_char scalar.
        if self.pos_emb_alpha_char != 1.0:
            char_positions_tt = ttnn.multiply(char_positions_tt, self.pos_emb_alpha_char)

        # Char-token embeddings (uint32 row-major ids -> tile layout output).
        tt_char_ids = ttnn.from_torch(
            char_input_ids.to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        char_embeds_tt = self.embed_char(tt_char_ids)

        # Upload host-side char_hidden_torch to device.
        char_hidden_tt = self._to_tt_tile(char_hidden_torch)

        # Sum: char_embed + char_pos + char_hidden.
        char_hidden_tt = ttnn.add(char_hidden_tt, char_embeds_tt)
        ttnn.deallocate(char_embeds_tt)
        char_hidden_tt = ttnn.add(char_hidden_tt, char_positions_tt)
        ttnn.deallocate(char_positions_tt)

        # ----------------------------------------------------------------
        # 4. Duration predictor (with char-padding mask).
        # ----------------------------------------------------------------
        char_pad_mask_tt = self._padding_mask_to_btN(char_padding_mask)
        log_dur_pred_tt = self.duration_predictor(char_hidden_tt, padding_mask=char_pad_mask_tt)
        ttnn.deallocate(char_pad_mask_tt)

        # Bring log_dur back to host to do the integer round/expm1/clamp/mask.
        # This is a host-resident step (#2) by necessity: ``dur_out`` is an
        # integer tensor whose values control the upsample length downstream.
        log_dur_pred = ttnn.to_torch(log_dur_pred_tt).to(torch.float32).reshape(batch, char_seq_len)
        ttnn.deallocate(log_dur_pred_tt)
        dur_out = torch.clamp(torch.round(torch.expm1(log_dur_pred)).long(), min=1)
        dur_out = dur_out.masked_fill(~char_padding_mask.bool(), 0)

        # ----------------------------------------------------------------
        # 5. Upsample char hidden states -> unit resolution (host gather).
        #    Host-resident step #3.
        # ----------------------------------------------------------------
        char_hidden_torch_for_unit = (
            ttnn.to_torch(char_hidden_tt).to(torch.float32).reshape(batch, char_seq_len, self.embed_dim)
        )
        ttnn.deallocate(char_hidden_tt)
        unit_hidden_torch = self._hard_upsample(char_hidden_torch_for_unit, dur_out)
        unit_seq_len = int(unit_hidden_torch.shape[1])

        # ----------------------------------------------------------------
        # 6. Unit positional embedding + sum with upsampled hidden.
        # ----------------------------------------------------------------
        # Tile-pad the upsampled hidden state along the seq dim so SDPA's
        # 32x32 tile reads see fully-defined data.
        pad_to = _pad_to_tile(unit_seq_len)
        if pad_to > 0:
            pad_block = torch.zeros(
                (batch, pad_to, self.embed_dim),
                dtype=unit_hidden_torch.dtype,
            )
            unit_hidden_padded = torch.cat([unit_hidden_torch, pad_block], dim=1)
        else:
            unit_hidden_padded = unit_hidden_torch

        full_seq_len = unit_seq_len + pad_to

        # Build the unit-level padding mask in HF semantics
        # (1.0 in [0, sum_durations), 0.0 elsewhere). The mask only sees the
        # logical (un-padded) sequence; the tile-pad tail is handled below
        # in the 4D additive mask.
        unit_padding_mask = self._compute_new_attention_mask(
            dur_out.sum(1),
            batch=batch,
            mask_seq_len=unit_seq_len,
        )

        # Embed positions: HF passes inputs_embeds (un-padded). But we feed
        # the model the padded hidden state, so the padded tail needs SOME
        # positional value. We build a position lookup at the *logical*
        # length and zero-pad the result to ``full_seq_len`` on host. (The
        # additive mask makes padded rows contribute nothing to attention
        # regardless of their values; zeroing here is the cleanest invariant
        # for the residual stream.)
        unit_positions_tt = self.embed_positions(
            input_ids=None,
            inputs_embeds_shape=(batch, unit_seq_len),
            past_key_values_length=0,
        )
        if self.pos_emb_alpha != 1.0:
            unit_positions_tt = ttnn.multiply(unit_positions_tt, self.pos_emb_alpha)

        # If we padded, also pad positions on the device side with zeros.
        unit_hidden_tt = self._to_tt_tile(unit_hidden_padded)
        if pad_to > 0:
            # Zero-pad positions to full_seq_len via host (small).
            positions_torch = (
                ttnn.to_torch(unit_positions_tt).to(torch.float32).reshape(batch, unit_seq_len, self.embed_dim)
            )
            ttnn.deallocate(unit_positions_tt)
            pos_pad = torch.zeros(
                (batch, pad_to, self.embed_dim),
                dtype=positions_torch.dtype,
            )
            positions_padded = torch.cat([positions_torch, pos_pad], dim=1)
            unit_positions_tt = self._to_tt_tile(positions_padded)

        hidden_states_tt = ttnn.add(unit_hidden_tt, unit_positions_tt)
        ttnn.deallocate(unit_hidden_tt)
        ttnn.deallocate(unit_positions_tt)

        # ----------------------------------------------------------------
        # 7. Build the 4D additive attention mask (with tile-pad tail).
        # ----------------------------------------------------------------
        attention_mask_torch = self._build_4d_additive_mask(
            unit_padding_mask,
            pad_to=pad_to,
            dtype=torch.float32,
        )
        attention_mask_tt = self._to_tt_tile(attention_mask_torch)

        # Float ``[B, T+pad_to, 1]`` multiplier for the conv branch (mirrors
        # the HF ``x.masked_fill(~pad_mask, 0)`` step). Padded entries are
        # 0.0 so they don't contribute to subsequent ops.
        unit_pad_mask_padded = torch.zeros((batch, full_seq_len), dtype=torch.float32)
        unit_pad_mask_padded[:, :unit_seq_len] = unit_padding_mask
        padding_mask_btN_tt = self._padding_mask_to_btN(unit_pad_mask_padded)

        # ----------------------------------------------------------------
        # 8. Stack of T2U decoder layers.
        # ----------------------------------------------------------------
        for layer in self.layers:
            hidden_states_tt = layer(
                hidden_states_tt,
                attention_mask=attention_mask_tt,
                padding_mask=padding_mask_btN_tt,
            )

        # ----------------------------------------------------------------
        # 9. Final LayerNorm.
        # ----------------------------------------------------------------
        hidden_states_tt = self.final_layer_norm(hidden_states_tt)

        # Scratch cleanup.
        ttnn.deallocate(attention_mask_tt)
        ttnn.deallocate(padding_mask_btN_tt)

        # Slice off the tile-pad tail.
        if pad_to > 0:
            hidden_states_tt = hidden_states_tt[:, :unit_seq_len, :]

        return {
            "last_hidden_state": hidden_states_tt,
            "padding_mask": unit_padding_mask,
            "dur_out": dur_out,
        }
