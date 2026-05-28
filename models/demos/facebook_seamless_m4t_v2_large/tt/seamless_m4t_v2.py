# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 top-level T2TT model.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::seamless_m4t_v2_forward``,
which reproduces the deterministic ``SeamlessM4Tv2ForTextToText`` forward
pass (Text-to-Text Translation). Op sequence::

    encoder_hidden = text_encoder(input_ids, attention_mask)
    decoder_hidden = text_decoder(decoder_input_ids,
                                   encoder_hidden_states=encoder_hidden,
                                   attention_mask=decoder_attention_mask,
                                   encoder_attention_mask=attention_mask)
    logits = decoder_hidden @ lm_head_weight.T  # bias=False in HF

Composed over already-verified TTNN sub-blocks:

    - :class:`TextEncoder`  (1 instance)
    - :class:`TextDecoder`  (1 instance)
    - inline ``ttnn.linear``  (LM head, bias=False)

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - vocab_size = 256102
    - encoder_layers = decoder_layers = 24 (golden uses 2)
    - encoder_attention_heads = decoder_attention_heads = 16 -> head_dim = 64
    - encoder_ffn_dim = decoder_ffn_dim = 8192
    - activation_function = "relu"
    - layer_norm_eps = 1e-5
    - scale_embedding = True -> embed_scale = sqrt(1024) = 32.0

**Tile padding for SDPA**

Encoder/decoder sub-blocks already handle their internal tile padding for
the SDPA mask read path (encoder takes ``attention_mask_torch``; decoder
takes 2D padding masks and builds the 4D causal mask + cross-attention
mask itself). This top-level composition forwards masks via those entry
points unchanged.

**HF mask convention**

The encoder self-attention mask and the decoder cross-attention encoder
mask are the same 2D ``attention_mask`` tensor in HF — we mirror that
exactly here: the same host tensor is fed to both the encoder (expanded
to a 4D additive log-mask) and the decoder (used as the 2D encoder
padding mask, expanded inside the decoder).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder import TextDecoder
from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder import TextEncoder


class SeamlessM4Tv2(LightweightModule):
    """Top-level T2TT (Text-to-Text Translation) composition in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of self/cross-attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        text_encoder_state_dict: nested dict consumed by
            :class:`TextEncoder` with keys
            ``{"embed_tokens": {"weight": ...},
               "embed_positions_weights": ...,
               "layers": [...],
               "final_layer_norm": {"weight", "bias"}}``.
        text_decoder_state_dict: nested dict consumed by
            :class:`TextDecoder` with keys
            ``{"embed_tokens": {"weight": ...},
               "embed_positions_weights": ...,
               "layers": [...],
               "layer_norm": {"weight", "bias"}}``.
        lm_head_state_dict: ``{"weight": (vocab_size, hidden_size)}`` —
            HF uses ``bias=False`` so no bias key is needed.
        eps: LayerNorm epsilon (default 1e-5).
        encoder_padding_idx: encoder padding token id (default 1).
        decoder_padding_idx: decoder padding token id (default 0).
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
        text_encoder_state_dict,
        text_decoder_state_dict,
        lm_head_state_dict,
        eps: float = 1e-5,
        encoder_padding_idx: int = 1,
        decoder_padding_idx: int = 0,
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
        self.eps = float(eps)
        if embed_scale is None:
            embed_scale = math.sqrt(embed_dim)
        self.embed_scale = float(embed_scale)

        # 1. Encoder (shares scale + activation + eps).
        self.text_encoder = TextEncoder(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            embed_tokens_weight=text_encoder_state_dict["embed_tokens"]["weight"],
            embed_positions_weights=text_encoder_state_dict["embed_positions_weights"],
            layers_state_dict=text_encoder_state_dict["layers"],
            final_layer_norm_state_dict=text_encoder_state_dict["final_layer_norm"],
            eps=self.eps,
            padding_idx=int(encoder_padding_idx),
            embed_scale=self.embed_scale,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. Decoder (separate position-aware embedding + stacked decoder layers).
        self.text_decoder = TextDecoder(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            embed_tokens_weight=text_decoder_state_dict["embed_tokens"]["weight"],
            embed_positions_weights=text_decoder_state_dict["embed_positions_weights"],
            layers_state_dict=text_decoder_state_dict["layers"],
            final_layer_norm_state_dict=text_decoder_state_dict["layer_norm"],
            eps=self.eps,
            padding_idx=int(decoder_padding_idx),
            embed_scale=self.embed_scale,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 3. LM head: nn.Linear(hidden_size, vocab_size, bias=False).
        # ttnn.linear expects the weight in (in_features, out_features) layout,
        # so we transpose the HF (vocab, hidden) weight on the host before
        # uploading.
        lm_head_weight = lm_head_state_dict["weight"]
        self.lm_head_weight = ttnn.from_torch(
            lm_head_weight.t().contiguous(),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

    def _expand_encoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        tgt_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Expand a 2D ``[B, S]`` padding mask to a 4D additive log-mask.

        Mirrors HF's ``_prepare_4d_attention_mask`` exactly: the 2D
        ``(1=keep, 0=pad)`` mask becomes ``[B, 1, T, S]`` where padded K
        columns are filled with ``finfo(dtype).min`` (broadcast over Q
        rows). The encoder block requires a 4D additive log-mask for its
        SDPA call.
        """
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

        return _prepare_4d_attention_mask(attention_mask, dtype, tgt_len=tgt_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the top-level T2TT forward pass.

        Args:
            input_ids: host-side torch int tensor of shape ``[B, S]`` of
                encoder token ids.
            decoder_input_ids: host-side torch int tensor of shape
                ``[B, T]`` of decoder token ids.
            attention_mask: optional 2D host ``[B, S]`` padding mask
                (1=keep, 0=pad). Used as both the encoder self-attention
                padding mask AND the decoder cross-attention encoder mask
                (HF semantics).
            decoder_attention_mask: optional 2D host ``[B, T]`` padding
                mask (1=keep, 0=pad). The decoder always adds the
                triangular causal mask internally on top.

        Returns:
            ttnn tensor of shape ``[B, T, vocab_size]`` (TILE_LAYOUT, DRAM)
            — the unnormalized token logits.
        """
        # --- 1. Encoder ---
        # Expand the 2D encoder padding mask to 4D for the encoder's SDPA.
        # When ``attention_mask`` is None, the encoder runs unmasked.
        if attention_mask is not None and attention_mask.dim() == 2:
            encoder_self_mask_torch = self._expand_encoder_attention_mask(
                attention_mask,
                tgt_len=int(input_ids.shape[-1]),
                dtype=torch.float32,
            )
        else:
            encoder_self_mask_torch = attention_mask  # None or already 4D

        encoder_hidden_tt = self.text_encoder(
            input_ids,
            attention_mask_torch=encoder_self_mask_torch,
        )

        # --- 2. Decoder ---
        # The decoder consumes the encoder hidden states as a host torch
        # tensor (so it can pad them along the src-seq dim for tile-aligned
        # SDPA reads). Pull the encoder output back to host once.
        encoder_hidden_torch = ttnn.to_torch(encoder_hidden_tt).to(torch.float32)
        ttnn.deallocate(encoder_hidden_tt)
        # Drop the leading singleton tile dim if present (ttnn.to_torch may
        # add one); the decoder expects exactly [B, S, H].
        if encoder_hidden_torch.dim() == 4 and encoder_hidden_torch.shape[0] == 1:
            encoder_hidden_torch = encoder_hidden_torch.squeeze(0)
        if encoder_hidden_torch.dim() != 3:
            raise RuntimeError(f"Unexpected encoder hidden shape after to_torch: {tuple(encoder_hidden_torch.shape)}")

        decoder_hidden_tt = self.text_decoder(
            decoder_input_ids,
            encoder_hidden_states_torch=encoder_hidden_torch,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
        )

        # --- 3. LM head: linear(hidden, vocab, bias=False) ---
        logits_tt = ttnn.linear(decoder_hidden_tt, self.lm_head_weight)
        ttnn.deallocate(decoder_hidden_tt)
        return logits_tt
