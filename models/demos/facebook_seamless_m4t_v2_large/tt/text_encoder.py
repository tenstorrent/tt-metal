# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 NLLB-style text encoder.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::text_encoder_forward``,
which reproduces the full ``SeamlessM4Tv2Encoder`` (NLLB text encoder branch,
``is_t2u_encoder=False``). Op sequence::

    inputs_embeds = scaled_word_embedding(input_ids)
    embed_pos     = sinusoidal_positional_embedding(input_ids, ...)
    hidden        = inputs_embeds + embed_pos
    for layer in self.layers:
        hidden = text_encoder_layer(hidden, attention_mask)
    hidden = final_layer_norm(hidden)

Composed over already-verified TTNN leaf modules:

    - :class:`ScaledWordEmbedding`             (1 instance)
    - :class:`SinusoidalPositionalEmbedding`   (1 instance)
    - :class:`TextEncoderLayer`                (``num_layers`` instances)
    - :class:`LayerNorm`                       (1 instance, final)

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - encoder_layers = 24 (golden uses 2 for size)
    - encoder_attention_heads = 16  (head_dim = 64)
    - encoder_ffn_dim = 8192
    - activation_function = "relu"
    - layer_norm_eps = 1e-5
    - scale_embedding = True  -> embed_scale = sqrt(1024) = 32.0
    - padding_idx = 1 in the real checkpoint, 0 in default config

The loading pattern (DRAM-resident weights, embedding tables in
ROW_MAJOR_LAYOUT, activations in TILE_LAYOUT) mirrors the Whisper
``ttnn_optimized_functional_whisper.py`` reference for absolute-position
embedding lookups.
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
from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder_layer import TextEncoderLayer


class TextEncoder(LightweightModule):
    """Full NLLB-style text encoder in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of self-attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        embed_tokens_weight: torch.Tensor of shape ``(vocab_size, embed_dim)``
            for the scaled word embedding table.
        embed_positions_weights: torch.Tensor of shape
            ``(num_embeddings, embed_dim)`` — the pre-computed sinusoidal
            positional embedding table (with row ``padding_idx`` zeroed).
        layers_state_dict: list of per-layer state dicts in the format
            consumed by :class:`TextEncoderLayer`.
        final_layer_norm_state_dict: ``{"weight", "bias"}`` for the
            terminal LayerNorm.
        eps: LayerNorm epsilon (default 1e-5).
        padding_idx: padding token id (default 1, matches SeamlessM4T-v2
            checkpoint; default config sets 0).
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
        padding_idx: int = 1,
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

        # 3. Stack of encoder layers.
        self.layers = [
            TextEncoderLayer(
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        attention_mask_torch: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the full text encoder.

        Args:
            input_ids: host-side torch int tensor of shape ``[B, T]``
                holding token ids. Kept on host so the padding-aware
                position-id arithmetic in ``SinusoidalPositionalEmbedding``
                can run on the (small) integer ids without a roundtrip;
                ids are uploaded to device internally for the embedding
                gathers.
            attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, T]`` representing an additive log-mask for the
                self-attention. ``None`` means full self-attention.
            attention_mask_torch: optional host-side ``[B, 1, T, T]`` (or
                broadcastable) float mask. When provided (and
                ``attention_mask`` is ``None``), the encoder uploads the
                mask itself and tile-pads it with the additive log-mask
                "minus" fill value so SDPA's tile-aligned read of the mask
                respects the logical sequence length. This is the
                recommended entry point when ``T`` is not a multiple of 32.

        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]`` in TILE_LAYOUT
            (DRAM) — the encoder's ``last_hidden_state``.
        """
        if attention_mask is not None and attention_mask_torch is not None:
            raise ValueError("Pass either attention_mask or attention_mask_torch, not both.")

        batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        pad_to = (32 - seq_len % 32) % 32

        # When the caller provides a host mask AND the sequence length is
        # off-tile, pad the input ids and the mask up to the next tile
        # boundary. SDPA reads both Q and the mask in 32x32 tiles -- if the
        # source mask is smaller than the tile boundary, the unpadded tail
        # gets stale bf16 values that leak into the softmax. Padding the
        # mask with the additive log "minus" fill masks those rows out, so
        # the padded Q rows have no influence on the un-padded output.
        if attention_mask_torch is not None and pad_to > 0:
            input_ids_eff = self._pad_input_ids(input_ids, pad_to)
            attention_mask = self._upload_attention_mask(attention_mask_torch, pad_to)
        else:
            input_ids_eff = input_ids
            if attention_mask_torch is not None:
                attention_mask = ttnn.from_torch(
                    attention_mask_torch,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

        # 1. Token embeddings (lookup + sqrt(hidden_size) scale folded in).
        #    ttnn.embedding wants uint32 ROW_MAJOR ids. Upload here so the
        #    rest of the forward path is fully on-device.
        tt_input_ids = ttnn.from_torch(
            input_ids_eff.to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inputs_embeds = self.embed_tokens(tt_input_ids)

        # 2. Sinusoidal positional embeddings — padding-aware index path
        #    derived from the host-side input_ids (small integer tensor).
        embed_pos = self.embed_positions(
            input_ids=input_ids_eff,
            past_key_values_length=0,
        )

        # 3. Sum embeddings.
        hidden_states = ttnn.add(inputs_embeds, embed_pos)
        ttnn.deallocate(inputs_embeds)
        ttnn.deallocate(embed_pos)

        # 4. Stack of encoder layers.
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # 5. Final LayerNorm.
        hidden_states = self.final_layer_norm(hidden_states)

        # 6. Slice back to the original (unpadded) sequence length if we
        #    padded the input on entry. SDPA needs Q and mask to match so
        #    we padded both together; trim the tail off the output here.
        if input_ids_eff.shape[1] != seq_len:
            hidden_states = hidden_states[:, :seq_len, :]
        return hidden_states

    def _pad_input_ids(self, input_ids: torch.Tensor, pad_to: int) -> torch.Tensor:
        """Pad ids on the right with ``padding_idx`` so the resulting embeds
        get the all-zero padding row (HF semantics) and the sinusoidal
        position lookup returns the (also zero) padding row too -- making
        the tail rows zero-init pre-attention.
        """
        batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        pad = torch.full(
            (batch, pad_to),
            self.padding_idx,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, pad], dim=1)

    def _upload_attention_mask(self, mask_torch: torch.Tensor, pad_to: int) -> ttnn.Tensor:
        """Upload a host attention mask, padding the trailing seq dims with
        the additive log "minus" fill so SDPA's 32x32 tile reads see fully
        masked-out rows/cols for any tile-boundary padding.
        """
        seq_dim_q, seq_dim_k = mask_torch.shape[-2], mask_torch.shape[-1]
        # Use HF's "minus" fill (``torch.finfo(dtype).min``) so the padded
        # region is masked out identically to a real pad position.
        fill = float(torch.finfo(mask_torch.dtype).min)
        new_shape = list(mask_torch.shape)
        new_shape[-2] = seq_dim_q + pad_to
        new_shape[-1] = seq_dim_k + pad_to
        padded = torch.full(new_shape, fill, dtype=mask_torch.dtype)
        padded[..., :seq_dim_q, :seq_dim_k] = mask_torch
        return ttnn.from_torch(
            padded,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
