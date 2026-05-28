# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Text-to-Unit (T2U) encoder.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::t2u_encoder_forward``,
which reproduces ``SeamlessM4Tv2Encoder(is_t2u_encoder=True)`` bit-for-bit.

Structurally identical to the NLLB text encoder (uses the same
``text_encoder_layer`` building block) BUT:
    - No token embedding.
    - No sinusoidal positional embedding.
    - Consumes pre-embedded ``inputs_embeds`` of shape ``[B, T, hidden]``
      directly.

Op sequence (matches HF exactly)::

    hidden = inputs_embeds                        # no embed lookup
                                                  # no positional add
    for layer in self.layers:                     # 6 layers in the full model
        hidden = text_encoder_layer(hidden, attention_mask)
    hidden = final_layer_norm(hidden)

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - t2u_encoder_layers = 6 (golden uses 2 to keep file size small)
    - t2u_encoder_attention_heads = 16  (head_dim = 64)
    - t2u_encoder_ffn_dim = 8192
    - activation_function = "relu"
    - layer_norm_eps = 1e-5

The block reuses the already-verified TTNN leaf modules:

    - :class:`TextEncoderLayer`  (``num_layers`` instances; the T2U encoder
      layer is structurally identical to the text encoder layer).
    - :class:`LayerNorm`         (1 instance, final).

Tile-padding fix (from text_encoder/text_decoder bring-up):
    When ``T`` is not a multiple of 32 (e.g. T=16 in the golden) the SDPA
    used inside ``SeamlessMHA`` reads Q and the attention mask in 32x32
    tiles. If the mask is supplied at the un-padded shape, the trailing
    sub-tile rows/cols pick up stale bf16 values that leak into the softmax
    and tank PCC. We pad ``inputs_embeds`` with zeros and the mask with the
    additive log "minus" fill (``finfo.min``) up to the next tile boundary
    so the padded rows are fully masked out, then slice the tail off the
    final-LN output before returning.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm
from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder_layer import TextEncoderLayer


class T2uEncoder(LightweightModule):
    """Full T2U encoder in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of self-attention heads (16 for v2-Large T2U).
        head_dim: per-head dim (64 for v2-Large T2U). Requires
            ``num_heads * head_dim == embed_dim``.
        layers_state_dict: list of per-layer state dicts in the format
            consumed by :class:`TextEncoderLayer`.
        final_layer_norm_state_dict: ``{"weight", "bias"}`` for the
            encoder's terminal LayerNorm.
        eps: LayerNorm epsilon (default 1e-5).
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM).
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        layers_state_dict,
        final_layer_norm_state_dict,
        eps: float = 1e-5,
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

        # 1. Stack of encoder layers (structurally identical to text encoder layer).
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

        # 2. Final LayerNorm.
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
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        attention_mask_torch: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the full T2U encoder.

        Args:
            inputs_embeds: host-side torch float tensor of shape
                ``[B, T, embed_dim]``. Uploaded to device internally. The
                T2U encoder consumes pre-embedded states directly (no token
                lookup).
            attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, T]`` representing an additive log-mask for the
                self-attention. ``None`` means full self-attention.
            attention_mask_torch: optional host-side ``[B, 1, T, T]`` (or
                broadcastable) float mask. When provided (and
                ``attention_mask`` is ``None``), the encoder uploads the
                mask itself and tile-pads it with the additive log-mask
                "minus" fill value so SDPA's tile-aligned read respects the
                logical sequence length. This is the recommended entry
                point when ``T`` is not a multiple of 32.

        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]`` in TILE_LAYOUT
            (DRAM) -- the T2U encoder's ``last_hidden_state``.
        """
        if attention_mask is not None and attention_mask_torch is not None:
            raise ValueError("Pass either attention_mask or attention_mask_torch, not both.")

        seq_len = int(inputs_embeds.shape[1])
        pad_to = (32 - seq_len % 32) % 32

        # Tile-pad inputs (zeros) + mask (finfo.min) so SDPA's 32x32 tile
        # reads respect the logical sequence length when T % 32 != 0.
        if pad_to > 0:
            inputs_embeds_eff = self._pad_inputs_embeds(inputs_embeds, pad_to)
            if attention_mask_torch is not None:
                attention_mask = self._upload_attention_mask(attention_mask_torch, pad_to)
            elif attention_mask is None:
                # Unmasked path with non-tile-aligned T: build a mask that
                # additively masks the padded tail so it can't influence the
                # un-padded outputs. Real rows attend to all real rows; padded
                # rows are isolated by the additive minus-fill columns.
                attention_mask = self._build_pad_only_mask(
                    batch=int(inputs_embeds.shape[0]),
                    seq_len=seq_len,
                    pad_to=pad_to,
                    dtype=inputs_embeds.dtype,
                )
        else:
            inputs_embeds_eff = inputs_embeds
            if attention_mask_torch is not None:
                attention_mask = ttnn.from_torch(
                    attention_mask_torch,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

        # Upload (possibly padded) inputs_embeds to device.
        hidden_states = ttnn.from_torch(
            inputs_embeds_eff.to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 1. Stack of encoder layers.
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # 2. Final LayerNorm.
        hidden_states = self.final_layer_norm(hidden_states)

        # 3. Slice back to the original (unpadded) sequence length.
        if pad_to > 0:
            hidden_states = hidden_states[:, :seq_len, :]
        return hidden_states

    def _pad_inputs_embeds(self, inputs_embeds: torch.Tensor, pad_to: int) -> torch.Tensor:
        """Right-pad inputs_embeds with zeros along the sequence dim."""
        batch, _, hidden = inputs_embeds.shape
        pad = torch.zeros(
            (batch, pad_to, hidden),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        return torch.cat([inputs_embeds, pad], dim=1)

    def _upload_attention_mask(self, mask_torch: torch.Tensor, pad_to: int) -> ttnn.Tensor:
        """Upload a host attention mask, padding the trailing seq dims with
        the additive log "minus" fill so SDPA's 32x32 tile reads see fully
        masked-out rows/cols for any tile-boundary padding.
        """
        seq_dim_q, seq_dim_k = mask_torch.shape[-2], mask_torch.shape[-1]
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

    def _build_pad_only_mask(
        self,
        batch: int,
        seq_len: int,
        pad_to: int,
        dtype: torch.dtype,
    ) -> ttnn.Tensor:
        """Build a ``[B, 1, T+pad, T+pad]`` additive mask that is zero in
        the real ``[T, T]`` corner and ``finfo.min`` everywhere the padded
        rows/cols overlap. Used for the "unmasked + non-tile-aligned T"
        path: real Q rows still attend to all real K cols, but padded rows
        are isolated and padded K cols contribute -inf to any softmax.
        """
        full = seq_len + pad_to
        fill = float(torch.finfo(dtype).min)
        mask = torch.full((batch, 1, full, full), fill, dtype=dtype)
        mask[..., :seq_len, :seq_len] = 0.0
        return ttnn.from_torch(
            mask,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
