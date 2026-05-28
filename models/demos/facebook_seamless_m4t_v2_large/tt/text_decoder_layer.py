# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 NLLB-style text decoder layer.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::text_decoder_layer_forward``,
which reproduces one full ``SeamlessM4Tv2DecoderLayer`` (NLLB-style BART
decoder block). It follows the standard pre-norm Transformer decoder block::

    residual = x
    x = self_attn_layer_norm(x)
    x = self_attn(x, attention_mask=self_attention_mask)  # causal self-attn
    x = residual + x

    if encoder_hidden_states is not None:
        residual = x
        x = cross_attention_layer_norm(x)
        x = cross_attention(x, encoder_hidden_states,
                            attention_mask=encoder_attention_mask)
        x = residual + x

    residual = x
    x = ffn_layer_norm(x)
    x = ffn(x)                # Linear -> ReLU -> Linear (both with bias)
    x = residual + x

    return x

This block is implemented as a thin composition over the already-verified
TTNN leaf modules:

    - :class:`LayerNorm`     (3 instances: self-attn pre-norm,
                              cross-attn pre-norm, ffn pre-norm)
    - :class:`SeamlessMHA`   (2 instances: self-attn, cross-attn)
    - :class:`SeamlessFfn`   (1 instance, Linear -> ReLU -> Linear)

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - decoder_attention_heads = 16  (head_dim = 64)
    - decoder_ffn_dim = 8192
    - activation_function = "relu"
    - layer_norm_eps = 1e-5

Note the HF v2 parameter naming uses ``cross_attention*`` (NOT
``encoder_attn*`` as in some BART forks) for the cross-attention block.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm
from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_ffn import SeamlessFfn
from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_mha import SeamlessMHA


class TextDecoderLayer(LightweightModule):
    """One full NLLB-style text decoder layer (pre-norm) in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        state_dict: nested mapping with keys
            ``{"self_attn_layer_norm", "self_attn",
                "cross_attention_layer_norm", "cross_attention",
                "ffn_layer_norm", "ffn"}``
            laid out as produced by ``_extract_text_decoder_layer_state_dict``
            in the reference test.
        eps: LayerNorm epsilon (1e-5).
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM).
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        state_dict,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        self.device = device
        self.embed_dim = embed_dim
        self.eps = float(eps)

        # 1. Pre-self-attention LayerNorm.
        self.self_attn_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["self_attn_layer_norm"]["weight"],
            bias=state_dict["self_attn_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. Self-attention (BART-style 4-proj MHA with bias).
        self.self_attn = SeamlessMHA(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            state_dict=state_dict["self_attn"],
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 3. Pre-cross-attention LayerNorm.
        self.cross_attention_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["cross_attention_layer_norm"]["weight"],
            bias=state_dict["cross_attention_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 4. Cross-attention (BART-style 4-proj MHA with bias).
        self.cross_attention = SeamlessMHA(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            state_dict=state_dict["cross_attention"],
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 5. Pre-FFN LayerNorm.
        self.ffn_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["ffn_layer_norm"]["weight"],
            bias=state_dict["ffn_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 6. Feed-forward (Linear -> ReLU -> Linear).
        ffn_sd = state_dict["ffn"]
        self.ffn = SeamlessFfn(
            device=device,
            fc1_weight=ffn_sd["fc1"]["weight"],
            fc1_bias=ffn_sd["fc1"]["bias"],
            fc2_weight=ffn_sd["fc2"]["weight"],
            fc2_bias=ffn_sd["fc2"]["bias"],
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        self_attention_mask: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run one NLLB-style text decoder layer.

        Args:
            hidden_states: ttnn tensor of shape ``[B, T, embed_dim]`` in
                TILE_LAYOUT.
            encoder_hidden_states: optional ttnn tensor of shape
                ``[B, S, embed_dim]``. If ``None`` the cross-attention block
                is skipped (matches HF behaviour where ``cross_attn_weights``
                is ``None`` and the residual stream is untouched between
                self-attn and FFN).
            self_attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, T]`` representing an additive log-mask for the
                causal self-attention.
            encoder_attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, S]`` representing an additive log-mask for the
                cross-attention.

        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]``.
        """
        # 1. Self-attention residual (pre-norm).
        residual = hidden_states
        x = self.self_attn_layer_norm(hidden_states)
        x = self.self_attn(x, encoder_hidden_states=None, attention_mask=self_attention_mask)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # 2. Cross-attention residual (pre-norm). Skipped when no encoder states.
        if encoder_hidden_states is not None:
            residual = x
            x = self.cross_attention_layer_norm(x)
            x = self.cross_attention(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            x = ttnn.add(x, residual)
            ttnn.deallocate(residual)

        # 3. FFN residual (pre-norm).
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)
        return x
