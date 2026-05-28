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
        past_key_values=None,
        position: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ) -> ttnn.Tensor:
        """Run one NLLB-style text decoder layer.

        Args:
            hidden_states: ttnn tensor of shape ``[B, T, embed_dim]`` in
                TILE_LAYOUT. When ``past_key_values`` is provided ``T`` is
                always 1 (the single new decoder token).
            encoder_hidden_states: optional ttnn tensor of shape
                ``[B, S, embed_dim]``. If ``None`` the cross-attention block
                is skipped (matches HF behaviour where ``cross_attn_weights``
                is ``None`` and the residual stream is untouched between
                self-attn and FFN). When ``past_key_values`` is provided
                this argument is unused — cross-attention K/V come from the
                already-populated cache.
            self_attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, T]`` (non-cached) or ``[B, 1, 1, max_seq]``
                (cached) representing an additive log-mask for the self
                attention.
            encoder_attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, S]`` representing an additive log-mask for the
                cross-attention.
            past_key_values: optional
                :class:`models.demos.facebook_seamless_m4t_v2_large.tt.kv_cache.TextDecoderKVCache`
                bundle. When provided the layer runs in incremental-decode
                mode: it expects ``T == 1``, writes the new self-attn K/V
                into the cache at ``position``, and reads cross-attn K/V
                from the already-populated cross-attention cache for the
                given ``layer_idx``.
            position: integer index (0-based) at which to write the new
                self-attn K/V. Required when ``past_key_values`` is given.
            layer_idx: this layer's index in the decoder stack. Required
                when ``past_key_values`` is given.

        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]``.
        """
        if past_key_values is not None:
            return self._forward_cached(
                hidden_states=hidden_states,
                self_attention_mask=self_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                position=position,
                layer_idx=layer_idx,
            )

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

    # ------------------------------------------------------------------ cached path
    def _forward_cached(
        self,
        hidden_states: ttnn.Tensor,
        self_attention_mask: Optional[ttnn.Tensor],
        encoder_attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        position: Optional[int],
        layer_idx: Optional[int],
    ) -> ttnn.Tensor:
        """Incremental-decode forward: one new token + KV cache.

        Matches the math of :meth:`forward` exactly for the single new
        decoder position. The self-attention K/V is computed only for
        the new token and merged into the per-layer cache at
        ``position``; the SDPA then attends over the FULL cache (the
        mask zeros out future positions). Cross-attention skips K and V
        projection entirely — those tensors were stored once in
        ``past_key_values.cross_attn`` by the encoder-prefill step.
        """
        if position is None:
            raise ValueError("position must be provided in cached forward.")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in cached forward.")

        # 1. Self-attention residual (pre-norm) with KV cache update.
        residual = hidden_states
        x = self.self_attn_layer_norm(hidden_states)

        # Project Q,K,V from the single new token.
        q_new, k_new, v_new = self.self_attn.project_qkv_single_token(x)
        ttnn.deallocate(x)

        # Update the per-layer self-attn cache at `position` and read the
        # full cache for SDPA. K/V tensors are tile-padded by SDPA's mask.
        past_key_values.self_attn.update(layer_idx, k_new, v_new, pos=position)
        ttnn.deallocate(k_new)
        ttnn.deallocate(v_new)
        k_full, v_full = past_key_values.self_attn.read(layer_idx)

        # SDPA + out_proj.
        x = self.self_attn.attend_and_out_project(q_new, k_full, v_full, attention_mask=self_attention_mask)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # 2. Cross-attention residual (pre-norm). Always present in cached
        #    mode — the caller already populated the cross-attn cache.
        residual = x
        x = self.cross_attention_layer_norm(x)
        # Q from the current decoder hidden; K/V from the static cross cache.
        q = self.cross_attention.project_q(x)
        ttnn.deallocate(x)
        k_cross, v_cross = past_key_values.cross_attn.read(layer_idx)
        x = self.cross_attention.attend_and_out_project(q, k_cross, v_cross, attention_mask=encoder_attention_mask)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # 3. FFN residual (pre-norm).
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)
        return x
