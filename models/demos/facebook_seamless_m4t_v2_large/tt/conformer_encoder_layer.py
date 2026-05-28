# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Conformer encoder layer.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::conformer_encoder_layer_forward``,
which reproduces one full block of the W2v-BERT-2.0 speech-encoder stack
(``SeamlessM4Tv2ConformerEncoderLayer``). It applies the macaron / half-step
FFN sandwich pattern around a self-attention + causal depthwise conv core::

    x = x + 0.5 * conformer_ffn(layer_norm(x))                # half-step FFN 1
    x = x + conformer_self_attention(layer_norm(x))           # MHA + relative_key bias
    x = x + conformer_convolution_module(x, conv_mask)        # causal depthwise conv
    x = x + 0.5 * conformer_ffn(layer_norm(x))                # half-step FFN 2
    x = layer_norm(x)                                          # final post-norm

This block is implemented as a thin composition over the already-verified
TTNN leaf modules:

    - :class:`LayerNorm`                       (4 instances: ffn1, self-attn, ffn2, final)
    - :class:`ConformerFfn`                    (2 instances: ffn1, ffn2)
    - :class:`ConformerSelfAttention`          (1 instance, owns the conv-module's
                                                inner LayerNorms internally)
    - :class:`ConformerConvolutionModule`      (1 instance, includes its own LNs)

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - speech_encoder_attention_heads = 16  (head_dim = 64)
    - position_embeddings_type = "relative_key"
    - left_max_position_embeddings = 64
    - right_max_position_embeddings = 8
    - conv_depthwise_kernel_size = 31
    - speech_encoder_hidden_act = "swish" (== SiLU)
    - layer_norm_eps = 1e-5
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_convolution_module import ConformerConvolutionModule
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_ffn import ConformerFfn
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_self_attention import ConformerSelfAttention
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm


class ConformerEncoderLayer(LightweightModule):
    """One full Conformer encoder layer (macaron FFN sandwich) in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        seq_len: sequence length (fixed at construction so the relative-key
            position bias table can be precomputed once).
        state_dict: nested mapping containing all sub-block weights with the
            same layout produced by ``_extract_encoder_layer_state_dict`` in
            the reference test - i.e. keys
            ``{"ffn1_layer_norm", "ffn1", "self_attn_layer_norm", "self_attn",
              "conv_module", "ffn2_layer_norm", "ffn2", "final_layer_norm"}``.
        distance_embedding_weight: ``[L+R+1, head_dim]`` relative-key embedding
            table for the self-attention bias.
        left_max_position_embeddings: ``L`` clamp for relative_key (64).
        right_max_position_embeddings: ``R`` clamp for relative_key (8).
        position_embeddings_type: ``"relative_key"`` (default) or ``None``.
        conv_kernel_size: depthwise kernel size for the conv module (31).
        eps: LayerNorm epsilon (1e-5).
        batch_size: forward batch size; used by the self-attention's
            relative-position-bias table layout.
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM).
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        seq_len: int,
        state_dict,
        distance_embedding_weight=None,
        left_max_position_embeddings: int = 64,
        right_max_position_embeddings: int = 8,
        position_embeddings_type: Optional[str] = "relative_key",
        conv_kernel_size: int = 31,
        eps: float = 1e-5,
        batch_size: int = 1,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        self.device = device
        self.embed_dim = embed_dim
        self.eps = float(eps)

        # 1. Pre-FFN1 LayerNorm.
        self.ffn1_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["ffn1_layer_norm"]["weight"],
            bias=state_dict["ffn1_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. FFN1 (Conformer macaron half-step).
        self.ffn1 = ConformerFfn(
            device=device,
            intermediate_weight=state_dict["ffn1"]["intermediate_dense"]["weight"],
            intermediate_bias=state_dict["ffn1"]["intermediate_dense"]["bias"],
            output_weight=state_dict["ffn1"]["output_dense"]["weight"],
            output_bias=state_dict["ffn1"]["output_dense"]["bias"],
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 3. Pre-self-attention LayerNorm.
        self.self_attn_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["self_attn_layer_norm"]["weight"],
            bias=state_dict["self_attn_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 4. Self-attention with relative-key positional bias.
        self.self_attn = ConformerSelfAttention(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            state_dict=state_dict["self_attn"],
            distance_embedding_weight=distance_embedding_weight,
            left_max_position_embeddings=left_max_position_embeddings,
            right_max_position_embeddings=right_max_position_embeddings,
            position_embeddings_type=position_embeddings_type,
            batch_size=batch_size,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 5. Convolution module (causal depthwise + GLU). The module owns its
        #    own internal LayerNorms (pre- and post-depthwise), so no extra
        #    LayerNorm at the encoder-layer level for this branch.
        conv_sd = state_dict["conv_module"]
        self.conv_module = ConformerConvolutionModule(
            device=device,
            layer_norm_weight=conv_sd["layer_norm"]["weight"],
            layer_norm_bias=conv_sd["layer_norm"]["bias"],
            pointwise_conv1_weight=conv_sd["pointwise_conv1"]["weight"],
            depthwise_conv_weight=conv_sd["depthwise_conv"]["weight"],
            depthwise_layer_norm_weight=conv_sd["depthwise_layer_norm"]["weight"],
            depthwise_layer_norm_bias=conv_sd["depthwise_layer_norm"]["bias"],
            pointwise_conv2_weight=conv_sd["pointwise_conv2"]["weight"],
            kernel_size=conv_kernel_size,
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 6. Pre-FFN2 LayerNorm.
        self.ffn2_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["ffn2_layer_norm"]["weight"],
            bias=state_dict["ffn2_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 7. FFN2 (second macaron half-step).
        self.ffn2 = ConformerFfn(
            device=device,
            intermediate_weight=state_dict["ffn2"]["intermediate_dense"]["weight"],
            intermediate_bias=state_dict["ffn2"]["intermediate_dense"]["bias"],
            output_weight=state_dict["ffn2"]["output_dense"]["weight"],
            output_bias=state_dict["ffn2"]["output_dense"]["bias"],
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 8. Final post-LayerNorm.
        self.final_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["final_layer_norm"]["weight"],
            bias=state_dict["final_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        conv_attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run one Conformer encoder layer.

        Args:
            hidden_states: ttnn tensor of shape ``[B, T, embed_dim]`` in
                TILE_LAYOUT.
            attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T, T]`` representing an additive log-mask for the
                self-attention.
            conv_attention_mask: optional ttnn tensor broadcastable to
                ``[B, T, 1]`` with 0.0 marking padded positions to zero out
                before the depthwise convolution.

        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]``.
        """
        # 1. Half-step FFN 1.
        residual = hidden_states
        x = self.ffn1_layer_norm(hidden_states)
        x = self.ffn1(x)
        x = ttnn.multiply(x, 0.5)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # 2. Self-attention residual.
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attention_mask=attention_mask)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # 3. Convolution-module residual.
        residual = x
        x = self.conv_module(x, attention_mask=conv_attention_mask)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # 4. Half-step FFN 2.
        residual = x
        x = self.ffn2_layer_norm(x)
        x = self.ffn2(x)
        x = ttnn.multiply(x, 0.5)
        x = ttnn.add(x, residual)
        ttnn.deallocate(residual)

        # 5. Final post-LayerNorm.
        out = self.final_layer_norm(x)
        ttnn.deallocate(x)
        return out
