# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Implementation of Bark Small GPT Block.

Implements the shared GPT-2 style transformer block used by all three Bark stages:
- Text-to-Semantic (causal)
- Semantic-to-Coarse (causal)
- Coarse-to-Fine (non-causal)

Architecture per block:
    hidden_states -> LayerNorm1 -> MultiHeadAttention -> + residual
                  -> LayerNorm2 -> MLP (Linear->GELU->Linear) -> + residual

Reference: HuggingFace transformers/models/bark/modeling_bark.py
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

from models.common.utility_functions import nearest_32

import torch

import ttnn


@dataclass
class BarkConfig:
    """Configuration for Bark Small model components."""

    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    block_size: int = 1024
    input_vocab_size: int = 10048
    output_vocab_size: int = 10048
    bias: bool = False
    n_codes_total: int = 8
    n_codes_given: int = 2

    layer_norm_epsilon: float = 1e-5

    # Optimization config (Stage 2 & 3)
    use_lofi: bool = False
    use_sharding: bool = True  # Enable sharding for Stage 3
    grid_size: Optional[Any] = None  # ttnn.CoreGrid


def preprocess_linear_weight(weight_tensor, device):
    """Convert a PyTorch weight tensor to a TTNN tensor for linear ops.

    PyTorch stores linear weights as (out_features, in_features).
    ttnn.linear expects [1, 1, in_features, out_features] (4D TILE).
    This is the standard TTNN transformer convention (Llama/GPT2/Falcon).

    Weights are stored in DRAM to avoid NCRISC compiler pressure when
    all 12 transformer layers are loaded simultaneously.
    """
    weight = weight_tensor.detach().float()
    if weight.dim() == 2:
        weight = weight.t()  # (out, in) -> (in, out) for ttnn.linear
        weight = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, in, out]
    tt_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_weight


def preprocess_bias_weight(bias_tensor, device):
    """Convert a 1D bias tensor to TTNN format [1, 1, 1, H].

    Unlike preprocess_linear_weight (which transposes 2D weights),
    biases must NOT be transposed — they are broadcast along the last dim.
    """
    bias = bias_tensor.detach().float()
    if bias.dim() == 1:
        bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H]
    elif bias.dim() == 2:
        # Already [1, H] from .unsqueeze(0) — just add batch dims
        bias = bias.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H]
    tt_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_bias


def preprocess_layernorm_weight(weight_tensor, device):
    """Convert LayerNorm weight/bias to TTNN tensor."""
    w = weight_tensor.detach().float()
    if w.dim() == 1:
        w = w.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, hidden]
    tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_w


def preprocess_embedding_weight(weight_tensor, device):
    """Preprocess embedding weights for ttnn.embedding (2D ROW_MAJOR).

    Stage 2 optimization: keeps embeddings on device for on-device lookup
    instead of CPU-side nn.Embedding.
    """
    w = weight_tensor.detach().float()
    return ttnn.from_torch(
        w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


class TtBarkMLP:
    def __init__(self, device, parameters, config: BarkConfig):
        self.device = device
        self.config = config

        # Preprocess weights
        self.in_proj_weight = preprocess_linear_weight(parameters["in_proj"]["weight"], device)
        self.in_proj_bias = (
            preprocess_bias_weight(parameters["in_proj"]["bias"], device)
            if config.bias and "bias" in parameters["in_proj"]
            else None
        )

        self.out_proj_weight = preprocess_linear_weight(parameters["out_proj"]["weight"], device)
        self.out_proj_bias = (
            preprocess_bias_weight(parameters["out_proj"]["bias"], device)
            if config.bias and "bias" in parameters["out_proj"]
            else None
        )

        # Optimization config (Stage 3)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(
        self, hidden_states: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = ttnn.L1_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        """Forward pass through MLP block with fused activation."""
        # Linear projection: hidden -> 4*hidden with fused GELU
        intermediate = ttnn.linear(
            hidden_states,
            self.in_proj_weight,
            bias=self.in_proj_bias,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Bark uses GELU_NEW (tanh approximation), NOT standard erf-based GELU.
        # gelu_new(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        # Using erf-based gelu causes PCC to decay ~0.03/layer across 12 layers.
        x = intermediate
        x_cubed = ttnn.multiply(ttnn.multiply(x, x), x)
        inner = ttnn.add(x, ttnn.multiply(x_cubed, 0.044715))
        ttnn.deallocate(x_cubed)
        inner = ttnn.multiply(inner, math.sqrt(2.0 / math.pi))
        tanh_out = ttnn.tanh(inner)
        ttnn.deallocate(inner)
        intermediate = ttnn.multiply(ttnn.multiply(x, 0.5), ttnn.add(tanh_out, 1.0))
        ttnn.deallocate(tanh_out)

        # Linear projection: 4*hidden -> hidden
        output = ttnn.linear(
            intermediate,
            self.out_proj_weight,
            bias=self.out_proj_bias,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(intermediate)

        return output


class TtBarkAttention:
    """Multi-head self-attention for Bark.

    Supports both causal (stages 1-2) and non-causal (stage 3) attention.
    Uses pre-allocated KV cache with write-in-place updates (O(n) vs O(n²) concat).

    Architecture:
        Q, K, V = att_proj(hidden_states).split(3)
        attn = SDPA(Q, K, V, is_causal)
        output = out_proj(attn)
    """

    def __init__(self, device, parameters, config: BarkConfig, is_causal: bool = True):
        self.device = device
        self.config = config
        self.is_causal = is_causal
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.embed_dim = config.hidden_size
        self.scale = 1.0 / (self.head_dim**0.5)
        self.max_seq_len = config.block_size  # 1024

        # Pre-allocated KV cache tensors (lazy-init on first use)
        self._kv_cache_k = None
        self._kv_cache_v = None

        assert self.embed_dim % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        # Optimization kernel configs (Stage 3)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # QKV projection: hidden -> 3*hidden
        self.att_proj_weight = preprocess_linear_weight(parameters["att_proj"]["weight"], device)
        self.att_proj_bias = (
            preprocess_bias_weight(parameters["att_proj"]["bias"], device)
            if "bias" in parameters["att_proj"] and parameters["att_proj"]["bias"] is not None
            else None
        )

        # Output projection: hidden -> hidden
        self.out_proj_weight = preprocess_linear_weight(parameters["out_proj"]["weight"], device)
        self.out_proj_bias = (
            preprocess_bias_weight(parameters["out_proj"]["bias"], device)
            if "bias" in parameters["out_proj"] and parameters["out_proj"]["bias"] is not None
            else None
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        layer_past: Optional[tuple] = None,
        use_cache: bool = False,
        memory_config: Optional[ttnn.MemoryConfig] = ttnn.L1_MEMORY_CONFIG,
    ) -> tuple:
        """Forward pass through attention. Fully on TTNN device."""
        # Ensure TILE layout before QKV projection
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # QKV projection in TTNN
        qkv = ttnn.linear(
            hidden_states,
            self.att_proj_weight,
            bias=self.att_proj_bias,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # 4D weights produce 4D output [B,1,S,3H] — squeeze to 3D [B,S,3H]
        if len(qkv.shape) == 4:
            qkv = ttnn.squeeze(qkv, dim=1)

        # On-device Q/K/V split — eliminates 2 PCIe roundtrips per layer.
        # qkv shape: [B, S, 3*H]
        B = qkv.shape[0]
        S = qkv.shape[1]
        H = self.embed_dim

        # Slice Q, K, V along last dim (all on device, no host transfer)
        query = qkv[:, :, :H]
        key = qkv[:, :, H : 2 * H]
        value = qkv[:, :, 2 * H :]
        ttnn.deallocate(qkv)

        # Reshape to multi-head: [B, S, H] → [B, S, num_heads, head_dim]
        query = ttnn.reshape(query, [B, S, self.num_heads, self.head_dim])
        key = ttnn.reshape(key, [B, S, self.num_heads, self.head_dim])
        value = ttnn.reshape(value, [B, S, self.num_heads, self.head_dim])

        # Transpose to [B, num_heads, S, head_dim] for SDPA
        query = ttnn.transpose(query, 1, 2)
        key = ttnn.transpose(key, 1, 2)
        value = ttnn.transpose(value, 1, 2)

        if layer_past is not None:
            kv_cache_k, kv_cache_v, cache_len = layer_past
            # Write-in-place: update pre-allocated cache at current position
            ttnn.kv_cache.update_cache_for_token_(kv_cache_k, key, cache_len)
            ttnn.kv_cache.update_cache_for_token_(kv_cache_v, value, cache_len)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            # Slice valid region from pre-allocated cache
            valid_len = nearest_32(cache_len + 1)
            key = kv_cache_k[:, :, :valid_len, : self.head_dim]
            value = kv_cache_v[:, :, :valid_len, : self.head_dim]

        if use_cache:
            if layer_past is not None:
                kv_cache_k, kv_cache_v, cache_len = layer_past
                layer_present = (kv_cache_k, kv_cache_v, cache_len + 1)
            else:
                # First call (prefill): initialize pre-allocated KV cache
                B = query.shape[0]
                if self._kv_cache_k is None:
                    torch_k_cache = torch.zeros(B, self.num_heads, self.max_seq_len, self.head_dim)
                    torch_v_cache = torch.zeros(B, self.num_heads, self.max_seq_len, self.head_dim)
                    self._kv_cache_k = ttnn.from_torch(
                        torch_k_cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    self._kv_cache_v = ttnn.from_torch(
                        torch_v_cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                # Fill cache with prefill KV
                ttnn.kv_cache.fill_cache_for_user_(self._kv_cache_k, key, 0)
                ttnn.kv_cache.fill_cache_for_user_(self._kv_cache_v, value, 0)
                prefill_len = key.shape[-2]
                layer_present = (self._kv_cache_k, self._kv_cache_v, prefill_len)
        else:
            layer_present = None

        # Fully on-device SDPA with mode selection:
        # - Prefill mode (seq_q>=32): use chunked SDPA on device
        # - Decode/small seq: PyTorch SDPA (TTNN SDPA requires chunk_size >= 32)
        q_seq_len = query.shape[-2]

        if q_seq_len >= 32:
            # Prefill mode: process full sequence with chunked SDPA
            chunk_size = min(128, q_seq_len)
            while q_seq_len % chunk_size != 0 and chunk_size > 32:
                chunk_size //= 2

            grid_size = self.config.grid_size if self.config.grid_size else self.device.compute_with_storage_grid_size()
            sdpa_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid_size.x, grid_size.y),
                q_chunk_size=chunk_size,
                k_chunk_size=chunk_size,
            )
            is_causal_mode = self.is_causal and layer_past is None and query.shape[-2] == key.shape[-2]

            attn_output = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=None,
                is_causal=is_causal_mode,
                memory_config=memory_config,
                program_config=sdpa_config,
                compute_kernel_config=self.compute_kernel_config,
            )
        else:
            # Decode/short-seq mode (seq < 32): matmul-based attention on-device.
            # TTNN SDPA requires chunk_size >= 32, so we use explicit matmul.
            # Q: [B, heads, q_seq, head_dim]
            # K: [B, heads, kv_seq, head_dim] (may be large from KV cache)
            # V: [B, heads, kv_seq, head_dim]
            key_t = ttnn.transpose(key, -2, -1)
            attn_scores = ttnn.matmul(query, key_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(key_t)  # Free transposed key immediately to reduce L1 pressure
            attn_scores = ttnn.multiply(attn_scores, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Causal masking for short prefills (layer_past is None, q == kv).
            # During decode with KV cache (q_seq=1), masking is not needed
            # because all KV positions are valid past positions.
            if self.is_causal and layer_past is None and q_seq_len > 1:
                kv_len = key.shape[-2]
                causal_mask = torch.tril(torch.ones(q_seq_len, kv_len, dtype=torch.float32))
                causal_mask = (1.0 - causal_mask) * -1e9
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q, kv]
                tt_mask = ttnn.from_torch(
                    causal_mask,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                attn_scores = ttnn.add(attn_scores, tt_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(tt_mask)

            attn_probs = ttnn.softmax(attn_scores, dim=-1)
            ttnn.deallocate(attn_scores)

            attn_output = ttnn.matmul(attn_probs, value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_probs)
        ttnn.deallocate(query)
        if not use_cache:
            ttnn.deallocate(key)
            ttnn.deallocate(value)

        # Merge heads
        temp_attn = attn_output
        attn_output = ttnn.transformer.concatenate_heads(temp_attn, memory_config=memory_config)
        ttnn.deallocate(temp_attn)

        # Output projection
        output = ttnn.linear(
            attn_output,
            self.out_proj_weight,
            bias=self.out_proj_bias,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_output)

        return output, layer_present


class TtBarkBlock:
    """Bark transformer block (pre-norm architecture).

    Flow:
        x -> LayerNorm1 -> Attention -> + x (residual)
          -> LayerNorm2 -> MLP       -> + (residual)
    """

    def __init__(self, device, parameters, config: BarkConfig, is_causal: bool = True):
        self.device = device
        self.config = config

        # Layer norms
        self.ln1_weight = preprocess_layernorm_weight(parameters["layernorm_1"]["weight"], device)
        self.ln1_bias = (
            preprocess_layernorm_weight(parameters["layernorm_1"]["bias"], device)
            if "bias" in parameters["layernorm_1"] and parameters["layernorm_1"]["bias"] is not None
            else None
        )

        self.ln2_weight = preprocess_layernorm_weight(parameters["layernorm_2"]["weight"], device)
        self.ln2_bias = (
            preprocess_layernorm_weight(parameters["layernorm_2"]["bias"], device)
            if "bias" in parameters["layernorm_2"] and parameters["layernorm_2"]["bias"] is not None
            else None
        )

        # Attention and MLP sub-modules
        self.attn = TtBarkAttention(device, parameters["attn"], config, is_causal=is_causal)
        self.mlp = TtBarkMLP(device, parameters["mlp"], config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        layer_past: Optional[tuple] = None,
        use_cache: bool = False,
        memory_config: Optional[ttnn.MemoryConfig] = ttnn.L1_MEMORY_CONFIG,
    ) -> tuple:
        """Forward pass through one transformer block."""
        # Fused MLP structure: Norm -> FC1 -> Activation -> FC2 -> Residual
        prev_hidden = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=self.config.layer_norm_epsilon,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            memory_config=memory_config,
        )
        # Self-Attention
        attn_output, layer_present = self.attn(
            hidden_states, layer_past=layer_past, use_cache=use_cache, memory_config=memory_config
        )
        ttnn.deallocate(hidden_states)

        # Residual 1
        hidden_states = ttnn.add(prev_hidden, attn_output, memory_config=memory_config)
        ttnn.deallocate(prev_hidden)
        ttnn.deallocate(attn_output)

        prev_hidden = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=self.config.layer_norm_epsilon,
            weight=self.ln2_weight,
            bias=self.ln2_bias,
            memory_config=memory_config,
        )
        # MLP
        mlp_output = self.mlp(hidden_states, memory_config=memory_config)
        ttnn.deallocate(hidden_states)

        # Residual 2
        hidden_states = ttnn.add(prev_hidden, mlp_output, memory_config=memory_config)
        ttnn.deallocate(prev_hidden)
        ttnn.deallocate(mlp_output)

        return hidden_states, layer_present


class TtBarkGPT:
    """Full Bark GPT model (used for semantic and coarse stages).

    Architecture:
        input_ids -> Embedding + PositionEmbedding -> N x BarkBlock -> LayerNorm -> lm_head

    This is the shared backbone for:
    - Text-to-Semantic (input_vocab=10048, output_vocab=10048, causal)
    - Semantic-to-Coarse (input_vocab=10048, output_vocab=10048, causal)
    """

    def __init__(self, device, parameters, config: BarkConfig, is_causal: bool = True):
        self.device = device
        self.config = config
        self.is_causal = is_causal

        # Embedding layers - dual strategy for maximum throughput:
        #
        # PREFILL: CPU nn.Embedding (batched, amortized cost, avoids NCRISC bug)
        # DECODE:  Device-side embedding table (eliminates per-token CPU→device transfer)
        #
        # The NCRISC kernel compilation bug affects ttnn.embedding() in some
        # tt-metal versions. We use CPU embedding as fallback when ttnn.embedding fails.
        # TODO(Ashutosh0x): Remove CPU fallback once NCRISC bug is resolved.
        #   Tracking: https://github.com/tenstorrent/tt-metal/issues/32069

        # CPU embeddings for prefill path
        self.input_embeds = torch.nn.Embedding.from_pretrained(
            parameters["input_embeds_layer"]["weight"].detach().float()
        )
        self.position_embeds = torch.nn.Embedding.from_pretrained(
            parameters["position_embeds_layer"]["weight"].detach().float()
        )

        # Device-side embedding tables for decode path (pre-transferred to DRAM)
        # This eliminates per-token CPU→device transfer (~0.1-1ms per token)
        self._tt_input_embeds_table = ttnn.from_torch(
            parameters["input_embeds_layer"]["weight"].detach().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._tt_position_embeds_table = ttnn.from_torch(
            parameters["position_embeds_layer"]["weight"].detach().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._use_device_embeds = True  # Set to False if ttnn.embedding fails at runtime

        # Transformer blocks
        self.blocks = []
        for i in range(config.num_layers):
            block = TtBarkBlock(device, parameters["layers"][str(i)], config, is_causal=is_causal)
            self.blocks.append(block)

        # Final layer norm
        self.ln_f_weight = preprocess_layernorm_weight(parameters["layernorm_final"]["weight"], device)
        self.ln_f_bias = (
            preprocess_layernorm_weight(parameters["layernorm_final"]["bias"], device)
            if "bias" in parameters["layernorm_final"] and parameters["layernorm_final"]["bias"] is not None
            else None
        )

        # LM head
        self.lm_head_weight = preprocess_linear_weight(parameters["lm_head"]["weight"], device)
        if self.lm_head_weight.shape[3] != config.output_vocab_size:
            raise ValueError(
                f"LM head weight shape mismatch: expected {config.output_vocab_size}, got {self.lm_head_weight.shape[3]}"
            )

        # Optimization config (Stage 3)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def reset_kv_cache(self):
        """Reset KV cache state for a new generation.

        The pre-allocated cache tensors are reused (no reallocation needed);
        only the cache_len tracking in layer_past tuples is reset by the
        caller passing layer_past=None on the next prefill call.

        Call this to reset the lazy-initialized cache tensors if batch size
        changes between generations.
        """
        for block in self.blocks:
            block.attn._kv_cache_k = None
            block.attn._kv_cache_v = None

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        layer_past: Optional[list] = None,
        use_cache: bool = False,
        memory_config: Optional[ttnn.MemoryConfig] = ttnn.L1_MEMORY_CONFIG,
    ) -> tuple:
        """Forward pass through the full GPT model.

        Args:
            input_ids: [batch, seq_len] token indices
            inputs_embeds: [batch, seq_len, hidden] pre-computed embeddings
            layer_past: List of (k_cache, v_cache, cache_len) tuples per layer
            use_cache: Whether to return the new KV cache
            memory_config: Memory configuration for activations

        Returns:
            logits: [batch, seq_len, vocab_size]
            layer_present: List of updated (k_cache, v_cache, cache_len) tuples
        """
        if inputs_embeds is None and input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                tok_ids = input_ids
            else:
                tok_ids = ttnn.to_torch(input_ids).to(torch.long)

            # Safety clamp: prevent index-out-of-range from bfloat16→uint32 rounding
            vocab_size = self.input_embeds.weight.shape[0]
            tok_ids = tok_ids.long().clamp(0, vocab_size - 1)
            seq_len = tok_ids.shape[-1]

            if layer_past is not None:
                past_len = layer_past[0][2]  # cache_len from first layer
                position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long)
            else:
                position_ids = torch.arange(0, seq_len, dtype=torch.long)

            # Clamp position to block_size to prevent overflow on long sequences
            max_pos = self.position_embeds.weight.shape[0] - 1
            position_ids = position_ids.clamp(0, max_pos)

            # Decode path (seq_len=1, has KV cache): on-device embedding lookup
            # Eliminates per-token CPU→device transfer (~0.1-1ms savings per token)
            if layer_past is not None and seq_len == 1 and self._use_device_embeds:
                try:
                    # ttnn.embedding: input [B, S] uint32, weight [V, H] bfloat16
                    tt_tok_ids = ttnn.from_torch(
                        tok_ids.to(torch.int32),
                        dtype=ttnn.uint32,
                        device=self.device,
                    )
                    tok_emb = ttnn.embedding(
                        tt_tok_ids,
                        self._tt_input_embeds_table,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(tt_tok_ids)

                    tt_pos_ids = ttnn.from_torch(
                        position_ids.unsqueeze(0).to(torch.int32),
                        dtype=ttnn.uint32,
                        device=self.device,
                    )
                    pos_emb = ttnn.embedding(
                        tt_pos_ids,
                        self._tt_position_embeds_table,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(tt_pos_ids)

                    # Add token + position embeddings on device
                    inputs_embeds = ttnn.add(tok_emb, pos_emb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(tok_emb)
                    ttnn.deallocate(pos_emb)
                    # Convert to TILE layout for transformer blocks
                    inputs_embeds = ttnn.to_layout(inputs_embeds, ttnn.TILE_LAYOUT)
                except Exception:
                    # NCRISC compilation failure — fall back to CPU path permanently
                    self._use_device_embeds = False
                    tok_emb = self.input_embeds(tok_ids)
                    pos_emb = self.position_embeds(position_ids)
                    hidden = (tok_emb + pos_emb).float()
                    inputs_embeds = ttnn.from_torch(
                        hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                    )
            else:
                # Prefill path: CPU embedding (batched, amortized cost)
                tok_emb = self.input_embeds(tok_ids)
                pos_emb = self.position_embeds(position_ids)
                hidden = (tok_emb + pos_emb).float()
                inputs_embeds = ttnn.from_torch(
                    hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
        elif inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        else:
            # inputs_embeds provided directly — determine seq_len
            if isinstance(inputs_embeds, torch.Tensor):
                seq_len = inputs_embeds.shape[1]
            else:
                seq_len = inputs_embeds.shape[-2]

        # Transformer blocks
        tt_hidden = inputs_embeds
        layer_present = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            block_past = layer_past[i] if layer_past is not None else None
            tt_hidden, present = block(
                tt_hidden, layer_past=block_past, use_cache=use_cache, memory_config=memory_config
            )
            if use_cache:
                layer_present.append(present)

        # Final layer norm
        prev_hidden = tt_hidden
        tt_hidden = ttnn.layer_norm(
            tt_hidden,
            epsilon=self.config.layer_norm_epsilon,
            weight=self.ln_f_weight,
            bias=self.ln_f_bias,
            memory_config=memory_config,
        )
        ttnn.deallocate(prev_hidden)

        # LM head
        logits = ttnn.linear(
            tt_hidden,
            self.lm_head_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(tt_hidden)

        # 4D weights ensure logits are [B, 1, seq, vocab] — no reshape needed

        return logits, layer_present


def preprocess_model_parameters(model, device, is_causal=True):
    """Extract and organize parameters from a HuggingFace BarkCausalModel.

    Args:
        model: HuggingFace BarkCausalModel (semantic or coarse)
        device: TTNN device
        is_causal: Whether this is a causal model

    Returns:
        dict: Organized parameter dictionary for TtBarkGPT
    """
    state_dict = model.state_dict()
    parameters = {}

    # Embedding layers (kept as torch tensors for CPU indexing)
    parameters["input_embeds_layer"] = {"weight": state_dict["input_embeds_layer.weight"].clone()}
    parameters["position_embeds_layer"] = {"weight": state_dict["position_embeds_layer.weight"].clone()}

    # Transformer layers
    parameters["layers"] = {}
    num_layers = model.config.num_layers
    for i in range(num_layers):
        prefix = f"layers.{i}"
        layer_params = {}

        # LayerNorm 1
        layer_params["layernorm_1"] = {
            "weight": state_dict[f"{prefix}.layernorm_1.weight"].clone(),
        }
        if f"{prefix}.layernorm_1.bias" in state_dict:
            layer_params["layernorm_1"]["bias"] = state_dict[f"{prefix}.layernorm_1.bias"].clone()

        # LayerNorm 2
        layer_params["layernorm_2"] = {
            "weight": state_dict[f"{prefix}.layernorm_2.weight"].clone(),
        }
        if f"{prefix}.layernorm_2.bias" in state_dict:
            layer_params["layernorm_2"]["bias"] = state_dict[f"{prefix}.layernorm_2.bias"].clone()

        # Attention
        layer_params["attn"] = {
            "att_proj": {"weight": state_dict[f"{prefix}.attn.att_proj.weight"].clone()},
            "out_proj": {"weight": state_dict[f"{prefix}.attn.out_proj.weight"].clone()},
        }
        if f"{prefix}.attn.att_proj.bias" in state_dict:
            layer_params["attn"]["att_proj"]["bias"] = state_dict[f"{prefix}.attn.att_proj.bias"].clone()
        if f"{prefix}.attn.out_proj.bias" in state_dict:
            layer_params["attn"]["out_proj"]["bias"] = state_dict[f"{prefix}.attn.out_proj.bias"].clone()

        # MLP
        layer_params["mlp"] = {
            "in_proj": {"weight": state_dict[f"{prefix}.mlp.in_proj.weight"].clone()},
            "out_proj": {"weight": state_dict[f"{prefix}.mlp.out_proj.weight"].clone()},
        }
        if f"{prefix}.mlp.in_proj.bias" in state_dict:
            layer_params["mlp"]["in_proj"]["bias"] = state_dict[f"{prefix}.mlp.in_proj.bias"].clone()
        if f"{prefix}.mlp.out_proj.bias" in state_dict:
            layer_params["mlp"]["out_proj"]["bias"] = state_dict[f"{prefix}.mlp.out_proj.bias"].clone()

        parameters["layers"][str(i)] = layer_params

    # Final layer norm
    parameters["layernorm_final"] = {
        "weight": state_dict["layernorm_final.weight"].clone(),
    }
    if "layernorm_final.bias" in state_dict:
        parameters["layernorm_final"]["bias"] = state_dict["layernorm_final.bias"].clone()

    # LM head
    parameters["lm_head"] = {"weight": state_dict["lm_head.weight"].clone()}

    return parameters
