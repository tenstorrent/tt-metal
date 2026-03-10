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
            preprocess_linear_weight(parameters["in_proj"]["bias"].unsqueeze(0), device)
            if config.bias and "bias" in parameters["in_proj"]
            else None
        )

        self.out_proj_weight = preprocess_linear_weight(parameters["out_proj"]["weight"], device)
        self.out_proj_bias = (
            preprocess_linear_weight(parameters["out_proj"]["bias"].unsqueeze(0), device)
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
    Now fully on-device with optional KV caching support.

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

        assert self.embed_dim % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        # Optimization kernel configs (Stage 3)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # SDPA Program Config for core grid utilization
        grid_size = config.grid_size if config.grid_size else self.device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid_size.x, grid_size.y),
            q_chunk_size=128,
            k_chunk_size=128,
        )

        # QKV projection: hidden -> 3*hidden
        self.att_proj_weight = preprocess_linear_weight(parameters["att_proj"]["weight"], device)
        self.att_proj_bias = (
            preprocess_linear_weight(parameters["att_proj"]["bias"].unsqueeze(0), device)
            if "bias" in parameters["att_proj"] and parameters["att_proj"]["bias"] is not None
            else None
        )

        # Output projection: hidden -> hidden
        self.out_proj_weight = preprocess_linear_weight(parameters["out_proj"]["weight"], device)
        self.out_proj_bias = (
            preprocess_linear_weight(parameters["out_proj"]["bias"].unsqueeze(0), device)
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
            past_key, past_value = layer_past
            # Update KV cache
            new_key = ttnn.concat([past_key, key], dim=-2, memory_config=memory_config)
            new_value = ttnn.concat([past_value, value], dim=-2, memory_config=memory_config)
            ttnn.deallocate(key)
            ttnn.deallocate(value)
            ttnn.deallocate(past_key)
            ttnn.deallocate(past_value)
            key, value = new_key, new_value

        layer_present = (key, value) if use_cache else None

        # Fully on-device SDPA with mode selection:
        # - Prefill mode (seq_q>=32): use chunked SDPA on device
        # - Decode/small seq: PyTorch SDPA (TTNN SDPA requires chunk_size >= 32)
        q_seq_len = query.shape[-2]
        k_seq_len = key.shape[-2]

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
            # Edge case fallback for small sequences (1 < seq < 32, no KV cache)
            q_torch = ttnn.to_torch(query)
            k_torch = ttnn.to_torch(key)
            v_torch = ttnn.to_torch(value)
            attn_output_torch = torch.nn.functional.scaled_dot_product_attention(
                q_torch,
                k_torch,
                v_torch,
                scale=None,
                is_causal=self.is_causal and layer_past is None,
            )
            attn_output = ttnn.from_torch(
                attn_output_torch, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT
            )
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

        # Embedding layers — use CPU nn.Embedding to avoid NCRISC kernel
        # compilation bug in ttnn 0.67.0rc5 (embedding NCRISC template fails).
        # CPU→device transfer for embeddings is ~0.1ms, negligible vs transformer.
        self.input_embeds = torch.nn.Embedding.from_pretrained(
            parameters["input_embeds_layer"]["weight"].detach().float()
        )
        self.position_embeds = torch.nn.Embedding.from_pretrained(
            parameters["position_embeds_layer"]["weight"].detach().float()
        )

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
            layer_past: List of (past_key, past_value) tuples for each layer
            use_cache: Whether to return the new KV cache
            memory_config: Memory configuration for activations

        Returns:
            logits: [batch, seq_len, vocab_size]
            layer_present: List of updated (key, value) tuples
        """
        if inputs_embeds is None and input_ids is not None:
            # CPU embedding lookup (avoids NCRISC kernel bug in ttnn 0.67.0rc5)
            if isinstance(input_ids, torch.Tensor):
                tok_ids = input_ids
            else:
                tok_ids = ttnn.to_torch(input_ids).to(torch.long)

            tok_emb = self.input_embeds(tok_ids.long())
            seq_len = tok_ids.shape[-1]

            if layer_past is not None:
                past_len = layer_past[0][0].shape[-2]
                position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long)
            else:
                position_ids = torch.arange(0, seq_len, dtype=torch.long)

            pos_emb = self.position_embeds(position_ids)
            hidden = (tok_emb + pos_emb).float()

            # Transfer combined embeddings to device
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
            tt_hidden, epsilon=1e-5, weight=self.ln_f_weight, bias=self.ln_f_bias, memory_config=memory_config
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
