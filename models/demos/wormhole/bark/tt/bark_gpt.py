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
from typing import Optional

import torch
import ttnn


@dataclass
class BarkConfig:
    """Configuration for Bark Small GPT model.

    All three stages share identical transformer architecture (80M params each).
    Differences: vocab sizes, causal vs non-causal attention, bias settings.
    """

    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    block_size: int = 1024
    dropout: float = 0.0  # No dropout at inference
    bias: bool = False  # Causal stages use no bias; fine stage uses bias for LN
    input_vocab_size: int = 10_048  # Varies per stage
    output_vocab_size: int = 10_048  # Varies per stage


def preprocess_linear_weight(weight_tensor, device):
    """Convert a PyTorch weight tensor to a TTNN tensor for linear ops."""
    weight = weight_tensor.detach().float()
    if weight.dim() == 2:
        weight = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, out, in]
    tt_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_weight


def preprocess_layernorm_weight(weight_tensor, device):
    """Convert LayerNorm weight/bias to TTNN tensor."""
    w = weight_tensor.detach().float()
    if w.dim() == 1:
        w = w.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, hidden]
    tt_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_w


class TtBarkMLP:
    """Bark MLP block: Linear(hidden, 4*hidden) -> GELU -> Linear(4*hidden, hidden).

    Follows HuggingFace BarkMLP exactly.
    """

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

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through MLP block."""
        # Linear projection: hidden -> 4*hidden
        hidden_states = ttnn.linear(hidden_states, self.in_proj_weight, bias=self.in_proj_bias)

        # GELU activation
        hidden_states = ttnn.gelu(hidden_states)

        # Linear projection: 4*hidden -> hidden
        hidden_states = ttnn.linear(hidden_states, self.out_proj_weight, bias=self.out_proj_bias)

        return hidden_states


class TtBarkAttention:
    """Multi-head self-attention for Bark.

    Supports both causal (stages 1-2) and non-causal (stage 3) attention.
    Uses a hybrid approach: TTNN for projections, PyTorch for SDPA.

    Architecture:
        Q, K, V = att_proj(hidden_states).split(3)
        attn = softmax(Q @ K^T / sqrt(d)) @ V  (with optional causal mask)
        output = out_proj(attn)
    """

    def __init__(self, device, parameters, config: BarkConfig, is_causal: bool = True):
        self.device = device
        self.config = config
        self.is_causal = is_causal
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.embed_dim = config.hidden_size

        assert self.embed_dim % self.num_heads == 0, "hidden_size must be divisible by num_heads"

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

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through attention.

        Uses PyTorch SDPA for the attention computation (hybrid approach),
        while TTNN handles the linear projections.
        """
        # QKV projection in TTNN
        qkv = ttnn.linear(hidden_states, self.att_proj_weight, bias=self.att_proj_bias)

        # Move to PyTorch for SDPA (reliable attention with causal mask)
        qkv_torch = ttnn.to_torch(qkv)
        ttnn.deallocate(qkv)

        # Shape: [batch, seq_len, 3*hidden] -> split into Q, K, V
        batch_size = qkv_torch.shape[0]
        seq_len = qkv_torch.shape[-2]

        if qkv_torch.dim() == 4:
            # [1, batch, seq, hidden*3]
            qkv_torch = qkv_torch.squeeze(0)

        q, k, v = qkv_torch.split(self.embed_dim, dim=-1)

        # Reshape: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (with optional causal mask)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            is_causal=self.is_causal,
        )

        # Merge heads: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Back to TTNN for output projection
        if attn_output.dim() == 3:
            attn_output = attn_output.unsqueeze(0)  # [1, batch, seq, hidden]

        tt_attn_output = ttnn.from_torch(attn_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        output = ttnn.linear(tt_attn_output, self.out_proj_weight, bias=self.out_proj_bias)
        ttnn.deallocate(tt_attn_output)

        return output


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

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through one transformer block."""
        # Pre-norm + attention + residual
        residual = hidden_states
        normed = ttnn.layer_norm(hidden_states, epsilon=1e-5, weight=self.ln1_weight, bias=self.ln1_bias)
        attn_output = self.attn(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(residual, attn_output)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_output)

        # Pre-norm + MLP + residual
        residual = hidden_states
        normed = ttnn.layer_norm(hidden_states, epsilon=1e-5, weight=self.ln2_weight, bias=self.ln2_bias)
        mlp_output = self.mlp(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(residual, mlp_output)
        ttnn.deallocate(residual)
        ttnn.deallocate(mlp_output)

        return hidden_states


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

        # Embedding layers (kept on CPU for token indexing)
        self.input_embeds_layer = torch.nn.Embedding(config.input_vocab_size, config.hidden_size)
        self.input_embeds_layer.weight = torch.nn.Parameter(parameters["input_embeds_layer"]["weight"])

        self.position_embeds_layer = torch.nn.Embedding(config.block_size, config.hidden_size)
        self.position_embeds_layer.weight = torch.nn.Parameter(parameters["position_embeds_layer"]["weight"])

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

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Forward pass through the full GPT model.

        Args:
            input_ids: [batch, seq_len] token indices
            inputs_embeds: [batch, seq_len, hidden] pre-computed embeddings (used by semantic stage)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.input_embeds_layer(input_ids)
        elif inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]

        # Position embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long)
        position_embeds = self.position_embeds_layer(position_ids)  # [seq_len, hidden]

        # Combine embeddings
        hidden = inputs_embeds + position_embeds.unsqueeze(0)

        # Convert to TTNN
        if hidden.dim() == 3:
            hidden = hidden.unsqueeze(0)  # [1, batch, seq, hidden]

        tt_hidden = ttnn.from_torch(hidden.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Transformer blocks
        for block in self.blocks:
            tt_hidden = block(tt_hidden)

        # Final layer norm
        tt_hidden = ttnn.layer_norm(tt_hidden, epsilon=1e-5, weight=self.ln_f_weight, bias=self.ln_f_bias)

        # LM head
        logits = ttnn.linear(tt_hidden, self.lm_head_weight)
        ttnn.deallocate(tt_hidden)

        return logits


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
