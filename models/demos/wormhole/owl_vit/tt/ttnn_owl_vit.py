# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict

import torch

import ttnn


@dataclass
class OwlViTTTNNConfig:
    """Configuration for TTNN OWL-ViT model."""

    # Vision config (ViT-B/32)
    vision_hidden_size: int = 768
    vision_num_heads: int = 12
    vision_head_dim: int = 64  # 768 // 12
    vision_layers: int = 12
    patch_size: int = 32
    image_size: int = 768

    # Text config
    text_hidden_size: int = 512
    text_num_heads: int = 8
    text_head_dim: int = 64  # 512 // 8
    text_layers: int = 12
    vocab_size: int = 49408

    # General
    layer_norm_eps: float = 1e-5

    # Stage 1 Optimization parameters
    use_lofi: bool = True  # Use LoFi math fidelity for faster matmuls
    use_l1_memory: bool = False  # Use L1 for activations (faster but limited size)
    weights_dtype: str = "bfloat16"  # "bfloat16" or "bfloat8_b" for weights

    @classmethod
    def from_huggingface(cls, hf_config, use_lofi: bool = True, use_l1_memory: bool = False):
        return cls(
            vision_hidden_size=hf_config.vision_config.hidden_size,
            vision_num_heads=hf_config.vision_config.num_attention_heads,
            vision_head_dim=hf_config.vision_config.hidden_size // hf_config.vision_config.num_attention_heads,
            vision_layers=hf_config.vision_config.num_hidden_layers,
            patch_size=hf_config.vision_config.patch_size,
            image_size=hf_config.vision_config.image_size,
            text_hidden_size=hf_config.text_config.hidden_size,
            text_num_heads=hf_config.text_config.num_attention_heads,
            text_head_dim=hf_config.text_config.hidden_size // hf_config.text_config.num_attention_heads,
            text_layers=hf_config.text_config.num_hidden_layers,
            vocab_size=hf_config.text_config.vocab_size,
            layer_norm_eps=hf_config.vision_config.layer_norm_eps,
            use_lofi=use_lofi,
            use_l1_memory=use_l1_memory,
        )

    def get_compute_kernel_config(self):
        """Get compute kernel config with appropriate math fidelity."""
        if self.use_lofi:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
            )
        else:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            )

    def get_memory_config(self):
        """Get memory config based on optimization settings."""
        if self.use_l1_memory:
            return ttnn.L1_MEMORY_CONFIG
        else:
            return ttnn.DRAM_MEMORY_CONFIG


def run_vision_encoder_layer(
    hidden_states: ttnn.Tensor,
    layer_params: Dict[str, Any],
    config: OwlViTTTNNConfig,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config: ttnn.WormholeComputeKernelConfig = None,
) -> ttnn.Tensor:
    """Run a single vision encoder layer with optional compute kernel config for optimization."""
    # Layer norm 1
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm1"]["weight"],
        bias=layer_params["layer_norm1"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # Self-attention with fused QKV
    qkv = ttnn.linear(
        hidden_states,
        layer_params["self_attn"]["qkv"]["weight"],
        bias=layer_params["self_attn"]["qkv"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(hidden_states)

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=memory_config,
        num_heads=config.vision_num_heads,
    )
    ttnn.deallocate(qkv)

    attention_scores = ttnn.matmul(query, key, memory_config=memory_config, compute_kernel_config=compute_kernel_config)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_scores = ttnn.mul(attention_scores, 1.0 / (config.vision_head_dim**0.5))
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    ttnn.deallocate(attention_scores)

    context = ttnn.matmul(
        attention_probs, value, memory_config=memory_config, compute_kernel_config=compute_kernel_config
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context = ttnn.transformer.concatenate_heads(context, memory_config=memory_config)

    attn_output = ttnn.linear(
        context,
        layer_params["self_attn"]["out_proj"]["weight"],
        bias=layer_params["self_attn"]["out_proj"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(context)

    hidden_states = ttnn.add(residual, attn_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(attn_output)

    # Layer norm 2
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm2"]["weight"],
        bias=layer_params["layer_norm2"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # MLP
    mlp_hidden = ttnn.linear(
        hidden_states,
        layer_params["mlp"]["fc1"]["weight"],
        bias=layer_params["mlp"]["fc1"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(hidden_states)
    mlp_hidden = ttnn.gelu(mlp_hidden)

    mlp_output = ttnn.linear(
        mlp_hidden,
        layer_params["mlp"]["fc2"]["weight"],
        bias=layer_params["mlp"]["fc2"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(mlp_hidden)

    hidden_states = ttnn.add(residual, mlp_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(mlp_output)

    return hidden_states


def run_text_encoder_layer(
    hidden_states: ttnn.Tensor,
    layer_params: Dict[str, Any],
    causal_mask: ttnn.Tensor,
    config: OwlViTTTNNConfig,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config: ttnn.WormholeComputeKernelConfig = None,
) -> ttnn.Tensor:
    """Run a single text encoder layer with causal attention and optional compute kernel config."""
    # Note: Text encoder uses Pre-Layernorm
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm1"]["weight"],
        bias=layer_params["layer_norm1"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # Self-attention
    # 1. Project Q, K, V (fused in verified implementation)
    qkv = ttnn.linear(
        hidden_states,
        layer_params["self_attn"]["qkv"]["weight"],
        bias=layer_params["self_attn"]["qkv"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(hidden_states)

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=memory_config,
        num_heads=config.text_num_heads,
    )
    ttnn.deallocate(qkv)

    # 2. Compute scores
    attention_scores = ttnn.matmul(query, key, memory_config=memory_config, compute_kernel_config=compute_kernel_config)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # 3. Scale
    attention_scores = ttnn.mul(attention_scores, 1.0 / (config.text_head_dim**0.5))

    # 4. Apply Causal Mask
    if causal_mask is not None:
        attention_scores = ttnn.add(attention_scores, causal_mask)

    # 5. Softmax
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    ttnn.deallocate(attention_scores)

    # 6. Context
    context = ttnn.matmul(
        attention_probs, value, memory_config=memory_config, compute_kernel_config=compute_kernel_config
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    # 7. Concatenate heads
    context = ttnn.transformer.concatenate_heads(context, memory_config=memory_config)

    # 8. Output projection
    attn_output = ttnn.linear(
        context,
        layer_params["self_attn"]["out_proj"]["weight"],
        bias=layer_params["self_attn"]["out_proj"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(context)

    # Residual 1
    hidden_states = ttnn.add(residual, attn_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(attn_output)

    # Layer norm 2
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm2"]["weight"],
        bias=layer_params["layer_norm2"]["bias"],
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
    )

    # MLP
    mlp_hidden = ttnn.linear(
        hidden_states,
        layer_params["mlp"]["fc1"]["weight"],
        bias=layer_params["mlp"]["fc1"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(hidden_states)
    # QuickGELU
    mlp_hidden = ttnn.gelu(mlp_hidden)

    mlp_output = ttnn.linear(
        mlp_hidden,
        layer_params["mlp"]["fc2"]["weight"],
        bias=layer_params["mlp"]["fc2"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(mlp_hidden)

    # Residual 2
    hidden_states = ttnn.add(residual, mlp_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(mlp_output)

    return hidden_states


def run_box_head(
    patch_features: ttnn.Tensor,
    parameters: Dict[str, Any],
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """
    Run box prediction head.

    Args:
        patch_features: [batch, num_patches, hidden_size]
        parameters: Box head weights including 'box_bias'
    """
    # MLP: hidden -> hidden -> hidden -> 4
    hidden = ttnn.linear(
        patch_features,
        parameters["box_head"]["dense0"]["weight"],
        bias=parameters["box_head"]["dense0"]["bias"],
        memory_config=memory_config,
    )
    hidden = ttnn.gelu(hidden)

    hidden = ttnn.linear(
        hidden,
        parameters["box_head"]["dense1"]["weight"],
        bias=parameters["box_head"]["dense1"]["bias"],
        memory_config=memory_config,
    )
    hidden = ttnn.gelu(hidden)

    pred_boxes = ttnn.linear(
        hidden,
        parameters["box_head"]["dense2"]["weight"],
        bias=parameters["box_head"]["dense2"]["bias"],
        memory_config=memory_config,
    )

    # Add box_bias before sigmoid
    pred_boxes = ttnn.add(pred_boxes, parameters["box_bias"])
    pred_boxes = ttnn.sigmoid(pred_boxes)

    return pred_boxes


def run_class_head(
    patch_features: ttnn.Tensor,
    text_embeds: torch.Tensor,
    parameters: Dict[str, Any],
    device: ttnn.Device,
    memory_config: ttnn.MemoryConfig,
    config: OwlViTTTNNConfig,
) -> ttnn.Tensor:
    """
    Run class prediction head.

    Args:
        patch_features: [batch, num_patches, hidden_size]
        text_embeds: [batch, num_queries, hidden_size] (torch tensor)
    """
    # Project patch features
    image_class_embeds = ttnn.linear(
        patch_features,
        parameters["class_head"]["dense0"]["weight"],
        bias=parameters["class_head"]["dense0"]["bias"],
        memory_config=memory_config,
    )

    # L2 normalize image embeddings
    # Using fallback to torch for correctness on normalization
    image_embeds_torch = ttnn.to_torch(image_class_embeds).float()
    image_norm = torch.nn.functional.normalize(image_embeds_torch, p=2, dim=-1, eps=1e-6)

    # L2 normalize text embeddings
    # text_embeds input is [batch, num_queries, dim] or [num_queries, dim]
    text_norm = torch.nn.functional.normalize(text_embeds.float(), p=2, dim=-1, eps=1e-6)
    if text_norm.dim() == 2:
        text_norm = text_norm.unsqueeze(0)  # [1, Q, D]

    # Compute similarity: image_embeds @ text_embeds^T
    # [B, N, D] @ [B, Q, D]^T -> [B, N, Q]
    pred_logits = torch.matmul(image_norm, text_norm.transpose(-2, -1))

    # Apply logit_shift and logit_scale
    patch_features_torch = ttnn.to_torch(patch_features).float()

    shift_weight = ttnn.to_torch(parameters["class_head"]["logit_shift"]["weight"]).float()
    shift_bias = ttnn.to_torch(parameters["class_head"]["logit_shift"]["bias"]).float()
    # shift_weight is [768, 1]
    logit_shift = torch.matmul(patch_features_torch, shift_weight.squeeze(0)) + shift_bias.squeeze()

    scale_weight = ttnn.to_torch(parameters["class_head"]["logit_scale"]["weight"]).float()
    scale_bias = ttnn.to_torch(parameters["class_head"]["logit_scale"]["bias"]).float()
    logit_scale = torch.matmul(patch_features_torch, scale_weight.squeeze(0)) + scale_bias.squeeze()
    logit_scale = torch.nn.functional.elu(logit_scale) + 1

    # Broadcast
    if logit_shift.dim() == 2:
        logit_shift = logit_shift.unsqueeze(-1)
    if logit_scale.dim() == 2:
        logit_scale = logit_scale.unsqueeze(-1)

    # Apply: (logits + shift) * scale
    pred_logits = (pred_logits + logit_shift) * logit_scale

    # Transfer back to device
    logits = ttnn.from_torch(
        pred_logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    return logits
