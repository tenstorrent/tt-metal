# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
OWL-ViT (Open-World Localization Vision Transformer) implementation using TTNN APIs.

This implements zero-shot text-conditioned object detection on Tenstorrent hardware.
OWL-ViT uses CLIP as its multi-modal backbone with:
- ViT-B/32 image encoder for visual features
- Masked self-attention Transformer for text features
- Box prediction head for bounding box regression
- Class prediction head for region-text similarity scoring

The implementation is optimized for Wormhole (N150/N300) hardware.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import OwlViTConfig
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn


@dataclass
class OwlViTTTNNConfig:
    """Configuration for TTNN OWL-ViT model."""

    # Vision config (ViT-B/32)
    vision_hidden_size: int = 768
    vision_intermediate_size: int = 3072
    vision_num_hidden_layers: int = 12
    vision_num_attention_heads: int = 12
    image_size: int = 768  # OWL-ViT uses 768x768 images
    patch_size: int = 32

    # Text config
    text_hidden_size: int = 512
    text_intermediate_size: int = 2048
    text_num_hidden_layers: int = 12
    text_num_attention_heads: int = 8
    vocab_size: int = 49408
    max_position_embeddings: int = 16

    # Projection dimension
    projection_dim: int = 512

    # Layer norm epsilon
    layer_norm_eps: float = 1e-5

    # Hardware-specific configs
    core_grid: Optional[ttnn.CoreGrid] = None

    @classmethod
    def from_huggingface(cls, hf_config: OwlViTConfig) -> "OwlViTTTNNConfig":
        """Create TTNN config from HuggingFace config."""
        return cls(
            vision_hidden_size=hf_config.vision_config.hidden_size,
            vision_intermediate_size=hf_config.vision_config.intermediate_size,
            vision_num_hidden_layers=hf_config.vision_config.num_hidden_layers,
            vision_num_attention_heads=hf_config.vision_config.num_attention_heads,
            image_size=hf_config.vision_config.image_size,
            patch_size=hf_config.vision_config.patch_size,
            text_hidden_size=hf_config.text_config.hidden_size,
            text_intermediate_size=hf_config.text_config.intermediate_size,
            text_num_hidden_layers=hf_config.text_config.num_hidden_layers,
            text_num_attention_heads=hf_config.text_config.num_attention_heads,
            vocab_size=hf_config.text_config.vocab_size,
            max_position_embeddings=hf_config.text_config.max_position_embeddings,
            projection_dim=hf_config.projection_dim,
        )


def update_model_config(config: OwlViTTTNNConfig, batch_size: int, device: ttnn.Device):
    """
    Update model configuration with hardware-specific sharding parameters.

    Args:
        config: OwlViTTTNNConfig
        batch_size: Batch size for inference
        device: TTNN device

    Returns:
        Updated config with program configs for sharding
    """
    TILE_HEIGHT = 32

    # Calculate patch count and sequence length for vision model
    # OWL-ViT: 768x768 image with 32x32 patches = 24x24 = 576 patches + 1 CLS token
    patch_count = config.image_size // config.patch_size  # 24 for 768/32
    num_patches = patch_count * patch_count  # 576
    vision_seq_len = num_patches + 1  # 577 with CLS token
    vision_seq_len_padded = ((vision_seq_len - 1) // TILE_HEIGHT + 1) * TILE_HEIGHT  # 608

    # Core grid configuration for Wormhole
    # Use 8x8 grid for maximum parallelism
    core_grid_8x8 = ttnn.CoreGrid(y=8, x=8)

    # Derive dimensions in tiles
    vision_dim_t = config.vision_hidden_size // TILE_HEIGHT  # 768/32 = 24
    vision_dim_t_per_x = vision_dim_t // core_grid_8x8.x  # 24/8 = 3
    vision_seq_t = vision_seq_len_padded // TILE_HEIGHT  # 608/32 = 19

    text_dim_t = config.text_hidden_size // TILE_HEIGHT  # 512/32 = 16
    text_dim_t_per_x = text_dim_t // core_grid_8x8.x  # 16/8 = 2

    # Vision encoder program configs
    vision_program_configs = {
        "layernorm_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            subblock_w=vision_dim_t_per_x,
            block_h=vision_seq_t,
            block_w=vision_dim_t_per_x,
            inplace=False,
        ),
        "qkv_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=vision_dim_t_per_x,
            out_subblock_h=1,
            out_subblock_w=3 * vision_dim_t_per_x,
            per_core_M=vision_seq_t,
            per_core_N=3 * vision_dim_t_per_x,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=vision_dim_t_per_x,
            out_subblock_h=1,
            out_subblock_w=vision_dim_t_per_x,
            per_core_M=vision_seq_t,
            per_core_N=vision_dim_t_per_x,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=vision_dim_t_per_x,
            out_subblock_h=1,
            out_subblock_w=(vision_dim_t_per_x * 4) // 2,
            per_core_M=vision_seq_t,
            per_core_N=vision_dim_t_per_x * 4,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=vision_dim_t_per_x * 4,
            out_subblock_h=1,
            out_subblock_w=vision_dim_t_per_x,
            per_core_M=vision_seq_t,
            per_core_N=vision_dim_t_per_x,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "ln_compute_config": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    }

    config.core_grid = core_grid_8x8
    config.vision_program_configs = vision_program_configs
    config.vision_seq_len = vision_seq_len
    config.vision_seq_len_padded = vision_seq_len_padded
    config.num_patches = num_patches
    config.patch_count = patch_count

    return config


# ============================================================================
# Vision Model Components
# ============================================================================


def owl_vit_vision_embeddings(
    config: OwlViTTTNNConfig,
    pixel_values: ttnn.Tensor,
    cls_token: ttnn.Tensor,
    position_embeddings: ttnn.Tensor,
    *,
    parameters,
) -> ttnn.Tensor:
    """
    Compute vision embeddings from pixel values.

    This includes:
    1. Patch embedding via convolution (simulated as linear on folded input)
    2. Prepending class token
    3. Adding position embeddings
    """
    batch_size, img_h, img_w, img_c = pixel_values.shape  # NHWC format
    patch_size = config.patch_size
    patch_count = img_h // patch_size
    patch_size_sq = patch_size * patch_size * 3  # 32*32*3 = 3072
    num_patches = patch_count * patch_count

    # Fold image into patches
    stride_h = patch_size
    stride_w = 1
    folded_pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    ttnn.deallocate(pixel_values)

    folded_pixel_values = ttnn.to_memory_config(folded_pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG)
    folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    # Linear projection of patches
    patch_embeddings = ttnn.linear(
        folded_pixel_values,
        parameters.patch_embedding.weight,
        bias=parameters.patch_embedding.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=config.core_grid,
    )
    ttnn.deallocate(folded_pixel_values)

    patch_embeddings = ttnn.to_layout(patch_embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embeddings = ttnn.reshape(patch_embeddings, (batch_size, num_patches, config.vision_hidden_size))

    # Prepend class token and add position embeddings
    embeddings = ttnn.concat([cls_token, patch_embeddings], -2, memory_config=ttnn.L1_MEMORY_CONFIG)
    embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
    embeddings = ttnn.add(embeddings, position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

    return embeddings


def owl_vit_vision_attention(
    config: OwlViTTTNNConfig,
    hidden_states: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """
    Multi-head self-attention for vision encoder.
    """
    num_heads = config.vision_num_attention_heads
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # Fused QKV projection
    query_key_value = ttnn.linear(
        hidden_states,
        parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.vision_program_configs["qkv_matmul_program_config"],
    )

    # Split into Q, K, V and reshape for multi-head attention
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    ttnn.deallocate(hidden_states)

    # Attention scores: Q @ K^T
    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # Scale attention scores
    scale = 1.0 / (head_size**0.5)
    attention_scores = ttnn.mul_(attention_scores, scale)

    # Softmax
    attention_probs = ttnn.softmax_in_place(attention_scores)

    # Attention output: attn_probs @ V
    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    # Concatenate heads
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    )

    # Output projection
    attention_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.vision_program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)

    return attention_output


def owl_vit_vision_mlp(
    config: OwlViTTTNNConfig,
    hidden_states: ttnn.Tensor,
    *,
    parameters,
) -> ttnn.Tensor:
    """
    Feed-forward network for vision encoder with GELU activation.
    """
    # First linear layer with fused GELU activation
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.vision_program_configs["ff1_matmul_program_config"],
    )

    # Second linear layer (no activation)
    output = ttnn.linear(
        hidden_states,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.vision_program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    return output


def owl_vit_vision_encoder_layer(
    config: OwlViTTTNNConfig,
    hidden_states: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """
    Single vision encoder layer with pre-norm architecture.
    """
    # Pre-attention layer norm
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layer_norm1.weight,
        bias=parameters.layer_norm1.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.vision_program_configs["layernorm_program_config"],
        compute_kernel_config=config.vision_program_configs["ln_compute_config"],
    )

    # Self-attention
    attention_output = owl_vit_vision_attention(config, hidden_states, parameters.self_attn)
    hidden_states = ttnn.add(
        attention_output, residual, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    # Pre-MLP layer norm
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layer_norm2.weight,
        bias=parameters.layer_norm2.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.vision_program_configs["layernorm_program_config"],
        compute_kernel_config=config.vision_program_configs["ln_compute_config"],
    )

    # MLP
    mlp_output = owl_vit_vision_mlp(config, hidden_states, parameters=parameters.mlp)
    hidden_states = ttnn.add(
        mlp_output, residual, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    return hidden_states


def owl_vit_vision_encoder(
    config: OwlViTTTNNConfig,
    embeddings: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """
    Vision encoder: stack of transformer layers.
    """
    TILE_HEIGHT = 32
    emb_shape = embeddings.shape

    # Convert to sharded memory for encoder
    encoder_input = ttnn.to_memory_config(
        embeddings,
        memory_config=ttnn.create_sharded_memory_config(
            list(emb_shape),
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(embeddings)

    # Process through encoder layers
    for layer_params in parameters.layers:
        encoder_output = owl_vit_vision_encoder_layer(config, encoder_input, layer_params)
        encoder_input = encoder_output

    return encoder_output


def owl_vit_vision_model(
    config: OwlViTTTNNConfig,
    pixel_values: ttnn.Tensor,
    cls_token: ttnn.Tensor,
    position_embeddings: ttnn.Tensor,
    parameters,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Full vision model forward pass.

    Returns:
        last_hidden_state: All patch features [batch, seq_len, hidden_size]
        pooled_output: CLS token output [batch, hidden_size]
    """
    # Pre-layer norm (OWL-ViT uses pre-norm)
    # Note: In OWL-ViT, pre_layernorm is applied before patch embedding

    # Get embeddings
    embeddings = owl_vit_vision_embeddings(
        config, pixel_values, cls_token, position_embeddings, parameters=parameters.embeddings
    )

    # Pre-layernorm
    embeddings = ttnn.layer_norm(
        embeddings,
        weight=parameters.pre_layernorm.weight,
        bias=parameters.pre_layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Encoder
    last_hidden_state = owl_vit_vision_encoder(config, embeddings, parameters.encoder)

    # Post-layernorm
    last_hidden_state = ttnn.layer_norm(
        last_hidden_state,
        weight=parameters.post_layernorm.weight,
        bias=parameters.post_layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.vision_program_configs["layernorm_program_config"],
    )

    # Pool from CLS token (first token)
    # Note: For OWL-ViT detection, we use all patch features, not just CLS
    pooled_output = last_hidden_state[:, 0, :]  # CLS token

    return last_hidden_state, pooled_output


# ============================================================================
# Detection Heads
# ============================================================================


def owl_vit_box_head(
    config: OwlViTTTNNConfig,
    image_features: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """
    Box prediction head: MLP that predicts bounding boxes.

    Args:
        image_features: Visual features [batch, num_patches, hidden_size]
        parameters: Box head parameters (dense0, dense1, dense2)

    Returns:
        pred_boxes: Predicted boxes [batch, num_patches, 4] in (cx, cy, w, h) format
    """
    # Skip CLS token, use only patch features for box prediction
    patch_features = image_features[:, 1:, :]

    # MLP: hidden -> hidden -> hidden -> 4
    hidden = ttnn.linear(
        patch_features,
        parameters.dense0.weight,
        bias=parameters.dense0.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    hidden = ttnn.gelu(hidden)

    hidden = ttnn.linear(
        hidden,
        parameters.dense1.weight,
        bias=parameters.dense1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    hidden = ttnn.gelu(hidden)

    pred_boxes = ttnn.linear(
        hidden,
        parameters.dense2.weight,
        bias=parameters.dense2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Apply sigmoid for normalized box coordinates
    pred_boxes = ttnn.sigmoid(pred_boxes)

    return pred_boxes


def owl_vit_class_head(
    config: OwlViTTTNNConfig,
    image_features: ttnn.Tensor,
    query_embeds: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """
    Class prediction head: computes region-text similarity.

    Args:
        image_features: Visual features [batch, num_patches+1, hidden_size]
        query_embeds: Text query embeddings [batch, num_queries, projection_dim]
        parameters: Class head parameters

    Returns:
        logits: Classification logits [batch, num_patches, num_queries]
    """
    # Skip CLS token for classification
    patch_features = image_features[:, 1:, :]

    # Project patch features to query space
    patch_features_projected = ttnn.linear(
        patch_features,
        parameters.dense0.weight,
        bias=parameters.dense0.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Normalize features
    patch_features_norm = ttnn.layer_norm(
        patch_features_projected,
        weight=None,
        bias=None,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Compute similarity: patch_features @ query_embeds^T
    query_embeds_t = ttnn.transpose(query_embeds, -2, -1)
    logits = ttnn.matmul(
        patch_features_norm,
        query_embeds_t,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Apply learned logit_scale and shift
    logits = ttnn.mul(logits, parameters.logit_scale)
    logits = ttnn.add(logits, parameters.logit_shift)

    return logits


# ============================================================================
# Text Model Components (simplified for initial bring-up)
# ============================================================================


def owl_vit_text_encoder(
    config: OwlViTTTNNConfig,
    input_ids: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """
    Text encoder for processing text queries.

    For initial bring-up, this uses a simplified implementation.
    Full optimization can follow the CLIPEncoder pattern from tt_dit.
    """
    # Token embeddings
    token_embeddings = ttnn.embedding(
        input_ids,
        parameters.embeddings.token_embedding.weight,
        layout=ttnn.TILE_LAYOUT,
    )

    # Position embeddings
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).expand((batch_size, -1))
    position_ids_ttnn = ttnn.from_torch(
        position_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=input_ids.device(),
    )
    position_embeddings = ttnn.embedding(
        position_ids_ttnn,
        parameters.embeddings.position_embedding.weight,
        layout=ttnn.TILE_LAYOUT,
    )

    hidden_states = ttnn.add(token_embeddings, position_embeddings)

    # Create causal attention mask
    # For text encoder, we use causal (masked) attention
    causal_mask = create_causal_mask(seq_len, hidden_states.device())

    # Encoder layers
    for layer_params in parameters.encoder.layers:
        hidden_states = owl_vit_text_encoder_layer(config, hidden_states, causal_mask, layer_params)

    # Final layer norm
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        bias=parameters.final_layer_norm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Pool from EOS token
    # For CLIP-like models, we take the feature at the EOS position
    pooled_output = gather_eos_token(hidden_states, input_ids)

    return hidden_states, pooled_output


def owl_vit_text_encoder_layer(
    config: OwlViTTTNNConfig,
    hidden_states: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """Single text encoder layer."""
    # Pre-norm attention
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layer_norm1.weight,
        bias=parameters.layer_norm1.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Self-attention with causal mask
    hidden_states = owl_vit_text_self_attention(config, hidden_states, attention_mask, parameters.self_attn)
    hidden_states = ttnn.add(hidden_states, residual)

    # Pre-norm MLP
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layer_norm2.weight,
        bias=parameters.layer_norm2.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    hidden_states = owl_vit_text_mlp(config, hidden_states, parameters.mlp)
    hidden_states = ttnn.add(hidden_states, residual)

    return hidden_states


def owl_vit_text_self_attention(
    config: OwlViTTTNNConfig,
    hidden_states: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """Text self-attention with causal masking."""
    num_heads = config.text_num_attention_heads
    head_dim = config.text_hidden_size // num_heads
    batch_size, seq_len, _ = hidden_states.shape

    # Q, K, V projections
    q = ttnn.linear(hidden_states, parameters.q_proj.weight, bias=parameters.q_proj.bias)
    k = ttnn.linear(hidden_states, parameters.k_proj.weight, bias=parameters.k_proj.bias)
    v = ttnn.linear(hidden_states, parameters.v_proj.weight, bias=parameters.v_proj.bias)

    # Reshape for multi-head attention
    q = ttnn.reshape(q, (batch_size, seq_len, num_heads, head_dim))
    k = ttnn.reshape(k, (batch_size, seq_len, num_heads, head_dim))
    v = ttnn.reshape(v, (batch_size, seq_len, num_heads, head_dim))

    q = ttnn.transpose(q, 1, 2)  # [B, H, S, D]
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    # Attention scores
    scale = head_dim**-0.5
    attention_scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
    attention_scores = ttnn.mul(attention_scores, scale)

    # Apply causal mask
    if attention_mask is not None:
        attention_scores = ttnn.add(attention_scores, attention_mask)

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    # Attention output
    attention_output = ttnn.matmul(attention_probs, v)
    attention_output = ttnn.transpose(attention_output, 1, 2)
    attention_output = ttnn.reshape(attention_output, (batch_size, seq_len, config.text_hidden_size))

    # Output projection
    output = ttnn.linear(attention_output, parameters.out_proj.weight, bias=parameters.out_proj.bias)

    return output


def owl_vit_text_mlp(
    config: OwlViTTTNNConfig,
    hidden_states: ttnn.Tensor,
    parameters,
) -> ttnn.Tensor:
    """Text encoder MLP with GELU activation."""
    hidden_states = ttnn.linear(hidden_states, parameters.fc1.weight, bias=parameters.fc1.bias)
    hidden_states = ttnn.gelu(hidden_states, fast_and_approximate=True)
    hidden_states = ttnn.linear(hidden_states, parameters.fc2.weight, bias=parameters.fc2.bias)
    return hidden_states


# ============================================================================
# Utility Functions
# ============================================================================


def create_causal_mask(seq_len: int, device: ttnn.Device) -> ttnn.Tensor:
    """Create a causal attention mask for text encoder."""
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask[None, None, :, :]  # [1, 1, S, S]
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)


def gather_eos_token(hidden_states: ttnn.Tensor, input_ids: ttnn.Tensor) -> ttnn.Tensor:
    """Gather the EOS token features from hidden states."""
    # For simplicity, take the last token position
    # In production, should find actual EOS token position
    return hidden_states[:, -1, :]


# ============================================================================
# Main Model
# ============================================================================


def owl_vit_for_object_detection(
    config: OwlViTTTNNConfig,
    pixel_values: ttnn.Tensor,
    input_ids: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    cls_token: ttnn.Tensor,
    position_embeddings: ttnn.Tensor,
    parameters,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """
    Full OWL-ViT object detection forward pass.

    Args:
        config: Model configuration
        pixel_values: Input images [batch, height, width, channels] NHWC format
        input_ids: Tokenized text queries [batch, num_queries, seq_len]
        attention_mask: Text attention mask
        cls_token: Vision CLS token
        position_embeddings: Vision position embeddings
        parameters: Model parameters

    Returns:
        pred_boxes: Predicted bounding boxes [batch, num_patches, 4]
        logits: Classification logits [batch, num_patches, num_queries]
        image_embeds: Image embeddings
        text_embeds: Text embeddings
    """
    # Vision model forward
    image_features, image_pooled = owl_vit_vision_model(
        config, pixel_values, cls_token, position_embeddings, parameters=parameters.owlvit.vision_model
    )

    # Get image embeddings via projection
    image_embeds = ttnn.linear(
        image_pooled,
        parameters.owlvit.visual_projection.weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Normalize image features for box/class heads
    image_features_norm = ttnn.layer_norm(
        image_features,
        weight=parameters.layer_norm.weight,
        bias=parameters.layer_norm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Text model forward for each query
    batch_size, num_queries, seq_len = input_ids.shape
    input_ids_flat = ttnn.reshape(input_ids, (batch_size * num_queries, seq_len))
    attention_mask_flat = ttnn.reshape(attention_mask, (batch_size * num_queries, seq_len))

    _, text_pooled = owl_vit_text_encoder(
        config, input_ids_flat, attention_mask_flat, parameters=parameters.owlvit.text_model
    )

    # Project text embeddings
    text_embeds = ttnn.linear(
        text_pooled,
        parameters.owlvit.text_projection.weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    text_embeds = ttnn.reshape(text_embeds, (batch_size, num_queries, config.projection_dim))

    # Box prediction head
    pred_boxes = owl_vit_box_head(config, image_features_norm, parameters.box_head)

    # Class prediction head
    logits = owl_vit_class_head(config, image_features_norm, text_embeds, parameters.class_head)

    return pred_boxes, logits, image_embeds, text_embeds


# ============================================================================
# Parameter Preprocessing
# ============================================================================


def custom_preprocessor(torch_model, name):
    """
    Preprocess PyTorch model weights for TTNN.

    This handles:
    - Weight transposition for linear layers
    - QKV weight fusion for attention
    - Data type conversion
    """
    parameters = {}

    # Handle QKV fusion for attention layers
    if hasattr(torch_model, "q_proj") and hasattr(torch_model, "k_proj") and hasattr(torch_model, "v_proj"):
        # Fuse Q, K, V weights
        q_weight = torch_model.q_proj.weight
        k_weight = torch_model.k_proj.weight
        v_weight = torch_model.v_proj.weight

        num_heads = q_weight.shape[0] // 64  # Assuming head_dim = 64
        head_dim = 64
        hidden_size = q_weight.shape[0]

        qkv_weight = torch.cat(
            [
                q_weight.reshape([num_heads, head_dim, -1]),
                k_weight.reshape([num_heads, head_dim, -1]),
                v_weight.reshape([num_heads, head_dim, -1]),
            ],
            dim=1,
        ).reshape([hidden_size * 3, -1])

        if hasattr(torch_model.q_proj, "bias") and torch_model.q_proj.bias is not None:
            q_bias = torch_model.q_proj.bias
            k_bias = torch_model.k_proj.bias
            v_bias = torch_model.v_proj.bias
            qkv_bias = torch.cat(
                [
                    q_bias.reshape([num_heads, head_dim]),
                    k_bias.reshape([num_heads, head_dim]),
                    v_bias.reshape([num_heads, head_dim]),
                ],
                dim=1,
            ).reshape([hidden_size * 3])
        else:
            qkv_bias = None

        parameters["query_key_value"] = {}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        if qkv_bias is not None:
            parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)

    # Handle standard linear layers
    elif isinstance(torch_model, torch.nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
        if torch_model.bias is not None:
            parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)

    return parameters
