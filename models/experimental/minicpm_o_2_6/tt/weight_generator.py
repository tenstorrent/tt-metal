# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Random weight generator for MiniCPM-o-2_6 TTNN PCC validation.

Generates random weights matching PyTorch reference component shapes for testing.
"""

import torch
from typing import Dict, Optional
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_resampler_weights(
    num_queries: int = 64,
    embed_dim: int = 3584,
    kv_dim: int = 1152,
    num_heads: int = 28,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for Vision/Audio Resampler component.

    Args:
        num_queries: Number of learnable queries (default 64)
        embed_dim: Output embedding dimension (default 3584 for Qwen2.5)
        kv_dim: Input key/value dimension (default 1152 for SigLip)
        num_heads: Number of attention heads (default 28)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    head_dim = embed_dim // num_heads
    weights = {}

    # Learnable queries
    weights["query"] = torch.randn(num_queries, embed_dim)

    # Cross-attention projections
    weights["attn.q_proj.weight"] = torch.randn(embed_dim, embed_dim)
    weights["attn.q_proj.bias"] = torch.randn(embed_dim)
    weights["attn.k_proj.weight"] = torch.randn(embed_dim, kv_dim)
    weights["attn.k_proj.bias"] = torch.randn(embed_dim)
    weights["attn.v_proj.weight"] = torch.randn(embed_dim, kv_dim)
    weights["attn.v_proj.bias"] = torch.randn(embed_dim)
    weights["attn.o_proj.weight"] = torch.randn(embed_dim, embed_dim)
    weights["attn.o_proj.bias"] = torch.randn(embed_dim)

    # Layer norms
    weights["ln_q.weight"] = torch.randn(embed_dim)
    weights["ln_q.bias"] = torch.randn(embed_dim)
    weights["ln_kv.weight"] = torch.randn(kv_dim)
    weights["ln_kv.bias"] = torch.randn(kv_dim)

    # MLP
    mlp_hidden = embed_dim * 4  # 14336 for embed_dim=3584
    weights["mlp.fc1.weight"] = torch.randn(mlp_hidden, embed_dim)
    weights["mlp.fc1.bias"] = torch.randn(mlp_hidden)
    weights["mlp.fc2.weight"] = torch.randn(embed_dim, mlp_hidden)
    weights["mlp.fc2.bias"] = torch.randn(embed_dim)

    # Final layer norm
    weights["ln_post.weight"] = torch.randn(embed_dim)
    weights["ln_post.bias"] = torch.randn(embed_dim)

    # Final projection layer
    weights["proj"] = torch.randn(embed_dim, embed_dim)

    return weights


def generate_audio_projector_weights(
    in_dim: int = 256,
    out_dim: int = 3584,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for Audio Projection Layer (MultiModalProjector).

    Args:
        in_dim: Input dimension (default 256 for Whisper // 4)
        out_dim: Output dimension (default 3584 for Qwen2.5)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    weights = {}

    # Two linear layers with ReLU
    weights["linear1.weight"] = torch.randn(out_dim, in_dim)
    weights["linear1.bias"] = torch.randn(out_dim)
    weights["linear2.weight"] = torch.randn(out_dim, out_dim)
    weights["linear2.bias"] = torch.randn(out_dim)

    return weights


def generate_cross_attention_weights(
    hidden_size: int = 3584,
    num_attention_heads: int = 28,
    num_key_value_heads: int = 4,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for Cross-Attention layer.

    Args:
        hidden_size: Hidden dimension (default 3584)
        num_attention_heads: Number of query heads (default 28)
        num_key_value_heads: Number of key/value heads for GQA (default 4)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    head_dim = hidden_size // num_attention_heads
    weights = {}

    # Query projection
    weights["q_proj.weight"] = torch.randn(num_attention_heads * head_dim, hidden_size)
    weights["q_proj.bias"] = torch.randn(num_attention_heads * head_dim)

    # Key projection
    weights["k_proj.weight"] = torch.randn(num_key_value_heads * head_dim, hidden_size)
    weights["k_proj.bias"] = torch.randn(num_key_value_heads * head_dim)

    # Value projection
    weights["v_proj.weight"] = torch.randn(num_key_value_heads * head_dim, hidden_size)
    weights["v_proj.bias"] = torch.randn(num_key_value_heads * head_dim)

    # Output projection
    weights["o_proj.weight"] = torch.randn(hidden_size, num_attention_heads * head_dim)

    # RMS norms for Q and K
    weights["q_norm.weight"] = torch.randn(head_dim)  # Applied per head
    weights["k_norm.weight"] = torch.randn(head_dim)  # Applied per head

    return weights


def generate_conditional_chattts_weights(
    llm_dim: int = 3584,
    hidden_size: int = 768,
    num_layers: int = 20,  # Production: 20 transformer layers
    num_heads: int = 12,
    intermediate_size: int = 3072,
    num_audio_tokens: int = 626,
    num_text_tokens: int = 21178,
    num_vq: int = 4,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for ConditionalChatTTS transformer.

    Args:
        llm_dim: LLM hidden dimension (default 3584)
        hidden_size: TTS hidden dimension (default 768)
        num_layers: Number of transformer layers (default 20)
        num_heads: Number of attention heads (default 12)
        intermediate_size: FFN intermediate size (default 3072)
        num_audio_tokens: Audio vocabulary size (default 626)
        num_text_tokens: Text vocabulary size (default 21178)
        num_vq: Number of VQ codebooks (default 4)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    weights = {}

    # LLM projector (MultiModalProjector or Linear)
    weights["projector.linear1.weight"] = torch.randn(hidden_size, llm_dim)
    weights["projector.linear1.bias"] = torch.randn(hidden_size)
    weights["projector.linear2.weight"] = torch.randn(hidden_size, hidden_size)
    weights["projector.linear2.bias"] = torch.randn(hidden_size)

    # Text embedding
    weights["emb_text.weight"] = torch.randn(num_text_tokens, hidden_size)

    # Audio code embeddings (4 codebooks)
    for i in range(num_vq):
        weights[f"emb_code.{i}.weight"] = torch.randn(num_audio_tokens, hidden_size)

    # Transformer layers (Llama-based)
    head_dim = hidden_size // num_heads
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # Self-attention
        weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)

        # MLP (Llama-style: gate_proj, up_proj, down_proj)
        # gate_proj: [intermediate_size, hidden_size] - expands to intermediate
        # up_proj: [intermediate_size, hidden_size] - expands to intermediate
        # down_proj: [hidden_size, intermediate_size] - reduces to hidden
        weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)

        # RMS norms
        weights[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)

    # Final norm
    weights["model.norm.weight"] = torch.randn(hidden_size)

    # Output heads (4 codebooks)
    for i in range(num_vq):
        weights[f"head_code.{i}.weight"] = torch.randn(num_audio_tokens, hidden_size)

    return weights


def generate_dvae_weights(
    num_encoder_layers: int = 12,  # Production: 12 layers
    num_decoder_layers: int = 12,  # Production: 12 layers
    hidden_dim: int = 256,
    num_mel_bins: int = 100,
    bn_dim: int = 128,  # Production: 128
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for DVAE (Discrete VAE).

    Using 2D convolutions for mel spectrogram processing.
    Input format: [batch, H=1, W=time_steps, C=num_mel_bins]

    Production Configuration (from MiniCPM-o-2_6):
    - Encoder: 12 ConvNeXt blocks, hidden_dim=256, bn_dim=128
    - Decoder: 12 ConvNeXt blocks, hidden_dim=256, bn_dim=128

    Args:
        num_encoder_layers: Number of encoder ConvNeXt blocks (default 12 for production)
        num_decoder_layers: Number of decoder ConvNeXt blocks (default 12 for production)
        hidden_dim: Hidden dimension (default 256)
        num_mel_bins: Number of mel bins (default 100)
        bn_dim: Bottleneck dimension for conv_in (default 128)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors in TTNN conv2d format
    """
    set_seed(seed)

    weights = {}

    # Coefficient parameter for mel scaling
    weights["coef"] = torch.randn(1, num_mel_bins, 1)

    # Downsample convolutions (2D: [out_channels, in_channels, kernel_h, kernel_w])
    # Input: [batch, num_mel_bins, time_steps] -> treated as [batch, H=1, W=time_steps, C=num_mel_bins]
    weights["downsample_conv.0.weight"] = torch.randn(512, num_mel_bins, 1, 3)  # Conv2d format
    weights["downsample_conv.0.bias"] = torch.randn(1, 1, 1, 512)  # TTNN bias format: [1, 1, 1, out_channels]
    weights["downsample_conv.2.weight"] = torch.randn(512, 512, 1, 4)  # Conv2d format
    weights["downsample_conv.2.bias"] = torch.randn(1, 1, 1, 512)  # TTNN bias format: [1, 1, 1, out_channels]

    # Encoder ConvNeXt blocks (2D depthwise conv)
    for i in range(num_encoder_layers):
        prefix = f"encoder.decoder_block.{i}"
        weights[f"{prefix}.dwconv.weight"] = torch.randn(hidden_dim, 1, 1, 7)  # Depthwise 2D conv
        weights[f"{prefix}.dwconv.bias"] = torch.randn(1, 1, 1, hidden_dim)  # TTNN bias format
        weights[f"{prefix}.norm.weight"] = torch.randn(hidden_dim)
        weights[f"{prefix}.norm.bias"] = torch.randn(hidden_dim)
        weights[f"{prefix}.pwconv1.weight"] = torch.randn(hidden_dim * 4, hidden_dim)
        weights[f"{prefix}.pwconv1.bias"] = torch.randn(hidden_dim * 4)
        weights[f"{prefix}.pwconv2.weight"] = torch.randn(hidden_dim, hidden_dim * 4)
        weights[f"{prefix}.pwconv2.bias"] = torch.randn(hidden_dim)

    # Encoder input/output convolutions (2D format)
    # Production: bn_dim=128 (not 64)
    weights["encoder.conv_in.0.weight"] = torch.randn(bn_dim, 512, 1, 3)  # [out, in, H, W]
    weights["encoder.conv_in.0.bias"] = torch.randn(1, 1, 1, bn_dim)  # TTNN bias format
    weights["encoder.conv_in.2.weight"] = torch.randn(hidden_dim, bn_dim, 1, 3)
    weights["encoder.conv_in.2.bias"] = torch.randn(1, 1, 1, hidden_dim)  # TTNN bias format
    weights["encoder.conv_out.weight"] = torch.randn(1024, hidden_dim, 1, 1)  # 1x1 conv

    # Decoder ConvNeXt blocks (2D depthwise conv)
    for i in range(num_decoder_layers):
        prefix = f"decoder.decoder_block.{i}"
        weights[f"{prefix}.dwconv.weight"] = torch.randn(hidden_dim, 1, 1, 7)  # Depthwise 2D conv
        weights[f"{prefix}.dwconv.bias"] = torch.randn(1, 1, 1, hidden_dim)  # TTNN bias format
        weights[f"{prefix}.norm.weight"] = torch.randn(hidden_dim)
        weights[f"{prefix}.norm.bias"] = torch.randn(hidden_dim)
        weights[f"{prefix}.pwconv1.weight"] = torch.randn(hidden_dim * 4, hidden_dim)
        weights[f"{prefix}.pwconv1.bias"] = torch.randn(hidden_dim * 4)
        weights[f"{prefix}.pwconv2.weight"] = torch.randn(hidden_dim, hidden_dim * 4)
        weights[f"{prefix}.pwconv2.bias"] = torch.randn(hidden_dim)

    # Decoder input/output convolutions (2D format)
    # Production: decoder input is 1024 channels from encoder
    weights["decoder.conv_in.0.weight"] = torch.randn(bn_dim, 1024, 1, 3)  # Production: 1024 input channels
    weights["decoder.conv_in.0.bias"] = torch.randn(1, 1, 1, bn_dim)  # TTNN bias format
    weights["decoder.conv_in.2.weight"] = torch.randn(hidden_dim, bn_dim, 1, 3)
    weights["decoder.conv_in.2.bias"] = torch.randn(1, 1, 1, hidden_dim)  # TTNN bias format

    # Decoder projection: hidden_dim -> 512 channels (NEW layer)
    weights["decoder.proj.weight"] = torch.randn(512, hidden_dim, 1, 1)  # 1x1 conv

    weights["out_conv.weight"] = torch.randn(num_mel_bins, 512, 1, 3)  # Production: 512 output channels

    return weights


def generate_audio_projector_weights(
    input_dim: int = 1024,  # Whisper hidden size
    output_dim: int = 3584,  # Qwen2.5 hidden size
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for Audio Projector component matching official MiniCPM-o-2_6 architecture.

    Official Architecture (from modeling_minicpmo.py):
    - audio_projection_layer = MultiModalProjector(in_dim=1024, out_dim=3584)
    - MultiModalProjector has Linear1: 1024 → 3584 and Linear2: 3584 → 3584
    - Pooling is applied externally in the forward pipeline

    Weight Format (matching HuggingFace checkpoint):
    - 'audio_projection_layer.linear1.weight': [3584, 1024] (out_features, in_features)
    - 'audio_projection_layer.linear1.bias': [3584]
    - 'audio_projection_layer.linear2.weight': [3584, 3584] (out_features, in_features)
    - 'audio_projection_layer.linear2.bias': [3584]

    Args:
        input_dim: Input dimension (default 1024 for Whisper)
        output_dim: Output dimension (default 3584 for Qwen2.5)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors in HuggingFace format
    """
    set_seed(seed)

    weights = {}

    # Linear1 weights: [output_dim, input_dim] format (PyTorch/HuggingFace format)
    # This is [3584, 1024] for the projection from Whisper to Qwen space
    weights["audio_projection_layer.linear1.weight"] = torch.randn(output_dim, input_dim)
    weights["audio_projection_layer.linear1.bias"] = torch.randn(output_dim)

    # Linear2 weights: [output_dim, output_dim] format (PyTorch/HuggingFace format)
    # This is [3584, 3584] for refinement in Qwen space
    weights["audio_projection_layer.linear2.weight"] = torch.randn(output_dim, output_dim)
    weights["audio_projection_layer.linear2.bias"] = torch.randn(output_dim)

    return weights


def generate_vision_weights(
    hidden_size: int = 1152,  # SigLip hidden size
    num_attention_heads: int = 12,
    num_hidden_layers: int = 27,  # SigLip layers
    image_size: int = 224,
    patch_size: int = 16,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for Vision (SigLip) component.

    Args:
        hidden_size: Hidden dimension (1152 for SigLip)
        num_attention_heads: Number of attention heads (12 for SigLip)
        num_hidden_layers: Number of transformer layers (27 for SigLip)
        image_size: Input image size (224)
        patch_size: Patch size (16)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    num_patches = (image_size // patch_size) ** 2  # 196
    seq_len = num_patches + 1  # +1 for CLS token

    weights = {}

    # Patch embedding
    weights["vision_model.embeddings.patch_embedding.weight"] = torch.randn(hidden_size, 3, patch_size, patch_size)
    weights["vision_model.embeddings.patch_embedding.bias"] = torch.randn(hidden_size)
    weights["vision_model.embeddings.position_embedding.weight"] = torch.randn(seq_len, hidden_size)

    # Transformer layers
    for i in range(num_hidden_layers):
        # Self-attention
        weights[f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = torch.randn(hidden_size)
        weights[f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = torch.randn(hidden_size, hidden_size)

        # Layer norms
        weights[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = torch.randn(hidden_size)
        weights[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = torch.randn(hidden_size)
        weights[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = torch.randn(hidden_size)
        weights[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = torch.randn(hidden_size)

        # MLP
        weights[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = torch.randn(hidden_size * 4, hidden_size)
        weights[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = torch.randn(hidden_size * 4)
        weights[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = torch.randn(hidden_size, hidden_size * 4)
        weights[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = torch.randn(hidden_size)

    # Final layer norm
    weights["vision_model.post_layernorm.weight"] = torch.randn(hidden_size)
    weights["vision_model.post_layernorm.bias"] = torch.randn(hidden_size)

    return weights


def generate_chattts_weights(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 20,
    intermediate_size: int = 3072,
    num_text_tokens: int = 21178,
    num_audio_tokens: int = 626,
    num_vq: int = 4,
    llm_dim: int = 3584,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for ChatTTS decoder component.

    Args:
        hidden_size: Hidden dimension (768 for ChatTTS)
        num_attention_heads: Number of attention heads (12)
        num_hidden_layers: Number of transformer layers (20)
        intermediate_size: FFN intermediate size (3072)
        num_text_tokens: Text vocabulary size (21178)
        num_audio_tokens: Audio token vocabulary size (626)
        num_vq: Number of audio codebooks (4)
        llm_dim: LLM hidden dimension for projector (3584)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    weights = {}

    # LLM projector
    weights["projector.linear1.weight"] = torch.randn(llm_dim, hidden_size)
    weights["projector.linear1.bias"] = torch.randn(hidden_size)
    weights["projector.linear2.weight"] = torch.randn(hidden_size, hidden_size)
    weights["projector.linear2.bias"] = torch.randn(hidden_size)

    # Text embeddings
    weights["emb_text.weight"] = torch.randn(num_text_tokens, hidden_size)

    # Audio code embeddings (4 codebooks)
    for i in range(num_vq):
        weights[f"emb_code.{i}.weight"] = torch.randn(num_audio_tokens, hidden_size)

    # Transformer layers
    for i in range(num_hidden_layers):
        # Self-attention
        weights[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)

        # Layer norms
        weights[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(hidden_size)
        weights[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(hidden_size)

        # MLP
        weights[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)

    # Final layer norm
    weights["model.norm.weight"] = torch.randn(hidden_size)

    # Output heads (4 for each codebook)
    for i in range(num_vq):
        weights[f"head_code.{i}.weight"] = torch.randn(num_audio_tokens, hidden_size)

    return weights


def generate_all_component_weights(seed: int = 42) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate random weights for all MiniCPM-o-2_6 components.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Nested dict with all component weights
    """
    all_weights = {
        "vision_resampler": generate_resampler_weights(seed=seed),
        "audio_resampler": generate_resampler_weights(kv_dim=1024, seed=seed + 1),  # Whisper is 1024d
        "audio_projector": generate_audio_projector_weights(seed=seed),
        "cross_attention_layer8": generate_cross_attention_weights(seed=seed),
        "cross_attention_layer16": generate_cross_attention_weights(seed=seed + 1),
        "cross_attention_layer24": generate_cross_attention_weights(seed=seed + 2),
        "conditional_chattts": generate_conditional_chattts_weights(seed=seed),
        "dvae": generate_dvae_weights(seed=seed),
        "whisper_encoder": generate_whisper_weights(seed=seed),
        "qwen_llm": generate_qwen_weights(seed=seed),
    }

    return all_weights


def generate_whisper_weights(
    d_model: int = 1024,
    encoder_layers: int = 24,
    encoder_attention_heads: int = 16,
    encoder_ffn_dim: int = 4096,
    num_mel_bins: int = 80,
    max_source_positions: int = 1500,
    vocab_size: int = 51865,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for Whisper encoder component.

    Based on Whisper architecture adapted for MiniCPM-o-2_6:
    - d_model: 1024
    - encoder_layers: 24
    - encoder_attention_heads: 16
    - encoder_ffn_dim: 4096

    Args:
        d_model: Hidden dimension
        encoder_layers: Number of encoder layers
        encoder_attention_heads: Number of attention heads
        encoder_ffn_dim: FFN intermediate dimension
        num_mel_bins: Number of mel frequency bins
        max_source_positions: Maximum sequence length
        vocab_size: Vocabulary size
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    weights = {}

    # Conv layers - use 'apm' prefix to match MiniCPM-o-2_6 key structure
    weights["apm.conv1.weight"] = torch.randn(d_model, num_mel_bins, 3)  # [out, in, kernel]
    weights["apm.conv1.bias"] = torch.randn(d_model)
    weights["apm.conv2.weight"] = torch.randn(d_model, d_model, 3)  # [out, in, kernel]
    weights["apm.conv2.bias"] = torch.randn(d_model)

    # Embeddings
    weights["apm.embed_positions.weight"] = torch.randn(max_source_positions, d_model)

    # Encoder layers - use 'apm.layers' prefix to match MiniCPM-o-2_6 key structure
    for i in range(encoder_layers):
        # Self-attention
        weights[f"apm.layers.{i}.self_attn.q_proj.weight"] = torch.randn(d_model, d_model)
        weights[f"apm.layers.{i}.self_attn.q_proj.bias"] = torch.randn(d_model)
        weights[f"apm.layers.{i}.self_attn.k_proj.weight"] = torch.randn(d_model, d_model)
        weights[f"apm.layers.{i}.self_attn.k_proj.bias"] = torch.randn(d_model)
        weights[f"apm.layers.{i}.self_attn.v_proj.weight"] = torch.randn(d_model, d_model)
        weights[f"apm.layers.{i}.self_attn.v_proj.bias"] = torch.randn(d_model)
        weights[f"apm.layers.{i}.self_attn.out_proj.weight"] = torch.randn(d_model, d_model)
        weights[f"apm.layers.{i}.self_attn.out_proj.bias"] = torch.randn(d_model)

        # Layer norms
        weights[f"apm.layers.{i}.self_attn_layer_norm.weight"] = torch.randn(d_model)
        weights[f"apm.layers.{i}.self_attn_layer_norm.bias"] = torch.randn(d_model)
        weights[f"apm.layers.{i}.final_layer_norm.weight"] = torch.randn(d_model)
        weights[f"apm.layers.{i}.final_layer_norm.bias"] = torch.randn(d_model)

        # Feed-forward
        weights[f"apm.layers.{i}.fc1.weight"] = torch.randn(encoder_ffn_dim, d_model)
        weights[f"apm.layers.{i}.fc1.bias"] = torch.randn(encoder_ffn_dim)
        weights[f"apm.layers.{i}.fc2.weight"] = torch.randn(d_model, encoder_ffn_dim)
        weights[f"apm.layers.{i}.fc2.bias"] = torch.randn(d_model)

    # Final layer norm
    weights["apm.layer_norm.weight"] = torch.randn(d_model)
    weights["apm.layer_norm.bias"] = torch.randn(d_model)

    return weights


def generate_qwen_weights(
    vocab_size: int = 151700,
    hidden_size: int = 3584,
    intermediate_size: int = 18944,
    num_hidden_layers: int = 28,
    num_attention_heads: int = 28,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 32768,
    cross_attention_layers: Optional[list] = None,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random weights for Qwen2.5 LLM component.

    Based on Qwen2.5 architecture adapted for MiniCPM-o-2_6:
    - hidden_size: 3584
    - num_layers: 28
    - num_attention_heads: 28
    - num_key_value_heads: 4 (GQA)

    Keys follow model_key_mapping.txt structure with 'llm.' prefix.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate size
        num_hidden_layers: Number of layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads
        max_position_embeddings: Maximum sequence length
        cross_attention_layers: Layers with cross-attention (for multimodal)
        seed: Random seed

    Returns:
        Dict[str, torch.Tensor]: Random weight tensors
    """
    set_seed(seed)

    if cross_attention_layers is None:
        cross_attention_layers = [8, 16, 24]  # MiniCPM specific layers

    weights = {}

    # Token embeddings
    weights["llm.model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)

    # Layers
    for i in range(num_hidden_layers):
        # Input layer norm
        weights[f"llm.model.layers.{i}.input_layernorm.weight"] = torch.randn(hidden_size)

        # Self-attention
        head_dim = hidden_size // num_attention_heads
        kv_dim = num_key_value_heads * head_dim

        weights[f"llm.model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"llm.model.layers.{i}.self_attn.q_proj.bias"] = torch.randn(hidden_size)
        weights[f"llm.model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(kv_dim, hidden_size)
        weights[f"llm.model.layers.{i}.self_attn.k_proj.bias"] = torch.randn(kv_dim)
        weights[f"llm.model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(kv_dim, hidden_size)
        weights[f"llm.model.layers.{i}.self_attn.v_proj.bias"] = torch.randn(kv_dim)
        weights[f"llm.model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        # Note: o_proj doesn't have bias in Qwen2

        # Post-attention layer norm
        weights[f"llm.model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(hidden_size)

        # MLP
        weights[f"llm.model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"llm.model.layers.{i}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"llm.model.layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)

        # Cross-attention (if this layer has it)
        if i in cross_attention_layers:
            # Cross-attention layer norm
            weights[f"llm.model.layers.{i}.cross_attn_layernorm.weight"] = torch.randn(hidden_size)

            # Cross-attention projections (Q uses full hidden_size, K/V use GQA dimensions)
            weights[f"llm.model.layers.{i}.cross_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
            weights[f"llm.model.layers.{i}.cross_attn.q_proj.bias"] = torch.randn(hidden_size)
            weights[f"llm.model.layers.{i}.cross_attn.k_proj.weight"] = torch.randn(kv_dim, hidden_size)
            weights[f"llm.model.layers.{i}.cross_attn.k_proj.bias"] = torch.randn(kv_dim)
            weights[f"llm.model.layers.{i}.cross_attn.v_proj.weight"] = torch.randn(kv_dim, hidden_size)
            weights[f"llm.model.layers.{i}.cross_attn.v_proj.bias"] = torch.randn(kv_dim)
            weights[f"llm.model.layers.{i}.cross_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)

            # Note: Q/K normalization is handled by Qwen2RMSNorm in the model, no separate weights needed

    # Final layer norm
    weights["llm.model.norm.weight"] = torch.randn(hidden_size)

    # LM head
    weights["llm.lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    return weights
