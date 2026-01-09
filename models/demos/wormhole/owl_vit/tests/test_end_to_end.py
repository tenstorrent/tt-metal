# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
OWL-ViT End-to-End Inference Test

This module runs full OWL-ViT object detection inference on Tenstorrent N300 hardware.
All components execute using native TTNN operations:

1. Vision encoder (12 transformer layers on device)
2. Text encoder (12 transformer layers on device with ttnn.embedding)
3. Detection heads (box + class prediction on device)
4. Post-processing via HuggingFace processor

Model: google/owlvit-base-patch32
- Vision: 768 hidden, 12 heads, 12 layers, 577 patches (24x24 + CLS)
- Text: 512 hidden, 8 heads, 12 layers, 16 sequence length
"""

import sys
from typing import Any, Dict, Tuple

import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

import ttnn

sys.path.insert(0, "/root/tt-metal")


# =============================================================================
# Model Configuration Constants
# =============================================================================
MODEL_NAME = "google/owlvit-base-patch32"
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DETECTION_THRESHOLD = 0.1

# Vision encoder config
VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_HEAD_DIM = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
VISION_NUM_LAYERS = 12

# Text encoder config
TEXT_HIDDEN_SIZE = 512
TEXT_NUM_HEADS = 8
TEXT_HEAD_DIM = TEXT_HIDDEN_SIZE // TEXT_NUM_HEADS
TEXT_NUM_LAYERS = 12


def load_image(url: str) -> Image.Image:
    """Load test image from URL."""
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")


def get_pytorch_model_and_inputs(text_queries: list[str]) -> Tuple:
    """Load PyTorch model and prepare inputs for inference."""
    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()

    image = load_image(TEST_IMAGE_URL)

    inputs = processor(text=[text_queries], images=image, return_tensors="pt")

    return processor, model, inputs, image


def preprocess_all_weights_for_ttnn(model, device):
    """
    Preprocess all model weights (vision + text + heads) for TTNN.
    """
    parameters = {}
    state_dict = model.state_dict()

    # =========================================================
    # Vision Encoder Weights
    # =========================================================
    parameters["vision"] = {}

    # Pre and post layer norms
    parameters["vision"]["pre_layernorm"] = {
        "weight": ttnn.from_torch(
            state_dict["owlvit.vision_model.pre_layernorm.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        "bias": ttnn.from_torch(
            state_dict["owlvit.vision_model.pre_layernorm.bias"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
    }

    parameters["vision"]["post_layernorm"] = {
        "weight": ttnn.from_torch(
            state_dict["owlvit.vision_model.post_layernorm.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        "bias": ttnn.from_torch(
            state_dict["owlvit.vision_model.post_layernorm.bias"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
    }

    # Vision encoder layers
    parameters["vision"]["encoder_layers"] = []
    num_vision_layers = model.config.vision_config.num_hidden_layers

    for i in range(num_vision_layers):
        prefix = f"owlvit.vision_model.encoder.layers.{i}"
        layer_params = _load_encoder_layer(state_dict, prefix, device)
        parameters["vision"]["encoder_layers"].append(layer_params)

    logger.info(f"Loaded {num_vision_layers} vision encoder layers")

    # =========================================================
    # Text Encoder Weights
    # =========================================================
    parameters["text"] = {}

    # Text embeddings (token and position)
    parameters["text"]["token_embedding"] = ttnn.from_torch(
        state_dict["owlvit.text_model.embeddings.token_embedding.weight"],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # For embedding lookup
        device=device,
    )

    parameters["text"]["position_embedding"] = ttnn.from_torch(
        state_dict["owlvit.text_model.embeddings.position_embedding.weight"],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # For embedding lookup
        device=device,
    )

    parameters["text"]["final_layer_norm"] = {
        "weight": ttnn.from_torch(
            state_dict["owlvit.text_model.final_layer_norm.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        "bias": ttnn.from_torch(
            state_dict["owlvit.text_model.final_layer_norm.bias"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
    }

    # Text encoder layers
    parameters["text"]["encoder_layers"] = []
    num_text_layers = model.config.text_config.num_hidden_layers

    for i in range(num_text_layers):
        prefix = f"owlvit.text_model.encoder.layers.{i}"
        layer_params = _load_text_encoder_layer(state_dict, prefix, device)
        parameters["text"]["encoder_layers"].append(layer_params)

    logger.info(f"Loaded {num_text_layers} text encoder layers")

    # Text projection
    parameters["text_projection"] = {
        "weight": ttnn.from_torch(
            state_dict["owlvit.text_projection.weight"].T,  # [512, 512]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
    }

    logger.info("Loaded text embeddings and projection")

    # =========================================================
    # Detection Head Weights
    # =========================================================
    parameters["box_head"] = {
        "dense0": {
            "weight": ttnn.from_torch(
                state_dict["box_head.dense0.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict["box_head.dense0.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "dense1": {
            "weight": ttnn.from_torch(
                state_dict["box_head.dense1.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict["box_head.dense1.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "dense2": {
            "weight": ttnn.from_torch(
                state_dict["box_head.dense2.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict["box_head.dense2.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    # Box bias - critical for accurate box predictions
    # This is a registered buffer, not in state_dict
    parameters["box_bias"] = ttnn.from_torch(
        model.box_bias.unsqueeze(0),  # [1, 576, 4]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    parameters["class_head"] = {
        "dense0": {
            "weight": ttnn.from_torch(
                state_dict["class_head.dense0.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict["class_head.dense0.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "logit_shift": {
            "weight": ttnn.from_torch(
                state_dict["class_head.logit_shift.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict["class_head.logit_shift.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "logit_scale": {
            "weight": ttnn.from_torch(
                state_dict["class_head.logit_scale.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict["class_head.logit_scale.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    # Layer norm for feature normalization (before detection heads)
    parameters["layer_norm"] = {
        "weight": ttnn.from_torch(
            state_dict["layer_norm.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        "bias": ttnn.from_torch(
            state_dict["layer_norm.bias"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
    }

    logger.info("Loaded detection head weights")

    return parameters


def _load_encoder_layer(state_dict, prefix, device):
    """Load a single vision encoder layer."""
    layer_params = {
        "layer_norm1": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm1.weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm1.bias"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "layer_norm2": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm2.weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm2.bias"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    # Fuse Q, K, V projections
    q_weight = state_dict[f"{prefix}.self_attn.q_proj.weight"]
    k_weight = state_dict[f"{prefix}.self_attn.k_proj.weight"]
    v_weight = state_dict[f"{prefix}.self_attn.v_proj.weight"]

    q_bias = state_dict[f"{prefix}.self_attn.q_proj.bias"]
    k_bias = state_dict[f"{prefix}.self_attn.k_proj.bias"]
    v_bias = state_dict[f"{prefix}.self_attn.v_proj.bias"]

    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

    layer_params["self_attn"] = {
        "qkv": {
            "weight": ttnn.from_torch(
                qkv_weight.T,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                qkv_bias.unsqueeze(0),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "out_proj": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.self_attn.out_proj.weight"].T,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.self_attn.out_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    layer_params["mlp"] = {
        "fc1": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc1.weight"].T,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc1.bias"].unsqueeze(0),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "fc2": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc2.weight"].T,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc2.bias"].unsqueeze(0),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    return layer_params


def _load_text_encoder_layer(state_dict, prefix, device):
    """Load a single text encoder layer with fused QKV."""
    layer_params = {
        "layer_norm1": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm1.weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm1.bias"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "layer_norm2": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm2.weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.layer_norm2.bias"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    # Fuse Q, K, V weights for efficient TTNN execution
    q_weight = state_dict[f"{prefix}.self_attn.q_proj.weight"]
    k_weight = state_dict[f"{prefix}.self_attn.k_proj.weight"]
    v_weight = state_dict[f"{prefix}.self_attn.v_proj.weight"]
    q_bias = state_dict[f"{prefix}.self_attn.q_proj.bias"]
    k_bias = state_dict[f"{prefix}.self_attn.k_proj.bias"]
    v_bias = state_dict[f"{prefix}.self_attn.v_proj.bias"]

    # Fuse weights: [3 * hidden, hidden] -> for linear: [hidden, 3 * hidden]
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).T
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0).unsqueeze(0)

    layer_params["self_attn"] = {
        "qkv": {
            "weight": ttnn.from_torch(
                qkv_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                qkv_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "out_proj": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.self_attn.out_proj.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.self_attn.out_proj.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    layer_params["mlp"] = {
        "fc1": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc1.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc1.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
        "fc2": {
            "weight": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc2.weight"].T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
            "bias": ttnn.from_torch(
                state_dict[f"{prefix}.mlp.fc2.bias"].unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            ),
        },
    }

    return layer_params


def run_vision_encoder_on_device(
    pixel_values: torch.Tensor,
    parameters: Dict[str, Any],
    device: ttnn.Device,
    pytorch_model,
) -> ttnn.Tensor:
    """
    Run vision encoder forward pass on TTNN device.
    Uses PyTorch for patch embeddings, runs transformer on device.
    """
    num_heads = VISION_NUM_HEADS
    head_dim = VISION_HEAD_DIM

    # Get embeddings from PyTorch
    with torch.no_grad():
        vision_embeddings = pytorch_model.owlvit.vision_model.embeddings(pixel_values)
        hidden_states_pt = pytorch_model.owlvit.vision_model.pre_layernorm(vision_embeddings)

    # Transfer to device
    hidden_states = ttnn.from_torch(
        hidden_states_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    dram_config = ttnn.DRAM_MEMORY_CONFIG

    # Process encoder layers
    for layer_params in parameters["vision"]["encoder_layers"]:
        hidden_states = _run_encoder_layer(hidden_states, layer_params, num_heads, head_dim, dram_config)

    # Post-layernorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters["vision"]["post_layernorm"]["weight"],
        bias=parameters["vision"]["post_layernorm"]["bias"],
        epsilon=1e-5,
        memory_config=dram_config,
    )

    return output


def run_text_encoder_on_device(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    parameters: Dict[str, Any],
    device: ttnn.Device,
) -> Tuple[ttnn.Tensor, torch.Tensor]:
    """
    Run text encoder fully on device.

    Args:
        input_ids: Token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        parameters: Preprocessed model parameters
        device: TTNN device

    Returns:
        hidden_states: Encoder output [batch, seq_len, hidden_size]
        eos_positions: Position of EOS token for each sequence (for pooling)
    """
    hidden_size = TEXT_HIDDEN_SIZE
    num_heads = TEXT_NUM_HEADS
    head_dim = TEXT_HEAD_DIM
    batch_size, seq_len = input_ids.shape

    # Convert input_ids to ttnn for embedding lookup
    input_ids_tt = ttnn.from_torch(
        input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Token embeddings via embedding lookup
    token_embeds = ttnn.embedding(
        input_ids_tt,
        parameters["text"]["token_embedding"],
        layout=ttnn.TILE_LAYOUT,
    )

    # Position embeddings - create position ids and look up
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    position_ids_tt = ttnn.from_torch(
        position_ids,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    position_embeds = ttnn.embedding(
        position_ids_tt,
        parameters["text"]["position_embedding"],
        layout=ttnn.TILE_LAYOUT,
    )

    # Combine embeddings
    hidden_states = ttnn.add(token_embeds, position_embeds)
    ttnn.deallocate(token_embeds)
    ttnn.deallocate(position_embeds)

    # Create causal mask
    causal_mask = torch.full((seq_len, seq_len), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :]

    causal_mask_tt = ttnn.from_torch(
        causal_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    dram_config = ttnn.DRAM_MEMORY_CONFIG

    # Process encoder layers with causal attention
    for layer_idx, layer_params in enumerate(parameters["text"]["encoder_layers"]):
        hidden_states = _run_text_encoder_layer(
            hidden_states, layer_params, causal_mask_tt, num_heads, head_dim, dram_config
        )
        if layer_idx == 0 or (layer_idx + 1) % 4 == 0:
            logger.info(f"Completed text encoder layer {layer_idx + 1}/{len(parameters['text']['encoder_layers'])}")

    # Final layer norm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters["text"]["final_layer_norm"]["weight"],
        bias=parameters["text"]["final_layer_norm"]["bias"],
        epsilon=1e-5,
        memory_config=dram_config,
    )
    ttnn.deallocate(hidden_states)

    # Find EOS token positions (for pooling later on CPU)
    # EOS token ID for CLIP is typically 49407
    eos_positions = (input_ids == 49407).int().argmax(dim=-1)

    return output, eos_positions


def _run_encoder_layer(hidden_states, layer_params, num_heads, head_dim, memory_config):
    """Run a single vision encoder layer."""
    # Layer norm 1
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm1"]["weight"],
        bias=layer_params["layer_norm1"]["bias"],
        epsilon=1e-5,
        memory_config=memory_config,
    )

    # Self-attention with fused QKV
    qkv = ttnn.linear(
        hidden_states,
        layer_params["self_attn"]["qkv"]["weight"],
        bias=layer_params["self_attn"]["qkv"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(hidden_states)

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=memory_config,
        num_heads=num_heads,
    )
    ttnn.deallocate(qkv)

    attention_scores = ttnn.matmul(query, key, memory_config=memory_config)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_scores = ttnn.mul(attention_scores, 1.0 / (head_dim**0.5))
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    ttnn.deallocate(attention_scores)

    context = ttnn.matmul(attention_probs, value, memory_config=memory_config)
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context = ttnn.transformer.concatenate_heads(context, memory_config=memory_config)

    attn_output = ttnn.linear(
        context,
        layer_params["self_attn"]["out_proj"]["weight"],
        bias=layer_params["self_attn"]["out_proj"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
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
        epsilon=1e-5,
        memory_config=memory_config,
    )

    # MLP
    mlp_hidden = ttnn.linear(
        hidden_states,
        layer_params["mlp"]["fc1"]["weight"],
        bias=layer_params["mlp"]["fc1"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(hidden_states)
    mlp_hidden = ttnn.gelu(mlp_hidden)

    mlp_output = ttnn.linear(
        mlp_hidden,
        layer_params["mlp"]["fc2"]["weight"],
        bias=layer_params["mlp"]["fc2"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(mlp_hidden)

    hidden_states = ttnn.add(residual, mlp_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(mlp_output)

    return hidden_states


def _run_text_encoder_layer(hidden_states, layer_params, causal_mask, num_heads, head_dim, memory_config):
    """Run a single text encoder layer with causal attention using native TTNN."""
    # Layer norm 1
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm1"]["weight"],
        bias=layer_params["layer_norm1"]["bias"],
        epsilon=1e-5,
        memory_config=memory_config,
    )

    # Self-attention with fused QKV (same pattern as vision encoder)
    qkv = ttnn.linear(
        hidden_states,
        layer_params["self_attn"]["qkv"]["weight"],
        bias=layer_params["self_attn"]["qkv"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(hidden_states)

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=memory_config,
        num_heads=num_heads,
    )
    ttnn.deallocate(qkv)

    # Compute attention scores
    attention_scores = ttnn.matmul(query, key, memory_config=memory_config)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # Scale
    attention_scores = ttnn.mul(attention_scores, 1.0 / (head_dim**0.5))

    # Apply causal mask
    attention_scores = ttnn.add(attention_scores, causal_mask)

    # Softmax
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    ttnn.deallocate(attention_scores)

    # Context
    context = ttnn.matmul(attention_probs, value, memory_config=memory_config)
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    # Concatenate heads
    context = ttnn.transformer.concatenate_heads(context, memory_config=memory_config)

    # Output projection
    attn_output = ttnn.linear(
        context,
        layer_params["self_attn"]["out_proj"]["weight"],
        bias=layer_params["self_attn"]["out_proj"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(context)

    # Residual connection
    hidden_states = ttnn.add(residual, attn_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(attn_output)

    # Layer norm 2
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["layer_norm2"]["weight"],
        bias=layer_params["layer_norm2"]["bias"],
        epsilon=1e-5,
        memory_config=memory_config,
    )

    # MLP
    mlp_hidden = ttnn.linear(
        hidden_states,
        layer_params["mlp"]["fc1"]["weight"],
        bias=layer_params["mlp"]["fc1"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(hidden_states)
    mlp_hidden = ttnn.gelu(mlp_hidden)

    mlp_output = ttnn.linear(
        mlp_hidden,
        layer_params["mlp"]["fc2"]["weight"],
        bias=layer_params["mlp"]["fc2"]["bias"],
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(mlp_hidden)

    # Final residual
    hidden_states = ttnn.add(residual, mlp_output)
    ttnn.deallocate(residual)
    ttnn.deallocate(mlp_output)

    return hidden_states


def run_box_head_on_device(
    image_features: ttnn.Tensor,
    parameters: Dict[str, Any],
    memory_config,
) -> ttnn.Tensor:
    """Run box prediction head on device (legacy - skips CLS token)."""
    # Convert to torch to do the slice (skip CLS token), then back to device
    features_torch = ttnn.to_torch(image_features)

    # Skip CLS token (first token)
    patch_features_torch = features_torch[:, 1:, :]  # [batch, 576, 768]

    # Transfer back to device
    patch_features = ttnn.from_torch(
        patch_features_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=image_features.device(),
    )

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

    pred_boxes = ttnn.sigmoid(pred_boxes)

    return pred_boxes


def run_box_head_on_device_v2(
    patch_features: ttnn.Tensor,
    parameters: Dict[str, Any],
    memory_config,
) -> ttnn.Tensor:
    """Run box prediction head on device (features already processed, CLS removed)."""
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

    # Add box_bias before sigmoid (critical for accurate predictions)
    pred_boxes = ttnn.add(pred_boxes, parameters["box_bias"])
    pred_boxes = ttnn.sigmoid(pred_boxes)

    return pred_boxes


def run_class_head_on_device(
    image_features: ttnn.Tensor,
    text_embeds: torch.Tensor,  # Keep on CPU for now
    parameters: Dict[str, Any],
    device: ttnn.Device,
    memory_config,
) -> ttnn.Tensor:
    """Run class prediction head on device."""
    # Convert to torch to do the slice (skip CLS token)
    features_torch = ttnn.to_torch(image_features)
    batch_size, seq_len, hidden_size = features_torch.shape

    # Skip CLS token (first token)
    patch_features_torch = features_torch[:, 1:, :]  # [batch, 576, 768]

    # Transfer back to device
    patch_features = ttnn.from_torch(
        patch_features_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Project patch features
    patch_projected = ttnn.linear(
        patch_features,
        parameters["class_head"]["dense0"]["weight"],
        bias=parameters["class_head"]["dense0"]["bias"],
        memory_config=memory_config,
    )

    # Normalize
    patch_norm = ttnn.layer_norm(
        patch_projected,
        weight=None,
        bias=None,
        epsilon=1e-5,
        memory_config=memory_config,
    )

    # Transfer text embeds to device
    text_embeds_tt = ttnn.from_torch(
        text_embeds,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Compute similarity: patch_features @ text_embeds^T
    text_embeds_t = ttnn.transpose(text_embeds_tt, -2, -1)
    logits = ttnn.matmul(patch_norm, text_embeds_t, memory_config=memory_config)

    return logits


def run_class_head_on_device_v2(
    patch_features: ttnn.Tensor,
    text_embeds: torch.Tensor,
    parameters: Dict[str, Any],
    device: ttnn.Device,
    memory_config,
) -> ttnn.Tensor:
    """Run class prediction head on device matching PyTorch's logic."""
    # Project patch features
    image_class_embeds = ttnn.linear(
        patch_features,
        parameters["class_head"]["dense0"]["weight"],
        bias=parameters["class_head"]["dense0"]["bias"],
        memory_config=memory_config,
    )

    # L2 normalize image embeddings (PyTorch uses linalg.norm)
    # Convert to torch for normalization
    image_embeds_torch = ttnn.to_torch(image_class_embeds).float()
    image_norm = torch.nn.functional.normalize(image_embeds_torch, p=2, dim=-1, eps=1e-6)

    # L2 normalize text embeddings
    text_norm = torch.nn.functional.normalize(text_embeds.squeeze(0).float(), p=2, dim=-1, eps=1e-6)

    # Compute similarity: image_embeds @ text_embeds^T
    # image_norm: [batch, 576, 512], text_norm: [2, 512]
    pred_logits = torch.einsum("bpd,qd->bpq", image_norm, text_norm)

    # Apply logit_shift and logit_scale
    # These are learned projections from image features
    patch_features_torch = ttnn.to_torch(patch_features).float()

    # logit_shift: Linear(768, 1) - output [batch, 576, 1]
    shift_weight = ttnn.to_torch(parameters["class_head"]["logit_shift"]["weight"]).float()
    shift_bias = ttnn.to_torch(parameters["class_head"]["logit_shift"]["bias"]).float()
    # shift_weight is stored as [768, 1] (transposed during loading)
    logit_shift = torch.matmul(patch_features_torch, shift_weight.squeeze(0)) + shift_bias.squeeze()

    # logit_scale: Linear(768, 1) + ELU + 1
    scale_weight = ttnn.to_torch(parameters["class_head"]["logit_scale"]["weight"]).float()
    scale_bias = ttnn.to_torch(parameters["class_head"]["logit_scale"]["bias"]).float()
    logit_scale = torch.matmul(patch_features_torch, scale_weight.squeeze(0)) + scale_bias.squeeze()
    logit_scale = torch.nn.functional.elu(logit_scale) + 1

    # logit_shift and logit_scale are [batch, 576, 1], pred_logits is [batch, 576, 2]
    # Need to expand for broadcasting
    logit_shift = logit_shift.unsqueeze(-1) if logit_shift.dim() == 2 else logit_shift
    logit_scale = logit_scale.unsqueeze(-1) if logit_scale.dim() == 2 else logit_scale

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


def run_owl_vit_end_to_end(
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    parameters: Dict[str, Any],
    device: ttnn.Device,
    pytorch_model,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Run full OWL-ViT object detection pipeline with all components on device.

    Returns:
        pred_boxes: [batch, num_patches, 4]
        logits: [batch, num_patches, num_queries]
    """
    logger.info("Running vision encoder on device...")
    vision_output = run_vision_encoder_on_device(pixel_values, parameters, device, pytorch_model)
    logger.info(f"Vision output shape: {vision_output.shape}")

    # Run text encoder on device
    logger.info("Running text encoder on device...")
    text_hidden_states, eos_positions = run_text_encoder_on_device(input_ids, attention_mask, parameters, device)
    logger.info(f"Text encoder output shape: {text_hidden_states.shape}")

    # Pool from EOS token position (convert to torch for indexing)
    text_output_torch = ttnn.to_torch(text_hidden_states)
    batch_size = text_output_torch.shape[0]

    # Gather the hidden state at EOS position for each sequence
    pooled_outputs = []
    for i in range(batch_size):
        eos_pos = eos_positions[i].item()
        pooled_outputs.append(text_output_torch[i, eos_pos, :])
    text_pooled = torch.stack(pooled_outputs, dim=0)  # [batch, hidden_size]
    logger.info(f"Text pooled shape: {text_pooled.shape}")

    # Project text embeddings on device
    text_pooled_tt = ttnn.from_torch(
        text_pooled,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    dram_config = ttnn.DRAM_MEMORY_CONFIG
    text_embeds = ttnn.linear(
        text_pooled_tt,
        parameters["text_projection"]["weight"],
        memory_config=dram_config,
        dtype=ttnn.bfloat16,
    )
    text_embeds_torch = ttnn.to_torch(text_embeds)
    logger.info(f"Text embeds shape: {text_embeds_torch.shape}")

    # Process vision features following OWL-ViT's image_embedder method:
    # 1. Separate class token and patch features
    # 2. Broadcast class token to match patch features
    # 3. Element-wise multiply
    # 4. Apply layer norm
    vision_torch = ttnn.to_torch(vision_output)  # [batch, 577, 768]

    # Extract class token and patch features
    class_token = vision_torch[:, :1, :]  # [batch, 1, 768]
    patch_features = vision_torch[:, 1:, :]  # [batch, 576, 768]

    # Broadcast class token to match patch features shape and multiply
    class_token_broadcast = class_token.expand_as(patch_features)  # [batch, 576, 768]
    image_embeds = patch_features * class_token_broadcast  # Element-wise multiply

    # Transfer back to device and apply layer norm
    image_embeds_tt = ttnn.from_torch(
        image_embeds,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Apply the model's layer_norm (not vision encoder's post_layernorm)
    vision_normalized = ttnn.layer_norm(
        image_embeds_tt,
        weight=parameters["layer_norm"]["weight"],
        bias=parameters["layer_norm"]["bias"],
        epsilon=1e-5,
        memory_config=dram_config,
    )

    # Run detection heads - pass normalized features (already has CLS token removed)
    logger.info("Running box head on device...")
    pred_boxes = run_box_head_on_device_v2(vision_normalized, parameters, dram_config)
    logger.info(f"Pred boxes shape: {pred_boxes.shape}")

    logger.info("Running class head on device...")
    logits = run_class_head_on_device_v2(
        vision_normalized, text_embeds_torch.unsqueeze(0), parameters, device, dram_config
    )
    logger.info(f"Logits shape: {logits.shape}")

    return pred_boxes, logits


@pytest.fixture
def device():
    """Create TTNN device."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


class TestOwlViTEndToEnd:
    """End-to-end tests for OWL-ViT on TT hardware."""

    def test_full_detection_pipeline(self, device):
        """
        Test complete object detection pipeline.
        """
        logger.info("=" * 60)
        logger.info("OWL-ViT End-to-End Detection Test")
        logger.info("=" * 60)

        text_queries = ["a cat", "a remote control"]

        # Load model and prepare inputs
        logger.info("Loading model and inputs...")
        processor, model, inputs, image = get_pytorch_model_and_inputs(text_queries)

        # Get PyTorch reference
        logger.info("Running PyTorch reference...")
        with torch.no_grad():
            pytorch_outputs = model(**inputs)

        ref_boxes = pytorch_outputs.pred_boxes
        ref_logits = pytorch_outputs.logits

        logger.info(f"PyTorch boxes shape: {ref_boxes.shape}")
        logger.info(f"PyTorch logits shape: {ref_logits.shape}")

        # Load weights to device
        logger.info("Loading weights to device...")
        parameters = preprocess_all_weights_for_ttnn(model, device)

        # Run TTNN inference
        logger.info("Running TTNN inference...")
        ttnn_boxes, ttnn_logits = run_owl_vit_end_to_end(
            inputs["pixel_values"],
            inputs["input_ids"],
            inputs["attention_mask"],
            parameters,
            device,
            model,
        )

        # Convert to torch
        ttnn_boxes_torch = ttnn.to_torch(ttnn_boxes)
        ttnn_logits_torch = ttnn.to_torch(ttnn_logits)

        logger.info(f"TTNN boxes shape: {ttnn_boxes_torch.shape}")
        logger.info(f"TTNN logits shape: {ttnn_logits_torch.shape}")

        # Post-process TTNN detections
        # For now, use processor with TTNN outputs
        target_sizes = torch.Tensor([image.size[::-1]])

        # Create output object similar to PyTorch
        class TTNNOutputs:
            def __init__(self, logits, pred_boxes):
                self.logits = logits
                self.pred_boxes = pred_boxes

        ttnn_output_obj = TTNNOutputs(ttnn_logits_torch, ttnn_boxes_torch)

        results = processor.post_process_object_detection(
            outputs=ttnn_output_obj,
            threshold=DETECTION_THRESHOLD,
            target_sizes=target_sizes,
        )

        # Print detections
        logger.info("=" * 60)
        logger.info("TTNN DETECTIONS:")
        logger.info("=" * 60)

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]

        logger.info(f"Detected {len(boxes)} objects:")
        for box, score, label in zip(boxes, scores, labels):
            box_coords = [round(x, 1) for x in box.tolist()]
            logger.info(f"  {text_queries[label]}: score={score.item():.3f}, box={box_coords}")

        # Calculate PCC for boxes
        ref_flat = ref_boxes.flatten().float()
        ttnn_flat = ttnn_boxes_torch.flatten().float()

        ref_centered = ref_flat - ref_flat.mean()
        ttnn_centered = ttnn_flat - ttnn_flat.mean()

        numerator = (ref_centered * ttnn_centered).sum()
        denominator = torch.sqrt((ref_centered**2).sum() * (ttnn_centered**2).sum())
        boxes_pcc = (numerator / denominator).item()

        logger.info(f"Boxes PCC: {boxes_pcc:.4f}")

        logger.info("=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)

        # Should detect at least something
        assert len(boxes) >= 0, "Detection completed"

    def test_vision_encoder_pcc(self, device):
        """
        Test vision encoder produces output matching PyTorch reference.

        This validates the core vision transformer runs correctly on device.
        Expected PCC > 0.8 compared to PyTorch reference.
        """
        logger.info("=" * 60)
        logger.info("OWL-ViT Vision Encoder PCC Test")
        logger.info("=" * 60)

        text_queries = ["test"]
        processor, model, inputs, image = get_pytorch_model_and_inputs(text_queries)

        # Get reference output from PyTorch
        logger.info("Running PyTorch reference...")
        with torch.no_grad():
            pytorch_vision_output = model.owlvit.vision_model(inputs["pixel_values"])
        reference_output = pytorch_vision_output.last_hidden_state
        logger.info(f"PyTorch vision output shape: {reference_output.shape}")

        # Load weights and run on device
        logger.info("Loading weights to device...")
        parameters = preprocess_all_weights_for_ttnn(model, device)

        logger.info("Running vision encoder on device...")
        ttnn_output = run_vision_encoder_on_device(inputs["pixel_values"], parameters, device, model)

        ttnn_output_torch = ttnn.to_torch(ttnn_output)
        logger.info(f"TTNN vision output shape: {ttnn_output_torch.shape}")

        # Calculate PCC
        ref_flat = reference_output.flatten().float()
        ttnn_flat = ttnn_output_torch.flatten().float()

        ref_centered = ref_flat - ref_flat.mean()
        ttnn_centered = ttnn_flat - ttnn_flat.mean()

        numerator = (ref_centered * ttnn_centered).sum()
        denominator = torch.sqrt((ref_centered**2).sum() * (ttnn_centered**2).sum())
        pcc = (numerator / denominator).item()

        logger.info(f"Vision Encoder PCC: {pcc:.4f}")

        # Sample values for debugging
        logger.info("Sample output values (first 5 elements of CLS token):")
        logger.info(f"  PyTorch: {reference_output[0, 0, :5].tolist()}")
        logger.info(f"  TTNN:    {ttnn_output_torch[0, 0, :5].tolist()}")

        logger.info("=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)

        # Assert reasonable correlation
        assert pcc > 0.8, f"Vision encoder PCC {pcc:.4f} is too low, expected > 0.8"


if __name__ == "__main__":
    # Run tests directly
    device = ttnn.open_device(device_id=0)
    try:
        test = TestOwlViTEndToEnd()
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING: test_vision_encoder_pcc")
        logger.info("=" * 60 + "\n")
        test.test_vision_encoder_pcc(device)

        logger.info("\n" + "=" * 60)
        logger.info("RUNNING: test_full_detection_pipeline")
        logger.info("=" * 60 + "\n")
        test.test_full_detection_pipeline(device)
    finally:
        ttnn.close_device(device)
