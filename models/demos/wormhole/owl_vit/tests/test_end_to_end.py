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
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
import torch
from loguru import logger
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.demos.wormhole.owl_vit.tt.ttnn_owl_vit import (
    OwlViTTTNNConfig,
    run_box_head,
    run_class_head,
    run_text_encoder_layer_sharded,
    run_vision_encoder_layer_sharded,
)

# Instantiate config
ttnn_config = OwlViTTTNNConfig()


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
    try:
        return Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to download image: {e}")
        pytest.skip("Test image could not be downloaded (offline?)")


def get_pytorch_model_and_inputs(text_queries: list[str], image: Image.Image = None) -> Tuple:
    """Load PyTorch model and prepare inputs for inference."""
    try:
        processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
        model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    except Exception as e:
        logger.warning(f"Failed to load HF model: {e}")
        pytest.skip("HuggingFace model could not be loaded coverage (offline/no token?)")
    model.eval()

    if image is None:
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

    The sharded version uses L1 memory and full core grid
    """
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

    compute_kernel_config = ttnn_config.get_compute_kernel_config()

    # Use sharded version for better performance
    # Process encoder layers with L1 sharding
    for layer_params in parameters["vision"]["encoder_layers"]:
        hidden_states = run_vision_encoder_layer_sharded(
            hidden_states, layer_params, ttnn_config, device, compute_kernel_config
        )

    # Post-layernorm - use L1 for consistency
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters["vision"]["post_layernorm"]["weight"],
        bias=parameters["vision"]["post_layernorm"]["bias"],
        epsilon=1e-5,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
    batch_size, seq_len = input_ids.shape

    # Pad to multiple of 128 for SDPA compatibility
    # SDPA chunk size is 128, so sequence length must be divisible by 128
    pad_to = 128
    if seq_len % pad_to != 0:
        new_len = ((seq_len // pad_to) + 1) * pad_to
        padding_len = new_len - seq_len

        # Pad input_ids with 0 (or any token, padding mask will handle it)
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_len), value=0)

        # Pad attention_mask with 0 (masked)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_len), value=0)

        seq_len = new_len

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
    # [1, 1, S, S]
    causal_mask = causal_mask[None, None, :, :]

    # Create padding mask from attention_mask
    # attention_mask is [batch, seq_len] (1 for keep, 0 for discard)
    # We want to add -inf to padded positions
    # [batch, 1, 1, seq_len]
    padding_mask = (1.0 - attention_mask) * -1e9
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)

    # Combine masks
    # Both are additive masks: -inf/large_neg for masked, 0 for visible
    combined_mask = causal_mask + padding_mask

    causal_mask_tt = ttnn.from_torch(
        combined_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    dram_config = ttnn.DRAM_MEMORY_CONFIG
    compute_kernel_config = ttnn_config.get_compute_kernel_config()

    # Process encoder layers with causal attention (Optimized L1 + SDPA)
    for layer_idx, layer_params in enumerate(parameters["text"]["encoder_layers"]):
        hidden_states = run_text_encoder_layer_sharded(
            hidden_states, layer_params, causal_mask_tt, ttnn_config, device, compute_kernel_config
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
    pred_boxes = run_box_head(vision_normalized, parameters, dram_config)
    logger.info(f"Pred boxes shape: {pred_boxes.shape}")

    logger.info("Running class head on device...")
    logits = run_class_head(
        vision_normalized,
        text_embeds_torch,
        parameters,
        device,
        dram_config,
        ttnn_config,
    )
    logger.info(f"Logits shape: {logits.shape}")

    return pred_boxes, logits


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
        assert boxes_pcc > 0.9, f"Box coordinates PCC too low: {boxes_pcc}"
        assert len(boxes) > 0, "No objects detected with default threshold"

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
