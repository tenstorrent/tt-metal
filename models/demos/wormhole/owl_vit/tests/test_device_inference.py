# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
OWL-ViT Device Inference Test

This module runs actual OWL-ViT inference on Tenstorrent N150/N300 hardware.
It loads model weights, preprocesses them for TTNN, and executes the model.
"""

import sys
from typing import Any, Dict

import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

import ttnn

sys.path.insert(0, "/root/tt-metal")


# Test constants
MODEL_NAME = "google/owlvit-base-patch32"
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
BATCH_SIZE = 1


def load_image(url: str) -> Image.Image:
    """Load test image from URL."""
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")


def get_pytorch_model_and_inputs():
    """Load PyTorch model and prepare inputs."""
    processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()

    image = load_image(TEST_IMAGE_URL)
    text_queries = ["a photo of a cat", "a photo of a dog"]

    inputs = processor(text=[text_queries], images=image, return_tensors="pt")

    return processor, model, inputs, text_queries


def preprocess_vision_weights_for_ttnn(model, device):
    """
    Preprocess vision model weights for TTNN.

    This converts PyTorch weights to TTNN tensors with appropriate layouts.
    """
    parameters = {}
    state_dict = model.state_dict()

    # Vision patch embedding (Conv2d in PyTorch, we convert to linear)
    patch_weight = state_dict["owlvit.vision_model.embeddings.patch_embedding.weight"]  # [768, 3, 32, 32]

    # Reshape for linear projection: Conv2d weight [out, in, kh, kw] -> [patch_size*patch_size*in_channels, out]
    out_channels, in_channels, kh, kw = patch_weight.shape
    # Flatten conv weight: [768, 3, 32, 32] -> [768, 3072] -> transpose -> [3072, 768]
    patch_weight_linear = patch_weight.reshape(out_channels, -1).T.contiguous()  # [3072, 768]

    parameters["patch_embedding"] = {
        "weight": ttnn.from_torch(
            patch_weight_linear,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        # No bias in OWL-ViT patch embedding
        "bias": None,
    }

    # Class token and position embeddings
    parameters["class_embedding"] = ttnn.from_torch(
        state_dict["owlvit.vision_model.embeddings.class_embedding"].unsqueeze(0).unsqueeze(0),  # [1, 1, 768]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    parameters["position_embedding"] = ttnn.from_torch(
        state_dict["owlvit.vision_model.embeddings.position_embedding.weight"].unsqueeze(0),  # [1, 577, 768]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Pre and post layer norms
    parameters["pre_layernorm"] = {
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

    parameters["post_layernorm"] = {
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

    # Encoder layers
    parameters["encoder_layers"] = []
    num_layers = model.config.vision_config.num_hidden_layers

    for i in range(num_layers):
        prefix = f"owlvit.vision_model.encoder.layers.{i}"

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

        # Fuse into single weight [3*hidden_size, hidden_size]
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

        layer_params["self_attn"] = {
            "qkv": {
                "weight": ttnn.from_torch(
                    qkv_weight.T,  # Transpose for TTNN
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

        # MLP
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

        parameters["encoder_layers"].append(layer_params)

    logger.info(f"Loaded {num_layers} vision encoder layers onto device")
    return parameters


def run_vision_encoder_ttnn(
    pixel_values: torch.Tensor,
    parameters: Dict[str, Any],
    device: ttnn.Device,
    pytorch_model,  # Add model for reference patch embedding
) -> ttnn.Tensor:
    """
    Run vision encoder forward pass on TTNN device.

    For initial bring-up, we compute patch embeddings using PyTorch and run
    the transformer encoder layers on the TT device.

    Args:
        pixel_values: Input images [batch, channels, height, width]
        parameters: Preprocessed model parameters
        device: TTNN device
        pytorch_model: PyTorch model for reference

    Returns:
        Vision encoder output tensor
    """
    batch_size, channels, height, width = pixel_values.shape
    patch_size = 32
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w
    hidden_size = 768

    logger.info(f"Input shape: {pixel_values.shape}")
    logger.info(f"Num patches: {num_patches} ({num_patches_h}x{num_patches_w})")

    # For initial bring-up: use PyTorch for patch embedding, run encoder on device
    # This allows us to isolate and test the transformer layers specifically
    with torch.no_grad():
        # Get embeddings from PyTorch (patch + position + pre_layernorm)
        vision_embeddings = pytorch_model.owlvit.vision_model.embeddings(pixel_values)
        # Apply pre-layernorm
        hidden_states_pt = pytorch_model.owlvit.vision_model.pre_layernorm(vision_embeddings)

    logger.info(f"PyTorch embedding shape: {hidden_states_pt.shape}")

    # Transfer embeddings to device
    hidden_states = ttnn.from_torch(
        hidden_states_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    logger.info(f"Transferred embeddings to device: {hidden_states.shape}")

    # Process through encoder layers on TT device
    num_heads = 12
    head_dim = hidden_size // num_heads

    # Use DRAM for large tensors to avoid L1 overflow
    dram_config = ttnn.DRAM_MEMORY_CONFIG

    for layer_idx, layer_params in enumerate(parameters["encoder_layers"]):
        # Layer norm 1
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=layer_params["layer_norm1"]["weight"],
            bias=layer_params["layer_norm1"]["bias"],
            epsilon=1e-5,
            memory_config=dram_config,
        )

        # Self-attention with fused QKV
        qkv = ttnn.linear(
            hidden_states,
            layer_params["self_attn"]["qkv"]["weight"],
            bias=layer_params["self_attn"]["qkv"]["bias"],
            memory_config=dram_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(hidden_states)

        # Split and reshape for multi-head attention
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            memory_config=dram_config,
            num_heads=num_heads,
        )
        ttnn.deallocate(qkv)

        # Attention scores
        attention_scores = ttnn.matmul(query, key, memory_config=dram_config)
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        attention_scores = ttnn.mul(attention_scores, 1.0 / (head_dim**0.5))
        attention_probs = ttnn.softmax(attention_scores, dim=-1)
        ttnn.deallocate(attention_scores)

        # Attention output
        context = ttnn.matmul(attention_probs, value, memory_config=dram_config)
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(value)

        context = ttnn.transformer.concatenate_heads(context, memory_config=dram_config)

        # Output projection
        attn_output = ttnn.linear(
            context,
            layer_params["self_attn"]["out_proj"]["weight"],
            bias=layer_params["self_attn"]["out_proj"]["bias"],
            memory_config=dram_config,
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
            memory_config=dram_config,
        )

        # MLP
        mlp_hidden = ttnn.linear(
            hidden_states,
            layer_params["mlp"]["fc1"]["weight"],
            bias=layer_params["mlp"]["fc1"]["bias"],
            memory_config=dram_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(hidden_states)
        mlp_hidden = ttnn.gelu(mlp_hidden)

        mlp_output = ttnn.linear(
            mlp_hidden,
            layer_params["mlp"]["fc2"]["weight"],
            bias=layer_params["mlp"]["fc2"]["bias"],
            memory_config=dram_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(mlp_hidden)

        # Residual connection
        hidden_states = ttnn.add(residual, mlp_output)
        ttnn.deallocate(residual)
        ttnn.deallocate(mlp_output)

        if layer_idx == 0 or (layer_idx + 1) % 4 == 0:
            logger.info(f"Completed encoder layer {layer_idx + 1}/{len(parameters['encoder_layers'])}")

    # Post-layernorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters["post_layernorm"]["weight"],
        bias=parameters["post_layernorm"]["bias"],
        epsilon=1e-5,
        memory_config=dram_config,
    )
    ttnn.deallocate(hidden_states)

    logger.info(f"Vision encoder output shape: {output.shape}")

    return output


@pytest.fixture
def device():
    """Create TTNN device."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


class TestOwlViTDeviceInference:
    """Tests for OWL-ViT running on TT hardware."""

    def test_vision_encoder_on_device(self, device):
        """
        Test vision encoder forward pass on TT device.

        This is the core test for Stage 1 bring-up.
        """
        logger.info("=" * 60)
        logger.info("OWL-ViT Vision Encoder Device Test")
        logger.info("=" * 60)

        # Load PyTorch model
        logger.info("Loading PyTorch model...")
        processor, model, inputs, text_queries = get_pytorch_model_and_inputs()

        # Get reference output from PyTorch
        logger.info("Running PyTorch reference inference...")
        with torch.no_grad():
            pytorch_vision_output = model.owlvit.vision_model(inputs["pixel_values"])

        reference_output = pytorch_vision_output.last_hidden_state
        logger.info(f"PyTorch output shape: {reference_output.shape}")

        # Preprocess weights for TTNN
        logger.info("Preprocessing weights for TTNN...")
        parameters = preprocess_vision_weights_for_ttnn(model, device)

        # Run TTNN inference
        logger.info("Running TTNN inference on device...")
        ttnn_output = run_vision_encoder_ttnn(
            inputs["pixel_values"],
            parameters,
            device,
            model,  # Pass model for PyTorch embedding computation
        )

        # Convert back to torch for comparison
        ttnn_output_torch = ttnn.to_torch(ttnn_output)
        logger.info(f"TTNN output shape: {ttnn_output_torch.shape}")

        # Compare outputs
        # Calculate PCC (Pearson Correlation Coefficient)
        ref_flat = reference_output.flatten().float()
        ttnn_flat = ttnn_output_torch.flatten().float()

        ref_centered = ref_flat - ref_flat.mean()
        ttnn_centered = ttnn_flat - ttnn_flat.mean()

        numerator = (ref_centered * ttnn_centered).sum()
        denominator = torch.sqrt((ref_centered**2).sum() * (ttnn_centered**2).sum())
        pcc = (numerator / denominator).item()

        logger.info(f"PCC between PyTorch and TTNN: {pcc:.6f}")

        # Print sample values
        logger.info("Sample output values (first 5 elements of CLS token):")
        logger.info(f"  PyTorch: {reference_output[0, 0, :5].tolist()}")
        logger.info(f"  TTNN:    {ttnn_output_torch[0, 0, :5].tolist()}")

        logger.info("=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)

        # Assert reasonable correlation
        assert pcc > 0.8, f"PCC {pcc} is too low, expected > 0.8"


if __name__ == "__main__":
    # Run the test directly
    device = ttnn.open_device(device_id=0)
    try:
        test = TestOwlViTDeviceInference()
        test.test_vision_encoder_on_device(device)
    finally:
        ttnn.close_device(device)
