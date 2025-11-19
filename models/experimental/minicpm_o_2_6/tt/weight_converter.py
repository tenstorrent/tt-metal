# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight Format Converter for MiniCPM-o-2_6

Converts HuggingFace weight formats to TTNN-compatible formats:
- Linear layers: transpose from [out, in] to [in, out]
- Conv layers: keep same format
- Handle bias terms correctly

This handles the format differences between PyTorch (HuggingFace) and TTNN tensor layouts.
"""

import torch
from typing import Dict
from loguru import logger


def convert_linear_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert linear layer weights from PyTorch to TTNN format.

    PyTorch format: [out_features, in_features]
    TTNN format: [in_features, out_features] (transposed)

    Args:
        weights: Dictionary of weights with PyTorch naming

    Returns:
        Dictionary with weights converted to TTNN format
    """
    converted = {}

    for key, tensor in weights.items():
        if "weight" in key and len(tensor.shape) == 2:
            # Skip embedding weights - they should remain [vocab_size, hidden_size]
            if "embed_tokens" in key:
                converted[key] = tensor
                logger.debug(f"Kept embedding {key}: {tensor.shape} (no transpose)")
            else:
                # Linear layer weights need transposition
                converted[key] = tensor.t()
                logger.debug(f"Transposed {key}: {tensor.shape} → {converted[key].shape}")
        else:
            # Bias terms and other tensors stay the same
            converted[key] = tensor

    return converted


def convert_conv_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert conv layer weights from PyTorch to TTNN format.

    PyTorch format: [out_channels, in_channels, kernel_size...]
    TTNN format: Same as PyTorch (no transpose needed)

    Args:
        weights: Dictionary of weights

    Returns:
        Dictionary with conv weights (unchanged)
    """
    # Conv weights don't need conversion - same format in PyTorch and TTNN
    return weights.copy()


def convert_attention_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert attention layer weights.

    Q/K/V/O projections: same as linear layers (transpose needed)
    Relative position embeddings: no transpose needed

    Args:
        weights: Dictionary of weights

    Returns:
        Dictionary with attention weights converted
    """
    converted = {}

    attention_patterns = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.out_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.k_proj.weight",
        "cross_attn.v_proj.weight",
        "cross_attn.out_proj.weight",
    ]

    for key, tensor in weights.items():
        if any(pattern in key for pattern in attention_patterns):
            # Attention projection weights need transposition
            converted[key] = tensor.t()
            logger.debug(f"Transposed attention {key}: {tensor.shape} → {converted[key].shape}")
        else:
            # Other attention weights (biases, embeddings) stay the same
            converted[key] = tensor

    return converted


def convert_mlp_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert MLP/feedforward layer weights.

    Gate/Up/Down projections: same as linear layers (transpose needed)

    Args:
        weights: Dictionary of weights

    Returns:
        Dictionary with MLP weights converted
    """
    converted = {}

    mlp_patterns = [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
    ]

    for key, tensor in weights.items():
        if any(pattern in key for pattern in mlp_patterns):
            # MLP projection weights need transposition
            converted[key] = tensor.t()
            logger.debug(f"Transposed MLP {key}: {tensor.shape} → {converted[key].shape}")
        else:
            # Other MLP weights (biases) stay the same
            converted[key] = tensor

    return converted


def convert_component_weights(
    component_weights: Dict[str, torch.Tensor], component_type: str
) -> Dict[str, torch.Tensor]:
    """
    Convert weights for a specific component type.

    Args:
        component_weights: Raw component weights from HuggingFace
        component_type: Type of component ('qwen', 'whisper', 'audio_projector', etc.)

    Returns:
        Weights converted to TTNN format
    """
    logger.info(f"Converting {component_type} weights: {len(component_weights)} tensors")

    # Special-case: resampler uses PyTorch-style combined in_proj (QKV) tensors.
    # Split combined in_proj_weight / in_proj_bias into separate q/k/v projection tensors
    # so downstream conversion routines (which expect q_proj/k_proj/v_proj names) work.
    if component_type == "resampler":
        try:
            # Work on a shallow copy to avoid mutating caller dict
            cw = dict(component_weights)
            in_proj_key = "resampler.attn.in_proj_weight"
            in_proj_bias_key = "resampler.attn.in_proj_bias"
            if in_proj_key in cw:
                w = cw.pop(in_proj_key)
                b = cw.pop(in_proj_bias_key, None)
                if w.ndim == 2 and w.shape[0] % 3 == 0:
                    E = w.shape[0] // 3
                    # Split Q/K/V
                    cw["resampler.attn.q_proj.weight"] = w[0:E, :].contiguous()
                    cw["resampler.attn.k_proj.weight"] = w[E : 2 * E, :].contiguous()
                    cw["resampler.attn.v_proj.weight"] = w[2 * E : 3 * E, :].contiguous()
                    if b is not None and b.ndim == 1 and b.shape[0] == 3 * E:
                        cw["resampler.attn.q_proj.bias"] = b[0:E].contiguous()
                        cw["resampler.attn.k_proj.bias"] = b[E : 2 * E].contiguous()
                        cw["resampler.attn.v_proj.bias"] = b[2 * E : 3 * E].contiguous()
                    logger.info(f"Split resampler in_proj into q/k/v with E={E}")
                    # Replace component_weights with the expanded dict for downstream conversion
                    component_weights = cw
                else:
                    logger.warning("Unexpected shape for resampler.in_proj_weight; skipping split")
        except Exception as e:
            logger.warning(f"Failed to split resampler combined QKV tensors: {e}")

    # First apply key mapping for Qwen weights
    if component_type == "qwen":
        component_weights = map_minicpm_qwen_keys(component_weights)

    if component_type in ["qwen", "whisper", "audio_projector", "vision", "siglip", "resampler", "chattts"]:
        # Apply general linear/attention/mlp conversions
        converted = convert_linear_weights(component_weights)
        converted = convert_attention_weights(converted)
        converted = convert_mlp_weights(converted)
        converted = convert_conv_weights(converted)

    elif component_type == "dvae":
        # DVAE has conv layers - no transpose needed
        converted = convert_conv_weights(component_weights)

    else:
        logger.warning(f"Unknown component type: {component_type}")
        converted = component_weights.copy()

    logger.info(f"Converted {component_type} weights: {len(converted)} tensors")
    return converted


def map_minicpm_qwen_keys(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map MiniCPM Qwen weight keys to TT transformers format.

    MiniCPM uses keys like: llm.model.embed_tokens.weight
    TT expects keys like: tok_embeddings.weight
    """
    import re

    # Define key mappings for MiniCPM Qwen component
    replacements = [
        # Remove llm.model prefix and map embed_tokens to tok_embeddings
        (r"^llm\.model\.embed_tokens", "tok_embeddings"),
        (r"^llm\.model\.layers\.(\d+)\.input_layernorm", r"layers.\1.attention_norm"),
        (r"^llm\.model\.layers\.(\d+)\.post_attention_layernorm", r"layers.\1.ffn_norm"),
        (r"^llm\.model\.layers\.(\d+)\.self_attn\.q_proj", r"layers.\1.attention.wq"),
        (r"^llm\.model\.layers\.(\d+)\.self_attn\.k_proj", r"layers.\1.attention.wk"),
        (r"^llm\.model\.layers\.(\d+)\.self_attn\.v_proj", r"layers.\1.attention.wv"),
        (r"^llm\.model\.layers\.(\d+)\.self_attn\.o_proj", r"layers.\1.attention.wo"),
        (r"^llm\.model\.layers\.(\d+)\.mlp\.gate_proj", r"layers.\1.feed_forward.w1"),
        (r"^llm\.model\.layers\.(\d+)\.mlp\.up_proj", r"layers.\1.feed_forward.w3"),
        (r"^llm\.model\.layers\.(\d+)\.mlp\.down_proj", r"layers.\1.feed_forward.w2"),
        (r"^llm\.model\.norm", "norm"),
        (r"^llm\.lm_head", "output"),
    ]

    mapped_weights = {}
    for key, value in weights.items():
        mapped_key = key
        for pattern, replacement in replacements:
            mapped_key = re.sub(pattern, replacement, mapped_key)
        mapped_weights[mapped_key] = value
        # Preserve original MiniCPM cross-attention keys with 'llm.' prefix so
        # downstream loaders (e.g. MiniCPMQwenModel) that expect the original
        # 'llm.model.layers.*.cross_attn.*' naming can find them.
        # If the original key contains 'cross_attn' but does not already start
        # with 'llm.', create a copy prefixed with 'llm.'.
        if "cross_attn" in key and not key.startswith("llm."):
            llm_prefixed = f"llm.{key}"
            if llm_prefixed not in mapped_weights:
                mapped_weights[llm_prefixed] = value

    logger.info(f"Mapped {len(weights)} Qwen keys, result has {len(mapped_weights)} keys")
    return mapped_weights


def validate_weight_shapes(
    original_weights: Dict[str, torch.Tensor], converted_weights: Dict[str, torch.Tensor]
) -> bool:
    """
    Validate that weight shapes are correct after conversion.

    Args:
        original_weights: Original weights from HuggingFace
        converted_weights: Converted weights for TTNN

    Returns:
        True if validation passes
    """
    logger.info("Validating weight shape conversions...")

    issues = []

    for key in original_weights.keys():
        if key not in converted_weights:
            issues.append(f"Missing key: {key}")
            continue

        orig_shape = original_weights[key].shape
        conv_shape = converted_weights[key].shape

        # Check if this should be transposed
        should_transpose = (
            "weight" in key
            and len(orig_shape) == 2
            and any(
                pattern in key
                for pattern in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "out_proj",
                    "o_proj",  # attention
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "fc1",
                    "fc2",  # MLP
                    "linear1",
                    "linear2",  # audio projector
                ]
            )
        )

        if should_transpose:
            expected_shape = (orig_shape[1], orig_shape[0])  # transposed
            if conv_shape != expected_shape:
                issues.append(f"Wrong transpose for {key}: {orig_shape} → {conv_shape}, expected {expected_shape}")
        else:
            # Should stay the same
            if conv_shape != orig_shape:
                issues.append(f"Unexpected shape change for {key}: {orig_shape} → {conv_shape}")

    if issues:
        logger.error(f"Weight validation failed with {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            logger.error(f"  {issue}")
        return False

    logger.info("✅ Weight shape validation passed")
    return True


def save_converted_weights(converted_weights: Dict[str, torch.Tensor], output_path: str) -> None:
    """
    Save converted weights to file.

    Args:
        converted_weights: Weights in TTNN format
        output_path: Output file path
    """
    torch.save(converted_weights, output_path)

    # Calculate size
    size_mb = sum(tensor.numel() * tensor.element_size() for tensor in converted_weights.values()) / (1024 * 1024)

    logger.info(f"✅ Saved converted weights: {len(converted_weights)} tensors, ~{size_mb:.1f} MB → {output_path}")


def convert_and_save_component_weights(
    component_weights: Dict[str, torch.Tensor], component_type: str, output_dir: str = "./minicpm_weights"
) -> str:
    """
    Convert component weights and save to file.

    Args:
        component_weights: Raw component weights
        component_type: Component type
        output_dir: Output directory

    Returns:
        Path to saved weights file
    """
    # Convert weights
    converted_weights = convert_component_weights(component_weights, component_type)

    # Validate shapes
    if not validate_weight_shapes(component_weights, converted_weights):
        raise ValueError(f"Weight validation failed for {component_type}")

    # Save converted weights
    import os

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{component_type}_weights_ttnn.pt")

    save_converted_weights(converted_weights, output_path)

    return output_path


# Convenience functions for common conversions


def convert_qwen_weights(qwen_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Qwen LLM weights to TTNN format."""
    return convert_component_weights(qwen_weights, "qwen")


def convert_whisper_weights(whisper_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Whisper encoder weights to TTNN format."""
    return convert_component_weights(whisper_weights, "whisper")


def convert_audio_projector_weights(audio_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Audio Projector weights to TTNN format."""
    return convert_component_weights(audio_weights, "audio_projector")


def convert_vision_weights(vision_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Vision (SigLip) weights to TTNN format."""
    return convert_component_weights(vision_weights, "vision")


if __name__ == "__main__":
    # Test with generated weights
    try:
        from .weight_generator import generate_audio_projector_weights
    except ImportError:
        from weight_generator import generate_audio_projector_weights

    logger.info("Testing weight converter with generated audio projector weights...")

    # Generate test weights
    original_weights = generate_audio_projector_weights()

    print("Original weights:")
    for key, tensor in original_weights.items():
        print(f"  {key}: {tensor.shape}")

    # Convert weights
    converted_weights = convert_audio_projector_weights(original_weights)

    print("\nConverted weights:")
    for key, tensor in converted_weights.items():
        print(f"  {key}: {tensor.shape}")

    # Validate
    if validate_weight_shapes(original_weights, converted_weights):
        print("\n✅ Weight conversion test passed!")
    else:
        print("\n❌ Weight conversion test failed!")
        exit(1)
