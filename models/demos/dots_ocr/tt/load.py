# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading utilities for Dots OCR with support for both text decoder
and full TTNN vision transformer weights from real dots.mocr checkpoints.
"""

from __future__ import annotations

from loguru import logger

from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta_no_qkv_permute,
    load_hf_state_dict_filtered,
    standardize_hf_keys_multimodal,
)


def load_dots_text_state_dict(hf_model_id_or_dir: str, *, head_dim: int, n_heads: int, n_kv_heads: int):
    """
    Load and convert the *text* (decoder) weights for Dots OCR.

    This is the original function enhanced with better error handling
    and logging for real dots.mocr checkpoints.
    """
    # Try to only pull language weights - Dots uses multiple naming patterns
    key_prefixes = (
        "model.",  # Standard HF pattern
        "language_model.",  # Some multimodal wrappers
        "text_model.",  # Alternative naming
    )

    logger.info(f"Loading text weights from {hf_model_id_or_dir} with prefixes: {key_prefixes}")

    loaded = load_hf_state_dict_filtered(hf_model_id_or_dir, key_prefixes)
    if not loaded:
        logger.warning(f"No weights found for prefixes {key_prefixes}, trying broader search...")
        # Fallback: load everything and filter
        loaded = load_hf_state_dict_filtered(hf_model_id_or_dir, [""])

    if not loaded:
        raise FileNotFoundError(f"No weights found for prefixes {key_prefixes} in {hf_model_id_or_dir}")

    logger.info(f"Loaded {len(loaded)} text-related tensors before standardization")

    loaded = standardize_hf_keys_multimodal(loaded)

    # Strip a leading "model." if present; convert expects HF-like keys.
    stripped = {}
    for k, v in loaded.items():
        if k.startswith("model."):
            stripped[k[len("model.") :]] = v
        elif k.startswith("language_model."):
            stripped[k[len("language_model.") :]] = v
        else:
            stripped[k] = v

    converted = convert_hf_to_meta_no_qkv_permute(stripped, head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)

    # Re-add "model." prefix that TTTransformer expects by default.
    final = {f"model.{k}": v for k, v in converted.items()}

    logger.info(f"Successfully loaded {len(final)} converted text weights for Dots OCR")
    logger.info(f"Key examples: {list(final.keys())[:5] if final else 'None'}")

    return final


def load_dots_vision_state_dict(hf_model_id_or_dir: str) -> dict:
    """
    Load vision weights for the full TTNN VisionTransformerTT.

    This extracts weights for:
    - Patch embedding
    - 42 vision blocks (attention, mlp, norms)
    - Final patch merger
    """
    logger.info(f"Loading vision weights from {hf_model_id_or_dir}")

    # Vision weights are typically under "vision_tower" or "visual" prefix
    vision_prefixes = (
        "vision_tower.",
        "visual.",
        "vision_model.",
        "model.vision_tower.",  # Some nested patterns
    )

    loaded = load_hf_state_dict_filtered(hf_model_id_or_dir, vision_prefixes)

    if not loaded:
        logger.warning("No vision weights found with standard prefixes, trying broader search...")
        # Try loading everything and filtering for vision-related keys
        all_weights = load_hf_state_dict_filtered(hf_model_id_or_dir, [""])
        loaded = {
            k: v
            for k, v in all_weights.items()
            if any(
                vision_term in k.lower() for vision_term in ["vision", "visual", "patch", "embed", "merger", "blocks"]
            )
        }

    logger.info(f"Loaded {len(loaded)} vision-related tensors")

    # Standardize keys for TTNN vision components
    vision_weights = {}
    for key, value in loaded.items():
        # Clean up key names for our TTNN components
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        if clean_key.startswith("vision_tower."):
            clean_key = clean_key[13:]  # Remove "vision_tower."
        elif clean_key.startswith("visual."):
            clean_key = clean_key[7:]  # Remove "visual."

        vision_weights[clean_key] = value

    logger.info(f"Processed {len(vision_weights)} vision weights for TTNN")
    if vision_weights:
        logger.info(f"Vision weight keys sample: {list(vision_weights.keys())[:5]}")

    return vision_weights


def load_dots_full_state_dict(hf_model_id_or_dir: str, *, head_dim: int, n_heads: int, n_kv_heads: int):
    """
    Load both text and vision weights for the complete Dots OCR model.

    Returns a combined state dict suitable for both the text decoder
    and the full TTNN vision transformer.
    """
    logger.info(f"Loading complete Dots OCR model weights from {hf_model_id_or_dir}")

    text_weights = load_dots_text_state_dict(
        hf_model_id_or_dir,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
    )

    vision_weights = load_dots_vision_state_dict(hf_model_id_or_dir)

    # Combine both
    combined = {}
    combined.update(text_weights)
    combined.update({f"vision.{k}": v for k, v in vision_weights.items()})

    logger.info(f"Total weights loaded: {len(combined)} " f"({len(text_weights)} text + {len(vision_weights)} vision)")

    return combined


def validate_dots_weight_loading(hf_model_id_or_dir: str):
    """
    Diagnostic function to validate weight loading for real dots.mocr checkpoints.

    This helps debug key mapping issues before full integration.
    """
    logger.info(f"Validating weight loading for {hf_model_id_or_dir}")

    try:
        # Test text weight loading
        text_weights = load_dots_text_state_dict(hf_model_id_or_dir, head_dim=128, n_heads=12, n_kv_heads=2)
        logger.info(f"✅ Text weights: {len(text_weights)} tensors loaded successfully")

        # Test vision weight loading
        vision_weights = load_dots_vision_state_dict(hf_model_id_or_dir)
        logger.info(f"✅ Vision weights: {len(vision_weights)} tensors loaded successfully")

        # Check for critical components
        critical_components = ["patch_embed", "blocks", "merger", "norm", "qkv", "o_proj"]

        found_components = []
        for key in vision_weights.keys():
            for component in critical_components:
                if component in key.lower():
                    found_components.append(component)
                    break

        unique_components = list(set(found_components))
        logger.info(f"Found vision components: {unique_components}")

        if len(vision_weights) > 50:  # Rough heuristic for a real vision model
            logger.info("✅ Weight loading validation PASSED - appears to be a real model")
            return True
        else:
            logger.warning("⚠️  Weight loading validation - limited weights found")
            return False

    except Exception as e:
        logger.error(f"❌ Weight loading validation failed: {e}")
        return False
