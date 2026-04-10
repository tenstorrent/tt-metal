# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Cache utilities for checking TTNN weight cache and reference output cache states.
"""

import os
from pathlib import Path

from loguru import logger


def check_ttnn_cache_complete(
    weight_cache_path: Path | None,
    num_layers: int,
    experts_per_chip: int = 8,
    mesh_shape: tuple[int, int] | None = None,
) -> bool:
    """
    Check if all TTNN weight cache files exist for the given configuration.

    Args:
        weight_cache_path: Path to TTNN weight cache directory (e.g., /cache/8x4/)
        num_layers: Number of transformer layers
        experts_per_chip: Number of routed experts per chip (default: 8 for 256 experts on 8x4 mesh)
        mesh_shape: Mesh shape (rows, cols) - used for logging only

    Returns:
        True if all expected .ttnn files exist, False otherwise
    """
    if not weight_cache_path or not weight_cache_path.exists():
        logger.debug(f"TTNN cache path does not exist: {weight_cache_path}")
        return False

    # Check embedding (actual format: embed_weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin)
    embed_files = list(weight_cache_path.glob("embed_weight*.tensorbin"))
    if not embed_files:
        logger.debug("TTNN cache missing: embed_weight")
        return False

    # Check norm (actual format: norm_weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin)
    norm_files = list(weight_cache_path.glob("norm*weight*.tensorbin"))
    if not norm_files:
        logger.debug("TTNN cache missing: norm weight files")
        return False

    # Check each layer's weights
    # First 3 layers are dense (no routed experts)
    # Remaining layers are MoE with routed experts
    first_k_dense = 3

    for layer_idx in range(num_layers):
        is_dense = layer_idx < first_k_dense

        # Check attention weights (all layers have these)
        # Actual format: layer_X.mla.q_a_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin
        mla_files = list(weight_cache_path.glob(f"layer_{layer_idx}.mla.*.tensorbin"))
        if not mla_files:
            logger.debug(f"TTNN cache missing: layer {layer_idx} MLA weights")
            return False

        if is_dense:
            # Dense layers: check for FFN weights
            # Actual format: layer_X.ffn.gate_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                ffn_files = list(weight_cache_path.glob(f"layer_{layer_idx}.ffn.{proj}*.tensorbin"))
                if not ffn_files:
                    logger.debug(f"TTNN cache missing: layer {layer_idx} FFN {proj}")
                    return False
        else:
            # MoE layers: check for routed expert weights
            # Actual format: layer_X.routed_expert.local_Y_gate_dtype_BFLOAT16_layout_TILE.tensorbin
            for local_expert_idx in range(experts_per_chip):
                for proj in ["gate", "up", "down"]:
                    pattern = f"layer_{layer_idx}.routed_expert.local_{local_expert_idx}_{proj}*.tensorbin"
                    expert_files = list(weight_cache_path.glob(pattern))
                    if not expert_files:
                        logger.debug(
                            f"TTNN cache missing: layer {layer_idx} " f"routed expert local_{local_expert_idx} {proj}"
                        )
                        return False

            # Check shared expert weights
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                shared_files = list(weight_cache_path.glob(f"layer_{layer_idx}.shared_expert.{proj}*.tensorbin"))
                if not shared_files:
                    logger.debug(f"TTNN cache missing: layer {layer_idx} shared expert {proj}")
                    return False

    logger.info(f"TTNN cache complete at {weight_cache_path} ({num_layers} layers)")
    return True


def check_reference_cache_exists(cache_key: str) -> bool:
    """
    Check if reference output cache exists for the given cache key.

    Reference cache contains forward pass outputs from HF model for PCC validation.
    This cache is machine-independent and can be generated once and shared.

    Args:
        cache_key: Cache identifier like "pretrained_json_prompts_isl1024_layers24_experts256"

    Returns:
        True if cache file exists, False otherwise
    """
    cache_dir = Path(os.environ.get("TT_DS_PREFILL_HOST_REF_CACHE", "/tmp/deepseek_v3_transformer_ref_cache"))
    cache_path = cache_dir / f"{cache_key}.pt"

    exists = cache_path.exists()

    if exists:
        logger.info(f"Reference cache found: {cache_path}")
    else:
        logger.debug(f"Reference cache not found: {cache_path}")

    return exists
