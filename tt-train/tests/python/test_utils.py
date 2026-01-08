# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared test utilities for BERT Python tests.
"""

import numpy as np
from pathlib import Path


def compute_pcc(golden: np.ndarray, actual: np.ndarray) -> float:
    """
    Compute Pearson Correlation Coefficient between two tensors.

    Args:
        golden: Reference/expected tensor
        actual: Actual/computed tensor

    Returns:
        PCC value between -1 and 1 (1 = perfect correlation)
    """
    # Convert to float32 to avoid precision issues with bfloat16/float16
    # bfloat16 has only 7 bits of mantissa which causes incorrect PCC values
    golden_flat = golden.flatten().astype(np.float32)
    actual_flat = actual.flatten().astype(np.float32)

    if len(golden_flat) != len(actual_flat):
        return 0.0

    mean_golden = np.mean(golden_flat)
    mean_actual = np.mean(actual_flat)

    numerator = np.sum((golden_flat - mean_golden) * (actual_flat - mean_actual))
    denominator = np.sqrt(
        np.sum((golden_flat - mean_golden) ** 2)
        * np.sum((actual_flat - mean_actual) ** 2)
    )

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return numerator / denominator


def get_safetensors_cache_path(model_name: str) -> Path:
    """
    Get the cache path for a HuggingFace model's safetensors file.

    Args:
        model_name: HuggingFace model name (e.g., "prajjwal1/bert-tiny")

    Returns:
        Path to the cached safetensors file
    """
    return Path(f"/tmp/{model_name.replace('/', '_')}.safetensors")


def save_hf_model_to_safetensors(model, model_name: str) -> Path:
    """
    Save a HuggingFace model to safetensors format if not already cached.

    Args:
        model: HuggingFace model instance
        model_name: Model name for cache path

    Returns:
        Path to the safetensors file
    """
    from safetensors.torch import save_file

    safetensors_path = get_safetensors_cache_path(model_name)
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)

    if not safetensors_path.exists():
        save_file(model.state_dict(), str(safetensors_path))

    return safetensors_path
