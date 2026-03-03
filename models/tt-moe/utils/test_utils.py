# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test utility functions for MoE implementation.
"""

import os
from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.weight_config import get_weight_config


def get_test_weight_config(
    ModuleClass: type[AbstractModule],
    hf_config: PretrainedConfig,
    state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
    cache_path: Path,
    mesh_device: ttnn.Device,
    force_recalculate: bool,
    *,
    test_name: str | None = None,
    real_weights: bool = True,
    layer_id: str | int | None = None,
) -> Any:
    """Get the weight config, either by loading from cache or recalculating.

    When ``test_name`` is provided the cache sub-directory is derived from
    weight-relevant parameters only (``test_name``, ``ModuleClass``,
    ``real_weights``, ``layer_id``).  Runtime parameters that do **not**
    affect weight conversion (mode, seq_len, batch_size, position_ids, …)
    are intentionally excluded so that e.g. decode and prefill variants share
    the same cached weights.

    ``num_hidden_layers`` and ``mesh_shape`` are already captured by
    :func:`get_weight_config` in its internal sub-path, so they are not
    needed here either.

    When ``test_name`` is ``None`` the function falls back to
    ``PYTEST_CURRENT_TEST`` for backward compatibility.

    Args:
        ModuleClass: The module class to convert weights for
        hf_config: HuggingFace model configuration
        state_dicts: Pre-loaded state dictionaries containing weights
        cache_path: Path to cache directory for weights
        mesh_device: TTNN mesh device
        force_recalculate: Force recalculation even if cached weights exist
        test_name: Test file name (e.g. ``"test_embedding"``).
        real_weights: ``True`` when using real model weights,
            ``False`` for randomly-initialised weights.
        layer_id: Identifies which layer / sub-module the weights come from.
            Typically a module-path string (``"model.layers.0.mlp"``), an
            integer layer index, or a descriptive qualifier for random weights
            (``"kv_lora_rank"``).  ``None`` when no further distinction is
            needed.

    Returns:
        Weight configuration dictionary
    """
    if test_name is not None:
        parts = [test_name, ModuleClass.__name__, "real" if real_weights else "random"]
        if layer_id is not None:
            parts.append(str(layer_id))
        weight_config_id = "/".join(parts)
    else:
        weight_config_id = os.environ.get("PYTEST_CURRENT_TEST", "unknown_test")
    per_test_weight_cache_path = cache_path / "tests_cache" / weight_config_id
    return get_weight_config(
        ModuleClass, hf_config, state_dicts, per_test_weight_cache_path, mesh_device, force_recalculate
    )


# Additional utility functions can be added here as needed
