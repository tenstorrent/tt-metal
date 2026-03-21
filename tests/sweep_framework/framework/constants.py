# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared constants for the sweep framework.

This module contains constants that are used across multiple modules
in the sweep framework to ensure consistency and avoid duplication.
"""

import logging
import re
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Lead models are models that are prioritized for sweep testing.
# These patterns are matched against the source path in traced operations
# to identify which vectors belong to lead model workloads.
#
# Source of truth: model_tracer/sweep_manifest.yaml (targets with scope: lead_models)
# This list is derived from the manifest at import time.
# Fallback to hardcoded list if manifest is unavailable (e.g., in CI without checkout).
#
# Used by:
#   - sweeps_parameter_generator.py: To filter vector generation for lead models only
#   - master_config_loader_v2.py: To filter configurations when loading from master JSON


def _load_lead_models_from_manifest():
    """Derive LEAD_MODELS patterns from sweep_manifest.yaml lead_models targets."""
    try:
        import yaml

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        manifest_path = os.path.join(repo_root, "model_tracer", "sweep_manifest.yaml")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                data = yaml.safe_load(f) or {}
            lead_entries = data.get("targets", {}).get("lead_models", [])
            patterns = [t["model"] for t in lead_entries if "model" in t]
            if patterns:
                return patterns
    except Exception as e:
        logger.debug("Could not load LEAD_MODELS from manifest: %s", e)
    return ["deepseek_v3"]


LEAD_MODELS = _load_lead_models_from_manifest()


# =============================================================================
# Mesh Shape Utilities
# =============================================================================
#
# These utilities handle mesh shape suffixes in module names and vector files.
# The suffix format is: __mesh_<rows>x<cols>
#
# Examples:
#   - model_traced.gelu__mesh_2x4.json
#   - model_traced.matmul__mesh_1x1.json
#
# Used by:
#   - sweeps_parameter_generator.py: To format mesh suffix for exported files
#   - compute_sweep_matrix.py: To parse mesh suffix from filenames and assign runners
#   - vector_source.py: To find mesh-variant files for a module

# Regex pattern to match mesh suffix at end of module name
# Captures the shape (e.g., "2x4") for extraction
MESH_SUFFIX_PATTERN = re.compile(r"__mesh_(\d+)x(\d+)$")


def format_mesh_suffix(mesh_shape: Tuple[int, int]) -> str:
    """Format mesh shape tuple as filename suffix.

    Args:
        mesh_shape: Tuple of (rows, cols), e.g., (2, 4)

    Returns:
        str: Formatted suffix like '__mesh_2x4'

    Examples:
        >>> format_mesh_suffix((2, 4))
        '__mesh_2x4'
        >>> format_mesh_suffix((1, 1))
        '__mesh_1x1'
    """
    return f"__mesh_{mesh_shape[0]}x{mesh_shape[1]}"


def parse_mesh_suffix(module_name: str) -> Optional[Tuple[int, int]]:
    """Extract mesh shape tuple from module name suffix.

    Args:
        module_name: Module name possibly containing mesh suffix

    Returns:
        Tuple of (rows, cols) if valid suffix found, None otherwise.

    Examples:
        >>> parse_mesh_suffix('op__mesh_2x4')
        (2, 4)
        >>> parse_mesh_suffix('op__mesh_1x1')
        (1, 1)
        >>> parse_mesh_suffix('op')
        None
        >>> parse_mesh_suffix('op__mesh_invalid')
        None
    """
    match = MESH_SUFFIX_PATTERN.search(module_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def get_mesh_shape_string(module_name: str) -> Optional[str]:
    """Extract mesh shape as string from module name suffix.

    Args:
        module_name: Module name possibly containing mesh suffix

    Returns:
        Mesh shape string (e.g., '2x4') or None if no valid suffix.

    Examples:
        >>> get_mesh_shape_string('op__mesh_2x4')
        '2x4'
        >>> get_mesh_shape_string('op')
        None
    """
    mesh_shape = parse_mesh_suffix(module_name)
    if mesh_shape:
        return f"{mesh_shape[0]}x{mesh_shape[1]}"
    return None


def strip_mesh_suffix(module_name: str) -> str:
    """Remove mesh suffix from module name.

    Args:
        module_name: Module name possibly containing mesh suffix

    Returns:
        Module name with mesh suffix removed.

    Examples:
        >>> strip_mesh_suffix('op__mesh_2x4')
        'op'
        >>> strip_mesh_suffix('op')
        'op'
        >>> strip_mesh_suffix('op__mesh_invalid')
        'op__mesh_invalid'
    """
    return MESH_SUFFIX_PATTERN.sub("", module_name)
