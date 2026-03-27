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

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        manifest_path = os.path.join(repo_root, "model_tracer", "sweep_manifest.yaml")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                data = yaml.safe_load(f) or {}
            lead_entries = data.get("targets", {}).get("lead_models", [])
            patterns = []
            for t in lead_entries:
                m = t.get("model")
                if isinstance(m, list):
                    patterns.extend(m)
                elif m:
                    patterns.append(m)
            if patterns:
                return patterns
    except Exception as e:
        logger.debug("Could not load LEAD_MODELS from manifest: %s", e)
    return ["deepseek_v3"]


LEAD_MODELS = _load_lead_models_from_manifest()


# =============================================================================
# Mesh Shape & Hardware Suffix Utilities
# =============================================================================
#
# These utilities handle mesh shape and hardware suffixes in module names
# and vector files. The suffix encodes both the mesh topology and the
# hardware the config was traced on, so the CI matrix can route each
# sub-job to the exact matching runner.
#
# Suffix format: __mesh_<rows>x<cols>__hw_<hardware_name>
#   (the __hw_ part is optional for backward compatibility)
#
# Examples:
#   - model_traced.gelu__mesh_4x8__hw_galaxy.json
#   - model_traced.add__mesh_1x2__hw_n300.json
#   - model_traced.matmul__mesh_1x1.json        (legacy, no hw)
#
# Used by:
#   - sweeps_parameter_generator.py: To format suffix for exported files
#   - compute_sweep_matrix.py: To parse suffix and assign runners
#   - vector_source.py: To find mesh-variant files for a module

# Regex pattern to match mesh + optional hardware suffix at end of module name.
# Group 1: rows, Group 2: cols, Group 3 (optional): hardware name
MESH_SUFFIX_PATTERN = re.compile(r"__mesh_(\d+)x(\d+)(?:__hw_([a-zA-Z0-9_-]+))?$")

# Maps a hardware identifier (stored in __hw_ suffix) to its CI runner config.
# The identifier comes from get_hardware_id_from_machine_info() and is embedded
# in filenames by format_mesh_suffix().  Keys are the exact strings returned by
# that function (e.g. "tt-galaxy-wh", "n300", "t3k") — NOT short aliases.
# The split-hw YAML step uses substring matching ('galaxy' in name) which works
# because "tt-galaxy-wh" contains "galaxy"; keep the map keys in sync if
# get_hardware_id_from_machine_info() return values ever change.
#
# Key: hardware identifier string (embedded in filenames as __hw_<id>)
# Value: dict with arch, runs_on, runner_label, tt_smi_cmd
HARDWARE_RUNNER_MAP = {
    # --- Wormhole ---
    "n150": {
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "runner_label": "N150",
        "tt_smi_cmd": "tt-smi -r",
    },
    "n300": {
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n300-stable",
        "runner_label": "N300",
        "tt_smi_cmd": "tt-smi -r",
    },
    "p150b": {
        "arch": "blackhole",
        "runs_on": "tt-ubuntu-2204-p150b-stable",
        "runner_label": "P150b",
        "tt_smi_cmd": "tt-smi -r",
    },
    "t3k": {
        "arch": "wormhole_b0",
        "runs_on": ["config-t3000", "arch-wormhole_b0", "in-service", "pipeline-functional"],
        "runner_label": "config-t3000",
        "tt_smi_cmd": "tt-smi -r",
    },
    "tt-galaxy-wh": {
        "arch": "wormhole_b0",
        "runs_on": ["topology-6u", "arch-wormhole_b0", "in-service", "pipeline-functional"],
        "runner_label": "topology-6u",
        "tt_smi_cmd": "tt-smi -r",
    },
    # --- Blackhole (future) ---
    # "tt-galaxy-bh": {
    #     "arch": "blackhole",
    #     "runs_on": ["topology-bh-galaxy", "arch-blackhole", "in-service", "pipeline-functional"],
    #     "runner_label": "topology-bh-galaxy",
    #     "tt_smi_cmd": "tt-smi -r",
    # },
}


def get_hardware_id_from_machine_info(machine_info) -> Optional[str]:
    """Read the hardware identifier directly from traced_machine_info.

    Returns device_series as-is (e.g. "n150", "n300", "t3k", "tt-galaxy-wh",
    "p150b").  The tracer records the exact hardware type in device_series so
    there is no need to infer it from device counts — both "n300" and "t3k"
    (and their device counts) are explicitly present in the trace.

    This value is embedded in vector filenames and later used to look up
    the runner config from HARDWARE_RUNNER_MAP.
    """
    if not machine_info:
        return None

    if isinstance(machine_info, list):
        machine_info = machine_info[0] if machine_info else {}

    device_series = machine_info.get("device_series", "")
    if isinstance(device_series, list):
        device_series = device_series[0] if device_series else ""

    return device_series if device_series else None


def get_runner_config_for_hardware(hardware_id: str) -> Optional[dict]:
    """Get CI runner configuration for a hardware identifier.

    Args:
        hardware_id: Value from get_hardware_id_from_machine_info() or
                     parsed from __hw_ filename suffix.

    Returns:
        Dict with arch, runs_on, runner_label, tt_smi_cmd; or None.
    """
    return HARDWARE_RUNNER_MAP.get(hardware_id)


def format_mesh_suffix(mesh_shape: Tuple[int, int], hardware_name: Optional[str] = None) -> str:
    """Format mesh shape (and optional hardware) as filename suffix.

    Args:
        mesh_shape: Tuple of (rows, cols), e.g., (2, 4)
        hardware_name: Short hardware name (e.g., 'galaxy', 'n300').
                       If None, only mesh shape is included (legacy format).

    Examples:
        >>> format_mesh_suffix((4, 8), 'galaxy')
        '__mesh_4x8__hw_galaxy'
        >>> format_mesh_suffix((1, 2))
        '__mesh_1x2'
    """
    suffix = f"__mesh_{mesh_shape[0]}x{mesh_shape[1]}"
    if hardware_name:
        suffix += f"__hw_{hardware_name}"
    return suffix


def parse_mesh_suffix(module_name: str) -> Optional[Tuple[int, int]]:
    """Extract mesh shape tuple from module name suffix.

    Examples:
        >>> parse_mesh_suffix('op__mesh_2x4__hw_n300')
        (2, 4)
        >>> parse_mesh_suffix('op__mesh_1x1')
        (1, 1)
        >>> parse_mesh_suffix('op')
        None
    """
    match = MESH_SUFFIX_PATTERN.search(module_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def parse_hardware_suffix(module_name: str) -> Optional[str]:
    """Extract hardware name from module name suffix.

    Examples:
        >>> parse_hardware_suffix('op__mesh_4x8__hw_galaxy')
        'galaxy'
        >>> parse_hardware_suffix('op__mesh_1x1')
        None
    """
    match = MESH_SUFFIX_PATTERN.search(module_name)
    if match and match.group(3):
        return match.group(3)
    return None


def get_mesh_shape_string(module_name: str) -> Optional[str]:
    """Extract mesh shape as string from module name suffix.

    Examples:
        >>> get_mesh_shape_string('op__mesh_2x4__hw_n300')
        '2x4'
        >>> get_mesh_shape_string('op')
        None
    """
    mesh_shape = parse_mesh_suffix(module_name)
    if mesh_shape:
        return f"{mesh_shape[0]}x{mesh_shape[1]}"
    return None


def strip_mesh_suffix(module_name: str) -> str:
    """Remove mesh (and hardware) suffix from module name.

    Examples:
        >>> strip_mesh_suffix('op__mesh_2x4__hw_galaxy')
        'op'
        >>> strip_mesh_suffix('op__mesh_2x4')
        'op'
        >>> strip_mesh_suffix('op')
        'op'
    """
    return MESH_SUFFIX_PATTERN.sub("", module_name)
