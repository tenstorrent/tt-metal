# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared constants for the sweep framework.

This module contains constants that are used across multiple modules
in the sweep framework to ensure consistency and avoid duplication.
"""

import re
from typing import Tuple, Optional

# Lead models are models that are prioritized for sweep testing.
# These patterns are matched against the source path in traced operations
# to identify which vectors belong to lead model workloads.
#
# Used by:
#   - sweeps_parameter_generator.py: To filter vector generation for lead models only
#   - master_config_loader.py: To filter configurations when loading from master JSON
#
# To add a new lead model, add the model directory name pattern here.
# Example: "llama3" would match source paths containing "llama3"
LEAD_MODELS = [
    "deepseek_v3",
]


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
# The identifier is the full device_series from the JSON (e.g. "tt-galaxy-wh")
# combined with device count when the same device_series spans multiple
# topologies (e.g. n300 cards are used in N300, T3K, and Galaxy setups).
#
# Key: hardware identifier string (embedded in filenames)
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
        "runs_on": ["topology-t3k", "arch-wormhole_b0", "in-service", "pipeline-functional"],
        "runner_label": "topology-t3k",
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
    """Derive the hardware identifier from traced_machine_info.

    Returns the full device_series (e.g. "tt-galaxy-wh", "p150b") or a
    disambiguated name for n300-based topologies ("n150", "n300", "t3k").
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

    if not device_series:
        return None

    # Non-n300 series: use device_series directly (e.g. "tt-galaxy-wh", "p150b")
    if device_series != "n300":
        return device_series

    # n300 series: disambiguate by device count
    # n300 cards are used in N150 (1 dev), N300 (2 dev), T3K (8 dev)
    mesh = machine_info.get("mesh_device_shape")
    if isinstance(mesh, str):
        try:
            import ast as _ast

            mesh = _ast.literal_eval(mesh)
        except (ValueError, SyntaxError):
            mesh = None
    if isinstance(mesh, (list, tuple)) and len(mesh) == 2:
        num_devices = mesh[0] * mesh[1]
    else:
        num_devices = machine_info.get("device_count", machine_info.get("card_count", 1))
        if isinstance(num_devices, list):
            num_devices = num_devices[0] if num_devices else 1

    if num_devices <= 1:
        return "n150"
    elif num_devices <= 2:
        return "n300"
    elif num_devices <= 8:
        return "t3k"
    # n300 cards in galaxy-scale topology (unusual but possible)
    return "tt-galaxy-wh"


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
