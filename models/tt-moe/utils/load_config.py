# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Load tt-moe JSON configs."""

import json
from pathlib import Path


def _configs_dir():
    """Directory containing config JSONs (next to this package)."""
    return Path(__file__).resolve().parent.parent / "configs"


def load_moe_config(path_or_name):
    """
    Load a tt-moe config from a file path or a preset name.

    Args:
        path_or_name: Either:
            - Path to a JSON file (str or Path), or
            - Preset name such as "glm4", "deepseek_v3", "gpt_oss" (loads configs/<name>.json).

    Returns:
        Full config dict (e.g. {"moe_block": {...}}).

    Example:
        config = load_moe_config("glm4")
        config = load_moe_config("models/tt-moe/configs/glm4.json")
    """
    p = Path(path_or_name)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    # Preset name
    configs = _configs_dir()
    json_path = configs / f"{path_or_name}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Config not found: {path_or_name} or {json_path}")
    with open(json_path) as f:
        return json.load(f)


def get_moe_block_config(path_or_name):
    """
    Load and return only the moe_block section of a tt-moe config.

    Args:
        path_or_name: Same as load_moe_config (path or preset like "glm4").

    Returns:
        dict: The "moe_block" section (model_params, router, experts, etc.).

    Example:
        moe_cfg = get_moe_block_config("glm4")
        hidden_size = moe_cfg["model_params"]["hidden_size"]
    """
    full = load_moe_config(path_or_name)
    if "moe_block" not in full:
        raise KeyError("Config has no 'moe_block' section")
    return full["moe_block"]
