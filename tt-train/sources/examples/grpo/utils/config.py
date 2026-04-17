# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import yaml
from ttml.trainers import GRPOConfig


def read_yaml(path: str):
    """Read a config YAML and return the four config sections.

    Any section missing from the YAML is returned as None.

    Returns:
        (transformer_config, device_config, optimizer_config, grpo_config)
        transformer_config, device_config, and optimizer_config are plain
        dicts (or None). grpo_config is a GRPOConfig instance (or None).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    transformer_config = raw.get("transformer_config")
    device_config = raw.get("device_config")
    optimizer_config = raw.get("optimizer_config")

    grpo_config = None
    if "grpo_config" in raw:
        grpo_fields = dict(raw["grpo_config"])
        grpo_fields.setdefault("output_dir", "")
        grpo_config = GRPOConfig(**grpo_fields)

    return transformer_config, device_config, optimizer_config, grpo_config
