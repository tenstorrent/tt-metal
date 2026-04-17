# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import yaml
from ttml.trainers import GRPOConfig


def read_yaml(path: str):
    """Read a config YAML and return the four config sections.

    Any section missing from the YAML is returned as None.

    ``transformer_config`` can be provided either inline or via a
    ``transformer_config_path`` key pointing to a separate YAML file
    (resolved relative to the config file's directory).  The external
    file must contain a top-level ``transformer_config`` mapping.

    Returns:
        (transformer_config, device_config, optimizer_config, grpo_config)
        transformer_config, device_config, and optimizer_config are plain
        dicts (or None). grpo_config is a GRPOConfig instance (or None).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(path))

    transformer_config = raw.get("transformer_config")
    if transformer_config is None and "transformer_config_path" in raw:
        tc_path = raw["transformer_config_path"]
        if not os.path.isabs(tc_path):
            tc_path = os.path.join(config_dir, tc_path)
        with open(tc_path) as f:
            transformer_config = yaml.safe_load(f)["transformer_config"]

    device_config = raw.get("device_config")
    optimizer_config = raw.get("optimizer_config")

    grpo_config = None
    if "grpo_config" in raw:
        grpo_fields = dict(raw["grpo_config"])
        grpo_fields.setdefault("output_dir", "")
        grpo_config = GRPOConfig(**grpo_fields)

    return transformer_config, device_config, optimizer_config, grpo_config
