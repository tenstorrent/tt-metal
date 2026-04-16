# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import yaml
from ttml.trainers import GRPOConfig


def read_yaml(path: str):
    """Read a GRPO training YAML and return the four config sections.

    Returns:
        (transformer_config, device_config, optimizer_config, grpo_config)
        The first three are plain dicts; grpo_config is a GRPOConfig instance.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    transformer_config = raw["transformer_config"]
    device_config = raw["device_config"]
    optimizer_config = raw["optimizer_config"]

    grpo_fields = dict(raw["grpo_config"])
    grpo_fields.setdefault("output_dir", "")
    grpo_config = GRPOConfig(**grpo_fields)

    return transformer_config, device_config, optimizer_config, grpo_config
