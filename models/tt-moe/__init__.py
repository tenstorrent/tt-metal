# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unified MoE (Mixture of Experts) config and block for tt-metal.

Use configs from models/tt-moe/configs/*.json (e.g. glm4.json) via load_moe_config.
When the full TT-MoE stack from PR #37920 is available, MoEBlock will be exposed here.
"""

from .utils import load_moe_config, get_moe_block_config

__all__ = ["load_moe_config", "get_moe_block_config"]
