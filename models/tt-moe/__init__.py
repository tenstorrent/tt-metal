# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unified MoE (Mixture of Experts) implementation for tt-metal.

This module provides a configurable MoE block that can support multiple
architectures including DeepSeek-V3, GPT-OSS, Mixtral, and others through
JSON configuration files.
"""

from .moe_block import MoEBlock

__all__ = [
    "MoEBlock",
]

__version__ = "0.1.0"
