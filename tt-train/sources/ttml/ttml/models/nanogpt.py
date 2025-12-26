# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT implementation using ttml operations.

This module provides backward compatibility imports for the NanoGPT model.
The implementation has been split into separate modules in the nanogpt package.
"""

# Import from the nanogpt package for backward compatibility
from .nanogpt import (
    GPTBlock,
    NanoGPT,
    NanoGPTConfig,
    create_nanogpt,
)

__all__ = [
    "GPTBlock",
    "NanoGPT",
    "NanoGPTConfig",
    "create_nanogpt",
]
