# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Language Model Head building block.

Provides output projection layers for language models with multi-device support.
"""

from .lm_head import (
    LMHeadSpec,
    LMHeadImplConfig,
    get_default_impl_config,
    prepare_weights,
    lm_head_forward,
    prefill_forward,
    decode_forward,
)

__all__ = [
    "LMHeadSpec",
    "LMHeadImplConfig",
    "get_default_impl_config",
    "prepare_weights",
    "lm_head_forward",
    "prefill_forward",
    "decode_forward",
]
