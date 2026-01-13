# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data utilities for pipeline parallel training.

This module re-exports data utilities from ttml.common.data for use in
pipeline parallel training examples.
"""

from ttml.common.data import (
    load_shakespeare_text,
    CharTokenizer,
    prepare_data,
    get_batch,
    build_causal_mask,
)

__all__ = [
    "load_shakespeare_text",
    "CharTokenizer",
    "prepare_data",
    "get_batch",
    "build_causal_mask",
]
