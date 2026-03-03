# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed training utilities for TTML.

This module provides distributed training primitives including:
- FSDPModule: Fully Sharded Data Parallel wrapper for memory-efficient training
"""

from .fsdp import (
    FSDPModule,
    fully_shard,
    setup_prefetching,
    synchronize_fsdp_gradients,
    clear_fsdp_registry,
)

__all__ = [
    "FSDPModule",
    "fully_shard",
    "setup_prefetching",
    "synchronize_fsdp_gradients",
    "clear_fsdp_registry",
]
