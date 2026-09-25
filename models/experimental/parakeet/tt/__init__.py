# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of nvidia/parakeet-tdt-0.6b-v3 (FastConformer encoder + TDT greedy decode)."""
from .ttnn_parakeet import (
    DEFAULT_ACT_DTYPE,
    DEVICE_OPTIONS,
    SUPPORTED_PRECISIONS,
    Backend,
    ParakeetConfig,
    create_backend,
    rel_positional_encoding,
)

__all__ = ["DEFAULT_ACT_DTYPE", "DEVICE_OPTIONS", "SUPPORTED_PRECISIONS", "Backend", "ParakeetConfig",
           "create_backend", "rel_positional_encoding"]
