# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Re-export shim for layer_norm TDD stage tests.

Stage test files import `from .layer_norm import layer_norm` (relative import
within this test package). This module re-exports from the canonical location.
"""

from ttnn.operations.layer_norm import layer_norm

__all__ = ["layer_norm"]
