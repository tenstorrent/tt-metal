# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test package for layer_norm_rm operation.

Re-exports layer_norm_rm so stage tests can use relative imports:
    from .layer_norm_rm import layer_norm_rm
"""

from ttnn.operations.layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
