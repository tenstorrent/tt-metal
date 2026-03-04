# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Bridge module: re-exports layer_norm from ttnn.operations.layer_norm.

This allows stage test files to use:
    from .layer_norm import layer_norm
without knowing the absolute import path.
"""

from ttnn.operations.layer_norm import layer_norm

__all__ = ["layer_norm"]
