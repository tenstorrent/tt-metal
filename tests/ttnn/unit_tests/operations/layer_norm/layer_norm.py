# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Re-export layer_norm from the operation package so stage tests can use:
    from .layer_norm import layer_norm
"""

from ttnn.operations.layer_norm import layer_norm

__all__ = ["layer_norm"]
