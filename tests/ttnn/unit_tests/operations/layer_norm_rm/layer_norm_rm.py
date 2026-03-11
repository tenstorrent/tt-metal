# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shim module for test imports.
Re-exports layer_norm_rm from the operation package so that
stage test files can use: from .layer_norm_rm import layer_norm_rm
"""

from ttnn.operations.layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
