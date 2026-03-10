# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Re-export layer_norm_rm for test imports.

Stage test files use: from .layer_norm_rm import layer_norm_rm
This module bridges from the test directory to the operation package.
"""

from ttnn.operations.layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
