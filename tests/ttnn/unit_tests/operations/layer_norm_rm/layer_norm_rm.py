# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Bridge module: re-exports layer_norm_rm from the operation package.
Stage tests import via `from .layer_norm_rm import layer_norm_rm`.
"""

from ttnn.operations.layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
