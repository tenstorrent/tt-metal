# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Re-export shim for stage tests.

Stage tests do `from .layer_norm_rm import layer_norm_rm`.
This module bridges to the actual implementation in ttnn.operations.layer_norm_rm.
"""

from ttnn.operations.layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
