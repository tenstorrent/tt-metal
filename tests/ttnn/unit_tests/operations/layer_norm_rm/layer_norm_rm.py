# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Re-export layer_norm_rm from the operation package.
Stage test files use `from .layer_norm_rm import layer_norm_rm` which
resolves to this module within the test package.
"""

from ttnn.operations.layer_norm_rm import layer_norm_rm

__all__ = ["layer_norm_rm"]
