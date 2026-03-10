# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Re-export layer_norm_rm for stage tests.

Stage tests use ``from .layer_norm_rm import layer_norm_rm`` which resolves
to this module inside the test package.
"""

from ttnn.operations.layer_norm_rm import layer_norm_rm  # noqa: F401

__all__ = ["layer_norm_rm"]
