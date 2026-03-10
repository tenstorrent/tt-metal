# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Re-export shim for stage tests.

Stage tests use `from .group_norm import group_norm` (relative import within the
test package). This module re-exports the operation from the ttnn package.
"""

from ttnn.operations.group_norm import group_norm

__all__ = ["group_norm"]
