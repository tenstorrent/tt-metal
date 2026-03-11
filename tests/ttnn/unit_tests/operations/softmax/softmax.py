# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Re-export softmax from the operation package for test convenience.

Stage tests import as: from .softmax import softmax
"""

from ttnn.operations.softmax import softmax

__all__ = ["softmax"]
