# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Re-export rms_norm so that stage test files can do:
    from .rms_norm import rms_norm
"""

from ttnn.operations.rms_norm import rms_norm

__all__ = ["rms_norm"]
