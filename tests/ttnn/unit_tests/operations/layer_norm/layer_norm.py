# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility shim for the generated stage tests.
"""

from ttnn.operations.layernorm import layer_norm

__all__ = ["layer_norm"]
