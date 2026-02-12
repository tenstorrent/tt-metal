# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation descriptors module.

Provides operation descriptors for creating and composing parallel operations.
"""

from .normalization.rms_norm import rms_norm
from .normalization.layer_norm import layer_norm

__all__ = ["rms_norm", "layer_norm"]
