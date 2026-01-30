# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Kernel Fusion Framework for tt-metal ops."""

from .kernel_fusion import GlobalProgram, SubProgram

__all__ = [
    "GlobalProgram",
    "SubProgram",
]
