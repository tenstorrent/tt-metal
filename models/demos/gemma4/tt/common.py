# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for Gemma4 TT modules.
"""

import torch

import ttnn


def create_placeholder_tensor(shape, dtype=ttnn.bfloat16):
    """Create a zero tensor on host for placeholder forward methods."""
    return torch.zeros(shape, dtype=torch.bfloat16)
