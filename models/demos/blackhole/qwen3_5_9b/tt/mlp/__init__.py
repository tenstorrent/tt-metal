# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 SwiGLU MLP. Forward pass lives in mlp.py, weight loading in weights.py.

Re-exports Qwen35MLP so callers import it straight from the package (e.g. tt/layer.py).
"""
from .mlp import Qwen35MLP

__all__ = ["Qwen35MLP"]
