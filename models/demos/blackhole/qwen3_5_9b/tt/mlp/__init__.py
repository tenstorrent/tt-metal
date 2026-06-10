# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MLP package: forward pass in mlp.py, weight loading in weights.py.

Re-exports Qwen35MLP so callers keep the import path from before the
single-file mlp.py was split into this package:
* tt/layer.py
* tests/test_mlp_tp.py, tests/unit/test_mlp.py, tests/unit/test_component_pcc.py
"""
from .mlp import Qwen35MLP

__all__ = ["Qwen35MLP"]
