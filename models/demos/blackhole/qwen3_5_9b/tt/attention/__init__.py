# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 full (softmax) attention. The module lives in attention.py; KV cache in
kv_cache.py, partial-RoPE in rope_tp.py, q/k/v/g projections in operations.py,
weight loading in weights.py.

Re-exports Qwen35Attention so callers import it straight from the package
(e.g. tests/unit/test_attention.py).
"""
from .attention import Qwen35Attention

__all__ = ["Qwen35Attention"]
