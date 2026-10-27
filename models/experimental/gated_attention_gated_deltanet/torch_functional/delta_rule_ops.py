# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Pure-torch implementations of the gated delta rule.

Extracted from FLA (Flash Linear Attention) library:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py

These are reference implementations with no CUDA/Triton dependencies.
They serve as the mathematical specification for TTNN conversion.

Tensor layout convention (FLA style):
  q, k: [B, T, H, K]   (batch, time, heads, key_dim)
  v:    [B, T, H, V]   (batch, time, heads, value_dim)
  beta: [B, T, H]      (batch, time, heads)
  g:    [B, T, H]      (batch, time, heads) -- log-space decay
  state:[B, H, K, V]   (batch, heads, key_dim, value_dim)
"""

from ttnn.operations.transformer_golden import (
    chunk_gated_delta_rule,
    l2_norm,
    recurrent_gated_delta_rule,
)


__all__ = ["chunk_gated_delta_rule", "l2_norm", "recurrent_gated_delta_rule"]
