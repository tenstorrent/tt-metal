# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated full-attention for Qwen3.5-9B, split into config/weights/prefill/decode.

The orchestrating layer lives in ``gated_attention.py``; this package re-exports it
(and ``AttentionConfig``) as the public API.
"""

from models.demos.blackhole.qwen36.tt.attention.config import AttentionConfig
from models.demos.blackhole.qwen36.tt.attention.gated_attention import Qwen36GatedAttention

__all__ = ["Qwen36GatedAttention", "AttentionConfig"]
