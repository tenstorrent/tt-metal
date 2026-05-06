# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN rotary tables for Devstral-2 (123B) text models.

The architecture matches Hugging Face ``Ministral3RotaryEmbedding`` (same as Devstral-Small-2). RoPE
parameters and cos/sin tables are implemented once under ``models.experimental.devstarl2_small``;
this module re-exports that implementation so ``devstral2_large`` can depend on it without a second
copy of the NumPy / YaRN logic.
"""

from __future__ import annotations

from models.experimental.devstarl2_small.tt.tt_ministral_rotary_emb import (
    TtMinistral3RotaryEmbedding,
    ministral3_hf_cos_sin_tables,
    ministral3_inv_freq_and_attention_scaling,
)

# Alias for bring-up code that names the class after the target checkpoint family.
TtDevstral2LargeRotaryEmbedding = TtMinistral3RotaryEmbedding

__all__ = [
    "TtDevstral2LargeRotaryEmbedding",
    "TtMinistral3RotaryEmbedding",
    "ministral3_hf_cos_sin_tables",
    "ministral3_inv_freq_and_attention_scaling",
]
