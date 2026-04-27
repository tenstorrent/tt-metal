# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-module weight dtype configuration for Gemma4.

Different modules have different precision tolerance and DRAM cost. A single
model-wide dtype forces a tradeoff that's wrong for at least one class — most
notably, expert weights are 85% of total DRAM and tolerate bfp8_b, while the
262k-vocab lm_head and embedding are accuracy-critical and must stay bf16.

Cache compatibility: cache filenames already encode dtype as
`..._dtype_BFLOAT8_B_layout_TILE.tensorbin`, so files for different dtypes
coexist in the same directory without colliding.
"""

from dataclasses import dataclass, fields, replace
from typing import Optional

import ttnn


@dataclass(frozen=True)
class Gemma4DTypes:
    """Per-module weight dtype config.

    Defaults reflect the current best-known precision tradeoff for the 26B-A4B
    variant: experts and shared MLP at bfp8_b (large weights, tolerant), the
    rest at bf16 (small or accuracy-critical).
    """

    attention: ttnn.DataType = ttnn.bfloat16  # QKV+O fused projections
    experts: ttnn.DataType = ttnn.bfloat4_b  # MoE gate/up/down — biggest weight class; gpt_oss / deepseek_v3 use bfp4
    shared_mlp: ttnn.DataType = ttnn.bfloat8_b  # Dense MLP gate/up/down
    router: ttnn.DataType = ttnn.bfloat16  # ~20 MB, sensitive (logits over 128 experts)
    embedding: ttnn.DataType = ttnn.bfloat16  # 262k vocab — bfp8 too lossy for argmax
    lm_head: ttnn.DataType = ttnn.bfloat16  # Same constraint as embedding
    pli: ttnn.DataType = ttnn.bfloat16  # E2B/E4B per-layer input gates / projections

    @classmethod
    def uniform(cls, dtype: ttnn.DataType) -> "Gemma4DTypes":
        """Apply a single dtype to every module — legacy escape hatch."""
        return cls(**{f.name: dtype for f in fields(cls)})

    def with_overrides(self, **overrides) -> "Gemma4DTypes":
        """Return a copy with selected fields overridden — for experimentation."""
        return replace(self, **overrides)


def resolve_dtypes(
    dtypes: Optional[Gemma4DTypes] = None,
    legacy_dtype: Optional[ttnn.DataType] = None,
) -> Gemma4DTypes:
    """Resolve which dtype config to use.

    `dtypes` wins. If only `legacy_dtype` is given, broadcast it to every field
    (preserves old `dtype=ttnn.bfloat16` behavior for any external caller still
    passing it). If neither is given, return defaults.
    """
    if dtypes is not None:
        return dtypes
    if legacy_dtype is not None:
        return Gemma4DTypes.uniform(legacy_dtype)
    return Gemma4DTypes()
