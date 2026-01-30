# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN DPT-Large (MiDaS 3.0) experimental implementation.

Modules are organized as:
    - tt/: implementation modules
    - demo/: runnable scripts
    - tests/: unit/e2e tests
"""

from .tt.config import DPTLargeConfig
from .tt.fallback import DPTFallbackPipeline, run_depth_cpu

# TT pipeline depends on TTNN + tt_lib; keep CPU-only environments importable.
try:  # pragma: no cover
    from .tt.pipeline import DPTTTPipeline, run_depth
except Exception:  # pragma: no cover
    DPTTTPipeline = None  # type: ignore
    run_depth = None  # type: ignore

__all__ = [
    "DPTLargeConfig",
    "DPTFallbackPipeline",
    "run_depth_cpu",
    "DPTTTPipeline",
    "run_depth",
]
