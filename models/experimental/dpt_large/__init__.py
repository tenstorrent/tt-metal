# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN DPT-Large (MiDaS 3.0) experimental implementation.

Modules are organized as:
    - config.py: model + TT specific configuration helpers
    - fallback.py: HF/CPU reference pipeline
    - vit_backbone.py: ViT-L backbone utilities (TT + CPU stubs)
    - reassembly.py, fusion_head.py: decoder / head pieces
    - pipeline.py: end-to-end wrapper
    - weights.py: weight loading / conversion helpers
    - runner.py: CLI entrypoint
"""

from .config import DPTLargeConfig
from .fallback import DPTFallbackPipeline, run_depth_cpu
from .pipeline import DPTTTPipeline, run_depth

__all__ = [
    "DPTLargeConfig",
    "DPTFallbackPipeline",
    "run_depth_cpu",
    "DPTTTPipeline",
    "run_depth",
]
