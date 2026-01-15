# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Performant Runner with Trace + 2CQ Optimizations.

This module provides optimized execution for PI0 inference using:
1. Full Model Tracing: Captures entire inference pipeline, eliminating dispatch overhead
2. Two Command Queue (2CQ): Overlaps input transfer with compute

IMPORTANT: This runner uses FIXED configuration for maximum performance.
See docs/TRACE_2CQ_OPTIMIZATION.md for limitations and alternative approaches.

Usage:
    from models.experimental.pi0.runner import PerformantRunner, PI0TraceConfig

    config = PI0TraceConfig()
    runner = PerformantRunner(model, device, config)
    runner.compile()

    actions = runner.execute(images, img_masks, lang_tokens, lang_masks, state)

    runner.cleanup()
"""

from .performant_runner import PerformantRunner, PI0TraceConfig

__all__ = ["PerformantRunner", "PI0TraceConfig"]
