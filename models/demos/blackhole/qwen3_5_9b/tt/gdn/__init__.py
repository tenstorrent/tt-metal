# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 Gated DeltaNet (linear attention). The module lives in gdn.py;
chunk/conv/triangular-solve helpers in operations.py, weight loading in weights.py.

Re-exports the public classes so callers import them straight from the package
(e.g. tt/layer.py uses Qwen35GatedDeltaNet).
"""
from .gdn import Mode, Qwen35GatedDeltaNet, Qwen35RMSNormGated

__all__ = ["Qwen35GatedDeltaNet", "Qwen35RMSNormGated", "Mode"]
