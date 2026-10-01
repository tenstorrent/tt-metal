# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 compute helpers — production defaults only (no A/B precision knobs).

GeLU: ``ttnn.GeluVariant.Accurate`` (exact SFPU path). Prefer this over
``Tanh`` / FastLut for accuracy; HF uses ``gelu_pytorch_tanh``, but Accurate
gives higher PCC on device.

SDPA uses HiFi4 + fp32_dest_acc (prefill) via the helpers below. Linear matmul
fidelity overrides are intentionally omitted: HiFi4+fp32 on MLP/QKV/O caused
unicode garbage on LB 12B decode.
"""

from __future__ import annotations

from loguru import logger

import ttnn

_GELU_LOGGED = False


def gelu_variant():
    """Always ``GeluVariant.Accurate``."""
    global _GELU_LOGGED
    if not _GELU_LOGGED:
        logger.info(f"Gemma4 GeLU variant={ttnn.GeluVariant.Accurate}")
        _GELU_LOGGED = True
    return ttnn.GeluVariant.Accurate


def sdpa_math_fidelity(default=ttnn.MathFidelity.HiFi4):
    """SDPA math fidelity (production: HiFi4; ``default`` kept for call-site clarity)."""
    return default


def sdpa_fp32_dest_acc_en(default: bool = True) -> bool:
    """SDPA fp32 dest accumulation (production: on for prefill softmax-reduce)."""
    return default
