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

import os

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


_SDPA_FIDELITY = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def sdpa_math_fidelity(default=ttnn.MathFidelity.HiFi4, *, scope="prefill"):
    """SDPA math fidelity (production: HiFi4 prefill / HiFi2 decode).

    Override per scope: ``GEMMA4_PREFILL_SDPA_FIDELITY`` /
    ``GEMMA4_DECODE_SDPA_FIDELITY`` = lofi|hifi2|hifi3|hifi4.

    Lowering prefill fidelity does NOT buy TTFT -- it costs it. 12B / T3K at
    64k measured HiFi2 at 91410 ms TTFT versus 20877 at HiFi4, a 4.4x
    regression, decode unchanged. The fast SDPA path evidently wants
    HiFi4 (+fp32 dest-acc); HiFi4 is already the right default and the long-ISL
    TTFT gap is not explained by attention fidelity.
    """
    override = os.environ.get(f"GEMMA4_{scope.upper()}_SDPA_FIDELITY", "").strip().lower()
    return _SDPA_FIDELITY.get(override, default)


def sdpa_fp32_dest_acc_en(default: bool = True, *, scope: str = "prefill") -> bool:
    """SDPA fp32 dest accumulation (production: on for prefill softmax-reduce).

    Required since #47311 stopped the softmax reduce forcing FP32 internally.
    Override per scope: ``GEMMA4_PREFILL_SDPA_FP32_ACC`` /
    ``GEMMA4_DECODE_SDPA_FP32_ACC`` = 0/1.

    Kept on by default in both scopes. Measured on 12B / T3K:

    * prefill off -> 64k TTFT 18735 ms versus 20877, ~10% faster, and the
      generation stayed coherent. Left ON anyway: it is a numerics change on
      the long-context softmax reduce that #47311 made the model's
      responsibility, and 10% does not justify spending that margin by default.
      Enable per-run if TTFT matters more than the accumulation headroom.
    * decode off -> the device wedged 4 tokens into a 64k run. Do not.
    """
    override = os.environ.get(f"GEMMA4_{scope.upper()}_SDPA_FP32_ACC", "").strip().lower()
    if override in ("0", "false", "no", "off"):
        return False
    if override in ("1", "true", "yes", "on"):
        return True
    return default
