# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 compute helpers and validated production defaults.

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


def sdpa_math_fidelity(default=ttnn.MathFidelity.HiFi4):
    """SDPA math fidelity (production: HiFi4; ``default`` kept for call-site clarity)."""
    return default


def sdpa_fp32_dest_acc_en(default: bool = True) -> bool:
    """SDPA fp32 dest accumulation (production: on for prefill softmax-reduce)."""
    return default


def prefill_sdpa_compute_kernel_config(device):
    """Prefill SDPA config, defaulting to main's validated HiFi4+FP32 policy.

    Alternative measured configurations remain explicit deployment opt-ins.
    """
    mode = os.environ.get("GEMMA4_PREFILL_SDPA_FIDELITY", "hifi4").lower()
    if mode == "hifi4_nodest":
        fidelity, dest_acc = ttnn.MathFidelity.HiFi4, False
    elif mode == "hifi3":
        fidelity, dest_acc = ttnn.MathFidelity.HiFi3, True
    else:
        fidelity, dest_acc = sdpa_math_fidelity(), sdpa_fp32_dest_acc_en()
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=dest_acc,
        packer_l1_acc=False,
    )


def decode_sdpa_compute_kernel_config(device):
    """Optional HiFi4 decode SDPA config; preserve the op default unless enabled."""
    mode = os.environ.get("GEMMA4_DECODE_SDPA_FIDELITY", "hifi2").lower()
    if mode in ("0", "false", "no", "hifi2"):
        return None
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
