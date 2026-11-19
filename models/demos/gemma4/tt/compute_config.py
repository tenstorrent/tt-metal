# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 compute helpers — production defaults, plus one opt-in A/B knob (GEMMA4_PREFILL_SDPA_GLOBAL_FIDELITY).

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


def prefill_linear_compute_kernel_config(device):
    """Prefill linear (QKV / O / MLP) compute config: HiFi2 with fp32 DEST accumulation.

    The DRAM-sharded and 2D program configs otherwise default to LoFi with a bf16 DEST, which adds 1-2% rel-L2 per
    prefill linear on Gemma-4-12B (HF bf16: 0.16%) and compounds through the layers. Decode keeps the op default.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def sdpa_math_fidelity(default=ttnn.MathFidelity.HiFi4):
    """SDPA math fidelity (production: HiFi4; ``default`` kept for call-site clarity)."""
    return default


GLOBAL_PREFILL_SDPA_FIDELITY_ENV = "GEMMA4_PREFILL_SDPA_GLOBAL_FIDELITY"
_SDPA_FIDELITIES = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2}


def global_prefill_sdpa_fidelity():
    """Math fidelity of the global-layer (head_dim 512) prefill SDPA: ``GEMMA4_PREFILL_SDPA_GLOBAL_FIDELITY`` =
    ``hifi4`` (default) or ``hifi2``. One Galaxy chip, 4 Q / 1 KV heads, bfp8 paged K/V, a 4096-token chunk on the
    whole grid (2026-11-10): 51 -> 36 ms after a 64K cached prefix, 101 -> 71 ms after 128K, PCC against torch
    0.99964 -> 0.99958 and 0.99928 -> 0.99923; HiFi2 failed the 2K prose gate on the 31B, so it stays opt-in. Read at call
    time (an unset or empty value is hifi4); the model build calls it once so a bad value fails at load."""
    name = (os.environ.get(GLOBAL_PREFILL_SDPA_FIDELITY_ENV) or "hifi4").strip().lower() or "hifi4"
    if name not in _SDPA_FIDELITIES:
        raise ValueError(f"{GLOBAL_PREFILL_SDPA_FIDELITY_ENV}={name!r}: expected one of {sorted(_SDPA_FIDELITIES)}")
    return _SDPA_FIDELITIES[name]


def sdpa_fp32_dest_acc_en(default: bool = True) -> bool:
    """SDPA fp32 dest accumulation (production: on for prefill softmax-reduce)."""
    return default
