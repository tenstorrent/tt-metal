# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 compute helpers — production defaults, one documented override each.

GeLU: ``ttnn.GeluVariant.Accurate`` (exact SFPU path). Prefer this over
``Tanh`` / FastLut for accuracy; HF uses ``gelu_pytorch_tanh``, but Accurate
gives higher PCC on device.

SDPA fidelity lives in ``prefill_sdpa_compute_kernel_config`` /
``decode_sdpa_compute_kernel_config`` below — both HiFi4, both without fp32
dest-acc. Linear matmul fidelity overrides are intentionally omitted: HiFi4+fp32
on MLP/QKV/O caused unicode garbage on LB 12B decode.
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


def prefill_sdpa_compute_kernel_config(device):
    """Compute-kernel config shared by every prefill SDPA call site.

    Fidelity and fp32 dest-accumulation are one decision, so they live behind
    one function and one knob (``GEMMA4_PREFILL_SDPA_FIDELITY``) rather than
    five inline literals that can drift apart:

      * ``hifi4`` (**default**) -- HiFi4 + fp32 dest-acc. Restores the
        softmax-reduce FP32 accumulation #47311 removed. On Wormhole B0 this is
        exactly the combination #38306 covers; SDPA never calls
        ``verify_numerical_configuration``, so it never warned about it.
      * ``hifi4_nodest`` -- HiFi4 without fp32 dest-acc, the loudbox default.
        MEASURED FASTER but LESS ACCURATE on 12B/T3K: prefill TTFT -5.2% at 4k
        and -7.6% at 32k, against ``test_attention_prefill`` full-attention PCC
        0.9932 -> 0.9861, which fails six unit tests at the 0.99 gate
        (seq1024-global tp=4/8, seq4096-global tp=1/2/4/8). Opt in only where
        the long-context TTFT is worth that.
      * ``hifi3`` -- HiFi3 + fp32 dest-acc, the runtime's own #38306
        recommendation for Wormhole.

    Dest-acc is not free: it halves the DEST register budget, and the chunked
    long-context SDPA pays for that in passes. Judge changes here on
    ``test_teacher_forcing_e2e``, which is bit-reproducible across repeats.
    """
    mode = os.environ.get("GEMMA4_PREFILL_SDPA_FIDELITY", "hifi4").strip().lower()
    if mode == "hifi4":
        fidelity, dest_acc = ttnn.MathFidelity.HiFi4, True
    elif mode == "hifi3":
        fidelity, dest_acc = ttnn.MathFidelity.HiFi3, True
    else:
        fidelity, dest_acc = ttnn.MathFidelity.HiFi4, False
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=dest_acc,
        packer_l1_acc=False,
    )


def decode_sdpa_compute_kernel_config(mesh_device):
    """HiFi4 compute config for decode SDPA, or ``None`` (op default) when opted out.

    Decode ships HiFi4 *without* dest-acc: fp32 dest-acc on this op is not a
    precision knob, it collapses batch-1 decode PCC. Do not enable it. Opt out
    with ``GEMMA4_DECODE_SDPA_FIDELITY=hifi2`` (or 0/false/no) to take the op
    default.
    """
    mode = os.environ.get("GEMMA4_DECODE_SDPA_FIDELITY", "hifi4").strip().lower()
    if mode in ("0", "false", "no", "hifi2"):
        return None
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
