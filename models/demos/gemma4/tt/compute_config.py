# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 compute helpers — production defaults only (no A/B precision knobs).

GeLU: ``ttnn.GeluVariant.Accurate`` (exact SFPU path). Prefer this over
``Tanh`` / FastLut for accuracy; HF uses ``gelu_pytorch_tanh``, but Accurate
gives higher PCC on device.

SDPA fidelity lives in ``prefill_sdpa_compute_kernel_config`` /
``decode_sdpa_compute_kernel_config`` below — both HiFi4, both without fp32
dest-acc. Linear matmul fidelity overrides are intentionally omitted: HiFi4+fp32
on MLP/QKV/O caused unicode garbage on LB 12B decode.
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


def prefill_sdpa_compute_kernel_config(device):
    """HiFi4 + fp32 dest-acc, shared by every prefill SDPA call site.

    Fidelity and fp32 dest-accumulation are one decision, so they live in one
    function rather than five inline literals that can drift apart. The fp32
    dest-acc restores the softmax-reduce accumulation #47311 removed; dropping
    it is faster but fails six unit tests at the 0.99 gate (full-attention PCC
    0.9932 -> 0.9861). Dest-acc is not free -- it halves the DEST register
    budget and the chunked long-context SDPA pays for that in passes -- so judge
    changes here on ``test_teacher_forcing_e2e``, which is bit-reproducible.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def decode_sdpa_compute_kernel_config(mesh_device):
    """HiFi4 compute config for decode SDPA.

    Decode ships HiFi4 *without* dest-acc: fp32 dest-acc on this op is not a
    precision knob, it collapses batch-1 decode PCC. Do not enable it.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
