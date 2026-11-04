# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dtype map and compute-kernel configs for the functional decoder.

Weight dtypes follow the pre-packed tile cache: routed experts bfloat4_b;
shared expert, dense MLP, KDA q/k/v/o and MLA q_a/q_b/kv_a/o_proj bfloat8_b; everything else bf16; A_log, dt_bias,
router bias, hc_base, hc_scale float32. Activations: residual streams and every mHC quantity in float32 (upstream fp32
islands), sublayer inputs bf16, router logits/sigmoid fp32, KDA decode state fp32.
"""

from __future__ import annotations

import ttnn

WEIGHT_DTYPES = {
    "routed_expert": ttnn.bfloat4_b,
    "projection_bfp8": ttnn.bfloat8_b,
    "bf16": ttnn.bfloat16,
    "fp32": ttnn.float32,
}

STREAM_DTYPE = ttnn.float32  # 4 x 4096 residual streams
ACT_DTYPE = ttnn.bfloat16  # sublayer inputs/outputs
LATENT_CACHE_DTYPE = ttnn.bfloat16  # MLA latent paged cache
KDA_STATE_DTYPE = ttnn.float32  # recurrent S per user


def hifi4_fp32(device, packer_l1_acc: bool = False):
    """HiFi4 + fp32 destination accumulation: router logits, mHC projection, fp32 recurrence matmuls."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def hifi2_fp32(device):
    """Projection matmuls with bfp8/bfp4 weights."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
