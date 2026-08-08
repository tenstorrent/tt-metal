# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Centralized, env-gated matmul precision for the VibeVoice diffusion head.

The diffusion head is dispatched S× per frame and every matmul is a tall-skinny
`[1,K]x[K,M]` matvec — i.e. memory-bound on reading the weight matrix. Storing the
weights in a narrower dtype (bf16: 2B, bfp8_b: ~1B vs fp32: 4B) cuts that DRAM
traffic ~2-4× and is the single biggest perf lever on this path.

Default is fp32 / HiFi4 (bring-up correctness), so the isolated per-component PCC
harnesses that import these stubs are byte-for-byte unchanged. The pipeline opts
into a faster mode by setting the env var BEFORE building the stubs:

    VIBEVOICE_DIFF_PREC = fp32 | bf16 | bfp8      (default fp32)

Everything is read at build/forward time, so toggling the env var picks the mode
for the next `VibeVoiceTTS(...)` build. All choices are e2e-PCC-gated.
"""

from __future__ import annotations

import os

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_TILE = ttnn.TILE_LAYOUT

_ENV = "VIBEVOICE_DIFF_PREC"


def _mode() -> str:
    m = os.environ.get(_ENV, "fp32").lower()
    return m if m in ("fp32", "bf16", "bfp8") else "fp32"


def weight_dtype():
    """Dtype the diffusion-head matmul weights are stored in on device."""
    return {"fp32": ttnn.float32, "bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[_mode()]


def act_dtype():
    """Dtype the matmul *activation* is cast to just before the matmul.

    fp32 mode keeps activations in fp32; the low-precision modes use bf16 activations
    (bfp8 weights pair with bf16 activations — bfp8 is a weights-only block format)."""
    return ttnn.float32 if _mode() == "fp32" else ttnn.bfloat16


def compute_config(device):
    """Matmul compute-kernel config matched to the active precision mode."""
    m = _mode()
    if m == "fp32":
        return ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
    if m == "bf16":
        return ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
    # bfp8: LoFi is the standard pairing for bfloat8_b weights
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def mm_weight(w_t_torch, device):
    """Upload a (already-transposed, contiguous) weight tensor in the active weight dtype."""
    return ttnn.from_torch(w_t_torch, dtype=weight_dtype(), layout=_TILE, device=device)


def matmul(a, w, cc):
    """Cast the activation to the active matmul dtype (if needed) and matmul in `cc`."""
    ad = act_dtype()
    if a.get_dtype() != ad:
        a = ttnn.typecast(a, ad)
    return ttnn.matmul(a, w, compute_kernel_config=cc, memory_config=_DRAM)
