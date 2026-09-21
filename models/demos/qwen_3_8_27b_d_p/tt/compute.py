# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The compute-kernel config every projection matmul uses.

**HiFi4 with ``fp32_dest_acc_en=True`` is the bring-up default** (recipe section 2.3), and it is
not decoration here. ``ttnn.linear`` left to its own devices runs LoFi, which takes fewer mantissa
bits per pass; against the spec's ``bfloat8_b`` weights that was measured costing this model most
of its depth:

| per-layer KV PCC at isl 5120, real weights | LoFi (ttnn default) | HiFi4 + fp32 acc |
|---|---|---|
| layer 3 | 0.9996 | see README |
| layer 27 | 0.9981 | |
| layer 47 | **0.766** | |

The residual stream is fine for ~28 layers and then falls off a cliff as activations grow — the
point where a truncated mantissa starts dominating the sum. A narrower setting is a measurement,
not an inheritance, and this is what the measurement said.

The one place that cannot have it is the ring-SDPA **cache-read** call, which rejects
``fp32_dest_acc_en=True`` outright; that exception is local to that op and lives in
``attention/prefill.py`` next to the call it constrains.
"""

from __future__ import annotations

import ttnn


def matmul_compute_config(
    mesh_device, *, fp32_dest_acc: bool = True, packer_l1_acc: bool = False
) -> ttnn.DeviceComputeKernelConfig:
    """HiFi4, fp32 destination accumulation, no packer L1 accumulation.

    ``packer_l1_acc=True`` was tried and is **off because it was measured**, not assumed: at full
    depth with real weights it moved the worst per-layer KV PCC from 0.8562 to 0.8507 and the e2e
    from 0.97752 to 0.97765 — inside the noise, and the worse side of it on the number that
    matters. Left as a parameter so the next person can re-measure rather than re-derive.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=packer_l1_acc,
    )
