# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The per-step inputs (tracer values, normal thickness flux f, high-order mask) in their natural layout.

These change every model time step, so the optimized version takes them as they are (row-major fp32 in DRAM,
the shapes of reference/optimized_ttnn.make_inputs without the trailing size-1 dimension) and rearranges them
on the chip inside the timed run.
"""
import ttnn

STEP_INPUTS = ("cell", "normalThicknessFlux1", "normalThicknessFlux2", "advMaskHighOrder1", "advMaskHighOrder2")


def natural_host(host):
    """cell [L, N+4, N+4]; f, mask [L, N+1, 2N+1] (slanted edges) and [L, N, N+1] (vertical edges)."""
    out = {}
    for k in STEP_INPUTS:
        t = host[k]
        if t.dim() == 4:
            t = t[..., 0]
        out[k] = t.contiguous().float()
    return out


def upload_natural(nat, device):
    return {
        k: ttnn.from_torch(
            v, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for k, v in nat.items()
    }
