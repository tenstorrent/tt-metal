# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in fused matmul + TP reduce-scatter for the prefill projections (DS_FUSED_MMRS=1).

ttnn.operations.matmul_reduce_scatter multiplies one output block at a time and sends each block's partial
around the TP ring while the next block is multiplied, so the reduce-scatter hides under the matmul and no
partial touches DRAM. It occupies the whole worker grid, so it cannot run on a sub-device.
"""

from __future__ import annotations

import os

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as _mrs_planner

import ttnn

# Allocate the op's L1 staging per call: a persistent one would sit under every later op's static CBs (the MoE
# expert matmul clashes with it), and per-call allocation follows the same order on every replay of a trace.
if "MRS_STAGING" not in os.environ:
    _mrs_planner.STAGING_POLICY = "ephemeral"


def fused_mmrs_enabled() -> bool:
    return os.environ.get("DS_FUSED_MMRS") == "1"


def fused_matmul_reduce_scatter(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    cluster_axis: int,
    dim: int,
    topology: ttnn.Topology,
    num_links: int,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
    extra_program=None,
    extra_tensors=(),
) -> ttnn.Tensor:
    """x @ weight summed over `cluster_axis` and scattered on `dim` (2 or 3, i.e. rows or columns)."""
    return matmul_reduce_scatter(
        x,
        weight,
        cluster_axis=cluster_axis,
        scatter_dim=dim - 4 if dim >= 0 else dim,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=True),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        extra_program=extra_program,
        extra_tensors=extra_tensors,
    )
