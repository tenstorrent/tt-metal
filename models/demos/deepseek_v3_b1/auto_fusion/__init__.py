# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Auto-fusion infrastructure for composing micro-ops into fused unified kernels.

This package provides a graph-based API for describing spatio-temporal dataflow
between micro-ops, and automatically generates a single unified C++ kernel file
plus the corresponding Python host descriptors (CBs, compile-time args, runtime
args, semaphores) to execute the fused kernel via ttnn.generic_op.

Usage:
    from models.demos.deepseek_v3_b1.auto_fusion import FusionGraph
    from models.demos.deepseek_v3_b1.auto_fusion.specs import RMSNORM, MATMUL, MCAST, GATHER

    g = FusionGraph()
    g.add("rmsnorm", RMSNORM, cores=input_core, ct_args={...})
    g.add("mcast",   MCAST,   cores=full_grid,  inputs={"src": ("rmsnorm", "out")}, ...)
    g.add("matmul",  MATMUL,  cores=mm_grid,    inputs={"in0": ("mcast", "dst")}, ...)
    fused = g.build(device, io_tensors={...})
    result = fused.run()
"""

from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
from models.demos.deepseek_v3_b1.auto_fusion.types import (
    CBConfig,
    CBPortSpec,
    MicroOpSpec,
    RISCContract,
    SDFRate,
)
from models.demos.deepseek_v3_b1.auto_fusion.sdf import SDFAnalyzer
from models.demos.deepseek_v3_b1.auto_fusion.polyhedral import PolyhedralAnalyzer
from models.demos.deepseek_v3_b1.auto_fusion.ilp_scheduler import ILPScheduler
from models.demos.deepseek_v3_b1.auto_fusion.software_pipeline import SoftwarePipeliner
from models.demos.deepseek_v3_b1.auto_fusion.graph_coloring import ChaitinBriggsAllocator

__all__ = [
    "FusionGraph",
    "CBConfig",
    "CBPortSpec",
    "MicroOpSpec",
    "RISCContract",
    "SDFRate",
    "SDFAnalyzer",
    "PolyhedralAnalyzer",
    "ILPScheduler",
    "SoftwarePipeliner",
    "ChaitinBriggsAllocator",
]
