# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Kernel Fusion Infrastructure.

Provides the public API for composing and fusing operations into single
dispatches. Re-exports key symbols from submodules for convenient access.
"""

from models.experimental.ops.descriptors.fusion.fusion import (
    FusedOp,
    Parallel,
    Sequential,
)
from models.experimental.ops.descriptors.fusion.graph import (
    CoreGroup,
    OpGraphBuilder,
    OpNode,
    build_op_graph,
)
from models.experimental.ops.descriptors.fusion.common import (
    BarrierConfig,
)
from models.experimental.ops.descriptors.fusion.cb_allocator import (
    CBInfo,
    CBPoolAllocator,
    PhaseInfo,
    extract_cb_info,
    extract_cb_names_from_kernel,
)
from models.experimental.ops.descriptors.fusion.codegen import (
    collect_defines,
    collect_includes,
    extract_kernel_body,
    inline_local_includes,
)

__all__ = [
    # High-level API
    "Sequential",
    "Parallel",
    "FusedOp",
    # Graph API
    "OpNode",
    "CoreGroup",
    "OpGraphBuilder",
    "build_op_graph",
    # CB management
    "CBPoolAllocator",
    "CBInfo",
    "PhaseInfo",
    "BarrierConfig",
    # Functions
    "extract_cb_info",
    "extract_cb_names_from_kernel",
    # C++ parsing
    "extract_kernel_body",
    "inline_local_includes",
    "collect_includes",
    "collect_defines",
]
