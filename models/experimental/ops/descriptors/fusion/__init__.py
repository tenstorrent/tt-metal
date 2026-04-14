# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    clear_build_cache,
)
from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.graph import (
    OpGraphBuilder,
    OpNode,
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
    num_cbs_for_device,
)
from models.experimental.ops.descriptors.fusion.codegen import (
    collect_defines,
    collect_includes,
    inline_local_includes,
)

__all__ = [
    # High-level API
    "Sequential",
    "Parallel",
    "FusedOp",
    "OpDescriptor",
    # Graph API
    "OpNode",
    "OpGraphBuilder",
    # CB management
    "CBPoolAllocator",
    "CBInfo",
    "PhaseInfo",
    "BarrierConfig",
    # Functions
    "extract_cb_info",
    "extract_cb_names_from_kernel",
    "num_cbs_for_device",
    "clear_build_cache",
    # C++ parsing
    "inline_local_includes",
    "collect_includes",
    "collect_defines",
]
