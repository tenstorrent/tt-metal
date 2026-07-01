# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Open a 1×tp Blackhole mesh for the TP=8 VLM-prefill pipeline.

FABRIC_1D is enabled at mesh open so the tensor-parallel AllReduce/AllGather
collectives can route across the chips. It is disabled again on context exit;
otherwise a subsequent ttnn.open_mesh_device call in the same process inherits a
stale fabric config and the underlying mesh fails to initialize.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import ttnn


@contextmanager
def open_prefill_tp8_mesh(
    tp: int = 4,
    l1_small_size: Optional[int] = None,
    trace_region_size: Optional[int] = None,
    enable_fabric: bool = True,
    num_command_queues: int = 1,
):
    """Open a 1×tp mesh for TP VLM prefill on an 8-chip (or larger) device.

    Yields the mesh device directly.  The caller is responsible for uploading
    inputs as ReplicateTensorToMesh and downloading outputs via
    ttnn.get_device_tensors(out)[0] + ttnn.to_torch().

    Args:
        tp:               tensor-parallel degree (number of chips, default 4).
        l1_small_size:    bytes per core for static CBs.
        trace_region_size: bytes per chip for trace capture.
        enable_fabric:    set FABRIC_1D for AllReduce collectives (default True).
    """
    if enable_fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    open_kwargs = {"mesh_shape": ttnn.MeshShape(1, tp)}
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    if trace_region_size is not None:
        open_kwargs["trace_region_size"] = trace_region_size
    if num_command_queues != 1:
        open_kwargs["num_command_queues"] = num_command_queues

    mesh = ttnn.open_mesh_device(**open_kwargs)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        if enable_fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
