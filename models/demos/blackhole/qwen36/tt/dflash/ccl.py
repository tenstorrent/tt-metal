# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Collectives for the DFlash drafter.

The drafter keeps its residual stream **replicated** and fractures only the projections, so
every row-parallel matmul needs a true all-reduce (a sum that leaves all devices holding the
whole tensor). That is not what ``tt_all_reduce`` gives on this hardware: when a mesh
dimension is 1 -- which covers T3K's (1, 8) and N300's (1, 2) -- it always takes the
``reduce_scatter_minimal_async`` branch and returns a tensor *scattered* along ``dim``
(``models/tt_transformers/tt/ccl.py``, the ``if 1 in list(mesh_device.shape)`` arm returns
early). The composite reduce-scatter-then-all-gather arm is only reachable on TG.

That scattered result is exactly what the target wants -- its residual stream stays
``dim/tp`` sharded and the reduce-scatter *is* the layout change. The drafter is only 5
layers on a 16-row block, so replication costs little and buys a much simpler port: no
distributed norm statistics, and no sharded-vs-replicated bookkeeping between modules.

So :func:`all_reduce_replicated` completes the reduce-scatter with an all-gather. Choosing
replication is a deliberate correctness-milestone trade; keeping the stream sharded (as the
target does) is the perf-oriented alternative and would remove one gather per call.
"""

from __future__ import annotations

import ttnn
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce


def ccl_topology(mesh_device) -> ttnn.Topology:
    """Ring on a full 8-chip T3K, Linear on narrower submeshes.

    Mirrors ``ModelArgs.ccl_topology`` (``models/tt_transformers/tt/model_config.py:2685``)
    rather than importing it, so the drafter needs no target ``ModelArgs`` instance.
    """
    num_devices = mesh_device.get_num_devices()
    cluster = ttnn.cluster.get_cluster_type()
    if cluster in (
        ttnn.cluster.ClusterType.P300_X2,
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.P150_X8,
    ):
        return ttnn.Topology.Ring
    if cluster in (
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.TG,
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
    ):
        return ttnn.Topology.Ring if num_devices >= 8 else ttnn.Topology.Linear
    return ttnn.Topology.Linear


def all_reduce_replicated(x, mesh_device, tt_ccl, topology=None, dim: int = 3):
    """Sum ``x`` across devices, leaving every device with the full result.

    ``x`` must be 4D with dims 0 and 1 equal to 1 -- ``tt_all_reduce`` reshapes anything else
    by indexing ``shape[-4]``, which is not present on a 3D tensor. ``x.shape[dim]`` must be
    divisible by the device count (the reduce-scatter splits it).
    """
    if mesh_device.get_num_devices() == 1:
        return x

    assert len(x.shape) == 4, f"all_reduce_replicated needs a 4D tensor, got {tuple(x.shape)}"
    assert x.shape[0] == 1 and x.shape[1] == 1, f"dims 0,1 must be 1, got {tuple(x.shape)}"
    n = mesh_device.get_num_devices()
    assert x.shape[dim] % n == 0, f"dim {dim} ({x.shape[dim]}) not divisible by {n} devices"

    topology = topology if topology is not None else ccl_topology(mesh_device)

    # The two calls take DIFFERENT cluster_axis values, and both are forced:
    #
    #  * tt_all_reduce needs cluster_axis=0. Its early-out is
    #    `cluster_axis == 1 and 1 in mesh_device.shape`, so on a (1, 8) T3K passing 1 would
    #    silently return the input UNREDUCED. Its reduce-scatter branch ignores cluster_axis
    #    entirely, so 0 is safe -- this is the same value the target's MLP passes.
    #  * tt_all_gather needs cluster_axis=None. It *honours* cluster_axis, and on a (1, 8)
    #    mesh the 8 devices lie along axis 1 while axis 0 has extent 1 -- passing 0 raises
    #    "all_gather_async op will only work for num_devices > 1, but has 1". Passing 1 hits
    #    the same early-out as above and returns un-gathered. None gathers over the whole
    #    mesh, which is what a 1-D device line wants.
    scattered = tt_all_reduce(
        x,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=dim,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gathered = tt_all_gather(
        scattered,
        mesh_device,
        tt_ccl,
        cluster_axis=None,
        dim=dim,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(scattered)
    return gathered
