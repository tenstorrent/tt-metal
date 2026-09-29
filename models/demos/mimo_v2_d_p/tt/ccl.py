# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL manager (the gpt_oss_d_p one is model-agnostic: semaphores, ring-gather buffers, CCL column). The fabric-link
count shared by the ring SDPA and the MoE dispatch / combine / reduce is ``MiMoRuntimeOptions.num_links`` (None:
resolved per system by :func:`resolve_num_links`)."""

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_num_links
from models.demos.gpt_oss_d_p.tt.ccl import CCLManager as _CCLManager

__all__ = ["CCLManager", "resolve_num_links"]


def resolve_num_links(mesh_device, num_links=None):
    """``num_links`` if given, else the system's (by the cluster type, so a submesh of a Galaxy is still a Galaxy):
    3 on the Blackhole QuietBox (4 x p150; measured best: MoE dispatch 2.59 / 1.31 / 0.88 ms for 1 / 2 / 3 links at
    640 tok/chip), 2 on a Blackhole Galaxy (2 ethernet channels between neighbours; more fail with "Requested link
    index 2 is out of bounds"), else the deepseek_v3_d_p per-system table."""
    if num_links is not None:
        return num_links
    cluster = ttnn.cluster.get_cluster_type()
    if cluster == ttnn.cluster.ClusterType.P150_X4:
        return 3
    if cluster == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
        return 2
    return get_num_links(mesh_device)


class CCLManager(_CCLManager):
    def __init__(self, mesh_device, num_links=None, **kw):
        super().__init__(mesh_device, num_links=resolve_num_links(mesh_device, num_links), **kw)
