# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL manager (the gpt_oss_d_p one is model-agnostic: semaphores, ring-gather buffers, CCL column). The fabric-link
count shared by the ring SDPA and the MoE dispatch / combine / reduce is ``MiMoRuntimeOptions.num_links`` (None:
resolved per system by :func:`resolve_num_links`)."""

from models.demos.deepseek_v3_d_p.tt.tt_ccl import _determine_device_name, get_num_links
from models.demos.gpt_oss_d_p.tt.ccl import CCLManager as _CCLManager

__all__ = ["CCLManager", "resolve_num_links"]


def resolve_num_links(mesh_device, num_links=None):
    """``num_links`` if given, else the system's: 3 on the Blackhole QuietBox (4 x p150; measured best: MoE dispatch
    2.59 / 1.31 / 0.88 ms for 1 / 2 / 3 links at 640 tok/chip), else the deepseek_v3_d_p table (BH Galaxy: 2 ethernet
    channels between neighbours; more fail with "Requested link index 2 is out of bounds")."""
    if num_links is not None:
        return num_links
    return 3 if _determine_device_name(mesh_device) == "P150x4" else get_num_links(mesh_device)


class CCLManager(_CCLManager):
    def __init__(self, mesh_device, num_links=None, **kw):
        super().__init__(mesh_device, num_links=resolve_num_links(mesh_device, num_links), **kw)
