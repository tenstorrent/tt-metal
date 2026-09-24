# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh bring-up: fabric selection, MeshConfig (SP rows x TP cols) and the CCL helpers.

Fabric default is the plain galaxy mesh descriptor with FABRIC_1D + Topology.Linear, which maps on
any galaxy. The torus is one env knob (``QWEN38_FABRIC=torus``) — a perf lever, never a correctness
gate. The descriptor must be chosen before the cluster initialises, so ``select_fabric_env()`` is
called from the package ``conftest.py`` at import time and by the standalone harnesses.

Collectives use the stateless ``ttnn.all_gather`` / ``ttnn.all_reduce`` with ``cluster_axis``; the
ring-joint SDPA needs global semaphores and a CCL core-grid offset, which come from the minimax_m3
``CCLManager`` (imported, same 8x4 galaxy, same op).
"""

from __future__ import annotations

import os

import ttnn

# L1_SMALL holds the semaphores of the stateless ttnn.all_gather / ttnn.all_reduce, one set per cached
# program. minimax_m3's 1152 B (sized for high_bw_all_gather only) runs out after ~a dozen distinct CCL
# shapes; 32 KiB (qwen36 reserves 24 KiB) leaves room for every shape the model and the test suite use.
L1_SMALL_SIZE = 32768

_DESC_DIR = "tt_metal/fabric/mesh_graph_descriptors"
_FABRICS = {
    # name: (fabric config, ccl topology, mesh graph descriptor)
    "linear": ("FABRIC_1D", "Linear", "single_bh_galaxy_mesh_graph_descriptor.textproto"),
    "torus": ("FABRIC_1D_RING", "Ring", "single_bh_galaxy_torus_xy_graph_descriptor.textproto"),
}


def fabric_name() -> str:
    name = os.environ.get("QWEN38_FABRIC", "linear")
    assert name in _FABRICS, f"QWEN38_FABRIC must be one of {list(_FABRICS)}, got {name}"
    return name


def select_fabric_env() -> str:
    """Point TT_MESH_GRAPH_DESC_PATH at the descriptor for the chosen fabric (before cluster init)."""
    name = fabric_name()
    home = os.environ.get("TT_METAL_HOME", os.getcwd())
    os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", os.path.join(home, _DESC_DIR, _FABRICS[name][2]))
    return name


def fabric_config():
    return getattr(ttnn.FabricConfig, _FABRICS[fabric_name()][0])


def ccl_topology():
    return getattr(ttnn.Topology, _FABRICS[fabric_name()][1])


def open_mesh(mesh_shape=(8, 4)):
    select_fabric_env()
    ttnn.set_fabric_config(fabric_config())
    return ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape), l1_small_size=L1_SMALL_SIZE)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


class MeshConfig:
    """(rows, cols) = (SP, TP). Rows carry sequence parallelism, columns carry tensor parallelism."""

    sp_axis = 0
    tp_axis = 1

    def __init__(self, mesh_device, sp: int, tp: int):
        shape = tuple(mesh_device.shape)
        assert shape == (sp, tp), f"mesh {shape} != spec (sp={sp}, tp={tp})"
        self.mesh_device = mesh_device
        self.sp, self.tp = sp, tp
        self.topology = ccl_topology()

    # ---- mappers ----
    def replicate(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device)

    def shard(self, sp_dim=None, tp_dim=None):
        """Shard ``sp_dim`` over mesh rows and ``tp_dim`` over mesh columns (None = replicate)."""
        return ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(sp_dim, tp_dim))

    def compose(self, sp_dim, tp_dim):
        return ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(sp_dim, tp_dim), mesh_shape=self.mesh_device.shape)

    # ---- collectives ----
    def all_reduce_tp(self, x, memory_config=None):
        return ttnn.all_reduce(
            x,
            cluster_axis=self.tp_axis,
            topology=self.topology,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    def all_gather(self, x, dim, axis, memory_config=None):
        return ttnn.all_gather(
            x,
            dim,
            cluster_axis=axis,
            topology=self.topology,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    def all_gather_sp(self, x, dim=2):
        return self.all_gather(x, dim, self.sp_axis)

    def all_gather_tp(self, x, dim=3):
        return self.all_gather(x, dim, self.tp_axis)


def make_ccl_manager(mesh_device):
    from models.demos.minimax_m3.tt.ccl import CCLManager

    # Blackhole galaxy: 2 links per neighbour (minimax_m3 utils.get_default_num_links).
    return CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
