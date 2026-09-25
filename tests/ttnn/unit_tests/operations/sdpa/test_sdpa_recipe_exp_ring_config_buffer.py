# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exp ring recipes at the default worker L1 (kernel config buffer regression).

The Tensix kernel config buffer holds every program's kernel binaries; it is the L1 between the kernel
config base and the allocator's unreserved base, so it shrinks as worker_l1_size grows: 70656 B at the
default Blackhole worker L1, ~187 KB at MiniMax H3's worker_l1_size=1344544 (the exp ring suites' default).
At the previously qualified geometries the exp ring COMPENSATED / LOW_PRECISION programs (pack-only -Os
for odd Q, all -O2 for even Q, plus the MUX writer) measure 71.3-76.0 KB, so at the default L1 they threw
``Program size (...) too large for kernel config buffer (70656)`` (e.g. the DiT parity cases
wan_exp_720p_4x32 and h3_exp_4x32). Such builds now take the generic-geometry size flags (pack and unpack
at -Os); builds with a large kernel config buffer keep their qualified flags.

Device tests run on a default-worker-L1 1x2 mesh: the exp ring recipe suite's bit-exact gate (each chip
equals the dense recipe in visiting order) at the qualified Q256/Q128/Q224 K512 geometries.
"""

import pytest
import ttnn

from models.common.utility_functions import is_blackhole
from . import test_sdpa_recipe_exp_ring as exp_ring
from .sdpa_recipe_test_utils import VARIANTS

# ---------------------------------------------------------------------------------------- device


@pytest.fixture(scope="module")
def exp_ring_mesh():
    """The exp ring recipe suite's 1x2 FABRIC_1D_RING mesh, at the default worker L1."""
    if not is_blackhole() or ttnn.GetNumAvailableDevices() != 2:
        pytest.skip("requires two connected Blackholes")
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    mesh = manager = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), trace_region_size=16777216)
        mesh.enable_program_cache()
        hardware = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hardware.x - 1, hardware.y - 1))})
        subdevice = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        yield mesh, subdevice, semaphores
    finally:
        if mesh is not None:
            if manager is not None:
                mesh.reset_sub_device_stall_group()
                mesh.clear_loaded_sub_device_manager()
                mesh.remove_sub_device_manager(manager)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# The exp ring suite's gate (bit-exact against the dense recipe in visiting order where chunk-aligned) on the
# qualified geometries whose builds did not fit, one and several passes, joint KV and padded tails.
QUALIFIED_CASES = [
    ("aligned", 256),
    ("joint", 256),
    ("padded-tails", 256),
    ("two-pass", 256),
    ("three-pass-joint-skip", 256),
    ("aligned", 128),
    ("two-pass", 128),
    ("aligned", 224),
    ("two-pass", 224),
]


@pytest.mark.parametrize("variant", VARIANTS[1:])
@pytest.mark.parametrize("case, q_chunk", QUALIFIED_CASES, ids=[f"{c}-q{q}" for c, q in QUALIFIED_CASES])
def test_qualified_geometry_default_l1(exp_ring_mesh, case, q_chunk, variant, record_property):
    exp_ring.test_recipe_exp_ring(exp_ring_mesh, case, variant, q_chunk, record_property)
