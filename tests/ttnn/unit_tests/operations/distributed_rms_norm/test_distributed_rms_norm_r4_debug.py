# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4 (TRANSPORT) probe: per chip / direction / link, the Ethernet channel setup_fabric_connection names,
its NoC coordinate, the current port core and the nearest-NoC1 port core. Host-only (no dispatch). DO NOT DELETE."""

import pytest
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd
from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as ag


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": production_router_config()}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
def test_port_placement_probe(mesh_device):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    G = 4
    cur = pd.port_cores(grid, 4)
    print(
        "\ncurrent ports (logical):",
        [(c.x, c.y) for c in cur],
        "virt:",
        [(mesh.worker_core_from_logical_core(c).x, mesh.worker_core_from_logical_core(c).y) for c in cur],
    )
    allowed = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    grp = pd.build_groups(tuple(mesh.shape), 1)[0]
    for p, coord in enumerate(grp):
        taken = set()
        for d, peer in ((pd.FWD, grp[(p + 1) % G]), (pd.BWD, grp[(p - 1) % G])):
            me, pn = pd._node(mesh, coord), pd._node(mesh, peer)
            links = ttnn.get_forwarding_link_indices(me, pn)
            for l in range(min(2, len(links))):
                prog = ttnn.ProgramDescriptor()
                args = list(ttnn.setup_fabric_connection(me, pn, links[l], prog, ttnn.CoreCoord(0, 0)))
                ch = args[0]
                eth_list, translated, t2p = ag._chip_maps(mesh, coord)
                exy = eth_list[ch] if eth_list is not None else None
                core = ag._worker_nearest_noc1(mesh, coord, exy, taken, allowed) if exy else None
                if core is not None:
                    taken.add((core.x, core.y))
                print(
                    f"chip {coord} p={p} d={d} link_idx={links[l]} chan={ch} eth_noc={exy} translated={translated} "
                    f"nearest={(core.x, core.y) if core else None} "
                    f"virt={(mesh.worker_core_from_logical_core(core).x, mesh.worker_core_from_logical_core(core).y) if core else None}"
                )
