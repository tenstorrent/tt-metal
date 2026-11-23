# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cache-hit host dispatch time of exp ring joint SDPA (untraced), the exp ring counterpart of
test_ring_joint_host_dispatch.py.

The case calls the op once to build and cache its program, then times calls that hit the program cache. The mesh is
synchronized before every timed call (calls on one global semaphore set must not overlap), so the time is the host's
work to dispatch one call (argument checks, program-cache lookup, runtime-argument and semaphore patching, output
allocation). Results are logged as "SDPA host dispatch <case>: ..." lines; the test only asserts that the op runs.
"""

import statistics
import time

import pytest
import torch
import ttnn
from loguru import logger

ITERS = 200


@pytest.fixture(scope="module")
def exp_ring_mesh():
    """One fabric ring with two links: a Galaxy row (1x8 submesh) or a QuietBox 2 (1x4)."""
    devices = ttnn.GetNumAvailableDevices()
    if devices < 4:
        pytest.skip("Requires a fabric ring of 4 or 8 Blackhole devices")
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
    # A Galaxy row is a submesh of the whole 4x8 mesh (the fabric routers of a smaller opened mesh are not all built).
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8) if devices >= 32 else ttnn.MeshShape(1, 4))
    ring = 8 if devices >= 32 else 4
    mesh = parent.create_submesh(ttnn.MeshShape(1, ring)) if devices >= 32 else parent
    mesh.enable_program_cache()
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
    mesh.load_sub_device_manager(manager)
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
    try:
        yield mesh, semaphores, ring
    finally:
        mesh.reset_sub_device_stall_group()
        mesh.clear_loaded_sub_device_manager()
        mesh.remove_sub_device_manager(manager)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# name, heads, local sequence, head dim, q chunk, k chunk (0: chosen by the op), grid (SDPA columns + 1 MUX column,
# rows)
CASES = [
    ("exp_ring_wan_like_s2368_nh10", 10, 2368, 128, 224, 512, (12, 10)),
    ("exp_ring_wan_like_s2368_nh10_op_chosen", 10, 2368, 128, 0, 0, (12, 10)),
]


def _deallocate(out):
    for t in out:
        t.deallocate(True)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_exp_ring_joint_host_dispatch(exp_ring_mesh, case):
    name, nh, s_local, d, q_chunk, k_chunk, grid = case
    mesh, semaphores, ring = exp_ring_mesh
    device_grid = mesh.compute_with_storage_grid_size()
    if device_grid.x < grid[0] or device_grid.y < grid[1]:
        pytest.skip(f"Requires a {grid[0]}x{grid[1]} worker grid")
    sq = s_local * ring

    def to_mesh(t, mapper):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

    q, k, v = (to_mesh(torch.randn(1, nh, sq, d), ttnn.ShardTensorToMesh(mesh, dim=2)) for _ in range(3))
    backing = [to_mesh(torch.zeros(1, nh, sq, d), ttnn.ReplicateTensorToMesh(mesh)) for _ in range(2)]
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
    )

    def call():
        return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=sq,
            program_config=program_config,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=2,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=ttnn.SubDeviceId(0),
            num_workers_per_link=grid[1] // 2,
            num_buffers_per_channel=16,
        )

    out = call()  # program-cache miss
    ttnn.synchronize_device(mesh)
    _deallocate(out)
    times = []
    for _ in range(ITERS):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        out = call()
        times.append((time.perf_counter() - t0) * 1e6)
        ttnn.synchronize_device(mesh)
        _deallocate(out)
    p10 = sorted(times)[len(times) // 10]
    logger.info(
        f"SDPA host dispatch {name}: median {statistics.median(times):.1f} us, p10 {p10:.1f} us, min {min(times):.1f} us "
        f"(ring {ring})"
    )
