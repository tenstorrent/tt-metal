#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-host fabric smoke test: Galaxy on host 40 <-> Galaxy on host 41.

Run under tt-run with dual_8x4_hosts_40_41_rank_bindings.yaml (one MPI rank per
host). The two Galaxies form ONE logical 16x4 mesh -- 8 rows per host -- which is
the same shape the repo's own DUAL_BH CCL tests use. Each rank shards a tensor
along the host-spanning axis, runs ttnn.all_reduce over the QSFP-DD fabric, and
verifies every local device shard against a torch reference.

A pass means: fabric came up across the 40<->41 cable bundle AND carried real
traffic correctly. Exit code 0 == pass.

Mirrors tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/
test_all_reduce_exabox.py: shard with ShardTensor2dMesh so the output topology
lets ttnn.get_device_tensors() return per-device shards, then compare each shard
to the torch reference with PCC.
"""

import os
import sys

import torch

import ttnn

# Full logical mesh: 16 rows x 4 cols, split 8 rows per host.
GLOBAL_MESH = (16, 4)

# The axis that spans the two hosts. host_topology is [2, 1], so the mesh is
# partitioned along dim 0 -> cluster_axis 0 crosses the host boundary.
CLUSTER_AXIS = 0

# Tile-aligned; same shape the repo's all_reduce exabox test uses.
INPUT_SHAPE = (1, 1, 32, 224)
PCC_THRESHOLD = 0.998


def build_inputs(mesh_shape, memory_config, device):
    """One unique shard per position along CLUSTER_AXIS, replicated across the other."""
    num_along_axis = mesh_shape[CLUSTER_AXIS]
    torch.manual_seed(0)
    shards = [torch.rand(INPUT_SHAPE).bfloat16() for _ in range(num_along_axis)]
    torch_input = torch.cat(shards, dim=0)

    shard_dims = (0, None) if CLUSTER_AXIS == 0 else (None, 0)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        device=device,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
    )
    torch_reference = torch.stack(shards, dim=0).sum(dim=0)
    return tt_input, torch_reference


def verify(tt_output, torch_reference, mesh_device, rank):
    """Compare every LOCAL device shard to the reference. Returns True on pass."""
    from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

    coords = list(tt_output.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_output)

    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [c for c in coords if view.is_local(c)]

    ok = True
    checked = 0
    for coord, tt_out in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        passed, message = comp_pcc(torch_reference, ttnn.to_torch(tt_out), PCC_THRESHOLD)
        checked += 1
        if not passed:
            print(f"[rank {rank}] device {coord}: {message}")
            ok = False

    if checked == 0:
        print(f"[rank {rank}] ERROR: no local shards were checked")
        return False
    print(f"[rank {rank}] verified {checked} local device shards")
    return ok


def main() -> int:
    # The MPI world is NOT initialized just by running under tt-run: opening a
    # mesh device normally does it, but this script needs rank/size before that.
    # Initialize explicitly so the rank/world-size calls below are valid.
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    rank = int(ttnn.distributed_context_get_rank())
    print(f"[rank {rank}/{world_size}] starting cross-host fabric check on hosts 40<->41", flush=True)

    if world_size != 2:
        print(f"ERROR: expected 2 ranks (one Galaxy per host), got {world_size}")
        return 2

    # Fabric must be configured before the mesh is opened.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    # COL dispatch is required on Blackhole unless fabric_tensix_config is also set.
    dispatch_core_config = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL, None)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*GLOBAL_MESH),
        dispatch_core_config=dispatch_core_config,
        trace_region_size=90112,
    )
    # get_num_devices() reports the GLOBAL mesh size (16x4 = 64), which spans both
    # hosts. The per-rank count comes from the view's local coordinate range.
    if ttnn.using_distributed_env():
        local_range = mesh_device.get_view().get_local_mesh_coord_range()
        local_count = 1
        for extent in local_range.shape():
            local_count *= extent
        print(
            f"[rank {rank}] opened mesh {tuple(mesh_device.shape)} globally; "
            f"{local_count} local devices (rows {tuple(local_range.start_coord())}..)"
        )
    else:
        print(f"[rank {rank}] opened mesh {tuple(mesh_device.shape)}; {mesh_device.get_num_devices()} devices")

    if ttnn.using_distributed_env():
        # MPI_Init_thread leaves OpenMP at 1 thread; restore for torch reference work.
        torch.set_num_threads(max(1, os.cpu_count()))

    try:
        mesh_shape = tuple(mesh_device.shape)
        if mesh_shape[CLUSTER_AXIS] < 2:
            print(f"ERROR: mesh shape {mesh_shape} cannot reduce along axis {CLUSTER_AXIS}")
            return 2

        for buffer_type, label in ((ttnn.BufferType.DRAM, "DRAM"), (ttnn.BufferType.L1, "L1")):
            memory_config = ttnn.MemoryConfig(buffer_type=buffer_type)
            tt_input, torch_reference = build_inputs(mesh_shape, memory_config, mesh_device)

            print(f"[rank {rank}] all_reduce across the host boundary ({label})...")
            tt_output = ttnn.all_reduce(
                tt_input,
                cluster_axis=CLUSTER_AXIS,
                topology=ttnn.Topology.Linear,
                memory_config=memory_config,
                num_links=2,
            )
            ttnn.synchronize_device(mesh_device)

            if not verify(tt_output, torch_reference, mesh_device, rank):
                print(f"[rank {rank}] FAIL ({label})")
                return 1
            print(f"[rank {rank}] PASS ({label})")

        ttnn.distributed_context_barrier()
        print(f"[rank {rank}] ALL CHECKS PASSED -- fabric 40<->41 is healthy")
        return 0

    finally:
        try:
            for submesh in mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
