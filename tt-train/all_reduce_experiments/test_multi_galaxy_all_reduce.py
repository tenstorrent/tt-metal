#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-galaxy all-reduce correctness test.

Each galaxy:
  1. Creates 32 per-device random tensors (seeded by rank)
  2. Local all-reduce (sum across 32 devices)
  3. Inter-galaxy ring all-reduce via sockets (32-to-32 connections)
  4. Compares result against torch golden computed from all ranks' data

Usage (via run_4galaxy.sh):
    bash run_4galaxy.sh
    bash run_4galaxy.sh --per-chip-shape 1 1 64 2048
"""

import argparse
import time
from dataclasses import dataclass

import torch
import ttnn


# ---------------------------------------------------------------------------
# Intra-galaxy all-reduce (reused from single-host test)
# ---------------------------------------------------------------------------


@dataclass
class AllReduceResources:
    mesh_device: ttnn.MeshDevice
    topology: ttnn.Topology
    worker_sub_device_id: ttnn.SubDeviceId
    sub_device_stall_group: list
    barrier_semaphores: list
    rs_global_semaphores: list
    ag_global_semaphores: list
    mem_config: ttnn.MemoryConfig


def setup_all_reduce_resources(
    mesh_device: ttnn.MeshDevice,
    topology: ttnn.Topology = ttnn.Topology.Linear,
) -> AllReduceResources:
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1),
            )
        }
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    return AllReduceResources(
        mesh_device=mesh_device,
        topology=topology,
        worker_sub_device_id=worker_sub_device_id,
        sub_device_stall_group=sub_device_stall_group,
        barrier_semaphores=[ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)],
        rs_global_semaphores=[ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)],
        ag_global_semaphores=[ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)],
        mem_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )


def local_all_reduce(tensor: ttnn.Tensor, resources: AllReduceResources) -> ttnn.Tensor:
    result = ttnn.experimental.all_reduce_async(
        tensor,
        cluster_axis=0,
        mesh_device=resources.mesh_device,
        barrier_semaphores=resources.barrier_semaphores,
        rs_global_semaphores=resources.rs_global_semaphores,
        ag_global_semaphores=resources.ag_global_semaphores,
        math_op=ttnn.ReduceType.Sum,
        memory_config=resources.mem_config,
        topology=resources.topology,
        subdevice_id=resources.worker_sub_device_id,
    )
    ttnn.synchronize_device(resources.mesh_device, sub_device_ids=resources.sub_device_stall_group)
    return result


# ---------------------------------------------------------------------------
# Inter-galaxy ring all-reduce via sockets
# ---------------------------------------------------------------------------


def create_socket_connections(mesh_shape: ttnn.MeshShape):
    """One connection per device: device (i,j) on sender <-> device (i,j) on receiver."""
    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
            )
        )
    return connections


def inter_galaxy_ring_all_reduce(
    tensor: ttnn.Tensor,
    rank: int,
    world_size: int,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: ttnn.MeshShape,
    per_chip_shape: list,
) -> ttnn.Tensor:
    """
    Ring all-reduce across galaxies using point-to-point sockets.

    Algorithm (N = world_size, N-1 rounds):
      accumulator = tensor
      to_send = tensor
      for each round:
          send to_send → next_rank
          recv received ← prev_rank
          accumulator += received
          to_send = received          (pass along what we just received)

    After N-1 rounds every rank holds the sum of all ranks' tensors.
    """
    if world_size == 1:
        return tensor

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    connections = create_socket_connections(mesh_shape)
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, 1024 * 1024)

    send_config = ttnn.SocketConfig(
        connections,
        socket_mem_config,
        sender_rank=rank,
        receiver_rank=next_rank,
    )
    recv_config = ttnn.SocketConfig(
        connections,
        socket_mem_config,
        sender_rank=prev_rank,
        receiver_rank=rank,
    )

    # Even ranks create send first, odd ranks create recv first.
    # This avoids a circular deadlock where every rank blocks on send handshake
    # waiting for the next rank, which is itself blocked on its own send handshake.
    if rank % 2 == 0:
        send_socket = ttnn.MeshSocket(mesh_device, send_config)
        recv_socket = ttnn.MeshSocket(mesh_device, recv_config)
    else:
        recv_socket = ttnn.MeshSocket(mesh_device, recv_config)
        send_socket = ttnn.MeshSocket(mesh_device, send_config)

    ttnn.distributed_context_barrier()

    recv_spec = ttnn.TensorSpec(per_chip_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT)
    accumulator = tensor
    to_send = tensor

    for step in range(world_size - 1):
        recv_buffer = ttnn.allocate_tensor_on_device(recv_spec, mesh_device)
        ttnn.experimental.send_async(to_send, send_socket)
        ttnn.experimental.recv_async(recv_buffer, recv_socket)
        ttnn.synchronize_device(mesh_device)

        accumulator = ttnn.add(accumulator, recv_buffer)
        to_send = recv_buffer

    ttnn.synchronize_device(mesh_device)

    del send_socket
    del recv_socket

    return accumulator


# ---------------------------------------------------------------------------
# Golden computation
# ---------------------------------------------------------------------------


def compute_golden(world_size: int, num_devices: int, per_chip_shape: list) -> torch.Tensor:
    """
    Reproduce what the device computation does, entirely in torch.

    For each rank r (seed=r): generate 32 per-device tensors, sum them (= local all-reduce).
    Then sum across all ranks (= inter-galaxy all-reduce).
    """
    golden = torch.zeros(per_chip_shape, dtype=torch.float32)
    for r in range(world_size):
        torch.manual_seed(r)
        for _ in range(num_devices):
            golden += torch.rand(per_chip_shape, dtype=torch.bfloat16).float()
    return golden.bfloat16()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Multi-galaxy all-reduce correctness test")
    parser.add_argument("--num-devices", type=int, default=32, help="Devices per galaxy (mesh is Nx1)")
    parser.add_argument(
        "--per-chip-shape",
        type=int,
        nargs=4,
        default=[1, 1, 32, 1024],
        help="Per-chip tensor shape (4D)",
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Total iterations (first is warmup + correctness check)"
    )
    parser.add_argument("--pcc-threshold", type=float, default=0.99)
    args = parser.parse_args()

    num_devices = args.num_devices
    per_chip_shape = args.per_chip_shape

    # Multi-mesh requires FABRIC_2D
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    mesh_shape = ttnn.MeshShape(num_devices, 1)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    resources = setup_all_reduce_resources(mesh_device)

    rank = int(ttnn.distributed_context_get_rank())
    world_size = int(ttnn.distributed_context_get_size())
    print(f"[Rank {rank}/{world_size}] Devices: {num_devices}, per-chip: {per_chip_shape}")

    try:
        # --- Generate this rank's per-device tensors (seed = rank) ---
        torch.manual_seed(rank)
        my_per_device_tensors = [torch.rand(per_chip_shape, dtype=torch.bfloat16) for _ in range(num_devices)]

        # --- Compute golden (all ranks can do this independently) ---
        golden = compute_golden(world_size, num_devices, per_chip_shape)
        print(f"[Rank {rank}] Golden shape: {list(golden.shape)}")

        # --- Shard per-device tensors onto mesh ---
        full_tensor = torch.cat(my_per_device_tensors, dim=0)
        tt_input = ttnn.from_torch(
            full_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=resources.mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                mesh_shape=(num_devices, 1),
                dims=(0, 1),
            ),
        )

        # --- Iteration 1: warmup + correctness check ---
        print(f"[Rank {rank}] Local all-reduce...")
        tt_local = local_all_reduce(tt_input, resources)

        print(f"[Rank {rank}] Inter-galaxy ring all-reduce ({world_size} ranks)...")
        tt_result = inter_galaxy_ring_all_reduce(
            tt_local,
            rank,
            world_size,
            mesh_device,
            mesh_shape,
            per_chip_shape,
        )

        print(f"[Rank {rank}] Verifying...")
        tt_out_tensors = ttnn.get_device_tensors(tt_result)
        num_mismatches = 0
        for i, t in enumerate(tt_out_tensors):
            tt_torch = ttnn.to_torch(t)
            pcc = torch.nn.functional.cosine_similarity(
                tt_torch.flatten().float(),
                golden.flatten().float(),
                dim=0,
            ).item()
            if pcc < args.pcc_threshold:
                print(f"  [Rank {rank}] Device {i:2d}: PCC = {pcc:.6f} [FAIL]")
                num_mismatches += 1

        if num_mismatches == 0:
            print(f"[Rank {rank}] ALL PASSED: {len(tt_out_tensors)} devices (PCC >= {args.pcc_threshold})")
        else:
            print(f"[Rank {rank}] FAILED: {num_mismatches}/{len(tt_out_tensors)} devices below threshold")

        ttnn.distributed_context_barrier()
        assert num_mismatches == 0, f"Rank {rank}: {num_mismatches} device(s) failed PCC check"

        # --- Remaining iterations: timing ---
        measure_iters = args.iterations - 1
        if measure_iters > 0:
            ttnn.synchronize_device(mesh_device)
            ttnn.distributed_context_barrier()

            t_start = time.perf_counter()
            for _ in range(measure_iters):
                tt_local = local_all_reduce(tt_input, resources)
                tt_result = inter_galaxy_ring_all_reduce(
                    tt_local,
                    rank,
                    world_size,
                    mesh_device,
                    mesh_shape,
                    per_chip_shape,
                )
            ttnn.synchronize_device(mesh_device)
            t_end = time.perf_counter()

            avg_ms = (t_end - t_start) / measure_iters * 1000
            print(f"[Rank {rank}] Avg local+global all-reduce time: {avg_ms:.3f} ms  ({measure_iters} iters)")

            ttnn.distributed_context_barrier()

    finally:
        mesh_device.reset_sub_device_stall_group()
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
