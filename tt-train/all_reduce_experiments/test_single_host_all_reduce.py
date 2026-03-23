#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-host all-reduce correctness test.

Creates per-device random tensors, shards them across an (N, 1) mesh along dim 0,
runs all-reduce (sum), reads back, and compares against a torch golden.

Usage:
    python test_single_host_all_reduce.py
    python test_single_host_all_reduce.py --num-devices 8
    python test_single_host_all_reduce.py --num-devices 32 --per-chip-shape 1 1 64 2048
    python test_single_host_all_reduce.py --topology ring
"""

import argparse
import time
from dataclasses import dataclass

import torch
import ttnn


@dataclass
class AllReduceResources:
    """Pre-allocated sub-device manager and semaphores needed by all_reduce_async."""

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


def all_reduce(tensor: ttnn.Tensor, resources: AllReduceResources) -> ttnn.Tensor:
    """Run all-reduce (sum) along cluster_axis=0 on an (N, 1) mesh and synchronize."""
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


def main():
    parser = argparse.ArgumentParser(description="Single-host all-reduce correctness test")
    parser.add_argument("--num-devices", type=int, default=32, help="Number of devices (mesh is Nx1)")
    parser.add_argument(
        "--per-chip-shape",
        type=int,
        nargs=4,
        default=[1, 1, 32, 1024],
        help="Per-chip output tensor shape (4D)",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="linear",
        choices=["linear", "ring"],
        help="CCL topology (linear or ring)",
    )
    parser.add_argument(
        "--fabric-config",
        type=str,
        default=None,
        choices=["1D", "1D_RING", "2D"],
        help="Fabric config override. If not set, inferred from --topology (linear→1D, ring→1D_RING)",
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Total iterations (first is warmup + correctness check)"
    )
    parser.add_argument("--pcc-threshold", type=float, default=0.999)
    args = parser.parse_args()

    num_devices = args.num_devices
    per_chip_shape = args.per_chip_shape
    topology = ttnn.Topology.Ring if args.topology == "ring" else ttnn.Topology.Linear

    fabric_config_map = {
        "1D": ttnn.FabricConfig.FABRIC_1D,
        "1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
        "2D": ttnn.FabricConfig.FABRIC_2D,
    }
    if args.fabric_config:
        fabric_config = fabric_config_map[args.fabric_config]
    else:
        fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if args.topology == "ring" else ttnn.FabricConfig.FABRIC_1D

    print(f"Num devices:    {num_devices}")
    print(f"Per-chip shape: {per_chip_shape}")
    print(f"Topology:       {args.topology}")
    print(f"Fabric config:  {fabric_config}")
    print()

    # --- Open mesh device ---
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.set_fabric_config(fabric_config)

    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(num_devices, 1))
    resources = setup_all_reduce_resources(mesh_device, topology=topology)

    try:
        # --- Create per-device input tensors in torch ---
        per_device_tensors = [torch.rand(per_chip_shape, dtype=torch.bfloat16) for _ in range(num_devices)]
        golden = torch.stack(per_device_tensors, dim=0).sum(dim=0)

        # Concatenate along dim 0 to form the full tensor that ShardTensor2dMesh will split back
        full_tensor = torch.cat(per_device_tensors, dim=0)
        print(f"Full input tensor shape: {list(full_tensor.shape)}")
        print(f"Golden shape:            {list(golden.shape)}")

        # --- Send to mesh, all-reduce, read back ---
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
        tt_output = all_reduce(tt_input, resources)

        tt_out_tensors = ttnn.get_device_tensors(tt_output)
        num_mismatches = 0
        for i, t in enumerate(tt_out_tensors):
            tt_result = ttnn.to_torch(t)
            pcc = torch.nn.functional.cosine_similarity(
                tt_result.flatten().float(), golden.flatten().float(), dim=0
            ).item()
            passed = pcc >= args.pcc_threshold
            print(f"  Device {i:2d}: PCC = {pcc:.6f}  [{'PASS' if passed else 'FAIL'}]")
            if not passed:
                num_mismatches += 1

        print()
        if num_mismatches == 0:
            print(f"ALL PASSED: {len(tt_out_tensors)} devices match golden (PCC >= {args.pcc_threshold})")
        else:
            print(f"FAILED: {num_mismatches}/{len(tt_out_tensors)} devices below PCC threshold")
        assert num_mismatches == 0, f"{num_mismatches} device(s) failed PCC check"

        # --- Remaining iterations: timing ---
        measure_iters = args.iterations - 1
        if measure_iters > 0:
            ttnn.synchronize_device(mesh_device, sub_device_ids=resources.sub_device_stall_group)
            t_start = time.perf_counter()
            for _ in range(measure_iters):
                all_reduce(tt_input, resources)
            t_end = time.perf_counter()
            avg_ms = (t_end - t_start) / measure_iters * 1000
            print(f"Avg all-reduce time: {avg_ms:.3f} ms  ({measure_iters} iters)")

    finally:
        mesh_device.reset_sub_device_stall_group()
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
