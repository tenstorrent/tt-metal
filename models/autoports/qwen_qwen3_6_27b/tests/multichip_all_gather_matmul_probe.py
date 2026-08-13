# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape-faithful TP4 fractured-residual all-gather + column matmul probe."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in0-block-w", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("fused", "separate"), required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    devices, m, hidden, local_n = 4, 32, 5120, 3584
    activation = (torch.randn(1, 1, m, hidden) * 0.2).bfloat16()
    weight = (torch.randn(1, 1, hidden, local_n * devices) / hidden**0.5).bfloat16()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, devices), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        subdevice = ttnn.SubDevice([cores])
        subdevice_id = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([subdevice], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice_id])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]

        tt_x = ttnn.from_torch(
            activation,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_w = ttnn.from_torch(
            weight,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        persistent = ttnn.from_torch(
            torch.zeros(1, 1, m, hidden),
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=args.in0_block_w,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=14,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        def run():
            if args.mode == "separate":
                gathered = ttnn.experimental.all_gather_async(
                    tt_x,
                    persistent_output_buffer=persistent,
                    dim=3,
                    multi_device_global_semaphore=semaphores,
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Ring,
                    subdevice_id=subdevice_id,
                )
                return gathered, ttnn.linear(
                    gathered,
                    tt_w,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=program,
                    compute_kernel_config=compute,
                )
            return ttnn.experimental.all_gather_matmul_async(
                tt_x,
                tt_w,
                persistent_output_buffer=persistent,
                dim=3,
                multi_device_global_semaphore=semaphores,
                all_gather_core_grid_offset=ttnn.CoreCoord(0, 6),
                num_links=1,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=subdevice_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=compute,
            )

        gathered, output = run()
        ttnn.synchronize_device(mesh, sub_device_ids=[subdevice_id])
        expected = torch.matmul(activation.float(), weight.float())
        actual = torch.cat([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(output)], dim=-1)
        passed, message = comp_pcc(expected, actual, 0.999)
        if not passed:
            raise AssertionError(message)
        for _ in range(3):
            run()
        ttnn.synchronize_device(mesh, sub_device_ids=[subdevice_id])
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            run()
            ttnn.synchronize_device(mesh, sub_device_ids=[subdevice_id])
            samples.append((time.perf_counter() - started) * 1000)
        result = {
            "mesh": [1, 4],
            "shape": [m, hidden // devices, local_n],
            "global_k": hidden,
            "dtype": "BF16",
            "fidelity": "HiFi2",
            "in0_block_w": args.in0_block_w,
            "grid": [8, 6],
            "all_gather_offset": [0, 6],
            "persistent_buffers": True,
            "async_ccl": True,
            "mode": args.mode,
            "pcc": str(message),
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
        }
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("MULTICHIP_ALL_GATHER_MATMUL", json.dumps(result, sort_keys=True))
    finally:
        try:
            mesh.reset_sub_device_stall_group()
            mesh.clear_loaded_sub_device_manager()
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
