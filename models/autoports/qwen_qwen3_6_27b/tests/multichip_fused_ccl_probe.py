# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape-faithful TP4 fused matmul/reduce-scatter probe for Qwen output projection."""

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
    parser.add_argument("--in0-block-w", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("fused", "separate", "both"), default="both")
    parser.add_argument("--grid-y", type=int, choices=(4, 6), default=6)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    devices, m, local_k, n = 4, 32, 1536, 5120
    global_k = devices * local_k
    activation = (torch.randn(1, 1, m, global_k) * 0.2).bfloat16()
    weight = (torch.randn(1, 1, global_k, n) / global_k**0.5).bfloat16()

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
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]

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
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        intermediate = ttnn.from_torch(
            torch.zeros(1, 1, m, n),
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        persistent = ttnn.from_torch(
            torch.zeros(1, 1, m, n // devices),
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, args.grid_y),
            in0_block_w=args.in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=20,
            out_block_w=10,
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

        def separate():
            mm = ttnn.linear(
                tt_x,
                tt_w,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=program,
                compute_kernel_config=compute,
            )
            rs = ttnn.experimental.reduce_scatter_minimal_async(
                mm,
                persistent_output_buffers=[intermediate, persistent],
                dim=3,
                multi_device_global_semaphore=semaphores,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=subdevice_id,
            )
            return mm, rs

        def fused():
            return ttnn.experimental.matmul_reduce_scatter_async(
                tt_x,
                tt_w,
                persistent_intermediate_buffer=intermediate,
                persistent_output_buffer=persistent,
                dim=3,
                multi_device_global_semaphore=semaphores,
                reduce_scatter_core_grid_offset=(0, args.grid_y),
                num_links=1,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=subdevice_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=compute,
            )

        sep_rs = fused_rs = None
        if args.mode in ("separate", "both"):
            _, sep_rs = separate()
        if args.mode in ("fused", "both"):
            _, fused_rs = fused()
        ttnn.synchronize_device(mesh, sub_device_ids=[subdevice_id])
        expected = torch.matmul(activation.float(), weight.float())
        actual_rs = fused_rs if fused_rs is not None else sep_rs
        actual = torch.cat([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(actual_rs)], dim=-1)
        passed, message = comp_pcc(expected, actual, 0.999)
        if not passed:
            raise AssertionError(message)

        def measure(fn):
            for _ in range(3):
                fn()
            ttnn.synchronize_device(mesh, sub_device_ids=[subdevice_id])
            samples = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                fn()
                ttnn.synchronize_device(mesh, sub_device_ids=[subdevice_id])
                samples.append((time.perf_counter() - started) * 1000)
            return samples

        separate_samples = measure(separate) if args.mode in ("separate", "both") else []
        fused_samples = measure(fused) if args.mode in ("fused", "both") else []
        result = {
            "mesh": [1, 4],
            "shape": [m, local_k, n],
            "global_k": global_k,
            "dtype": "BF16",
            "fidelity": "HiFi2",
            "in0_block_w": args.in0_block_w,
            "grid": [8, args.grid_y],
            "reduce_scatter_offset": [0, args.grid_y],
            "persistent_buffers": True,
            "async_ccl": True,
            "pcc": str(message),
            "mode": args.mode,
            "separate_median_ms": statistics.median(separate_samples) if separate_samples else None,
            "fused_median_ms": statistics.median(fused_samples) if fused_samples else None,
            "separate_samples_ms": separate_samples,
            "fused_samples_ms": fused_samples,
        }
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("MULTICHIP_FUSED_CCL", json.dumps(result, sort_keys=True))
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
