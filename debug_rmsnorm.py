#!/usr/bin/env python3
"""Debug script to compare standalone vs auto-fused RMSNorm descriptors."""

import math
import sys
import traceback

import torch
import ttnn

from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
from models.demos.deepseek_v3_b1.auto_fusion.specs.rmsnorm import RMSNORM
from models.demos.deepseek_v3_b1.micro_ops.rmsnorm.op import RMSNormSingleCore
from models.demos.deepseek_v3_b1.utils import float_to_uint32


def debug_rmsnorm():
    width = 7168
    epsilon = 1e-6
    shape = (1, width)
    tile = ttnn.Tile([1, 32])
    FULL_32x32_TILE = ttnn.Tile((32, 32))
    HALF_16x32_TILE = ttnn.Tile((16, 32))
    is_16x32 = (width // 32) % 32 != 0
    interpreted_tile = HALF_16x32_TILE if is_16x32 else FULL_32x32_TILE
    tile_height, tile_width = interpreted_tile.tile_shape
    num_tiles = (shape[0] * shape[1]) // (tile_height * tile_width)
    numel = shape[0] * shape[1]

    print(f"width={width}, tile_height={tile_height}, tile_width={tile_width}, num_tiles={num_tiles}, numel={numel}")
    print(f"interpreted_tile={interpreted_tile}, is_16x32={is_16x32}")

    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(42)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        torch_gamma = torch.randn(shape, dtype=torch.bfloat16)

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            (shape[0], width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem_config,
            tile=tile,
        )
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem_config,
            tile=tile,
        )
        ttnn_output = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem_config,
            tile=tile,
        )

        # Step 1: Check compile
        print("\n=== Step 1: Compile ===")
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        g = FusionGraph()
        g.add(
            "rmsnorm",
            RMSNORM,
            cores=core_grid,
            ct_args={
                "fp32_acc": 0,
                "num_tiles": num_tiles,
                "rsqrt_fast_approx": 0,
                "input_num_pages": num_tiles,
                "gamma_num_pages": num_tiles,
                "epsilon": float_to_uint32(epsilon),
                "scalar": float_to_uint32(1.0 / math.sqrt(float(numel))),
            },
        )

        external_ports = {("rmsnorm", "input"), ("rmsnorm", "gamma"), ("rmsnorm", "output")}
        source, schedule, allocator = g.compile(external_ports=external_ports)
        print(f"Schedule: {schedule}")
        print(f"Allocations:")
        for key, alloc in allocator._allocations.items():
            print(f"  {key}: index={alloc.index}, is_external={alloc.is_external}")
        print(f"\nCB bindings: {g._nodes['rmsnorm'].cb_bindings}")

        # Print generated kernel source
        print(f"\n=== Generated kernel source (first 80 lines) ===")
        for i, line in enumerate(source.split("\n")[:80]):
            print(f"  {i+1}: {line}")

        # Step 2: Build host descriptor step by step
        print("\n=== Step 2: Build host descriptor ===")
        from models.demos.deepseek_v3_b1.auto_fusion.host_gen import HostGenerator

        io_tensors = {
            ("rmsnorm", "input"): ttnn_input,
            ("rmsnorm", "gamma"): ttnn_gamma,
            ("rmsnorm", "output"): ttnn_output,
        }

        host_gen = HostGenerator(g, schedule, allocator, device, io_tensors, g._cb_configs)

        # Build CB descriptors
        print("\n--- CB Descriptors ---")
        pool_size = host_gen._create_l1_pool()
        print(f"Pool size: {pool_size}")

        cb_descs = host_gen._build_cb_descriptors()
        print(f"Number of CB descriptors: {len(cb_descs)}")
        for i, cb in enumerate(cb_descs):
            print(f"  CB {i}:")
            print(f"    type: {type(cb)}")
            if hasattr(cb, "format_descriptors"):
                for j, fd in enumerate(cb.format_descriptors):
                    print(f"    format_descriptor[{j}]:")
                    print(f"      buffer_index={fd.buffer_index}")
                    try:
                        print(f"      data_format={fd.data_format}")
                    except Exception:
                        print(f"      data_format=<cannot print>")
                    print(f"      page_size={fd.page_size}")
                    try:
                        if hasattr(fd, "tile"):
                            print(f"      tile={fd.tile}")
                    except Exception:
                        print(f"      tile=<cannot print>")

        # Build compile-time args
        print("\n--- Compile-time args ---")
        ncrisc_ct, brisc_ct, trisc_ct = host_gen._build_compile_time_args()
        print(f"NCRISC CT args ({len(ncrisc_ct)}):")
        for name, val in ncrisc_ct:
            print(f"  {name} = {val}")
        print(f"BRISC CT args ({len(brisc_ct)}):")
        for name, val in brisc_ct:
            print(f"  {name} = {val}")
        print(f"TRISC CT args ({len(trisc_ct)}):")
        for name, val in trisc_ct:
            print(f"  {name} = {val}")

        # Build common runtime args
        print("\n--- Common runtime args ---")
        trisc_common_rt = host_gen._build_trisc_common_runtime_args()
        print(f"TRISC common RT args: {trisc_common_rt}")

        # Build core descriptors
        print("\n--- Core descriptors ---")
        core_descs = host_gen._build_core_descriptors()
        print(f"Core descriptors: {len(core_descs)}")
        for cd in core_descs:
            print(f"  {cd.named_compile_time_arg}: value={cd.value}, other={cd.other_value}")

        # Build compute config
        print("\n--- Compute config ---")
        compute_config = host_gen._build_compute_config()
        print(f"Compute config: {compute_config}")

        # Build semaphores
        print("\n--- Semaphores ---")
        sems = host_gen._build_semaphores()
        print(f"Semaphores: {len(sems)}")

        # Now compare with standalone
        print("\n=== Standalone comparison ===")
        # Recreate what standalone does
        tile_size = interpreted_tile.get_tile_size(ttnn.bfloat16)
        tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

        standalone_in_cb = ttnn.cb_descriptor_from_sharded_tensor(0, ttnn_input)
        standalone_in_cb.format_descriptors[0].tile = tile_descriptor
        standalone_in_cb.format_descriptors[0].page_size = tile_size

        standalone_gamma_cb = ttnn.cb_descriptor_from_sharded_tensor(1, ttnn_gamma)
        standalone_gamma_cb.format_descriptors[0].tile = tile_descriptor
        standalone_gamma_cb.format_descriptors[0].page_size = tile_size

        standalone_out_cb = ttnn.cb_descriptor_from_sharded_tensor(2, ttnn_output)
        standalone_out_cb.format_descriptors[0].tile = tile_descriptor
        standalone_out_cb.format_descriptors[0].page_size = tile_size

        print(f"tile_size = {tile_size}")
        print(
            f"Standalone in CB: page_size={standalone_in_cb.format_descriptors[0].page_size}, tile={standalone_in_cb.format_descriptors[0].tile}"
        )
        print(
            f"Auto-fused in CB: page_size={cb_descs[0].format_descriptors[0].page_size}, tile={cb_descs[0].format_descriptors[0].tile}"
        )

        # Compare named CT args
        standalone_ncrisc_ct = [
            ("rmsnorm_input_cb", 0),
            ("rmsnorm_gamma_cb", 1),
            ("rmsnorm_num_tiles", num_tiles),
            ("rmsnorm_num_faces", interpreted_tile.num_faces),
        ]
        standalone_trisc_ct = [
            ("rmsnorm_input_cb", 0),
            ("rmsnorm_gamma_cb", 1),
            ("rmsnorm_output_cb", 2),
            ("rmsnorm_fp32_acc", 0),
            ("rmsnorm_num_tiles", num_tiles),
            ("rmsnorm_rsqrt_fast_approx", 0),
        ]
        print(f"\nStandalone NCRISC CT: {standalone_ncrisc_ct}")
        print(f"Auto-fused NCRISC CT: {ncrisc_ct}")
        print(f"\nStandalone TRISC CT: {standalone_trisc_ct}")
        print(f"Auto-fused TRISC CT: {trisc_ct}")

        # Standalone common RT args
        epsilon_packed = float_to_uint32(epsilon)
        inv_sqrt_numel = 1.0 / math.sqrt(float(numel))
        scalar_packed = float_to_uint32(inv_sqrt_numel)
        standalone_trisc_common_rt = [epsilon_packed, scalar_packed]
        print(f"\nStandalone TRISC common RT: {standalone_trisc_common_rt}")
        print(f"Auto-fused TRISC common RT: {trisc_common_rt}")

        # Now try building the UnifiedKernelDescriptor
        print("\n=== Step 3: Build UnifiedKernelDescriptor ===")
        import os
        from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
            UnifiedCompileTimeCoreDescriptor,
            UnifiedKernelDescriptor,
        )

        output_dir = os.path.join(os.path.dirname(__file__), "models/demos/deepseek_v3_b1/auto_fusion/kernels")
        os.makedirs(output_dir, exist_ok=True)
        kernel_path = os.path.join(output_dir, "auto_fused_kernel.cpp")
        with open(kernel_path, "w") as f:
            f.write(source)

        tt_metal_root = "/localdev/rmiller/tt-metal"
        rel_kernel_path = os.path.relpath(kernel_path, tt_metal_root)
        print(f"Kernel path (rel): {rel_kernel_path}")

        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        try:
            unified_kernel = UnifiedKernelDescriptor(
                kernel_source=rel_kernel_path,
                core_ranges=all_cores,
                ncrisc_named_compile_time_args=ncrisc_ct,
                brisc_named_compile_time_args=brisc_ct,
                trisc_named_compile_time_args=trisc_ct,
                trisc_common_runtime_args=trisc_common_rt,
                trisc_compute_config=compute_config,
                unified_compile_time_core_descriptors=core_descs,
                noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            )
            print("UnifiedKernelDescriptor created OK")

            result = unified_kernel.get_kernel_descriptors()
            print(f"Got {len(result.kernels)} kernel descriptors")
            for i, kd in enumerate(result.kernels):
                print(f"  Kernel {i}: source={kd.kernel_source}, core_ranges={kd.core_ranges}")
                if hasattr(kd, "named_compile_time_args"):
                    print(f"    named_ct_args: {kd.named_compile_time_args}")
                if hasattr(kd, "common_runtime_args"):
                    print(f"    common_rt_args: {kd.common_runtime_args}")
        except Exception as e:
            print(f"ERROR creating UnifiedKernelDescriptor: {e}")
            traceback.print_exc()
            return

        # Step 4: Build ProgramDescriptor
        print("\n=== Step 4: Build ProgramDescriptor ===")
        try:
            pd = ttnn.ProgramDescriptor(
                kernels=result.kernels,
                cbs=cb_descs,
                semaphores=sems,
            )
            print("ProgramDescriptor created OK")
        except Exception as e:
            print(f"ERROR creating ProgramDescriptor: {e}")
            traceback.print_exc()
            return

        # Step 5: Run
        print("\n=== Step 5: Run generic_op ===")
        io_tensor_list = [ttnn_input, ttnn_gamma, ttnn_output]
        try:
            result = ttnn.generic_op(io_tensor_list, pd)
            print("generic_op completed OK")
            result_torch = ttnn.to_torch(result)[:, :width]
            print(f"Result shape: {result_torch.shape}")
            print(f"Result sample: {result_torch[0, :5]}")
        except Exception as e:
            print(f"ERROR running generic_op: {e}")
            traceback.print_exc()

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    debug_rmsnorm()
