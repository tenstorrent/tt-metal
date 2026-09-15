# SPDX-License-Identifier: Apache-2.0
"""T0.2 zone-tax micro-benchmark: cost per zone boundary pair on each RISC type.

Launches analysis/kernels/zone_tax_{dm,compute}.cpp through ttnn.generic_op on a small core grid:
reader slot (RISCV_1 = NCRISC), writer slot (RISCV_0 = BRISC) and compute (TRISC0/1/2 all run the
same loop). One program per (mode, n) in ZT_GRID env ("mode:n:loop,mode:n:loop,..."). The KERNEL
zone duration per RISC divided by n, minus the mode-0 baseline at the same n, is the per-zone cost.

Run:
  ZT_GRID="0:2000:16,3:2000:16,2:2000:16,0:100:16,1:100:16" python -m tracy --enable-sum-profiling \
      -m pytest analysis/zone_tax.py::test_zone_tax -s
"""
import os

import torch

# kernels sit next to this harness, so the path follows the file and not a fixed workspace
_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def test_zone_tax(device):
    import ttnn

    grid = os.environ.get("ZT_GRID", "0:2000:16,3:2000:16,2:2000:16,0:100:16,1:100:16")
    cores_env = os.environ.get("ZT_CORES", "1x1")
    cx, cy = [int(v) for v in cores_env.split("x")]
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cx - 1, cy - 1))])
    io = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    for spec in grid.split(","):
        mode, n, loop = [int(v) for v in spec.split(":")]
        defines = [("ZT_MODE", str(mode)), ("ZT_N", str(n)), ("ZT_LOOP", str(loop))]
        print(f"\n[zone_tax] mode={mode} n={n} loop={loop} cores={cores_env}", flush=True)
        rd = ttnn.KernelDescriptor(
            kernel_source=_KDIR + "/zone_tax_dm.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=[],
            defines=defines,
            runtime_args=[],
            config=ttnn.ReaderConfigDescriptor(),
        )
        wr = ttnn.KernelDescriptor(
            kernel_source=_KDIR + "/zone_tax_dm.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=[],
            defines=defines,
            runtime_args=[],
            config=ttnn.WriterConfigDescriptor(),
        )
        cp = ttnn.KernelDescriptor(
            kernel_source=_KDIR + "/zone_tax_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=[],
            defines=defines,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=[])
        out = ttnn.generic_op([io, out_t], prog)
        ttnn.synchronize_device(device)
        _ = out.shape
        print(f"[zone_tax] done mode={mode} n={n}", flush=True)
