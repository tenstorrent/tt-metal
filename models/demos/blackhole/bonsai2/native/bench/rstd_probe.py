# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: rstd in every lane of a DST tile via two ones-tile matmuls (kernels/rstd_probe_kernel.cpp), with the
S / Z intermediates as bf16 or fp32 CBs. Prints the max rel error of every lane vs float64."""
import os

import numpy as np
import torch


def main():
    import ttnn
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        x = (torch.randn(32, 32) * 3).bfloat16()
        want = 1 / np.sqrt((x.double().numpy() ** 2).mean() + 1e-6)
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ot = ttnn.from_torch(torch.ones(32, 32).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        core = ttnn.CoreCoord(0, 0)
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

        def cb(i, fmt):
            size = 4096 if fmt == ttnn.float32 else 2048
            return ttnn.CBDescriptor(total_size=size, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=i, data_format=fmt, page_size=size)])

        for name, mid in (("bf16 S/Z", ttnn.bfloat16),):
            y = ttnn.from_torch(torch.full((32, 32), -7.0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for fid, fname, hifi in ((ttnn.MathFidelity.HiFi4, "HiFi4 kernel", 0), (ttnn.MathFidelity.LoFi, "LoFi kernel", 0),
                                     (ttnn.MathFidelity.LoFi, "LoFi kernel + explicit HiFi4 matmuls", 1)):
                cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False,
                                                   dst_full_sync_en=True)
                uk = UnifiedKernelDescriptor(
                    kernel_source="models/demos/blackhole/bonsai2/native/bench/kernels/rstd_probe_kernel.cpp",
                    core_ranges=grid, trisc_compute_config=cfg, trisc_named_compile_time_args=[("pr_hifi", hifi)],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(core, [xt.buffer_address(), ot.buffer_address()])],
                        brisc_args=[(core, [y.buffer_address()])]))
                try:
                    ttnn.generic_op([xt, ot, y], ttnn.ProgramDescriptor(
                        kernels=uk.get_kernel_descriptors().kernels, semaphores=[],
                        cbs=[cb(0, ttnn.bfloat16), cb(1, ttnn.bfloat16), cb(2, mid), cb(3, mid), cb(16, ttnn.float32)]))
                    got = ttnn.to_torch(y).double().numpy()
                    rel = np.abs(got - want).max() / want
                    print(f"{name} {fname}: every lane within {rel:.2e} of float64 rstd {want:.6f}  "
                          f"(lanes min {got.min():.6f} max {got.max():.6f})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{name} {fname} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
