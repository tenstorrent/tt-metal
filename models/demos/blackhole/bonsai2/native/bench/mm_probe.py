# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: does a plain matmul_tiles work inside our unified kernels (with/without deepseek init, pack_tile vs
pack_block_contiguous)? A = random bf16, B = the +-1 order-32 Hadamard; fp32 DST, HiFi4."""
import os

import numpy as np
import torch


def main():
    import ttnn
    from models.demos.blackhole.bonsai2.native.hadamard import sylvester
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        a = torch.randn(32, 32).bfloat16()
        h = torch.from_numpy(np.round(sylvester(32) * np.sqrt(32)).astype(np.float32)).bfloat16()
        at = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ht = ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        want = a.double().numpy() @ h.double().numpy()
        core = ttnn.CoreCoord(0, 0)
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

        def cb(i, fmt, size):
            return ttnn.CBDescriptor(total_size=size, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=i, data_format=fmt, page_size=size)])

        for mode in (0, 1, 2, 3):
            y = ttnn.from_torch(torch.full((32, 32), -7.0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False, dst_full_sync_en=True)
            uk = UnifiedKernelDescriptor(
                kernel_source="models/demos/blackhole/bonsai2/native/bench/kernels/mm_probe_kernel.cpp",
                core_ranges=grid, ncrisc_named_compile_time_args=[("pr_mode", mode)],
                brisc_named_compile_time_args=[("pr_mode", mode)], trisc_named_compile_time_args=[("pr_mode", mode)],
                trisc_compute_config=cfg,
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=[(core, [at.buffer_address(), ht.buffer_address()])], brisc_args=[(core, [y.buffer_address()])]))
            try:
                ttnn.generic_op([at, ht, y], ttnn.ProgramDescriptor(
                    kernels=uk.get_kernel_descriptors().kernels, semaphores=[],
                    cbs=[cb(0, ttnn.bfloat16, 2048), cb(1, ttnn.bfloat16, 2048), cb(16, ttnn.float32, 4096)]))
                got = ttnn.to_torch(y).double().numpy()
                rel = np.abs(got - want).max() / np.abs(want).max()
                print(f"mode {mode} (deepseek init {mode & 1}, pack_block {mode >> 1 & 1}): max rel err {rel:.2e}  "
                      f"got[0,:3] {got[0, :3].round(3)} want {want[0, :3].round(3)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"mode {mode} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
