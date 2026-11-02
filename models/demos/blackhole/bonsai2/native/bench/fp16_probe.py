# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fp16 scales probe: UInt16 tile of fp16 bits -> fp32 DST -> SFPU fp32 (F16a load, or LO16 + rebias)."""
import os

import numpy as np
import torch


def main():
    import ttnn
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        rng = np.random.default_rng(0)
        vals = (rng.random((32, 32)) * 0.03 + 0.003).astype(np.float16)
        vals[0, 0], vals[0, 1] = 0, -0.0123
        bits = vals.view(np.uint16)
        x = ttnn.from_torch(torch.from_numpy(bits.astype(np.int32)), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for mode in (0, 1):
            y = ttnn.from_torch(torch.full((32, 32), -7.0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            core = ttnn.CoreCoord(0, 0)
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
            cbs = [ttnn.CBDescriptor(total_size=2048, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
                       buffer_index=0, data_format=ttnn.uint16, page_size=2048)]),
                   ttnn.CBDescriptor(total_size=4096, core_ranges=grid, format_descriptors=[ttnn.CBFormatDescriptor(
                       buffer_index=16, data_format=ttnn.float32, page_size=4096)])]
            cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                               math_approx_mode=False)
            uk = UnifiedKernelDescriptor(
                kernel_source="models/demos/blackhole/bonsai2/native/bench/kernels/fp16_probe_kernel.cpp",
                core_ranges=grid, ncrisc_named_compile_time_args=[("pr_mode", mode)],
                brisc_named_compile_time_args=[("pr_mode", mode)], trisc_named_compile_time_args=[("pr_mode", mode)],
                trisc_compute_config=cfg,
                per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                    ncrisc_args=[(core, [x.buffer_address()])], brisc_args=[(core, [y.buffer_address()])]))
            try:
                ttnn.generic_op([x, y], ttnn.ProgramDescriptor(kernels=uk.get_kernel_descriptors().kernels,
                                                               semaphores=[], cbs=cbs))
                got = ttnn.to_torch(y).float().numpy()
                want = vals.astype(np.float32)
                print(f"mode {mode} ({'F16a load' if mode == 0 else 'LO16 + rebias'}): exact {int((got == want).sum())}/1024 "
                      f"max|err| {float(np.abs(got - want).max()):.3e}  got[0,:3] {got[0, :3]} want {want[0, :3]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"mode {mode} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
