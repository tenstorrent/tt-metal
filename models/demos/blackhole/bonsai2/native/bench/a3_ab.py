# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN a/b projection (dense, 5120 x 96) at decode: auto config vs explicit 1D configs, device time."""
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us
from .a1_calib import device_kernel_us


def main():
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)
    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        k, n = 5120, 96
        x = ttnn.from_torch(torch.randn(1, 1, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        wt = torch.randn(k, n)
        ref = x_t = None
        ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                               packer_l1_acc=True)
        for dt in (ttnn.bfloat8_b, ttnn.bfloat16):
            w = ttnn.from_torch(wt, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cfgs = [("auto", None)]
            for bw in (8, 16, 32, 40, 80, 160):
                cfgs.append((f"1D 3 cores bw={bw}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(3, 1), in0_block_w=bw, out_subblock_h=1, out_subblock_w=1,
                    per_core_M=1, per_core_N=1, fuse_batch=True, fused_activation=None, mcast_in0=True)))
            base = None
            for name, pc in cfgs:
                try:
                    f = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc,  # noqa: E731
                                            memory_config=ttnn.L1_MEMORY_CONFIG)
                    y = ttnn.to_torch(f()).float().flatten()
                    base = y if base is None else base
                    device_kernel_us(ttnn, device)
                    for _ in range(9):
                        ttnn.deallocate(f())
                    ttnn.synchronize_device(device)
                    print(f"{str(dt):22s} {name:22s} {np.median(all_kernel_us(ttnn, device)):7.1f} us  "
                          f"max|y-auto| {float((y - base).abs().max()):.3e}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{str(dt):22s} {name:22s} FAILED {str(e)[:100]}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
