# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bisect step 6: compute reads a sharded-CB input and packs it back out.

Removes NCRISC from the path. The const test proved compute + pack + BRISC write
work; the NCRISC-read copy does not. This puts the input directly in core 0's L1 as
a sharded tensor, which is the arrangement the paused ternary_matmul op used.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.bfp2_src
"""

from __future__ import annotations

import os

import numpy as np
import torch


def main() -> int:
    import ttnn
    from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    rng = np.random.default_rng(0)
    vals = rng.standard_normal((32, 32)).astype(np.float32)
    rc = 0
    try:
        a_t = ttnn.from_torch(
            torch.from_numpy(vals).to(torch.bfloat16).reshape(1, 1, 32, 32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(core, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)))
        o_t = ttnn.from_torch(
            torch.full((1, 1, 32, 32), -7.0, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cbs = [
            ttnn.cb_descriptor_from_sharded_tensor(0, a_t),
            ttnn.CBDescriptor(
                total_size=2048, core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)]),
        ]
        uk = UnifiedKernelDescriptor(
            kernel_source="models/demos/blackhole/bonsai2/native/bench/bfp2_srckernel.cpp",
            core_ranges=core,
            brisc_common_runtime_args=[o_t.buffer_address()],
        )
        kds = uk.get_kernel_descriptors()
        prog = ttnn.ProgramDescriptor(kernels=kds.kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op([a_t, o_t], prog)
        ttnn.synchronize_device(device)

        got = ttnn.to_torch(o_t).float().numpy().reshape(32, 32)
        exp = torch.from_numpy(vals).to(torch.bfloat16).float().numpy()
        err = float(np.abs(got - exp).max())
        print(f"src test: max|got-exp|={err:.6f} nonzero={int((got != 0).sum())}/1024 "
              f"still_-7={int((np.abs(got + 7) < 1e-3).sum())}", flush=True)
        print(f"  got[0,:4]={got[0,:4]}  exp[0,:4]={exp[0,:4]}", flush=True)
        if err < 1e-2:
            print("  -> compute reads a sharded CB correctly; only the NCRISC DRAM read fails")
        elif (got != 0).sum() == 0:
            print("  -> zeros: the compute-side tile path is broken, not just the read")
            rc = 1
        else:
            print("  -> wrong values: layout mismatch between the shard and the pack")
            rc = 1
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
