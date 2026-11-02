# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bisect step 4: is the DRAM read path working?

After adding the missing ``compute_kernel_hw_startup`` call, a constant write comes
back exactly (1024/1024 cells) but a reader->pack->write copy still returns zeros.
So the pack and output are fine and the fault is the read or the handoff. This
drops compute entirely: NCRISC reads one 2048-byte tile and writes it straight back
out with a NOC write, using exact addresses.

Outcomes:

  * values equal to the input -> the read path is fine, and the fault is in the CB
    handoff to the compute threads (or in the Dst/pack path after them).
  * zeros -> the read itself is not landing, and the page-size/format derivation is
    the thing to fix.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.bfp2_io
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o_t = ttnn.from_torch(
            torch.full((1, 1, 32, 32), -7.0, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048, core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)]),
            ttnn.CBDescriptor(
                total_size=2048, core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)]),
        ]
        uk = UnifiedKernelDescriptor(
            kernel_source="models/demos/blackhole/bonsai2/native/bench/bfp2_io_kernel.cpp",
            core_ranges=core,
            ncrisc_common_runtime_args=[a_t.buffer_address()],
            brisc_common_runtime_args=[o_t.buffer_address()],
        )
        kds = uk.get_kernel_descriptors()
        prog = ttnn.ProgramDescriptor(kernels=kds.kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op([a_t, o_t], prog)
        ttnn.synchronize_device(device)

        got = ttnn.to_torch(o_t).float().numpy().reshape(32, 32)
        exp = torch.from_numpy(vals).to(torch.bfloat16).float().numpy()
        err = float(np.abs(got - exp).max())
        print(f"io test: max|got-exp| = {err:.6f}, nonzero {int((got != 0).sum())}/1024", flush=True)
        print(f"  got[0,:4]={got[0,:4]}", flush=True)
        print(f"  exp[0,:4]={exp[0,:4]}", flush=True)
        all5 = int((np.abs(got - 5.0) < 1e-2).sum())
        print(f"  cells equal to 5.0: {all5}/1024", flush=True)
        if all5 >= 1000:
            print("  -> NCRISC runs, its L1 write works, CB handoff and compute copy work;")
            print("     only the DRAM read failed")
        elif err < 1e-2:
            print("  -> real data came back: the read works and the overwrite did not")
        elif (got != 0).sum() == 0:
            print("  -> zeros: the CB handoff or the compute copy is broken (not the read)")
            rc = 1
        else:
            print("  -> something else; inspect the samples")
            rc = 1
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
