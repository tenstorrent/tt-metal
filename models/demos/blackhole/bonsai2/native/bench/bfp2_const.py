# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bisect step 3: does a kernel write reach the host at all?

Run after the copy test returned identical wrong values across a source change.
This writes a constant (1.0) from the compute core with no input, packs it and has
BRISC copy it to an ordinary DRAM output tensor. Nothing upstream can affect it.

If the host reads 1.0 everywhere, the output path works and the earlier garbage
was a stale-buffer artifact of the sharded-CB arrangement. If it reads zeros or
garbage, the fault is in how this program writes memory, and every matmul result
before this point should be treated as meaningless.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.bfp2_const
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
    rc = 0
    try:
        # A distinctive fill so "did anything I wrote land" is unambiguous.
        # generic_op requires at least one input tensor besides the output, and
        # treats the LAST tensor as the output, so a trivial input rides along.
        dummy = ttnn.from_torch(
            torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o_t = ttnn.from_torch(
            torch.full((1, 1, 32, 32), -7.0, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cbs = [
            ttnn.CBDescriptor(
                total_size=2048, core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)]),
        ]
        uk = UnifiedKernelDescriptor(
            kernel_source="models/demos/blackhole/bonsai2/native/bench/bfp2_const_kernel.cpp",
            core_ranges=core,
            brisc_common_runtime_args=[o_t.buffer_address()],
        )
        kds = uk.get_kernel_descriptors()
        prog = ttnn.ProgramDescriptor(kernels=kds.kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op([dummy, o_t], prog)
        ttnn.synchronize_device(device)

        got = ttnn.to_torch(o_t).float().numpy()
        ones = int((np.abs(got - 1.0) < 1e-3).sum())
        kept = int((np.abs(got + 7.0) < 1e-3).sum())
        print(f"const write test: {ones}/1024 cells are 1.0, {kept}/1024 still -7.0", flush=True)
        print(f"  sample: {got.reshape(-1)[:8]}", flush=True)
        if ones == 0 and kept == 1024:
            print("  -> NOTHING was written; the program never touched the output buffer")
            rc = 1
        elif ones >= 1000:
            print("  -> output path OK: a constant packed and written back correctly")
        else:
            print(f"  -> partial/incorrect write ({ones} correct) - geometry or pack issue")
            rc = 1
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
