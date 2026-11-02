# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bisect: does a single tile survive reader -> CB -> pack -> host?

The matmul probe returned exact zeros for BOTH the BFP2 cases and a plain bf16
control, so before blaming the format the plumbing has to be shown working. This
does the smallest possible round trip: read one bf16 tile from DRAM into a CB,
copy it to Dst, pack it back out into a CB that *is* the sharded output tensor's
buffer, and read it on the host. No matmul.

Three outcomes, three different fixes:

  * values come back  -> the plumbing is fine, and the matmul is what broke.
  * zeros come back   -> the pack is not landing in the buffer the host reads.
  * garbage comes back -> the CB page/tile sizes disagree somewhere.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.bfp2_copy
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
        # Sentinel fill, not zeros: a zero-filled output cannot distinguish "the
        # kernel wrote zero" from "the kernel never wrote", which made an earlier
        # run of this test ambiguous.
        o_t = ttnn.from_torch(
            torch.full((1, 1, 32, 32), -7.0, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            kernel_source="models/demos/blackhole/bonsai2/native/bench/bfp2_copy_kernel.cpp",
            core_ranges=core,
            ncrisc_common_runtime_args=[a_t.buffer_address()],
            brisc_common_runtime_args=[o_t.buffer_address()],
            trisc_common_runtime_args=[],
        )
        kds = uk.get_kernel_descriptors()
        prog = ttnn.ProgramDescriptor(kernels=kds.kernels, cbs=cbs, semaphores=[])
        ttnn.generic_op([a_t, o_t], prog)
        ttnn.synchronize_device(device)

        got = ttnn.to_torch(o_t).float().numpy().reshape(32, 32)
        exp = torch.from_numpy(vals).to(torch.bfloat16).float().numpy()
        err = np.abs(got - exp).max()
        print(f"copy test: max|got-exp| = {err:.6f}", flush=True)
        print(f"  got nonzero cells: {int((got != 0).sum())}/1024, "
              f"exp nonzero: {int((exp != 0).sum())}/1024", flush=True)
        print(f"  got[0,:6]={got[0,:6]}", flush=True)
        print(f"  exp[0,:6]={exp[0,:6]}", flush=True)
        # Is the data present but transposed / offset?
        for name, cand in (("transpose", got.T), ("flat", got.reshape(-1).reshape(32, 32))):
            print(f"  vs {name}: max err {np.abs(cand - exp).max():.4f}", flush=True)
        if (got != 0).sum() and (got != 0).sum() != (exp != 0).sum():
            nz = np.argwhere(got != 0)
            print(f"  first/last nonzero index in got: {nz[0]} .. {nz[-1]}", flush=True)
        if got.max() == 0:
            print("  -> exact zeros: the pack is not reaching the buffer the host reads")
            rc = 1
        elif err > 1e-2:
            print("  -> values present but wrong: page size or shard geometry disagrees")
            rc = 1
        else:
            print("  -> plumbing OK: reader, CB, Dst, pack and the sharded output all work")
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
