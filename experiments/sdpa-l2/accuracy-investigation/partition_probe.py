# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: split interleaved dot-product lanes, sum outputs in FP64."""

import argparse
import json
from pathlib import Path

import torch
import ttnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    try:
        with args.output.open("x") as log:
            for pattern in ("normal128", "outlier128"):
                gen = torch.Generator().manual_seed(71)
                a = torch.randn((1, 1, 32, 128), generator=gen).bfloat16()
                b = torch.randn((1, 1, 128, 32), generator=gen).bfloat16()
                if pattern == "outlier128":
                    a[..., 0] *= 32
                    b[..., 0, :] *= 32
                gold = a.double() @ b.double()
                tb = ttnn.from_torch(
                    b,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for parts in (1, 2, 4, 8, 16, 32, 128):
                    total = torch.zeros_like(gold)
                    for offset in range(parts):
                        part = torch.zeros_like(a)
                        part[..., offset::parts] = a[..., offset::parts]
                        ta = ttnn.from_torch(
                            part,
                            device=device,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        result = ttnn.matmul(ta, tb, dtype=ttnn.float32, compute_kernel_config=config)
                        total += ttnn.to_torch(result).double()
                        ttnn.deallocate(result)
                        ttnn.deallocate(ta)
                    delta = total - gold
                    row = dict(
                        pattern=pattern,
                        parts=parts,
                        l2_pct=float(100 * delta.norm() / gold.norm()),
                        max_abs=float(delta.abs().max()),
                    )
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                ttnn.deallocate(tb)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
