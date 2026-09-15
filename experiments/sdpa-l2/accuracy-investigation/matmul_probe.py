# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate HiFi4 product and accumulation precision without SDPA."""

import argparse
import json
from pathlib import Path

import torch
import ttnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-format", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    try:
        with args.output.open("x") as log:
            for pattern in ("single_product", "normal32", "normal128", "outlier128"):
                depth = 32 if pattern in ("single_product", "normal32") else 128
                gen = torch.Generator().manual_seed(71)
                a = torch.randn((1, 1, 32, depth), generator=gen).bfloat16()
                b = torch.randn((1, 1, depth, 32), generator=gen).bfloat16()
                if pattern == "single_product":
                    a.zero_()
                    b.zero_()
                    a[..., 0] = 1 + 1 / 128  # SrcB: last BF16 bit only.
                    b[..., 0, :] = 1 + 7 / 128  # SrcA: low three bits set.
                elif pattern == "outlier128":
                    a[..., 0] *= 32
                    b[..., 0, :] *= 32
                gold = a.double() @ b.double()
                dtype = ttnn.bfloat16 if args.input_format == "bf16" else ttnn.float32
                ta, tb = [
                    ttnn.from_torch(
                        x if args.input_format == "bf16" else x.float(),
                        device=device,
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for x in (a, b)
                ]
                for name, fidelity in (
                    ("HiFi2", ttnn.MathFidelity.HiFi2),
                    ("HiFi3", ttnn.MathFidelity.HiFi3),
                    ("HiFi4", ttnn.MathFidelity.HiFi4),
                ):
                    result = ttnn.matmul(
                        ta,
                        tb,
                        dtype=ttnn.float32,
                        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                            math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
                        ),
                    )
                    actual = ttnn.to_torch(result).float()
                    delta = actual.double() - gold
                    bits = actual.contiguous().view(torch.int32)
                    row = dict(
                        pattern=pattern,
                        fidelity=name,
                        input_format=args.input_format,
                        l2_pct=float(100 * delta.norm() / gold.norm()),
                        max_abs=float(delta.abs().max()),
                        first_actual=actual.flatten()[:8].tolist(),
                        first_gold=gold.flatten()[:8].tolist(),
                        low_zero_fraction={
                            str(n): float(((bits & ((1 << n) - 1)) == 0).float().mean())
                            for n in (8, 9, 10, 11, 12, 13, 16)
                        },
                    )
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                    ttnn.deallocate(result)
                ttnn.deallocate(ta)
                ttnn.deallocate(tb)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
