#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Profile matmul compute kernel pack strategy.

Measures throughput + prints the compile-time args so we can see
the subblock dims and in1_num_subblocks the factory selected.
"""

import time
import torch
import ttnn

# Shapes chosen to likely give in1_num_subblocks=1 on BH
# (per_core_N = subblock_w = 4 or less)
SHAPES = [
    (2048, 2048, 2048),  # Large: may have in1_num_subblocks > 1
    (2048, 2048, 1024),  # Narrower N
    (2048, 4096, 512),  # Even narrower N, tall K
    (1024, 2048, 256),  # Small N, forces per_core_N small
    (512, 1024, 512),  # Medium square-ish
]

NUM_WARMUP = 20
NUM_ITERS = 200


def profile_shape(device, M, K, N):
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)

    # Warmup
    for _ in range(NUM_WARMUP):
        c_tt = ttnn.matmul(a_tt, b_tt)
        ttnn.deallocate(c_tt)
    ttnn.synchronize_device(device)

    # Timed
    start = time.perf_counter()
    for _ in range(NUM_ITERS):
        c_tt = ttnn.matmul(a_tt, b_tt)
        ttnn.deallocate(c_tt)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start
    avg_us = elapsed / NUM_ITERS * 1e6

    # Correctness
    c_tt = ttnn.matmul(a_tt, b_tt)
    c_torch = ttnn.to_torch(c_tt)
    c_ref = a @ b
    pcc = torch.corrcoef(torch.stack([c_torch.flatten().float(), c_ref.flatten().float()]))[0, 1].item()

    ttnn.deallocate(c_tt)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)
    return avg_us, pcc


def main():
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    print(f"Arch: {device.arch()}, Warmup: {NUM_WARMUP}, Iters: {NUM_ITERS}\n")
    print(f"{'Shape':>25s}  {'us/op':>10s}  {'PCC':>10s}")
    print("-" * 50)

    for M, K, N in SHAPES:
        avg_us, pcc = profile_shape(device, M, K, N)
        print(f"  {M:4d}x{K:4d}x{N:4d}  {avg_us:10.1f}  {pcc:10.6f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
