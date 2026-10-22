# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""DRAM address aliasing check: write tensor X at the bottom of the DRAM banks, reserve S MiB per bank (not
written), write tensor Y above that, then re-read X. X changing means writes at base+S land on base."""
import torch
import ttnn

MiB = 1 << 20


def main():
    dev = ttnn.open_device(device_id=0)
    v = ttnn.device.get_memory_view(dev, ttnn.BufferType.DRAM)
    print(f"DRAM banks={v.num_banks} bytes/bank={v.total_bytes_per_bank}", flush=True)
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cols = 32 * 64  # 64 tile pages: 32 per bank
    for s_mib in (16, 32, 64, 128, 256, 512):
        x_t = torch.full((1, 1, 32, cols), 1.0).bfloat16()
        x = up(x_t)
        # Filler of 64-tile rows (64 KiB per bank each), sized so y starts exactly S MiB above x.
        rows = (s_mib * MiB - 32 * 2048) // (32 * 2048) * 32
        filler = ttnn.empty([1, 1, rows, cols], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        y = up(torch.full((1, 1, 32, cols), -7.0).bfloat16())
        back = ttnn.to_torch(x)
        bad = int((back != x_t).sum())
        print(
            f"S={s_mib:4d} MiB/bank x@{x.buffer_address():#x} y@{y.buffer_address():#x} "
            f"(y-x={(y.buffer_address() - x.buffer_address()) / MiB:.1f} MiB) x_changed={bad}/{back.numel()}",
            flush=True,
        )
        for t in (x, filler, y):
            ttnn.deallocate(t)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
