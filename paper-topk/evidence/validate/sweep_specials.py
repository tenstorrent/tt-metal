#!/usr/bin/env python3
"""Bisection sweep for the reported topk specials hang.

One device process; each cell prints CELL_START/CELL_OK markers so an
in-kernel hang identifies the offending cell (outer timeout kills us).
"""
import torch
import ttnn


def bf16_from_bits(bits):
    b = torch.as_tensor(bits, dtype=torch.int64)
    b = torch.where(b >= 0x8000, b - 0x10000, b).to(torch.int16)
    return b.view(torch.bfloat16)


ALL8 = [0x7FC0, 0x7FC1, 0xFFC1, 0x8000, 0x0000, 0x7F80, 0xFF80, 0x0001]
SINGLES = [[s] for s in ALL8]
PAIRS = [
    [0x7FC0, 0x7F80],  # +NaN with +Inf
    [0xFFC1, 0xFF80],  # -NaN with -Inf
    [0x7F80, 0xFF80],  # +-Inf in-row (interacts with -inf padding)
    [0x7FC1, 0xFFC1],  # +-NaN payloads
    [0x7FC0, 0xFFC1],
    [0x8000, 0x0000],  # +-0
    [0x0001, 0x8000],  # subnormal + -0
]


def run_cell(dev, name, w, k, largest, specials, rows=32, reps=1):
    row = torch.linspace(-1, 1, w).to(torch.bfloat16)
    if specials:
        row[100 : 100 + len(specials)] = bf16_from_bits(specials)
    x = row.expand(rows, w).clone().reshape(1, 1, rows, w)
    tt_in = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=dev)
    for r in range(reps):
        print(f"CELL_START {name} rep={r}", flush=True)
        vals, idx = ttnn.topk(tt_in, k, dim=-1, largest=largest, sorted=True)
        _ = ttnn.to_torch(vals)
        _ = ttnn.to_torch(idx)
        print(f"CELL_OK {name} rep={r}", flush=True)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        for i, s in enumerate(SINGLES):
            for lg in (True, False):
                run_cell(dev, f"single[{s[0]:04x}]-L{int(lg)}", 10000, 32, lg, s)
        for i, p in enumerate(PAIRS):
            for lg in (True, False):
                run_cell(dev, f"pair[{p[0]:04x},{p[1]:04x}]-L{int(lg)}", 10000, 32, lg, p)
        # engine/W/k variants with all 8 specials
        run_cell(dev, "all8-W10000-k32-L1", 10000, 32, True, ALL8)
        run_cell(dev, "all8-W10000-k32-L0", 10000, 32, False, ALL8)
        run_cell(dev, "all8-W8192-k32-L1-multicore", 8192, 32, True, ALL8)
        run_cell(dev, "all8-W8192-k32-L0-multicore", 8192, 32, False, ALL8)
        run_cell(dev, "all8-W8192-k96-L1-routed", 8192, 96, True, ALL8)
        run_cell(dev, "all8-W16384-k32-L1", 16384, 32, True, ALL8)
        # stress: 20 reps of the reported trigger
        run_cell(dev, "all8-W10000-k32-L1-stress", 10000, 32, True, ALL8, reps=20)
        run_cell(dev, "all8-W10000-k32-L1-rows1-stress", 10000, 32, True, ALL8, rows=1, reps=20)
        print("SWEEP_DONE", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
