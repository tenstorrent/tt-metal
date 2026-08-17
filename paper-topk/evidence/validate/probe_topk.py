#!/usr/bin/env python3
"""Minimal parametrized ttnn.topk special-values probe.

Builds torch.linspace(-1,1,W) bf16, plants the given special bit patterns at
contiguous indices starting at --pos (default 100), runs ttnn.topk, converts
outputs back, prints value/index bits, prints PROBE_DONE.  A hang shows as
outer-timeout kill with no PROBE_DONE line.
"""
import argparse
import sys

import torch
import ttnn


def bf16_from_bits(bits):
    b = torch.as_tensor(bits, dtype=torch.int64)
    b = torch.where(b >= 0x8000, b - 0x10000, b).to(torch.int16)
    return b.view(torch.bfloat16)


def bits_of(t):
    return t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=10000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--largest", type=int, default=1)
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--specials", type=str, default="7FC0,7FC1,FFC1,8000,0000,7F80,FF80,0001")
    ap.add_argument("--base", type=str, default="linspace", choices=["linspace", "zeros"])
    ap.add_argument("--print-bits", type=int, default=0, help="print first N value/index lanes of row 0")
    ap.add_argument("--repeat", type=int, default=1, help="run topk N times in one process (program-cache-hit path)")
    args = ap.parse_args()

    specials = [int(s, 16) for s in args.specials.split(",") if s]
    if args.base == "linspace":
        row = torch.linspace(-1, 1, args.w).to(torch.bfloat16)
    else:
        row = torch.zeros(args.w, dtype=torch.bfloat16)
    if specials:
        row[args.pos : args.pos + len(specials)] = bf16_from_bits(specials)
    x = row.expand(args.rows, args.w).clone().reshape(1, 1, args.rows, args.w)

    print(f"PROBE_START w={args.w} k={args.k} largest={args.largest} specials={args.specials} pos={args.pos}", flush=True)
    dev = ttnn.open_device(device_id=0)
    try:
        tt_in = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=dev)
        for it in range(args.repeat):
            vals, idx = ttnn.topk(tt_in, args.k, dim=-1, largest=bool(args.largest), sorted=True)
            print(f"DISPATCH_RETURNED iter={it}", flush=True)
            v = ttnn.to_torch(vals)
            print(f"VALUES_READ iter={it}", flush=True)
        i = ttnn.to_torch(idx, dtype=torch.uint16 if idx.dtype == ttnn.uint16 else torch.uint32).to(torch.int64)
        vb = bits_of(v).reshape(-1, args.k)
        ib = i.reshape(-1, args.k)
        if args.print_bits:
            n = min(args.print_bits, args.k)
            print("row0 val bits:", [f"{b:04x}" for b in vb[0, :n].tolist()], flush=True)
            print("row0 indices :", ib[0, :n].tolist(), flush=True)
            # gather check against input bits
            inb = bits_of(x.reshape(-1, args.w))
            g = []
            for lane in range(n):
                j = int(ib[0, lane])
                g.append(f"{inb[0, j]:04x}" if 0 <= j < args.w else "OOR")
            print("row0 in[idx] :", g, flush=True)
        rows_same = all(torch.equal(vb[r], vb[0]) for r in range(vb.shape[0]))
        print(f"rows_identical={rows_same}", flush=True)
        print("PROBE_DONE", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
